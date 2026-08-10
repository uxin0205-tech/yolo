"""Host verification, systemd lifecycle, and formal pipeline execution."""

from __future__ import annotations

import json
import importlib.metadata
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import faster_coco_eval
import torch
import ultralytics

from masf_yolo.artifacts.io import PipelineLock, atomic_write_json
from masf_yolo.contracts import (
    DatasetManifest,
    EnvironmentManifest,
    Phase1Config,
    sha256_file,
    sha256_value,
)

GPU_MIN_FREE_MIB = 24 * 1024
GPU_MAX_UTILIZATION_PERCENT = 10
GPU_READY_CONSECUTIVE_POLLS = 3
GPU_POLL_INTERVAL_SECONDS = 60
PREDECESSOR_POLL_INTERVAL_SECONDS = 60


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    free_memory_mib: int
    utilization_percent: int


def query_systemd_unit_active(unit: str) -> bool:
    """Return whether a user service is still active or transitioning to active."""
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() in {"active", "activating", "reloading"}


def wait_for_predecessor_units(
    units: tuple[str, ...],
    *,
    query: Callable[[str], bool] = query_systemd_unit_active,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_seconds: int = PREDECESSOR_POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Wait until every explicitly ordered predecessor service has finished."""
    if poll_interval_seconds < 1:
        raise ValueError("invalid predecessor polling policy")
    normalized = tuple(dict.fromkeys(unit.strip() for unit in units if unit.strip()))
    observations: list[dict[str, bool]] = []
    while True:
        states = {unit: query(unit) for unit in normalized}
        observations.append(states)
        if not any(states.values()):
            return {
                "units": list(normalized),
                "poll_interval_seconds": poll_interval_seconds,
                "observations": observations[-3:],
                "ready": True,
            }
        sleep(float(poll_interval_seconds))


def query_gpu_snapshot(device: int = 0) -> GpuSnapshot:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU status row, got {rows!r}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 2:
        raise RuntimeError(f"invalid GPU status row: {rows[0]!r}")
    return GpuSnapshot(free_memory_mib=int(fields[0]), utilization_percent=int(fields[1]))


def wait_for_gpu_idle(
    *,
    query: Callable[[], GpuSnapshot] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    minimum_free_mib: int = GPU_MIN_FREE_MIB,
    maximum_utilization_percent: int = GPU_MAX_UTILIZATION_PERCENT,
    consecutive_polls: int = GPU_READY_CONSECUTIVE_POLLS,
    poll_interval_seconds: int = GPU_POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    if minimum_free_mib < 1 or not 0 <= maximum_utilization_percent <= 100:
        raise ValueError("invalid GPU idle thresholds")
    if consecutive_polls < 1 or poll_interval_seconds < 1:
        raise ValueError("invalid GPU polling policy")
    query = query or query_gpu_snapshot
    ready_count = 0
    observations: list[dict[str, int | bool]] = []
    while ready_count < consecutive_polls:
        snapshot = query()
        ready = (
            snapshot.free_memory_mib >= minimum_free_mib
            and snapshot.utilization_percent <= maximum_utilization_percent
        )
        ready_count = ready_count + 1 if ready else 0
        observations.append(asdict(snapshot) | {"ready": ready})
        observations = observations[-consecutive_polls:]
        if ready_count < consecutive_polls:
            sleep(float(poll_interval_seconds))
    return {
        "device": 0,
        "minimum_free_mib": minimum_free_mib,
        "maximum_utilization_percent": maximum_utilization_percent,
        "consecutive_polls": consecutive_polls,
        "poll_interval_seconds": poll_interval_seconds,
        "observations": observations,
        "ready": True,
    }


def _load_config(config_path: Path) -> Phase1Config:
    from masf_yolo.cli import load_config

    return load_config(config_path)


def _work_root(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def _artifact_root(config_path: Path, config: Phase1Config) -> Path:
    return _work_root(config_path) / config.values["artifacts_root"]


def verify_environment(config_path: Path, *, require_cuda: bool) -> EnvironmentManifest:
    config = _load_config(config_path)
    values = config.values
    pins = values["environment"]
    actual = {
        "python": ".".join(platform.python_version().split(".")[:2]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "faster_coco_eval": faster_coco_eval.__version__,
        "onnx": importlib.metadata.version("onnx"),
        "ml_dtypes": importlib.metadata.version("ml_dtypes"),
        "protobuf": importlib.metadata.version("protobuf"),
    }
    for name, expected in pins.items():
        if actual[name] != expected:
            raise RuntimeError(f"environment pin mismatch for {name}: expected {expected}, got {actual[name]}")
    if require_cuda and (not torch.cuda.is_available() or torch.cuda.device_count() < 1):
        raise RuntimeError("CUDA device 0 is unavailable")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    package_root = Path(ultralytics.__file__).resolve().parent
    model_yaml = package_root / "cfg" / "models" / "11" / "yolo11.yaml"
    default_yaml = package_root / "cfg" / "default.yaml"
    work_root = _work_root(config_path)
    source_weights = (work_root / values["model"]["source_weights"]).resolve()
    hashes = {
        "official_model_yaml_hash": sha256_file(model_yaml),
        "official_default_yaml_hash": sha256_file(default_yaml),
        "source_weights_hash": sha256_file(source_weights),
    }
    expected_hashes = {
        "official_model_yaml_hash": values["model"]["official_model_yaml_sha256"],
        "official_default_yaml_hash": values["model"]["official_default_yaml_sha256"],
        "source_weights_hash": values["model"]["source_weights_sha256"],
    }
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise RuntimeError(f"pinned asset hash mismatch for {name}")
    return EnvironmentManifest(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        ultralytics=ultralytics.__version__,
        faster_coco_eval=faster_coco_eval.__version__,
        onnx=actual["onnx"],
        ml_dtypes=actual["ml_dtypes"],
        protobuf=actual["protobuf"],
        device_name=device_name,
        **hashes,
    )


def pipeline_identity(config_hash: str, data_hash: str, environment_hash: str) -> str:
    return sha256_value(
        {"config": config_hash, "data": data_hash, "environment": environment_hash}
    )[:12]


def launch_python_path(executable: str) -> Path:
    """Return an absolute interpreter path without dereferencing a venv symlink."""
    return Path(executable).absolute()


def _dataset_manifest(artifact_root: Path) -> DatasetManifest:
    path = artifact_root / "dataset" / "manifest.json"
    if not path.is_file():
        raise RuntimeError("dataset audit manifest is missing; run audit first")
    return DatasetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def pipeline_start(config_path: Path) -> dict[str, Any]:
    from masf_yolo.cli import build_systemd_command, ensure_tracked_clean

    config_path = config_path.resolve()
    config = _load_config(config_path)
    artifact_root = _artifact_root(config_path, config)
    dataset = _dataset_manifest(artifact_root)
    environment = verify_environment(config_path, require_cuda=True)
    pipeline_id = pipeline_identity(config.config_hash, dataset.dataset_hash, environment.manifest_hash)
    # A stable unit name makes it impossible to launch two MASF pipelines in parallel.
    unit = str(config.values["pipeline"]["systemd_unit_prefix"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    with PipelineLock(artifact_root / "start.lock"):
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "."],
            cwd=_work_root(config_path),
            check=True,
            capture_output=True,
            text=True,
        )
        ensure_tracked_clean(status.stdout)
        active = subprocess.run(
            ["systemctl", "--user", "is-active", f"{unit}.service"],
            check=False,
            capture_output=True,
            text=True,
        )
        if active.returncode == 0:
            metadata_path = artifact_root / "pipeline.json"
            if metadata_path.is_file():
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                if existing.get("pipeline_id") != pipeline_id:
                    raise RuntimeError("the single MASF service is active with a different pipeline identity")
            return {"pipeline_id": pipeline_id, "unit": f"{unit}.service", "active": True, "existing": True}
        atomic_write_json(artifact_root / "environment.json", environment.to_dict())
        metadata = {
            "pipeline_id": pipeline_id,
            "unit": f"{unit}.service",
            "config": str(config_path),
            "config_hash": config.config_hash,
            "data_hash": dataset.dataset_hash,
            "environment_hash": environment.manifest_hash,
        }
        atomic_write_json(artifact_root / "pipeline.json", metadata)
        command = build_systemd_command(
            config_path=config_path,
            unit=unit,
            python=launch_python_path(sys.executable),
        )
        launched = subprocess.run(command, check=True, capture_output=True, text=True)
    return metadata | {"active": True, "existing": False, "systemd_output": launched.stdout.strip()}


def pipeline_status(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    artifact_root = _artifact_root(config_path, config)
    metadata_path = artifact_root / "pipeline.json"
    if not metadata_path.is_file():
        return {"started": False}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    status = subprocess.run(
        ["systemctl", "--user", "is-active", metadata["unit"]],
        check=False,
        capture_output=True,
        text=True,
    )
    state_path = artifact_root / "state.json"
    return metadata | {
        "started": True,
        "active": status.returncode == 0,
        "service_state": status.stdout.strip(),
        "state": json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None,
    }


def execute_pipeline(config_path: Path) -> None:
    from masf_yolo.pipeline import execute_formal_pipeline

    execute_formal_pipeline(config_path.resolve())
