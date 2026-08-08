"""Host verification, systemd lifecycle, and formal pipeline execution."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    unit = f"{config.values['pipeline']['systemd_unit_prefix']}-{pipeline_id}"
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
