"""Fail-closed Full35 then Partial75 autonomous training queue."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .selection import phase_c_candidate, phase_gate

ARCHITECTURE_PREFIXES = {"full35": "a1-full35", "partial75": "a2-partial75"}
PREPARED_CHECKPOINTS = {
    "full35": Path("artifacts/prepared/a1-full35.pt"),
    "partial75": Path("artifacts/prepared/a2-partial75.pt"),
}
PHASE_C_PROFILE_PEAK_BYTES = {
    "full35": 18_397_301_760,
    "partial75": 17_761_890_304,
}
PHASE_C_HEADROOM_BYTES = 1 << 30
MIN_AVAILABLE_RAM_BYTES = 3 << 30
FRACTION03_MIN_AVAILABLE_RAM_BYTES = 8 << 30
FRACTION03_MIN_FREE_VRAM_BYTES = 12 << 30
FRACTION10_MIN_FREE_VRAM_BYTES = 9 << 30
RESOURCE_POLL_SECONDS = 30
FRACTION03_SOURCE_DESCRIPTORS = {
    "full35": Path("inputs/continuation/full35-accepted-a2/candidate.json"),
    "partial75": Path("inputs/continuation/partial75-accepted-a2/candidate.json"),
}
FULL35_SOURCE_A1_RUN = "a1-full35-phase-a1-rtx4080super-batch16-workers4-r1"


def _fixed_queue_settings(
    workers: int,
    *,
    training_batch: int = 16,
    nbs: int = 16,
    phase_c_training_batch: int | None = None,
    phase_c_nbs: int | None = None,
    validation_batch: int = 16,
    minimum_available_ram_bytes: int | None = None,
    minimum_free_vram_bytes: int = 0,
    fraction: float | None = None,
    amp: bool | None = None,
    phase_c_patience: int | None = None,
) -> dict[str, int | bool | float]:
    phase_c_training_batch = phase_c_training_batch or training_batch
    phase_c_nbs = phase_c_nbs or nbs
    if workers not in range(1, 9):
        raise ValueError("queue workers must be between 1 and 8")
    if training_batch < 1 or phase_c_training_batch < 1 or validation_batch < 1:
        raise ValueError("training and validation batches must be positive")
    if nbs < training_batch or nbs % training_batch:
        raise ValueError("nbs must be an integer multiple of the training batch")
    if phase_c_nbs < phase_c_training_batch or phase_c_nbs % phase_c_training_batch:
        raise ValueError("Phase C nbs must be an integer multiple of its training batch")
    if minimum_available_ram_bytes is None:
        minimum_available_ram_bytes = MIN_AVAILABLE_RAM_BYTES + max(0, workers - 4) * (1 << 30)
    if minimum_available_ram_bytes < 1 or minimum_free_vram_bytes < 0:
        raise ValueError("資源安全門檻不得為負數")
    if phase_c_patience is not None and phase_c_patience < 1:
        raise ValueError("Phase C patience 必須為正整數")
    result: dict[str, int | bool | float] = {
        "batch": training_batch,
        "nbs": nbs,
        "effective_batch": nbs,
        "gradient_accumulation": nbs != training_batch,
        "phase_c_batch": phase_c_training_batch,
        "phase_c_nbs": phase_c_nbs,
        "phase_c_effective_batch": phase_c_nbs,
        "phase_c_gradient_accumulation": phase_c_nbs != phase_c_training_batch,
        "validation_batch": validation_batch,
        "training_workers": workers,
        "in_training_validation_workers": 0,
        "maximum_concurrent_data_workers": workers,
        "standalone_validation_workers": workers,
        "minimum_available_ram_bytes": minimum_available_ram_bytes,
        "minimum_free_vram_bytes": minimum_free_vram_bytes,
    }
    if fraction is not None:
        result["fraction"] = fraction
    if amp is not None:
        result["amp"] = amp
    if phase_c_patience is not None:
        result["phase_c_patience"] = phase_c_patience
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_run_id(architecture: str, phase: str, run_tag: str) -> str:
    try:
        prefix = ARCHITECTURE_PREFIXES[architecture]
    except KeyError as exc:
        raise ValueError(f"unknown architecture: {architecture}") from exc
    if phase not in {"a1", "a2", "b", "c"}:
        raise ValueError(f"unknown phase: {phase}")
    return f"{prefix}-phase-{phase}-{run_tag}"


def phase_c_has_capacity(architecture: str, total_vram_bytes: int) -> bool:
    try:
        measured_peak = PHASE_C_PROFILE_PEAK_BYTES[architecture]
    except KeyError as exc:
        raise ValueError(f"unknown architecture: {architecture}") from exc
    return total_vram_bytes >= measured_peak + PHASE_C_HEADROOM_BYTES


class QueueJournal:
    """Atomically persist queue state and a compact event history."""

    def __init__(
        self,
        path: Path,
        *,
        run_tag: str,
        source_a1_run: str = FULL35_SOURCE_A1_RUN,
        workers: int = 4,
        training_batch: int = 16,
        nbs: int = 16,
        phase_c_training_batch: int | None = None,
        phase_c_nbs: int | None = None,
        validation_batch: int = 16,
        minimum_available_ram_bytes: int | None = None,
        minimum_free_vram_bytes: int = 0,
        fraction: float | None = None,
        amp: bool | None = None,
        phase_c_patience: int | None = None,
    ) -> None:
        self.path = path.resolve()
        self.workers = workers
        self.training_batch = training_batch
        self.nbs = nbs
        self.phase_c_training_batch = phase_c_training_batch or training_batch
        self.phase_c_nbs = phase_c_nbs or nbs
        self.validation_batch = validation_batch
        self.fixed_settings = _fixed_queue_settings(
            workers,
            training_batch=training_batch,
            nbs=nbs,
            phase_c_training_batch=self.phase_c_training_batch,
            phase_c_nbs=self.phase_c_nbs,
            validation_batch=validation_batch,
            minimum_available_ram_bytes=minimum_available_ram_bytes,
            minimum_free_vram_bytes=minimum_free_vram_bytes,
            fraction=fraction,
            amp=amp,
            phase_c_patience=phase_c_patience,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            if self.payload.get("run_tag") != run_tag:
                raise ValueError("existing queue state belongs to a different run tag")
        else:
            self.payload: dict[str, Any] = {
                "schema_version": 1,
                "created_at": _utc_now(),
                "run_tag": run_tag,
                "source_a1_run": source_a1_run,
                "fixed_settings": self.fixed_settings,
                "events": [],
            }

    def update(self, status: str, **values: Any) -> None:
        timestamp = _utc_now()
        self.payload.update(status=status, updated_at=timestamp, **values)
        event = {"at": timestamp, "status": status}
        if "current_job" in values:
            event["current_job"] = values["current_job"]
        self.payload.setdefault("events", []).append(event)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)


@dataclass(frozen=True)
class ValidatedCandidate:
    label: str
    float_checkpoint: Path
    bittrue_checkpoint: Path
    metrics_path: Path
    map50_95: float

    def payload(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("float_checkpoint", "bittrue_checkpoint", "metrics_path"):
            result[key] = str(result[key])
        return result


def _memory_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo does not report MemAvailable")


def _gpu_memory_bytes() -> tuple[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    try:
        free_mib, total_mib = (int(item.strip()) for item in line.split(","))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"無法解析 nvidia-smi 記憶體資訊：{line!r}") from exc
    return free_mib << 20, total_mib << 20


def _release_parent_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _wait_for_resources(
    journal: QueueJournal,
    job: str,
    *,
    require_vram: bool,
    waiting_status: str = "waiting_for_resources",
) -> dict[str, int | None]:
    required_ram = int(journal.fixed_settings["minimum_available_ram_bytes"])
    required_vram = int(journal.fixed_settings.get("minimum_free_vram_bytes", 0)) if require_vram else 0
    while True:
        available_ram = _memory_available_bytes()
        free_vram: int | None = None
        total_vram: int | None = None
        gpu_error: str | None = None
        if require_vram:
            try:
                free_vram, total_vram = _gpu_memory_bytes()
            except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                gpu_error = str(exc)
        if available_ram >= required_ram and (
            not require_vram or (free_vram is not None and free_vram >= required_vram)
        ):
            return {
                "available_ram_bytes": available_ram,
                "free_vram_bytes": free_vram,
                "total_vram_bytes": total_vram,
            }
        journal.update(
            waiting_status,
            current_job=job,
            available_ram_bytes=available_ram,
            required_ram_bytes=required_ram,
            free_vram_bytes=free_vram,
            total_vram_bytes=total_vram,
            required_free_vram_bytes=required_vram,
            gpu_probe_error=gpu_error,
        )
        ram_text = f"RAM {available_ram / (1 << 30):.2f}/{required_ram / (1 << 30):.2f} GiB"
        vram_text = ""
        if require_vram:
            current = "無法讀取" if free_vram is None else f"{free_vram / (1 << 30):.2f} GiB"
            vram_text = f"，VRAM {current}/{required_vram / (1 << 30):.2f} GiB"
        print(f"[{_utc_now()}] {job} 等待資源：{ram_text}{vram_text}", flush=True)
        time.sleep(RESOURCE_POLL_SECONDS)


def _run_worker(root: Path, journal: QueueJournal, job: str, *arguments: str) -> None:
    require_vram = bool(arguments and arguments[0] in {"train", "validate", "profile-phase-c"})
    resources = _wait_for_resources(journal, job, require_vram=require_vram)
    journal.update("running", current_job=job, **resources)
    print(
        f"[{_utc_now()}] 啟動 {job}；可用 RAM "
        f"{int(resources['available_ram_bytes']) / (1 << 30):.2f} GiB",
        flush=True,
    )
    environment = os.environ.copy()
    sources = [
        str((root / "src").resolve()),
        str((root.parents[1] / "yolo_attention/src").resolve()),
    ]
    inherited_pythonpath = environment.get("PYTHONPATH")
    if inherited_pythonpath:
        sources.append(inherited_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(sources)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "achitechure_1.queue_worker",
                "--project-root",
                str(root),
                "--workers",
                str(journal.workers),
                *arguments,
            ],
            cwd=root,
            env=environment,
            check=True,
        )
    finally:
        _release_parent_memory()
    recovered = _wait_for_resources(
        journal,
        f"recover:{job}",
        require_vram=require_vram,
        waiting_status="waiting_for_resource_recovery",
    )
    journal.update("running", current_job=f"recovered:{job}", **recovered)
    print(f"[{_utc_now()}] 完成 {job}，RAM/VRAM 已回到安全門檻", flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _verify_completed_run(
    root: Path,
    *,
    run_id: str,
    variant: str,
    phase: str,
    parent_checkpoint: Path | None = None,
    expected_batch: int = 16,
    expected_nbs: int = 16,
    expected_workers: int = 4,
    expected_validation_batch: int | None = None,
    expected_fraction: float | None = None,
    expected_amp: bool | None = None,
    expected_patience: int | None = None,
) -> Path:
    run_dir = root / "artifacts/runs" / run_id
    complete_path = run_dir / "training-complete.json"
    manifest_path = run_dir / "manifest.json"
    if run_dir.exists() and (not complete_path.is_file() or not manifest_path.is_file()):
        raise RuntimeError(f"incomplete run directory requires manual review: {run_dir}")
    if not complete_path.is_file():
        raise FileNotFoundError(complete_path)
    complete = _read_json(complete_path)
    manifest = _read_json(manifest_path)
    common = manifest.get("common", {})
    if complete.get("status") != "completed":
        raise RuntimeError(f"run is not completed: {run_id}")
    if manifest.get("variant") != variant or manifest.get("phase", {}).get("name") != phase:
        raise RuntimeError(f"run metadata does not match {variant}/{phase}: {run_id}")
    actual_settings = (common.get("batch"), common.get("nbs"), common.get("workers"))
    expected_settings = (expected_batch, expected_nbs, expected_workers)
    if actual_settings != expected_settings:
        raise RuntimeError(
            "run does not use "
            f"batch={expected_batch}, nbs={expected_nbs}, workers={expected_workers}: {run_id}"
        )
    if (
        expected_validation_batch is not None
        and manifest.get("in_training_validation_batch", common.get("batch"))
        != expected_validation_batch
    ):
        raise RuntimeError(
            "run does not use in-training validation "
            f"batch={expected_validation_batch}: {run_id}"
        )
    if expected_fraction is not None and not math.isclose(
        float(common.get("fraction", -1.0)), expected_fraction, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"run does not use fraction={expected_fraction}: {run_id}")
    if expected_amp is not None and common.get("amp") is not expected_amp:
        raise RuntimeError(f"run does not use amp={expected_amp}: {run_id}")
    if expected_patience is not None and manifest.get("phase", {}).get("patience") != expected_patience:
        raise RuntimeError(f"run does not use patience={expected_patience}: {run_id}")
    if parent_checkpoint is not None:
        parent = manifest.get("parent", {})
        if Path(parent.get("path", "")).resolve() != parent_checkpoint.resolve():
            raise RuntimeError(f"run parent path does not match accepted checkpoint: {run_id}")
        if parent.get("sha256") != _file_sha256(parent_checkpoint):
            raise RuntimeError(f"run parent checksum does not match accepted checkpoint: {run_id}")
    checkpoint = Path(complete.get("best_checkpoint", "")).resolve()
    if checkpoint != (run_dir / "ultralytics/weights/best.pt").resolve() or not checkpoint.is_file():
        raise RuntimeError(f"completed run has no valid best checkpoint: {run_id}")
    return checkpoint


def _train_phase(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    phase: str,
    parent: Path,
    run_tag: str,
) -> tuple[str, Path]:
    if phase == "c":
        phase_batch = journal.phase_c_training_batch
        phase_nbs = journal.phase_c_nbs
    else:
        phase_batch = journal.training_batch
        phase_nbs = journal.nbs
    run_id = _phase_run_id(architecture, phase, run_tag)
    run_dir = root / "artifacts/runs" / run_id
    complete = run_dir / "training-complete.json"
    if not complete.is_file():
        resume_incomplete = run_dir.exists()
        if resume_incomplete and not (
            (run_dir / "manifest.json").is_file()
            and (run_dir / "ultralytics/weights/last.pt").is_file()
        ):
            raise RuntimeError(f"incomplete run 無有效 last.pt，需人工檢查：{run_dir}")
        arguments = [
            "train",
            "--variant",
            architecture,
            "--phase",
            phase,
            "--weights",
            str(parent.resolve()),
            "--run-id",
            run_id,
            "--batch",
            str(phase_batch),
            "--nbs",
            str(phase_nbs),
            "--validation-batch",
            str(journal.validation_batch),
        ]
        if "fraction" in journal.fixed_settings:
            arguments.extend(("--fraction", str(journal.fixed_settings["fraction"])))
        if phase == "c" and "phase_c_patience" in journal.fixed_settings:
            arguments.extend(("--patience", str(journal.fixed_settings["phase_c_patience"])))
        if resume_incomplete:
            arguments.append("--resume-incomplete")
        _run_worker(
            root,
            journal,
            f"train:{architecture}:{phase}",
            *arguments,
        )
    checkpoint = _verify_completed_run(
        root,
        run_id=run_id,
        variant=architecture,
        phase=phase,
        parent_checkpoint=parent,
        expected_batch=phase_batch,
        expected_nbs=phase_nbs,
        expected_workers=journal.workers,
        expected_validation_batch=journal.validation_batch,
        expected_fraction=(
            float(journal.fixed_settings["fraction"])
            if "fraction" in journal.fixed_settings
            else None
        ),
        expected_amp=(
            bool(journal.fixed_settings["amp"])
            if "amp" in journal.fixed_settings
            else None
        ),
        expected_patience=(
            int(journal.fixed_settings["phase_c_patience"])
            if phase == "c" and "phase_c_patience" in journal.fixed_settings
            else None
        ),
    )
    return run_id, checkpoint


def _validate_candidate(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    phase: str,
    run_id: str,
    float_checkpoint: Path,
) -> ValidatedCandidate:
    bittrue = (root / "artifacts/checkpoints" / f"{run_id}-bittrue.pt").resolve()
    lineage_path = bittrue.with_suffix(".lineage.json")
    validation_dir = (root / "artifacts/validation" / f"{run_id}-bittrue").resolve()
    metrics_path = validation_dir / "metrics.json"
    if not bittrue.is_file():
        _run_worker(
            root,
            journal,
            f"materialize:{architecture}:{phase}",
            "materialize",
            "--checkpoint",
            str(float_checkpoint.resolve()),
            "--output",
            str(bittrue),
        )
        lineage = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "variant": architecture,
            "phase": phase,
            "float_checkpoint": {
                "path": str(float_checkpoint.resolve()),
                "sha256": _file_sha256(float_checkpoint),
            },
            "bittrue_checkpoint": {
                "path": str(bittrue),
                "sha256": _file_sha256(bittrue),
            },
        }
        temporary = lineage_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(lineage_path)
    require_lineage = "fraction" in journal.fixed_settings
    if lineage_path.is_file():
        lineage = _read_json(lineage_path)
        float_lineage = lineage.get("float_checkpoint", {})
        bittrue_lineage = lineage.get("bittrue_checkpoint", {})
        if (
            lineage.get("variant") != architecture
            or lineage.get("phase") != phase
            or Path(str(float_lineage.get("path", ""))).resolve() != float_checkpoint.resolve()
            or float_lineage.get("sha256") != _file_sha256(float_checkpoint)
            or Path(str(bittrue_lineage.get("path", ""))).resolve() != bittrue
            or bittrue_lineage.get("sha256") != _file_sha256(bittrue)
        ):
            raise RuntimeError(f"Bit-True checkpoint lineage 不符：{lineage_path}")
    elif require_lineage:
        raise RuntimeError(f"目前 queue 契約要求 Bit-True lineage：{lineage_path}")
    if not metrics_path.is_file():
        _run_worker(
            root,
            journal,
            f"validate:{architecture}:{phase}",
            "validate",
            "--checkpoint",
            str(bittrue),
            "--run-dir",
            str(validation_dir),
            "--batch",
            str(journal.validation_batch),
        )
    metrics = _read_json(metrics_path)
    recorded_checkpoint = Path(metrics.get("checkpoint", {}).get("path", "")).resolve()
    if (
        recorded_checkpoint != bittrue
        or metrics.get("checkpoint", {}).get("sha256") != _file_sha256(bittrue)
        or metrics.get("selection_backend") != "bit_true_pwl"
    ):
        raise RuntimeError(f"validation metadata does not match candidate: {metrics_path}")
    score = float(metrics["map50_95"])
    if not math.isfinite(score):
        raise FloatingPointError(f"non-finite mAP50-95 in {metrics_path}")
    candidate = ValidatedCandidate(
        label=f"{architecture}:{phase}",
        float_checkpoint=float_checkpoint.resolve(),
        bittrue_checkpoint=bittrue,
        metrics_path=metrics_path,
        map50_95=score,
    )
    journal.update(
        "running",
        current_job=f"accepted-metric:{architecture}:{phase}",
        candidate=candidate.payload(),
    )
    return candidate


def _run_and_validate(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    phase: str,
    parent: Path,
    run_tag: str,
) -> ValidatedCandidate:
    run_id, checkpoint = _train_phase(
        root,
        journal,
        architecture=architecture,
        phase=phase,
        parent=parent,
        run_tag=run_tag,
    )
    return _validate_candidate(
        root,
        journal,
        architecture=architecture,
        phase=phase,
        run_id=run_id,
        float_checkpoint=checkpoint,
    )


def _rollback_gate(
    journal: QueueJournal,
    *,
    architecture: str,
    phase: str,
    parent: ValidatedCandidate,
    child: ValidatedCandidate,
) -> ValidatedCandidate:
    selected = phase_gate(parent.label, parent.map50_95, child.label, child.map50_95)
    accepted = child if selected == child.label else parent
    journal.update(
        "running",
        current_job=f"gate:{architecture}:{phase}",
        gate={
            "policy": "rollback-0.001",
            "parent": parent.payload(),
            "child": child.payload(),
            "selected": accepted.payload(),
        },
    )
    return accepted


def _through_phase_b(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    a1: ValidatedCandidate,
    run_tag: str,
) -> ValidatedCandidate:
    a2 = _run_and_validate(
        root,
        journal,
        architecture=architecture,
        phase="a2",
        parent=a1.float_checkpoint,
        run_tag=run_tag,
    )
    accepted_a2 = _rollback_gate(journal, architecture=architecture, phase="a2", parent=a1, child=a2)
    phase_b = _run_and_validate(
        root,
        journal,
        architecture=architecture,
        phase="b",
        parent=accepted_a2.float_checkpoint,
        run_tag=run_tag,
    )
    return _rollback_gate(journal, architecture=architecture, phase="b", parent=accepted_a2, child=phase_b)


def _maybe_phase_c(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    parent: ValidatedCandidate,
    run_tag: str,
    gpu: dict[str, Any],
) -> tuple[ValidatedCandidate, dict[str, object] | None]:
    total = int(gpu["total_vram_bytes"])
    required = PHASE_C_PROFILE_PEAK_BYTES[architecture] + PHASE_C_HEADROOM_BYTES
    if not phase_c_has_capacity(architecture, total):
        pending = {
            "architecture": architecture,
            "phase": "c",
            "reason": "insufficient_vram_for_measured_batch16_peak_plus_headroom",
            "gpu": gpu,
            "required_vram_bytes": required,
            "parent": parent.payload(),
        }
        journal.update("running", current_job=f"deferred:{architecture}:c", deferred_phase_c=pending)
        return parent, pending
    child = _run_and_validate(
        root,
        journal,
        architecture=architecture,
        phase="c",
        parent=parent.float_checkpoint,
        run_tag=run_tag,
    )
    selected = phase_c_candidate(parent.label, parent.map50_95, child.label, child.map50_95)
    accepted = child if selected == child.label else parent
    journal.update(
        "running",
        current_job=f"gate:{architecture}:c",
        gate={
            "policy": "phase-c-strict-improvement",
            "parent": parent.payload(),
            "child": child.payload(),
            "selected": accepted.payload(),
        },
    )
    return accepted, None


def _probe_gpu(root: Path, journal: QueueJournal) -> dict[str, Any]:
    output = journal.path.parent / "gpu-probe.json"
    _run_worker(root, journal, "probe:gpu", "probe", "--output", str(output))
    return _read_json(output)


def _candidate_from_payload(payload: dict[str, Any]) -> ValidatedCandidate:
    candidate = ValidatedCandidate(
        label=str(payload["label"]),
        float_checkpoint=Path(payload["float_checkpoint"]).resolve(),
        bittrue_checkpoint=Path(payload["bittrue_checkpoint"]).resolve(),
        metrics_path=Path(payload["metrics_path"]).resolve(),
        map50_95=float(payload["map50_95"]),
    )
    for path in (candidate.float_checkpoint, candidate.bittrue_checkpoint, candidate.metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics = _read_json(candidate.metrics_path)
    if not math.isclose(float(metrics["map50_95"]), candidate.map50_95, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("source queue candidate score does not match its metrics file")
    return candidate


def _descriptor_checkpoint(descriptor: Path, architecture: str) -> tuple[Path, dict[str, Any]]:
    payload = _read_json(descriptor)
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"不支援的續跑來源格式：{descriptor}")
    if payload.get("variant") != architecture:
        raise RuntimeError(f"續跑來源架構不是 {architecture}：{descriptor}")
    if payload.get("boundary") != "accepted_after_phase_a2":
        raise RuntimeError(f"續跑來源未聲明為 accepted Phase A2 boundary：{descriptor}")
    checkpoint_payload = payload.get("float_checkpoint", {})
    if not isinstance(checkpoint_payload, dict):
        raise TypeError(f"float_checkpoint 必須是物件：{descriptor}")
    checkpoint = Path(str(checkpoint_payload.get("path", "")))
    if not checkpoint.is_absolute():
        checkpoint = descriptor.parent / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_sha256 = str(checkpoint_payload.get("sha256", ""))
    actual_sha256 = _file_sha256(checkpoint)
    if expected_sha256 != actual_sha256:
        raise RuntimeError(f"續跑 checkpoint SHA256 不符：{checkpoint}")
    return checkpoint, payload


def _inspect_source_checkpoint(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    checkpoint: Path,
) -> dict[str, Any]:
    checkpoint_sha256 = _file_sha256(checkpoint)
    output = (
        root
        / "artifacts/queues"
        / "source-inspection"
        / f"{architecture}-{checkpoint_sha256[:16]}.json"
    ).resolve()
    if not output.is_file():
        _run_worker(
            root,
            journal,
            f"inspect-source:{architecture}",
            "inspect-candidate",
            "--checkpoint",
            str(checkpoint),
            "--kind",
            "float",
            "--output",
            str(output),
        )
    inspection = _read_json(output)
    if (
        Path(str(inspection.get("checkpoint", ""))).resolve() != checkpoint.resolve()
        or inspection.get("kind") != "float"
        or inspection.get("variant") != architecture
        or inspection.get("attention_normalizations") != ["piecewise_linear", "piecewise_linear"]
    ):
        raise RuntimeError(f"續跑 checkpoint 檢查結果不符：{output}")
    return inspection


def _validate_source_candidate(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    checkpoint: Path,
    run_tag: str,
) -> ValidatedCandidate:
    prefix = ARCHITECTURE_PREFIXES[architecture]
    run_id = f"{prefix}-source-accepted-a2-{run_tag}"
    return _validate_candidate(
        root,
        journal,
        architecture=architecture,
        phase="source-a2",
        run_id=run_id,
        float_checkpoint=checkpoint,
    )


def _profile_phase_c_candidate(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    parent: ValidatedCandidate,
    run_tag: str,
    gpu: dict[str, Any],
    amp: bool = False,
) -> dict[str, Any]:
    """Run and validate a real-loss Phase-C microbatch capacity probe."""

    prefix = ARCHITECTURE_PREFIXES[architecture]
    amp_suffix = "-amp" if amp else ""
    output = (
        root / "artifacts/profiles" / f"{prefix}-phase-c-{run_tag}-training-smoke{amp_suffix}.json"
    ).resolve()
    if not output.is_file():
        profile_arguments = [
            "profile-phase-c",
            "--checkpoint",
            str(parent.float_checkpoint),
            "--output",
            str(output),
            "--batch",
            str(journal.phase_c_training_batch),
            "--accumulate",
            str(journal.phase_c_nbs // journal.phase_c_training_batch),
            "--steps",
            "2",
        ]
        if amp:
            profile_arguments.append("--amp")
        _run_worker(
            root,
            journal,
            f"profile:{architecture}:c",
            *profile_arguments,
        )
    profile = _read_json(output)
    expected_checkpoint = parent.float_checkpoint.resolve()
    recorded_checkpoint = Path(profile.get("checkpoint", "")).resolve()
    expected_accumulate = journal.phase_c_nbs // journal.phase_c_training_batch
    if profile.get("status") == "oom":
        if (
            recorded_checkpoint != expected_checkpoint
            or profile.get("batch") != journal.phase_c_training_batch
            or profile.get("accumulate") != expected_accumulate
            or profile.get("amp") is not amp
        ):
            raise RuntimeError(f"Phase-C OOM profile settings do not match recovery queue: {output}")
        return profile
    if profile.get("status") != "passed":
        raise RuntimeError(f"Phase-C profile has no successful terminal status: {output}")
    if (
        recorded_checkpoint != expected_checkpoint
        or profile.get("batch") != journal.phase_c_training_batch
        or profile.get("accumulate") != expected_accumulate
        or profile.get("effective_batch") != journal.phase_c_nbs
        or profile.get("amp") is not amp
    ):
        raise RuntimeError(f"Phase-C profile metadata does not match recovery queue: {output}")
    peak = int(profile["peak_vram_bytes"])
    required = peak + PHASE_C_HEADROOM_BYTES
    profile["profile_path"] = str(output)
    profile["required_vram_bytes"] = required
    profile["detected_total_vram_bytes"] = int(gpu["total_vram_bytes"])
    return profile


def _run_phase_c_recovery_candidate(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    parent: ValidatedCandidate,
    run_tag: str,
    gpu: dict[str, Any],
) -> tuple[ValidatedCandidate, dict[str, object] | None]:
    profile = _profile_phase_c_candidate(
        root,
        journal,
        architecture=architecture,
        parent=parent,
        run_tag=run_tag,
        gpu=gpu,
    )
    if profile.get("status") == "oom":
        pending: dict[str, object] = {
            "architecture": architecture,
            "phase": "c",
            "reason": "microbatch8_real_loss_profile_oom",
            "gpu": gpu,
            "profile": profile,
            "parent": parent.payload(),
        }
        journal.update("running", current_job=f"deferred:{architecture}:c", deferred_phase_c=pending)
        return parent, pending

    total = int(gpu["total_vram_bytes"])
    required = int(profile["required_vram_bytes"])
    journal.update(
        "running",
        current_job=f"profile-passed:{architecture}:c",
        phase_c_profile=profile,
    )
    if total < required:
        pending = {
            "architecture": architecture,
            "phase": "c",
            "reason": "insufficient_vram_for_microbatch8_profile_peak_plus_headroom",
            "gpu": gpu,
            "profile": profile,
            "required_vram_bytes": required,
            "parent": parent.payload(),
        }
        journal.update("running", current_job=f"deferred:{architecture}:c", deferred_phase_c=pending)
        return parent, pending

    child = _run_and_validate(
        root,
        journal,
        architecture=architecture,
        phase="c",
        parent=parent.float_checkpoint,
        run_tag=run_tag,
    )
    selected = phase_c_candidate(parent.label, parent.map50_95, child.label, child.map50_95)
    accepted = child if selected == child.label else parent
    journal.update(
        "running",
        current_job=f"gate:{architecture}:c",
        gate={
            "policy": "phase-c-strict-improvement",
            "parent": parent.payload(),
            "child": child.payload(),
            "selected": accepted.payload(),
        },
    )
    return accepted, None


def _run_fraction03_phase_c_candidate(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    parent: ValidatedCandidate,
    run_tag: str,
    gpu: dict[str, Any],
) -> tuple[ValidatedCandidate, dict[str, object] | None]:
    profile = _profile_phase_c_candidate(
        root,
        journal,
        architecture=architecture,
        parent=parent,
        run_tag=run_tag,
        gpu=gpu,
        amp=True,
    )
    if profile.get("status") == "oom":
        pending: dict[str, object] = {
            "architecture": architecture,
            "phase": "c",
            "reason": "formal_amp_real_loss_profile_oom",
            "gpu": gpu,
            "profile": profile,
            "parent": parent.payload(),
        }
        journal.update("running", current_job=f"deferred:{architecture}:c", deferred_phase_c=pending)
        return parent, pending

    total = int(gpu["total_vram_bytes"])
    required = int(profile["required_vram_bytes"])
    journal.update(
        "running",
        current_job=f"profile-passed:{architecture}:c",
        phase_c_profile=profile,
    )
    if total < required:
        pending = {
            "architecture": architecture,
            "phase": "c",
            "reason": "insufficient_vram_for_formal_amp_profile_peak_plus_headroom",
            "gpu": gpu,
            "profile": profile,
            "required_vram_bytes": required,
            "parent": parent.payload(),
        }
        journal.update("running", current_job=f"deferred:{architecture}:c", deferred_phase_c=pending)
        return parent, pending

    child = _run_and_validate(
        root,
        journal,
        architecture=architecture,
        phase="c",
        parent=parent.float_checkpoint,
        run_tag=run_tag,
    )
    selected = phase_c_candidate(parent.label, parent.map50_95, child.label, child.map50_95)
    accepted = child if selected == child.label else parent
    journal.update(
        "running",
        current_job=f"gate:{architecture}:c",
        gate={
            "policy": "phase-c-strict-improvement",
            "parent": parent.payload(),
            "child": child.payload(),
            "selected": accepted.payload(),
        },
    )
    return accepted, None


def _finish_after_full_b(
    root: Path,
    journal: QueueJournal,
    *,
    full_b: ValidatedCandidate,
    run_tag: str,
    gpu: dict[str, Any],
) -> None:
    full_final, full_pending = _maybe_phase_c(
        root,
        journal,
        architecture="full35",
        parent=full_b,
        run_tag=run_tag,
        gpu=gpu,
    )

    partial_a1 = _run_and_validate(
        root,
        journal,
        architecture="partial75",
        phase="a1",
        parent=(root / PREPARED_CHECKPOINTS["partial75"]).resolve(),
        run_tag=run_tag,
    )
    partial_b = _through_phase_b(
        root,
        journal,
        architecture="partial75",
        a1=partial_a1,
        run_tag=run_tag,
    )
    partial_final, partial_pending = _maybe_phase_c(
        root,
        journal,
        architecture="partial75",
        parent=partial_b,
        run_tag=run_tag,
        gpu=gpu,
    )

    pending = [item for item in (full_pending, partial_pending) if item is not None]
    accepted = {"full35": full_final.payload(), "partial75": partial_final.payload()}
    if pending:
        journal.update(
            "waiting_for_phase_c_gpu",
            current_job=None,
            accepted_candidates=accepted,
            pending_phase_c=pending,
        )
    else:
        journal.update("completed", current_job=None, accepted_candidates=accepted, pending_phase_c=[])


def run_architecture_queue(root: Path, run_tag: str, *, workers: int = 4) -> Path:
    """Run Full35 first, then Partial75, while deferring unsafe Phase-C jobs."""

    root = root.resolve()
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_tag) is None:
        raise ValueError(
            "run tag must contain only lowercase letters, digits, dots, underscores, and hyphens"
        )
    state_path = root / "artifacts/queues" / f"architecture-{run_tag}" / "state.json"
    journal = QueueJournal(state_path, run_tag=run_tag, workers=workers)
    try:
        journal.update(
            "starting",
            current_job="verify:inputs",
            error=None,
            fixed_settings=journal.fixed_settings,
        )
        for checkpoint in PREPARED_CHECKPOINTS.values():
            if not (root / checkpoint).is_file():
                raise FileNotFoundError(root / checkpoint)
        attention_source = root.parents[1] / "yolo_attention/src/yolo_attention"
        if not attention_source.is_dir():
            raise FileNotFoundError(attention_source)
        gpu = _probe_gpu(root, journal)
        journal.update("running", current_job="verify:full35:a1", gpu=gpu)

        full_a1_checkpoint = _verify_completed_run(
            root,
            run_id=FULL35_SOURCE_A1_RUN,
            variant="full35",
            phase="a1",
            parent_checkpoint=root / PREPARED_CHECKPOINTS["full35"],
        )
        full_a1 = _validate_candidate(
            root,
            journal,
            architecture="full35",
            phase="a1",
            run_id=FULL35_SOURCE_A1_RUN,
            float_checkpoint=full_a1_checkpoint,
        )
        full_b = _through_phase_b(
            root,
            journal,
            architecture="full35",
            a1=full_a1,
            run_tag=run_tag,
        )
        _finish_after_full_b(root, journal, full_b=full_b, run_tag=run_tag, gpu=gpu)
        return state_path
    except BaseException as exc:
        journal.update(
            "failed",
            current_job=None,
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise


def run_partial75_continuation_queue(
    root: Path,
    run_tag: str,
    *,
    source_full35_state: Path,
    workers: int = 8,
) -> Path:
    """Continue with Partial75 from a formally gated Full35 Phase-B candidate."""

    root = root.resolve()
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_tag) is None:
        raise ValueError(
            "run tag must contain only lowercase letters, digits, dots, underscores, and hyphens"
        )
    state_path = root / "artifacts/queues" / f"architecture-{run_tag}" / "state.json"
    journal = QueueJournal(state_path, run_tag=run_tag, workers=workers)
    try:
        journal.update(
            "starting",
            current_job="verify:full35-b-source",
            error=None,
            fixed_settings=journal.fixed_settings,
            source_full35_state=str(source_full35_state.resolve()),
        )
        source = _read_json(source_full35_state.resolve())
        gate = source.get("gate", {})
        child = gate.get("child", {})
        if gate.get("policy") != "rollback-0.001" or child.get("label") != "full35:b":
            raise RuntimeError("source queue has not completed the Full35 Phase-B rollback gate")
        full_b = _candidate_from_payload(gate["selected"])
        if not (root / PREPARED_CHECKPOINTS["partial75"]).is_file():
            raise FileNotFoundError(root / PREPARED_CHECKPOINTS["partial75"])
        gpu = _probe_gpu(root, journal)
        journal.update("running", current_job="accepted-source:full35:b", gpu=gpu, candidate=full_b.payload())
        _finish_after_full_b(root, journal, full_b=full_b, run_tag=run_tag, gpu=gpu)
        return state_path
    except BaseException as exc:
        journal.update(
            "failed",
            current_job=None,
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise


def run_phase_c_recovery_queue(
    root: Path,
    run_tag: str,
    *,
    source_state: Path,
    workers: int = 6,
) -> Path:
    """Try deferred Phase-C jobs with batch 8 and two-step accumulation."""

    root = root.resolve()
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_tag) is None:
        raise ValueError(
            "run tag must contain only lowercase letters, digits, dots, underscores, and hyphens"
        )
    state_path = root / "artifacts/queues" / f"phase-c-{run_tag}" / "state.json"
    journal = QueueJournal(
        state_path,
        run_tag=run_tag,
        workers=workers,
        training_batch=8,
        nbs=16,
        phase_c_training_batch=8,
        phase_c_nbs=16,
        validation_batch=16,
    )
    try:
        journal.update(
            "starting",
            current_job="verify:phase-c-source",
            error=None,
            fixed_settings=journal.fixed_settings,
            source_state=str(source_state.resolve()),
        )
        source = _read_json(source_state.resolve())
        if source.get("status") not in {"waiting_for_phase_c_gpu", "completed"}:
            raise RuntimeError("source queue has not reached a terminal Phase-C handoff state")
        accepted_payload = source.get("accepted_candidates", {})
        if set(accepted_payload) != {"full35", "partial75"}:
            raise RuntimeError("source queue does not contain both accepted architecture candidates")
        parents = {
            architecture: _candidate_from_payload(accepted_payload[architecture])
            for architecture in ("full35", "partial75")
        }
        gpu = _probe_gpu(root, journal)
        accepted: dict[str, dict[str, object]] = {}
        pending: list[dict[str, object]] = []
        for architecture in ("full35", "partial75"):
            final, deferred = _run_phase_c_recovery_candidate(
                root,
                journal,
                architecture=architecture,
                parent=parents[architecture],
                run_tag=run_tag,
                gpu=gpu,
            )
            accepted[architecture] = final.payload()
            if deferred is not None:
                pending.append(deferred)
        if pending:
            journal.update(
                "waiting_for_phase_c_capacity",
                current_job=None,
                accepted_candidates=accepted,
                pending_phase_c=pending,
            )
        else:
            journal.update(
                "completed",
                current_job=None,
                accepted_candidates=accepted,
                pending_phase_c=[],
            )
        return state_path
    except BaseException as exc:
        journal.update(
            "failed",
            current_job=None,
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise


def run_fraction03_continuation_queue(
    root: Path,
    run_tag: str,
    *,
    full35_source: Path | None = None,
    partial75_source: Path | None = None,
    workers: int = 6,
) -> Path:
    """由兩個 accepted-A2 checkpoint 對稱續跑 B，再依序續跑 C。"""

    root = root.resolve()
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_tag) is None:
        raise ValueError("run tag 只能包含小寫英數字、點、底線與連字號")
    descriptors = {
        "full35": (full35_source or (root / FRACTION03_SOURCE_DESCRIPTORS["full35"])).resolve(),
        "partial75": (
            partial75_source or (root / FRACTION03_SOURCE_DESCRIPTORS["partial75"])
        ).resolve(),
    }
    state_path = root / "artifacts/queues" / f"fraction03-{run_tag}" / "state.json"
    journal = QueueJournal(
        state_path,
        run_tag=run_tag,
        source_a1_run="external-accepted-a2-checkpoints",
        workers=workers,
        training_batch=16,
        nbs=16,
        phase_c_training_batch=8,
        phase_c_nbs=16,
        validation_batch=8,
        minimum_available_ram_bytes=FRACTION03_MIN_AVAILABLE_RAM_BYTES,
        minimum_free_vram_bytes=FRACTION03_MIN_FREE_VRAM_BYTES,
        fraction=0.3,
        amp=True,
    )
    try:
        if journal.payload.get("fixed_settings") != journal.fixed_settings:
            raise RuntimeError("既有 queue state 的固定設定與本次 fraction=0.3 契約不同")
        journal.update(
            "starting",
            current_job="verify:contract-and-inputs",
            error=None,
            fixed_settings=journal.fixed_settings,
            source_descriptors={key: str(value) for key, value in descriptors.items()},
            execution_order=["full35:b", "partial75:b", "full35:c", "partial75:c"],
        )

        from .config import CommonTrainingConfig

        common = CommonTrainingConfig.from_yaml(root / "configs/training/common.yaml")
        contract = (common.batch, common.nbs, common.workers, common.fraction, common.amp)
        expected_contract = (16, 16, workers, 0.3, True)
        if contract != expected_contract or common.gradient_accumulation:
            raise RuntimeError(
                "common.yaml 不符合 queue 契約；預期 "
                f"A1/A2/B batch=16, nbs=16, workers={workers}, fraction=0.3, amp=true"
            )

        missing = [str(path) for path in descriptors.values() if not path.is_file()]
        if missing:
            journal.update(
                "waiting_for_inputs",
                current_job=None,
                missing_inputs=missing,
                next_action="匯入 Full35 與 Partial75 的 accepted Phase A2 Float checkpoint",
            )
            return state_path

        source_checkpoints: dict[str, Path] = {}
        source_metadata: dict[str, dict[str, Any]] = {}
        try:
            for architecture in ("full35", "partial75"):
                checkpoint, metadata = _descriptor_checkpoint(descriptors[architecture], architecture)
                source_checkpoints[architecture] = checkpoint
                source_metadata[architecture] = metadata
        except FileNotFoundError as exc:
            journal.update(
                "waiting_for_inputs",
                current_job=None,
                missing_inputs=[str(exc.filename or exc)],
                next_action="補齊 candidate.json 所指向的 accepted Phase A2 Float checkpoint",
            )
            return state_path

        attention_source = root.parents[1] / "yolo_attention/src/yolo_attention"
        if not attention_source.is_dir():
            raise FileNotFoundError(attention_source)
        inspections = {
            architecture: _inspect_source_checkpoint(
                root,
                journal,
                architecture=architecture,
                checkpoint=source_checkpoints[architecture],
            )
            for architecture in ("full35", "partial75")
        }
        journal.update(
            "running",
            current_job="validate:accepted-a2-sources",
            source_metadata=source_metadata,
            source_inspections=inspections,
        )

        gpu = _probe_gpu(root, journal)
        source_candidates = {
            architecture: _validate_source_candidate(
                root,
                journal,
                architecture=architecture,
                checkpoint=source_checkpoints[architecture],
                run_tag=run_tag,
            )
            for architecture in ("full35", "partial75")
        }
        journal.update(
            "running",
            current_job="accepted-sources",
            gpu=gpu,
            accepted_a2_sources={
                architecture: candidate.payload()
                for architecture, candidate in source_candidates.items()
            },
        )

        accepted_b: dict[str, ValidatedCandidate] = {}
        for architecture in ("full35", "partial75"):
            child = _run_and_validate(
                root,
                journal,
                architecture=architecture,
                phase="b",
                parent=source_candidates[architecture].float_checkpoint,
                run_tag=run_tag,
            )
            accepted_b[architecture] = _rollback_gate(
                journal,
                architecture=architecture,
                phase="b",
                parent=source_candidates[architecture],
                child=child,
            )

        accepted_final: dict[str, dict[str, object]] = {}
        pending: list[dict[str, object]] = []
        for architecture in ("full35", "partial75"):
            final, deferred = _run_fraction03_phase_c_candidate(
                root,
                journal,
                architecture=architecture,
                parent=accepted_b[architecture],
                run_tag=run_tag,
                gpu=gpu,
            )
            accepted_final[architecture] = final.payload()
            if deferred is not None:
                pending.append(deferred)
        if pending:
            journal.update(
                "waiting_for_phase_c_capacity",
                current_job=None,
                accepted_candidates=accepted_final,
                pending_phase_c=pending,
            )
        else:
            journal.update(
                "completed",
                current_job=None,
                accepted_candidates=accepted_final,
                pending_phase_c=[],
            )
        return state_path
    except BaseException as exc:
        journal.update(
            "failed",
            current_job=None,
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise


def run_fraction10_phase_b_control_queue(
    root: Path,
    run_tag: str,
    *,
    full35_source: Path | None = None,
    partial75_source: Path | None = None,
    workers: int = 6,
    minimum_free_vram_bytes: int = FRACTION03_MIN_FREE_VRAM_BYTES,
) -> Path:
    """由相同 accepted-A2 checkpoint 以 fraction=1.0 對稱重跑 Phase B。"""

    root = root.resolve()
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_tag) is None:
        raise ValueError("run tag 只能包含小寫英數字、點、底線與連字號")
    descriptors = {
        "full35": (full35_source or (root / FRACTION03_SOURCE_DESCRIPTORS["full35"])).resolve(),
        "partial75": (
            partial75_source or (root / FRACTION03_SOURCE_DESCRIPTORS["partial75"])
        ).resolve(),
    }
    state_path = root / "artifacts/queues" / f"fraction10-phase-b-{run_tag}" / "state.json"
    journal = QueueJournal(
        state_path,
        run_tag=run_tag,
        source_a1_run="external-accepted-a2-checkpoints",
        workers=workers,
        training_batch=16,
        nbs=16,
        phase_c_training_batch=16,
        phase_c_nbs=16,
        validation_batch=8,
        minimum_available_ram_bytes=FRACTION03_MIN_AVAILABLE_RAM_BYTES,
        minimum_free_vram_bytes=minimum_free_vram_bytes,
        fraction=1.0,
        amp=True,
    )
    try:
        existing_settings = journal.payload.get("fixed_settings")
        if existing_settings != journal.fixed_settings:
            comparable_settings = dict(journal.fixed_settings)
            if isinstance(existing_settings, dict):
                comparable_settings["minimum_free_vram_bytes"] = existing_settings.get(
                    "minimum_free_vram_bytes"
                )
            only_vram_threshold_changed = existing_settings == comparable_settings
            previous_threshold = (
                int(existing_settings["minimum_free_vram_bytes"])
                if isinstance(existing_settings, dict)
                and "minimum_free_vram_bytes" in existing_settings
                else None
            )
            if (
                not only_vram_threshold_changed
                or previous_threshold is None
                or minimum_free_vram_bytes >= previous_threshold
            ):
                raise RuntimeError("既有 queue state 的固定設定與本次 fraction=1.0 契約不同")
            journal.payload["fixed_settings"] = journal.fixed_settings
            journal.update(
                "resource_threshold_overridden",
                current_job=None,
                previous_minimum_free_vram_bytes=previous_threshold,
                minimum_free_vram_bytes=minimum_free_vram_bytes,
                override_reason="使用者要求在外部 GPU 工作負載存在時直接續跑",
            )
        journal.update(
            "starting",
            current_job="verify:contract-and-inputs",
            error=None,
            fixed_settings=journal.fixed_settings,
            source_descriptors={key: str(value) for key, value in descriptors.items()},
            execution_order=["full35:b", "partial75:b"],
            purpose="以相同 accepted A2 與 Phase-B 超參數隔離 fraction=0.3 的影響",
        )

        from .config import CommonTrainingConfig

        common = CommonTrainingConfig.from_yaml(root / "configs/training/common.yaml")
        base_contract = (common.batch, common.nbs, common.workers, common.amp)
        expected_base_contract = (16, 16, workers, True)
        if base_contract != expected_base_contract or common.gradient_accumulation:
            raise RuntimeError(
                "common.yaml 不符合 Phase-B 對照基礎契約；預期 "
                f"batch=16, nbs=16, workers={workers}, amp=true"
            )

        missing = [str(path) for path in descriptors.values() if not path.is_file()]
        if missing:
            journal.update(
                "waiting_for_inputs",
                current_job=None,
                missing_inputs=missing,
                next_action="補齊 Full35 與 Partial75 accepted Phase A2 descriptor",
            )
            return state_path

        source_checkpoints: dict[str, Path] = {}
        source_metadata: dict[str, dict[str, Any]] = {}
        for architecture in ("full35", "partial75"):
            checkpoint, metadata = _descriptor_checkpoint(descriptors[architecture], architecture)
            source_checkpoints[architecture] = checkpoint
            source_metadata[architecture] = metadata

        attention_source = root.parents[1] / "yolo_attention/src/yolo_attention"
        if not attention_source.is_dir():
            raise FileNotFoundError(attention_source)
        inspections = {
            architecture: _inspect_source_checkpoint(
                root,
                journal,
                architecture=architecture,
                checkpoint=source_checkpoints[architecture],
            )
            for architecture in ("full35", "partial75")
        }
        journal.update(
            "running",
            current_job="validate:accepted-a2-sources",
            source_metadata=source_metadata,
            source_inspections=inspections,
            base_common_fraction=common.fraction,
            overridden_training_fraction=1.0,
        )

        gpu = _probe_gpu(root, journal)
        source_candidates = {
            architecture: _validate_source_candidate(
                root,
                journal,
                architecture=architecture,
                checkpoint=source_checkpoints[architecture],
                run_tag=run_tag,
            )
            for architecture in ("full35", "partial75")
        }
        journal.update(
            "running",
            current_job="accepted-sources",
            gpu=gpu,
            accepted_a2_sources={
                architecture: candidate.payload()
                for architecture, candidate in source_candidates.items()
            },
        )

        accepted: dict[str, dict[str, object]] = {}
        phase_b_candidates: dict[str, dict[str, object]] = {}
        for architecture in ("full35", "partial75"):
            child = _run_and_validate(
                root,
                journal,
                architecture=architecture,
                phase="b",
                parent=source_candidates[architecture].float_checkpoint,
                run_tag=run_tag,
            )
            selected = _rollback_gate(
                journal,
                architecture=architecture,
                phase="b",
                parent=source_candidates[architecture],
                child=child,
            )
            phase_b_candidates[architecture] = child.payload()
            accepted[architecture] = selected.payload()
        journal.update(
            "completed",
            current_job=None,
            phase_b_candidates=phase_b_candidates,
            accepted_candidates=accepted,
            pending_phase_c=[],
        )
        return state_path
    except BaseException as exc:
        journal.update(
            "failed",
            current_job=None,
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise


def run_fraction10_phase_c_queue(
    root: Path,
    run_tag: str,
    *,
    source_state: Path,
    workers: int = 6,
    patience: int = 7,
    minimum_free_vram_bytes: int = FRACTION10_MIN_FREE_VRAM_BYTES,
) -> Path:
    """從 fraction=1.0 Phase-B gate 結果執行 C batch8×accumulate2。"""

    root = root.resolve()
    source_state = source_state.resolve()
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_tag) is None:
        raise ValueError("run tag 只能包含小寫英數字、點、底線與連字號")
    if patience < 1:
        raise ValueError("patience 必須為正整數")
    state_path = root / "artifacts/queues" / f"fraction10-phase-c-{run_tag}" / "state.json"
    journal = QueueJournal(
        state_path,
        run_tag=run_tag,
        source_a1_run="fraction10-phase-b-gated-candidates",
        workers=workers,
        training_batch=8,
        nbs=16,
        phase_c_training_batch=8,
        phase_c_nbs=16,
        validation_batch=8,
        minimum_available_ram_bytes=FRACTION03_MIN_AVAILABLE_RAM_BYTES,
        minimum_free_vram_bytes=minimum_free_vram_bytes,
        fraction=1.0,
        amp=True,
        phase_c_patience=patience,
    )
    try:
        if journal.payload.get("fixed_settings") != journal.fixed_settings:
            raise RuntimeError("既有 queue state 與本次 fraction=1.0 Phase-C 契約不同")
        journal.update(
            "starting",
            current_job="verify:phase-b-source-and-contract",
            error=None,
            fixed_settings=journal.fixed_settings,
            source_state=str(source_state),
            execution_order=["full35:c", "partial75:c"],
            purpose="fraction=1.0 Phase C，batch8×accumulate2，patience=7",
        )
        if not source_state.is_file():
            raise FileNotFoundError(source_state)
        source = _read_json(source_state)
        if source.get("status") != "completed":
            raise RuntimeError("fraction=1.0 Phase-B source queue 尚未完成")
        source_settings = source.get("fixed_settings", {})
        if (
            float(source_settings.get("fraction", -1.0)) != 1.0
            or source_settings.get("batch") != 16
            or source_settings.get("nbs") != 16
        ):
            raise RuntimeError("來源 queue 不是 fraction=1.0、Phase-B batch16 契約")
        accepted_payload = source.get("accepted_candidates", {})
        if set(accepted_payload) != {"full35", "partial75"}:
            raise RuntimeError("來源 queue 缺少兩個架構的 accepted candidate")
        parents = {
            architecture: _candidate_from_payload(accepted_payload[architecture])
            for architecture in ("full35", "partial75")
        }

        from .config import CommonTrainingConfig

        common = CommonTrainingConfig.from_yaml(root / "configs/training/common.yaml")
        if common.workers != workers or common.amp is not True:
            raise RuntimeError(f"common.yaml 必須維持 workers={workers}、amp=true")
        gpu = _probe_gpu(root, journal)
        journal.update(
            "running",
            current_job="accepted-phase-b-gates",
            gpu=gpu,
            phase_b_candidates=source.get("phase_b_candidates"),
            accepted_phase_b_candidates={
                architecture: candidate.payload()
                for architecture, candidate in parents.items()
            },
        )

        accepted: dict[str, dict[str, object]] = {}
        phase_c_candidates: dict[str, dict[str, object]] = {}
        pending: list[dict[str, object]] = []
        for architecture in ("full35", "partial75"):
            final, deferred = _run_fraction03_phase_c_candidate(
                root,
                journal,
                architecture=architecture,
                parent=parents[architecture],
                run_tag=run_tag,
                gpu=gpu,
            )
            accepted[architecture] = final.payload()
            if deferred is None:
                candidate_payload = journal.payload.get("gate", {}).get("child")
                if isinstance(candidate_payload, dict):
                    phase_c_candidates[architecture] = candidate_payload
            else:
                pending.append(deferred)
        if pending:
            journal.update(
                "waiting_for_phase_c_capacity",
                current_job=None,
                phase_c_candidates=phase_c_candidates,
                accepted_candidates=accepted,
                pending_phase_c=pending,
            )
        else:
            journal.update(
                "completed",
                current_job=None,
                phase_c_candidates=phase_c_candidates,
                accepted_candidates=accepted,
                pending_phase_c=[],
            )
        return state_path
    except BaseException as exc:
        journal.update(
            "failed",
            current_job=None,
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise
