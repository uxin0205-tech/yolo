"""Fail-closed Full35 then Partial75 autonomous training queue."""

from __future__ import annotations

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
FULL35_SOURCE_A1_RUN = "a1-full35-phase-a1-rtx4080super-batch16-workers4-r1"


def _fixed_queue_settings(
    workers: int,
    *,
    training_batch: int = 16,
    nbs: int = 16,
    validation_batch: int = 16,
) -> dict[str, int | bool]:
    if workers not in range(1, 9):
        raise ValueError("queue workers must be between 1 and 8")
    if training_batch < 1 or validation_batch < 1:
        raise ValueError("training and validation batches must be positive")
    if nbs < training_batch or nbs % training_batch:
        raise ValueError("nbs must be an integer multiple of the training batch")
    return {
        "batch": training_batch,
        "nbs": nbs,
        "effective_batch": nbs,
        "gradient_accumulation": nbs != training_batch,
        "validation_batch": validation_batch,
        "training_workers": workers,
        "in_training_validation_workers": 0,
        "maximum_concurrent_data_workers": workers,
        "standalone_validation_workers": workers,
        "minimum_available_ram_bytes": MIN_AVAILABLE_RAM_BYTES + max(0, workers - 4) * (1 << 30),
    }


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
        validation_batch: int = 16,
    ) -> None:
        self.path = path.resolve()
        self.workers = workers
        self.training_batch = training_batch
        self.nbs = nbs
        self.validation_batch = validation_batch
        self.fixed_settings = _fixed_queue_settings(
            workers,
            training_batch=training_batch,
            nbs=nbs,
            validation_batch=validation_batch,
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


def _wait_for_ram(journal: QueueJournal, job: str) -> None:
    required = journal.fixed_settings["minimum_available_ram_bytes"]
    while True:
        available = _memory_available_bytes()
        if available >= required:
            return
        journal.update(
            "waiting_for_ram",
            current_job=job,
            available_ram_bytes=available,
            required_ram_bytes=required,
        )
        print(
            f"[{_utc_now()}] waiting for RAM before {job}: "
            f"{available / (1 << 30):.2f} GiB available",
            flush=True,
        )
        time.sleep(60)


def _run_worker(root: Path, journal: QueueJournal, job: str, *arguments: str) -> None:
    _wait_for_ram(journal, job)
    available = _memory_available_bytes()
    journal.update("running", current_job=job, available_ram_bytes=available)
    print(f"[{_utc_now()}] starting {job} with {available / (1 << 30):.2f} GiB RAM available", flush=True)
    environment = os.environ.copy()
    sources = [
        str((root / "src").resolve()),
        str((root.parents[1] / "yolo_attention/src").resolve()),
    ]
    inherited_pythonpath = environment.get("PYTHONPATH")
    if inherited_pythonpath:
        sources.append(inherited_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(sources)
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
    print(f"[{_utc_now()}] completed {job}", flush=True)


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
    run_id = _phase_run_id(architecture, phase, run_tag)
    complete = root / "artifacts/runs" / run_id / "training-complete.json"
    if not complete.is_file():
        _run_worker(
            root,
            journal,
            f"train:{architecture}:{phase}",
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
            str(journal.training_batch),
            "--nbs",
            str(journal.nbs),
        )
    checkpoint = _verify_completed_run(
        root,
        run_id=run_id,
        variant=architecture,
        phase=phase,
        parent_checkpoint=parent,
        expected_batch=journal.training_batch,
        expected_nbs=journal.nbs,
        expected_workers=journal.workers,
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
    if recorded_checkpoint != bittrue or metrics.get("selection_backend") != "bit_true_pwl":
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


def _profile_phase_c_candidate(
    root: Path,
    journal: QueueJournal,
    *,
    architecture: str,
    parent: ValidatedCandidate,
    run_tag: str,
    gpu: dict[str, Any],
) -> dict[str, Any]:
    """Run and validate a real-loss Phase-C microbatch capacity probe."""

    prefix = ARCHITECTURE_PREFIXES[architecture]
    output = (
        root / "artifacts/profiles" / f"{prefix}-phase-c-{run_tag}-training-smoke.json"
    ).resolve()
    if not output.is_file():
        _run_worker(
            root,
            journal,
            f"profile:{architecture}:c",
            "profile-phase-c",
            "--checkpoint",
            str(parent.float_checkpoint),
            "--output",
            str(output),
            "--batch",
            str(journal.training_batch),
            "--accumulate",
            str(journal.nbs // journal.training_batch),
            "--steps",
            "2",
        )
    profile = _read_json(output)
    expected_checkpoint = parent.float_checkpoint.resolve()
    recorded_checkpoint = Path(profile.get("checkpoint", "")).resolve()
    expected_accumulate = journal.nbs // journal.training_batch
    if profile.get("status") == "oom":
        if (
            profile.get("batch") != journal.training_batch
            or profile.get("accumulate") != expected_accumulate
        ):
            raise RuntimeError(f"Phase-C OOM profile settings do not match recovery queue: {output}")
        return profile
    if profile.get("status") != "passed":
        raise RuntimeError(f"Phase-C profile has no successful terminal status: {output}")
    if (
        recorded_checkpoint != expected_checkpoint
        or profile.get("batch") != journal.training_batch
        or profile.get("accumulate") != expected_accumulate
        or profile.get("effective_batch") != journal.nbs
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
