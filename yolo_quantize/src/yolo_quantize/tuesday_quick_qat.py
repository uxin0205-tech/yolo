"""Materialize and execute the V34 qSiLU head-three LSQ-SD4 paired QAT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .progressive_queue import _foreign_gpu_pids
from .qat_plan import Full35QATPlan
from .qat_runtime import Full35QATRuntime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_PLAN = (
    PROJECT_ROOT
    / "artifacts/queues/v30-qsilu-complete-quantization-lane-v1/generated/"
    "v30-qsilu-all-w8-qat-pilot-v1.yaml"
)
BASE_PLAN_SHA256 = "25f8f50a4a0e83b14716ecf48d0e6411d89b2dc171cdb6bb1b7b991a8ad0035c"
PTQ_PLAN = (
    PROJECT_ROOT
    / "artifacts/queues/v34-qsilu-tuesday-quick-v1/generated/"
    "v34-qsilu-layer-and-structured-ptq-v1.yaml"
)
PTQ_PLAN_SHA256 = "3b29ad7a36998534615ce6e6d070d842a1f1c6f4668b88fc9562001c556fe469"
PTQ_REPORT = (
    PROJECT_ROOT / "artifacts/reports/v34-qsilu-layer-and-structured-ptq-v1.json"
)
PTQ_REPORT_SHA256 = "dfa12f851d0d659a45a80c658c7884f96a63b08dc234caf1cf9d706d05358b83"
DUAL_REPORT = (
    PROJECT_ROOT
    / "artifacts/reports/v34-qsilu-layer-and-structured-ptq-dual-v1.json"
)
DUAL_REPORT_SHA256 = "552633cd3c76affde781d5134411f3975ba4ce374459f615fe3b4ebe4319ce83"
CANDIDATE_ID = "fixed-sd4-head-three"
PLAN_ID = "v34-qsilu-head-three-lsq-sd4-short-qat-v1"
QUEUE_ROOT = (
    PROJECT_ROOT
    / "artifacts/queues/v34-qsilu-tuesday-quick-v1/"
    "short-qat-head-three-lsq-sd4-v1"
)
GENERATED_PLAN = QUEUE_ROOT / "generated/qat-plan.yaml"
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/qat/v34-qsilu-head-three-lsq-sd4-short-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 drifted: {actual} != {expected}")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _write_same_or_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite drifted artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def _transition(status: str, **values: object) -> None:
    payload = {
        "schema_version": 1,
        "status": status,
        "current_index": 0,
        "current_candidate": CANDIDATE_ID,
        "current_arm": values.pop("current_arm", None),
        "completed_jobs": values.pop("completed_jobs", 0),
        "error": values.pop("error", None),
        "time_unix": time.time(),
        "values": values,
    }
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    with (QUEUE_ROOT / "execution-events.jsonl").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    _atomic_json(QUEUE_ROOT / "execution-status.json", payload)


def _lsq_sd4_assignments(ptq: Mapping[str, object]) -> list[dict[str, object]]:
    results = _mapping(ptq.get("results"), "V34 PTQ results")
    candidate = _mapping(results.get(CANDIDATE_ID), "V34 head-three candidate")
    gate = _mapping(candidate.get("gate"), "V34 head-three gate")
    if candidate.get("status") != "completed" or gate.get("decision") != "green":
        raise ValueError("V34 head-three candidate is not completed green evidence")
    weight = _mapping(candidate.get("weight_quantization"), "V34 weight evidence")
    raw_assignments = weight.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ValueError("V34 head-three candidate has no path assignments")
    assignments: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_assignments:
        record = _mapping(raw, "V34 PTQ assignment")
        format_record = _mapping(record.get("format"), "V34 PTQ format")
        selector = _mapping(record.get("selector"), "V34 PTQ selector")
        paths = selector.get("paths")
        if (
            format_record.get("format_id") != "fixed-sd4"
            or format_record.get("scale_method") != "optimal_scaled_codebook"
            or selector.get("kind") != "paths"
            or not isinstance(paths, list)
            or not paths
        ):
            raise ValueError("V34 head-three PTQ assignment semantics drifted")
        encoded_paths = [str(path) for path in paths]
        if seen.intersection(encoded_paths):
            raise ValueError("V34 head-three PTQ paths overlap")
        seen.update(encoded_paths)
        assignments.append(
            {
                "region": str(record["region"]),
                "paths": encoded_paths,
                "format": {
                    "family": "ls_sd4",
                    "scale_method": "optimal_scaled_codebook",
                },
            }
        )
    if len(seen) != 8:
        raise ValueError("V34 head-three LSQ-SD4 route must contain exactly 8 layers")
    return assignments


def build_qat_plan_payload() -> dict[str, Any]:
    """Derive one 4-epoch matched QAT plan from immutable V34 PTQ evidence."""

    for path, expected, label in (
        (BASE_PLAN, BASE_PLAN_SHA256, "V30 QAT base plan"),
        (PTQ_PLAN, PTQ_PLAN_SHA256, "V34 PTQ plan"),
        (PTQ_REPORT, PTQ_REPORT_SHA256, "V34 PTQ report"),
        (DUAL_REPORT, DUAL_REPORT_SHA256, "V34 dual report"),
    ):
        _verified(path, expected, label)
    payload = yaml.safe_load(BASE_PLAN.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V30 QAT base plan must be a mapping")
    ptq = _mapping(
        json.loads(PTQ_REPORT.read_text(encoding="utf-8")), "V34 PTQ report"
    )
    dual = _mapping(
        json.loads(DUAL_REPORT.read_text(encoding="utf-8")), "V34 dual report"
    )
    dual_candidate = _mapping(
        _mapping(dual.get("candidates"), "V34 dual candidates").get(CANDIDATE_ID),
        "V34 head-three dual result",
    )
    if (
        ptq.get("status") != "completed"
        or dual.get("status") != "completed"
        or dual_candidate.get("decision") != "green"
        or _mapping(dual.get("source"), "V34 dual source").get("report_sha256")
        != PTQ_REPORT_SHA256
    ):
        raise ValueError("V34 head-three dual evidence is not completed green")

    payload["plan_id"] = PLAN_ID
    payload["date"] = "2026-09-06"
    payload["execution_authorization"] = {
        "authorization_id": "user-2026-09-06-tuesday-quick-lsq-sd4-paired-qat",
        "arms": ["sham", "qat"],
        "scope": "paired_qsilu_head_three_lsq_sd4_short_qat_search_only",
    }
    sources = _mapping(payload.get("sources"), "V34 generated QAT sources")
    sources["weight_plan"] = {
        "path": str(PTQ_PLAN.relative_to(PROJECT_ROOT)),
        "sha256": PTQ_PLAN_SHA256,
    }
    sources["candidate_evidence"] = {
        "path": str(PTQ_REPORT.relative_to(PROJECT_ROOT)),
        "sha256": PTQ_REPORT_SHA256,
    }
    sources["candidate_dual_regate"] = {
        "path": str(DUAL_REPORT.relative_to(PROJECT_ROOT)),
        "sha256": DUAL_REPORT_SHA256,
    }
    regions = list(
        _mapping(
            _mapping(payload["graph_inventory"], "V34 graph inventory").get("master"),
            "V34 master inventory",
        )["deployment_regions"]
    )
    payload["weight_policy"] = {
        "candidate_id": CANDIDATE_ID,
        "float_regions": regions,
        "assignments": _lsq_sd4_assignments(ptq),
    }
    training = _mapping(payload.get("training"), "V34 QAT training")
    training.update(
        {
            "epochs": 4,
            "patience": 5,
            "warmup_epochs": 1,
            "scale_only_epochs": 1,
            "progressive_start_epoch": 0,
            "progressive_full_epoch": 1,
            "detect_logical_batch": 128,
            "detect_microbatch": 16,
            "pose_batch": 16,
            "added_noise": False,
        }
    )
    payload["run_root"] = str(RUN_ROOT.relative_to(PROJECT_ROOT))
    payload["quick_recovery_provenance"] = {
        "selection": "largest_material_fixed_sd4_candidate_inside_dual_gate",
        "quantized_layers": 8,
        "quantized_weight_elements": 933888,
        "ptq_worst_total_map50_delta": dual_candidate["worst_map50_delta"],
        "ptq_worst_total_map50_95_delta": dual_candidate["worst_map50_95_delta"],
        "other_weight_layers": "float",
        "optimizer": "accepted_full35_adamw_lr01",
        "augmentation": "accepted_full35_exact_no_added_noise",
        "formal_validation": False,
    }
    return payload


def materialize_qat_plan(path: Path = GENERATED_PLAN) -> Full35QATPlan:
    encoded = yaml.safe_dump(
        build_qat_plan_payload(), allow_unicode=True, sort_keys=False
    )
    _write_same_or_new(path, encoded)
    return Full35QATPlan.from_yaml(path)


def _completion(runtime: Full35QATRuntime, arm: str) -> dict[str, object] | None:
    run_dir = runtime.plan.run_root / runtime.run_name(arm)  # type: ignore[arg-type]
    manifest = run_dir / "qat-experiment.json"
    if not manifest.is_file():
        return None
    payload = _mapping(
        json.loads(manifest.read_text(encoding="utf-8")), "V34 QAT completion"
    )
    expected = {
        "schema_version": 1,
        "plan_sha256": runtime.plan.config_sha256,
        "candidate_id": CANDIDATE_ID,
        "arm": arm,
        "run_name": runtime.run_name(arm),  # type: ignore[arg-type]
        "completed_stages": ["j3"],
        "formal_validation": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V34 {arm} completion manifest drifted")
    epochs = int(payload.get("epochs_completed", -1))
    if not 0 < epochs <= runtime.plan.training.epochs:
        raise ValueError(f"V34 {arm} completion epoch count is invalid")
    paths = _mapping(payload.get("checkpoint_paths"), "V34 QAT checkpoint paths")
    digests = _mapping(payload.get("checkpoint_sha256"), "V34 QAT checkpoint hashes")
    checkpoint = Path(str(paths.get("best_joint", ""))).resolve()
    if run_dir.resolve() not in checkpoint.parents or not checkpoint.is_file():
        raise FileNotFoundError(f"V34 {arm} best_joint checkpoint is missing")
    digest = _sha256(checkpoint)
    if digests.get("best_joint") != digest:
        raise ValueError(f"V34 {arm} best_joint checkpoint SHA-256 drifted")
    return {
        "manifest": str(manifest),
        "epochs_completed": epochs,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
    }


def _resume_checkpoint(runtime: Full35QATRuntime, arm: str) -> Path | None:
    run_dir = runtime.plan.run_root / runtime.run_name(arm)  # type: ignore[arg-type]
    if not run_dir.exists():
        return None
    checkpoint = run_dir / "checkpoints/last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"incomplete V34 {arm} run exists without last checkpoint: {run_dir}"
        )
    return checkpoint.resolve()


def run_queue(*, device_index: int, poll_seconds: int) -> dict[str, object]:
    if poll_seconds < 30:
        raise ValueError("V34 QAT poll interval must be at least 30 seconds")
    runtime = Full35QATRuntime.from_yaml(materialize_qat_plan().config_path)
    try:
        preflight = runtime.preflight(verify_graph=True, verify_sample_files=True)
        preflight_payload = preflight.to_dict()
        _atomic_json(QUEUE_ROOT / "preflight.json", preflight_payload)
        if not preflight.ready:
            raise RuntimeError("V34 QAT CPU preflight failed")
        completed = 0
        for arm in ("sham", "qat"):
            existing = _completion(runtime, arm)
            if existing is not None:
                completed += 1
                _transition(
                    "arm_reused",
                    current_arm=arm,
                    completed_jobs=completed,
                    completion=existing,
                )
                continue
            while True:
                foreign = _foreign_gpu_pids(device_index)
                if not foreign:
                    break
                _transition(
                    "waiting_for_gpu",
                    current_arm=arm,
                    completed_jobs=completed,
                    foreign_pids=list(foreign),
                    poll_seconds=poll_seconds,
                )
                time.sleep(poll_seconds)
            resume = _resume_checkpoint(runtime, arm)
            _transition(
                "arm_started",
                current_arm=arm,
                completed_jobs=completed,
                resume=None if resume is None else str(resume),
            )
            runtime.run(arm, device_index=device_index, resume=resume)  # type: ignore[arg-type]
            completion = _completion(runtime, arm)
            if completion is None:
                raise RuntimeError(f"V34 {arm} returned without completion evidence")
            completed += 1
            _transition(
                "arm_completed",
                current_arm=arm,
                completed_jobs=completed,
                completion=completion,
            )
        result = {
            "status": "complete",
            "candidate_id": CANDIDATE_ID,
            "plan": str(runtime.plan.config_path),
            "plan_sha256": runtime.plan.config_sha256,
            "completed_arms": 2,
        }
        _transition("complete", completed_jobs=2, result=result)
        return result
    except Exception as error:
        _transition(
            "error",
            completed_jobs=0,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V34 qSiLU LSQ-SD4 paired QAT")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--execute-reviewed-queue", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_reviewed_queue:
        parser.error("V34 QAT execution requires --execute-reviewed-queue")
    result = run_queue(device_index=args.device, poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
