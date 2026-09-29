"""Materialize and execute the V35 qSiLU heterogeneous paired QAT."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .progressive_queue import _foreign_gpu_pids
from .qat_plan import Full35QATPlan
from .qat_runtime import Full35QATRuntime
from .tuesday_quick_qat import (
    _atomic_json,
    _mapping,
    _sha256,
    _verified,
    _write_same_or_new,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_QAT_PLAN = (
    PROJECT_ROOT / "artifacts/queues/v34-qsilu-tuesday-quick-v1/"
    "short-qat-head-three-lsq-sd4-v1/generated/qat-plan.yaml"
)
BASE_QAT_PLAN_SHA256 = (
    "289d3730d94f376ff54ff88cba0e0083e062af9d9f3c85e8c58ddd2285d5dbe3"
)
MIXED_PLAN = (
    PROJECT_ROOT / "artifacts/queues/v35-qsilu-mixed-layer-successor-v1/generated/"
    "v35-cumulative-head-v1.yaml"
)
MIXED_PLAN_SHA256 = "c40a4ff33585d1460b538f3df911c94074a6347f4a10c8b00b68dc1eafc82a2c"
MIXED_SUMMARY = (
    PROJECT_ROOT / "artifacts/queues/v35-qsilu-mixed-layer-successor-v1/summary.json"
)
MIXED_SUMMARY_SHA256 = (
    "9c88914cc69a624cb1e9221e9e9e80bc10af0545ce15690fc196a2310298082c"
)
PTQ_REPORT = PROJECT_ROOT / "artifacts/reports/v35-cumulative-head-v1.json"
PTQ_REPORT_SHA256 = "c00114d440f84adcfc0dbe810c5f33668e5c417da0e967923d728b3efe7b65b2"
DUAL_REPORT = PROJECT_ROOT / "artifacts/reports/v35-cumulative-head-dual-v1.json"
DUAL_REPORT_SHA256 = "71eed7f59afbdeae9f3cd2efbefa05951258ab0099b116cece75462bcad49dc8"
CANDIDATE_ID = "cumulative-head-w4"
PLAN_ID = "v35-qsilu-mixed-lssd4-w6-w4-short-qat-v1"
QUEUE_ROOT = (
    PROJECT_ROOT / "artifacts/queues/v35-qsilu-mixed-layer-successor-v1/"
    "short-qat-mixed-final-v1"
)
GENERATED_PLAN = QUEUE_ROOT / "generated/qat-plan.yaml"
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/qat/v35-qsilu-mixed-final-short-v1"


def _training_format(raw: Mapping[str, object]) -> dict[str, object]:
    format_id = str(raw.get("format_id"))
    scale_method = str(raw.get("scale_method"))
    if scale_method != "optimal_scaled_codebook":
        raise ValueError(
            f"V35 format {format_id} does not use the exact scale baseline"
        )
    if format_id == "fixed-sd4":
        return {"family": "ls_sd4", "scale_method": scale_method}
    if format_id in {"w4", "w6"}:
        return {
            "family": "uniform",
            "bits": int(format_id.removeprefix("w")),
            "scale_method": scale_method,
        }
    raise ValueError(f"V35 locked an unsupported QAT format: {format_id}")


def _mixed_assignments(
    report: Mapping[str, object], base: Mapping[str, object]
) -> list[dict[str, object]]:
    results = _mapping(report.get("results"), "V35 PTQ results")
    candidate = _mapping(results.get(CANDIDATE_ID), "V35 final candidate")
    gate = _mapping(candidate.get("gate"), "V35 final candidate gate")
    if candidate.get("status") != "completed" or gate.get("decision") != "green":
        raise ValueError("V35 final candidate is not completed green evidence")
    weight = _mapping(candidate.get("weight_quantization"), "V35 weight evidence")
    raw_assignments = weight.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ValueError("V35 final candidate has no path assignments")

    policy = _mapping(base.get("weight_policy"), "V34 base QAT weight policy")
    base_assignments = policy.get("assignments")
    if not isinstance(base_assignments, list):
        raise TypeError("V34 base QAT assignments must be a list")
    base_paths = {
        str(path)
        for raw in base_assignments
        for path in _mapping(raw, "V34 base assignment").get("paths", [])
    }
    if len(base_paths) != 8:
        raise ValueError("V34 base QAT must contain exactly eight head-three paths")

    assignments: list[dict[str, object]] = []
    seen: set[str] = set()
    path_formats: dict[str, str] = {}
    for raw in raw_assignments:
        record = _mapping(raw, "V35 PTQ assignment")
        encoded = _mapping(record.get("format"), "V35 PTQ assignment format")
        selector = _mapping(record.get("selector"), "V35 PTQ assignment selector")
        paths = selector.get("paths")
        if selector.get("kind") != "paths" or not isinstance(paths, list) or not paths:
            raise ValueError("V35 QAT accepts explicit non-empty path routes only")
        encoded_paths = [str(path) for path in paths]
        if seen.intersection(encoded_paths):
            raise ValueError("V35 final candidate contains overlapping paths")
        seen.update(encoded_paths)
        format_id = str(encoded.get("format_id"))
        path_formats.update({path: format_id for path in encoded_paths})
        assignments.append(
            {
                "region": str(record["region"]),
                "paths": encoded_paths,
                "format": _training_format(encoded),
            }
        )

    if len(seen) != 11 or not base_paths.issubset(seen):
        raise ValueError(
            "V35 mixed QAT must extend the eight-path base to eleven paths"
        )
    expected_counts = {"fixed-sd4": 9, "w6": 1, "w4": 1}
    actual_counts = {
        format_id: sum(value == format_id for value in path_formats.values())
        for format_id in expected_counts
    }
    if actual_counts != expected_counts or set(path_formats.values()) != set(
        expected_counts
    ):
        raise ValueError("V35 mixed path-format distribution drifted")
    if any(path_formats[path] != "fixed-sd4" for path in base_paths):
        raise ValueError("V35 final policy changed the immutable head-three base")
    return assignments


def build_qat_plan_payload() -> dict[str, Any]:
    """Derive one paired mixed-QAT plan from immutable V35 evidence."""

    for path, expected, label in (
        (BASE_QAT_PLAN, BASE_QAT_PLAN_SHA256, "V34 base QAT plan"),
        (MIXED_PLAN, MIXED_PLAN_SHA256, "V35 cumulative head plan"),
        (MIXED_SUMMARY, MIXED_SUMMARY_SHA256, "V35 mixed summary"),
        (PTQ_REPORT, PTQ_REPORT_SHA256, "V35 cumulative head report"),
        (DUAL_REPORT, DUAL_REPORT_SHA256, "V35 cumulative head dual report"),
    ):
        _verified(path, expected, label)

    payload = yaml.safe_load(BASE_QAT_PLAN.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V34 base QAT plan must be a mapping")
    summary = _mapping(
        json.loads(MIXED_SUMMARY.read_text(encoding="utf-8")), "V35 summary"
    )
    report = _mapping(
        json.loads(PTQ_REPORT.read_text(encoding="utf-8")), "V35 PTQ report"
    )
    dual = _mapping(
        json.loads(DUAL_REPORT.read_text(encoding="utf-8")), "V35 dual report"
    )
    dual_candidate = _mapping(
        _mapping(dual.get("candidates"), "V35 dual candidates").get(CANDIDATE_ID),
        "V35 final dual candidate",
    )
    stages = summary.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("V35 summary must contain backbone, neck and head stages")
    final_stage = _mapping(stages[-1], "V35 final stage")
    selected = _mapping(final_stage.get("selected"), "V35 final selected candidate")
    expected_locked = [
        ("layer-backbone-high", "fixed_sd4"),
        ("layer-neck-high", "uniform", 6),
        ("layer-head-high", "uniform", 4),
    ]
    locked = summary.get("locked_assignments")
    if not isinstance(locked, list):
        raise TypeError("V35 locked assignments must be a list")
    actual_locked: list[tuple[object, ...]] = []
    for raw in locked:
        item = _mapping(raw, "V35 locked assignment")
        encoded = _mapping(item.get("format"), "V35 locked assignment format")
        row: tuple[object, ...] = (item.get("route_id"), encoded.get("family"))
        if encoded.get("family") == "uniform":
            row += (int(encoded["bits"]),)
        actual_locked.append(row)
    if (
        summary.get("status") != "complete_ready_for_one_final_mixed_qat"
        or summary.get("activation") != "qsilu_pq_a8"
        or actual_locked != expected_locked
        or final_stage.get("segment") != "head"
        or selected.get("candidate_id") != CANDIDATE_ID
        or selected.get("decision") != "green"
        or dual.get("status") != "completed"
        or dual_candidate.get("decision") != "green"
        or _mapping(dual.get("source"), "V35 dual source").get("report_sha256")
        != PTQ_REPORT_SHA256
    ):
        raise ValueError("V35 final mixed evidence drifted")

    payload["plan_id"] = PLAN_ID
    payload["date"] = "2026-09-06"
    payload["execution_authorization"] = {
        "authorization_id": "user-2026-09-06-continue-mixed-layer-queue",
        "arms": ["sham", "qat"],
        "scope": "paired_qsilu_mixed_lssd4_w6_w4_short_qat_search_only",
    }
    sources = _mapping(payload.get("sources"), "V35 generated QAT sources")
    sources["weight_plan"] = {
        "path": str(MIXED_PLAN.relative_to(PROJECT_ROOT)),
        "sha256": MIXED_PLAN_SHA256,
    }
    sources["candidate_evidence"] = {
        "path": str(PTQ_REPORT.relative_to(PROJECT_ROOT)),
        "sha256": PTQ_REPORT_SHA256,
    }
    sources["candidate_dual_regate"] = {
        "path": str(DUAL_REPORT.relative_to(PROJECT_ROOT)),
        "sha256": DUAL_REPORT_SHA256,
    }
    base_policy = _mapping(payload.get("weight_policy"), "V34 base weight policy")
    float_regions = base_policy.get("float_regions")
    if not isinstance(float_regions, list) or len(float_regions) != 10:
        raise ValueError("V34 base must preserve all ten regions as float defaults")
    payload["weight_policy"] = {
        "candidate_id": CANDIDATE_ID,
        "float_regions": list(float_regions),
        "assignments": _mixed_assignments(report, payload),
    }
    training = _mapping(payload.get("training"), "V35 QAT training")
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
        "selection": "v35_progressive_backbone_neck_head_smallest_green",
        "summary": str(MIXED_SUMMARY.relative_to(PROJECT_ROOT)),
        "summary_sha256": MIXED_SUMMARY_SHA256,
        "quantized_layers": 11,
        "path_format_counts": {"ls_sd4": 9, "w6": 1, "w4": 1},
        "ptq_quantized_bytes": int(selected["quantized_bytes"]),
        "ptq_worst_total_map50_delta": float(selected["worst_map50_delta"]),
        "ptq_worst_total_map50_95_delta": float(selected["worst_map50_95_delta"]),
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
    with (QUEUE_ROOT / "execution-events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    _atomic_json(QUEUE_ROOT / "execution-status.json", payload)


def _completion(runtime: Full35QATRuntime, arm: str) -> dict[str, object] | None:
    run_dir = runtime.plan.run_root / runtime.run_name(arm)  # type: ignore[arg-type]
    manifest = run_dir / "qat-experiment.json"
    if not manifest.is_file():
        return None
    payload = _mapping(
        json.loads(manifest.read_text(encoding="utf-8")), "V35 QAT completion"
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
        raise ValueError(f"V35 {arm} completion manifest drifted")
    epochs = int(payload.get("epochs_completed", -1))
    if not 0 < epochs <= runtime.plan.training.epochs:
        raise ValueError(f"V35 {arm} completion epoch count is invalid")
    paths = _mapping(payload.get("checkpoint_paths"), "V35 QAT checkpoint paths")
    digests = _mapping(payload.get("checkpoint_sha256"), "V35 QAT checkpoint hashes")
    checkpoint = Path(str(paths.get("best_joint", ""))).resolve()
    if run_dir.resolve() not in checkpoint.parents or not checkpoint.is_file():
        raise FileNotFoundError(f"V35 {arm} best_joint checkpoint is missing")
    digest = _sha256(checkpoint)
    if digests.get("best_joint") != digest:
        raise ValueError(f"V35 {arm} best_joint checkpoint SHA-256 drifted")
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
            f"incomplete V35 {arm} run exists without last checkpoint: {run_dir}"
        )
    return checkpoint.resolve()


def run_queue(*, device_index: int, poll_seconds: int) -> dict[str, object]:
    if poll_seconds < 30:
        raise ValueError("V35 QAT poll interval must be at least 30 seconds")
    runtime = Full35QATRuntime.from_yaml(materialize_qat_plan().config_path)
    completed = 0
    try:
        preflight = runtime.preflight(verify_graph=True, verify_sample_files=True)
        _atomic_json(QUEUE_ROOT / "preflight.json", preflight.to_dict())
        if not preflight.ready:
            raise RuntimeError("V35 mixed QAT CPU preflight failed")
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
                raise RuntimeError(f"V35 {arm} returned without completion evidence")
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
            completed_jobs=completed,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V35 qSiLU mixed paired QAT")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--execute-reviewed-queue", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_reviewed_queue:
        parser.error("V35 mixed QAT execution requires --execute-reviewed-queue")
    result = run_queue(device_index=args.device, poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
