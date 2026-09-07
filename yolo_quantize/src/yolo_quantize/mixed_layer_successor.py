"""V35 successor: compare per-layer formats, then lock backbone to head."""

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

from .dual_metric_regate import regate_candidate_report
from .mixed_policy_search import Full35MixedPolicySearchPlan, run_mixed_policy_search
from .progressive_queue import _foreign_gpu_pids

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_PLAN = (
    PROJECT_ROOT
    / "artifacts/queues/v34-qsilu-tuesday-quick-v1/generated/"
    "v34-qsilu-layer-and-structured-ptq-v1.yaml"
)
BASE_PLAN_SHA256 = "3b29ad7a36998534615ce6e6d070d842a1f1c6f4668b88fc9562001c556fe469"
DEPENDENCY_PLAN = (
    PROJECT_ROOT
    / "artifacts/queues/v34-qsilu-tuesday-quick-v1/"
    "short-qat-head-three-lsq-sd4-v1/generated/qat-plan.yaml"
)
DEPENDENCY_PLAN_SHA256 = "289d3730d94f376ff54ff88cba0e0083e062af9d9f3c85e8c58ddd2285d5dbe3"
DEPENDENCY_STATUS = (
    PROJECT_ROOT
    / "artifacts/queues/v34-qsilu-tuesday-quick-v1/"
    "short-qat-head-three-lsq-sd4-v1/execution-status.json"
)
QUEUE_ROOT = PROJECT_ROOT / "artifacts/queues/v35-qsilu-mixed-layer-successor-v1"
PHASE_A_PLAN = QUEUE_ROOT / "generated/v35-independent-layer-format-search-v1.yaml"
PHASE_A_REPORT = (
    PROJECT_ROOT / "artifacts/reports/v35-independent-layer-format-search-v1.json"
)
PHASE_A_DUAL = (
    PROJECT_ROOT / "artifacts/reports/v35-independent-layer-format-dual-v1.json"
)

SEGMENT_ROUTES = (
    ("backbone", "layer-backbone-high"),
    ("neck", "layer-neck-high"),
    ("head", "layer-head-high"),
)
BASE_HEAD_ROUTES = (
    "fixed-sd4-neck-attention",
    "fixed-sd4-detect-tower",
    "fixed-sd4-detect-predictor",
)
FIXED_SD4 = {"family": "fixed_sd4", "scale_method": "optimal_scaled_codebook"}
FORMAT_SPECS: tuple[tuple[str, dict[str, object]], ...] = (
    ("w8", {"family": "uniform", "bits": 8, "scale_method": "optimal_scaled_codebook"}),
    ("w7", {"family": "uniform", "bits": 7, "scale_method": "optimal_scaled_codebook"}),
    ("w6", {"family": "uniform", "bits": 6, "scale_method": "optimal_scaled_codebook"}),
    ("w5", {"family": "uniform", "bits": 5, "scale_method": "optimal_scaled_codebook"}),
    ("w4", {"family": "uniform", "bits": 4, "scale_method": "optimal_scaled_codebook"}),
    ("fixed-sd4", dict(FIXED_SD4)),
    (
        "exact-ternary",
        {"family": "exact_scaled_ternary", "scale_method": "optimal_scaled_codebook"},
    ),
    ("twn-v3", {"family": "twn_filterwise", "threshold_multiplier": 0.75}),
    ("paper-twn-v2", {"family": "paper_twn", "threshold_multiplier": 0.7}),
)
FORMAT_BY_ID = dict(FORMAT_SPECS)


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
        "current_index": values.pop("current_index", None),
        "current_candidate": values.pop("current_candidate", None),
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


def _candidate(
    candidate_id: str,
    assignments: Sequence[tuple[str, Mapping[str, object]]],
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "region_defaults": [],
        "path_routes": [
            {"route_id": route_id, "format": dict(format_spec)}
            for route_id, format_spec in assignments
        ],
    }


def _plan_payload(
    *,
    plan_id: str,
    candidates: Sequence[Mapping[str, object]],
    phase: str,
) -> dict[str, Any]:
    _verified(BASE_PLAN, BASE_PLAN_SHA256, "V35 base plan")
    payload = yaml.safe_load(BASE_PLAN.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V35 base plan must be a mapping")
    payload["plan_id"] = plan_id
    payload["date"] = "2026-09-06"
    payload["candidates"] = [dict(item) for item in candidates]
    route_ids: list[str] = []
    for candidate in candidates:
        path_routes = candidate.get("path_routes")
        if not isinstance(path_routes, list):
            raise TypeError("V35 candidate path_routes must be a list")
        for raw in path_routes:
            route = _mapping(raw, "V35 candidate route")
            route_id = str(route["route_id"])
            if route_id not in route_ids:
                route_ids.append(route_id)
    base_routes = _mapping(payload.get("routes"), "V35 base routes")
    payload["routes"] = {route_id: base_routes[route_id] for route_id in route_ids}
    authorization = _mapping(
        payload.get("execution_authorization"), "V35 authorization"
    )
    authorization.update(
        {
            "authorization_id": "user-2026-09-06-mixed-layer-successor",
            "candidate_ids": [str(item["candidate_id"]) for item in candidates],
            "route_ids": route_ids,
            "training": False,
            "stop_after": f"{phase}_dual_regate",
        }
    )
    payload["successor_provenance"] = {
        "phase": phase,
        "dependency_plan": str(DEPENDENCY_PLAN.relative_to(PROJECT_ROOT)),
        "dependency_plan_sha256": DEPENDENCY_PLAN_SHA256,
        "activation": "qsilu_pq_a8",
        "selection_order": "backbone_then_neck_then_head",
        "cpu_metrics_are_ranking_only": True,
        "map50_max_drop": 0.015,
        "map50_95_max_drop": 0.04,
        "formal_training": False,
        "formal_validation": False,
    }
    payload.pop("quick_queue_provenance", None)
    return payload


def build_independent_plan_payload() -> dict[str, Any]:
    candidates = [
        _candidate(
            f"independent-{segment}-{format_id}",
            ((route_id, format_spec),),
        )
        for segment, route_id in SEGMENT_ROUTES
        for format_id, format_spec in FORMAT_SPECS
    ]
    return _plan_payload(
        plan_id="v35-independent-layer-format-search-v1",
        candidates=candidates,
        phase="independent_3_layers_x_9_formats",
    )


def build_cumulative_plan_payload(
    *,
    segment: str,
    locked: Sequence[tuple[str, Mapping[str, object]]],
    format_ids: Sequence[str],
) -> dict[str, Any]:
    route_id = dict(SEGMENT_ROUTES)[segment]
    base = [(route, FIXED_SD4) for route in BASE_HEAD_ROUTES]
    candidates = [
        _candidate(
            f"cumulative-{segment}-{format_id}",
            (*base, *locked, (route_id, FORMAT_BY_ID[format_id])),
        )
        for format_id in format_ids
    ]
    return _plan_payload(
        plan_id=f"v35-cumulative-{segment}-mixed-format-search-v1",
        candidates=candidates,
        phase=f"cumulative_{segment}",
    )


def _materialize(
    path: Path, payload: Mapping[str, object]
) -> Full35MixedPolicySearchPlan:
    encoded = yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False)
    _write_same_or_new(path, encoded)
    return Full35MixedPolicySearchPlan.from_yaml(path)


def _run_search(
    *,
    plan_path: Path,
    payload: Mapping[str, object],
    report_path: Path,
    dual_path: Path,
    device_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _materialize(plan_path, payload)
    completed = False
    if report_path.is_file():
        existing = _mapping(
            json.loads(report_path.read_text(encoding="utf-8")), "V35 saved report"
        )
        completed = existing.get("status") == "completed"
    if not completed:
        code = run_mixed_policy_search(
            plan=plan,
            output=report_path,
            device_index=device_index,
            resume=report_path.exists(),
        )
        if code:
            raise RuntimeError(f"V35 search returned {code}")
    report = _mapping(
        json.loads(report_path.read_text(encoding="utf-8")), "V35 report"
    )
    if report.get("status") != "completed":
        raise RuntimeError("V35 search report is incomplete")
    report_sha = _sha256(report_path)
    dual = regate_candidate_report(
        source_report=report_path,
        expected_source_sha256=report_sha,
        metric_contract_id=plan.metric_contract_id,
    )
    encoded_dual = json.dumps(
        dual, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    _write_same_or_new(dual_path, encoded_dual)
    return report, dual


def _quantized_bytes(record: Mapping[str, object]) -> int:
    weight = _mapping(record.get("weight_quantization"), "V35 weight result")
    formats = weight.get("formats")
    if not isinstance(formats, list) or not formats:
        raise ValueError("V35 weight result has no formats")
    total = 0
    for raw in formats:
        numeric = _mapping(_mapping(raw, "V35 format").get("numeric"), "V35 numeric")
        total += int(numeric["weight_code_bytes"]) + int(numeric["scale_bytes"])
    return total


def _summaries(
    report: Mapping[str, object],
    dual: Mapping[str, object],
    metadata: Mapping[str, tuple[str, str]],
) -> list[dict[str, object]]:
    results = _mapping(report.get("results"), "V35 results")
    dual_candidates = _mapping(dual.get("candidates"), "V35 dual candidates")
    summaries: list[dict[str, object]] = []
    for candidate_id, (segment, format_id) in metadata.items():
        result = _mapping(results.get(candidate_id), f"V35 result {candidate_id}")
        gate = _mapping(
            dual_candidates.get(candidate_id), f"V35 dual {candidate_id}"
        )
        summaries.append(
            {
                "candidate_id": candidate_id,
                "segment": segment,
                "format_id": format_id,
                "decision": gate["decision"],
                "worst_map50_delta": gate["worst_map50_delta"],
                "worst_map50_95_delta": gate["worst_map50_95_delta"],
                "quantized_bytes": _quantized_bytes(result),
            }
        )
    return summaries


def _shortlist(rows: Sequence[Mapping[str, object]]) -> list[str]:
    green = [row for row in rows if row["decision"] == "green"]
    if not green:
        return []
    accuracy = max(
        green,
        key=lambda row: (
            float(row["worst_map50_delta"]),
            float(row["worst_map50_95_delta"]),
            -int(row["quantized_bytes"]),
        ),
    )
    close = [
        row
        for row in green
        if float(row["worst_map50_delta"])
        >= float(accuracy["worst_map50_delta"]) - 0.002
        and float(row["worst_map50_95_delta"])
        >= float(accuracy["worst_map50_95_delta"]) - 0.005
    ]
    hardware = min(green, key=lambda row: (int(row["quantized_bytes"]), str(row["format_id"])))
    balanced = min(close, key=lambda row: (int(row["quantized_bytes"]), str(row["format_id"])))
    result: list[str] = []
    for row in (hardware, balanced, accuracy):
        format_id = str(row["format_id"])
        if format_id not in result:
            result.append(format_id)
    return result


def _wait_for_dependency(poll_seconds: int) -> None:
    _verified(DEPENDENCY_PLAN, DEPENDENCY_PLAN_SHA256, "V35 dependency plan")
    while True:
        if DEPENDENCY_STATUS.is_file():
            payload = _mapping(
                json.loads(DEPENDENCY_STATUS.read_text(encoding="utf-8")),
                "V35 dependency status",
            )
            status = str(payload.get("status", ""))
            if status == "complete":
                if int(payload.get("completed_jobs", -1)) != 2:
                    raise ValueError("V35 dependency completed_jobs differs")
                return
            if status == "error":
                raise RuntimeError(
                    "V35 dependency failed: "
                    + json.dumps(payload.get("error"), ensure_ascii=False)
                )
        _transition(
            "waiting_for_dependency",
            completed_jobs=0,
            dependency_status=str(DEPENDENCY_STATUS),
            poll_seconds=poll_seconds,
        )
        time.sleep(poll_seconds)


def _wait_for_gpu(device_index: int, poll_seconds: int, completed_jobs: int) -> None:
    while True:
        pids = _foreign_gpu_pids(device_index)
        if not pids:
            return
        _transition(
            "waiting_for_gpu",
            completed_jobs=completed_jobs,
            foreign_pids=list(pids),
            poll_seconds=poll_seconds,
        )
        time.sleep(poll_seconds)


def run_queue(*, device_index: int, poll_seconds: int) -> dict[str, object]:
    if poll_seconds < 30:
        raise ValueError("V35 poll interval must be at least 30 seconds")
    completed_jobs = 0
    try:
        _wait_for_dependency(poll_seconds)
        _wait_for_gpu(device_index, poll_seconds, completed_jobs)
        _transition("independent_started", completed_jobs=completed_jobs)
        phase_a_payload = build_independent_plan_payload()
        phase_a_report, phase_a_dual = _run_search(
            plan_path=PHASE_A_PLAN,
            payload=phase_a_payload,
            report_path=PHASE_A_REPORT,
            dual_path=PHASE_A_DUAL,
            device_index=device_index,
        )
        completed_jobs += 1
        phase_a_metadata = {
            f"independent-{segment}-{format_id}": (segment, format_id)
            for segment, _ in SEGMENT_ROUTES
            for format_id, _ in FORMAT_SPECS
        }
        phase_a_rows = _summaries(
            phase_a_report, phase_a_dual, phase_a_metadata
        )
        shortlists = {
            segment: _shortlist(
                [row for row in phase_a_rows if row["segment"] == segment]
            )
            for segment, _ in SEGMENT_ROUTES
        }
        locked: list[tuple[str, Mapping[str, object]]] = []
        stages: list[dict[str, object]] = []
        for index, (segment, route_id) in enumerate(SEGMENT_ROUTES):
            formats = shortlists[segment]
            if not formats:
                stages.append(
                    {"segment": segment, "status": "skipped_no_independent_green"}
                )
                continue
            plan_path = QUEUE_ROOT / f"generated/v35-cumulative-{segment}-v1.yaml"
            report_path = (
                PROJECT_ROOT / f"artifacts/reports/v35-cumulative-{segment}-v1.json"
            )
            dual_path = (
                PROJECT_ROOT
                / f"artifacts/reports/v35-cumulative-{segment}-dual-v1.json"
            )
            _wait_for_gpu(device_index, poll_seconds, completed_jobs)
            _transition(
                "cumulative_stage_started",
                current_index=index,
                current_candidate=segment,
                completed_jobs=completed_jobs,
                formats=formats,
            )
            cumulative_payload = build_cumulative_plan_payload(
                segment=segment,
                locked=locked,
                format_ids=formats,
            )
            report, dual = _run_search(
                plan_path=plan_path,
                payload=cumulative_payload,
                report_path=report_path,
                dual_path=dual_path,
                device_index=device_index,
            )
            completed_jobs += 1
            metadata = {
                f"cumulative-{segment}-{format_id}": (segment, format_id)
                for format_id in formats
            }
            rows = _summaries(report, dual, metadata)
            green = [row for row in rows if row["decision"] == "green"]
            if green:
                selected = min(
                    green,
                    key=lambda row: (
                        int(row["quantized_bytes"]),
                        -float(row["worst_map50_delta"]),
                        -float(row["worst_map50_95_delta"]),
                        str(row["format_id"]),
                    ),
                )
                selected_format = str(selected["format_id"])
                locked.append((route_id, FORMAT_BY_ID[selected_format]))
                stage_status = "locked"
            else:
                selected = None
                selected_format = None
                stage_status = "retained_previous_policy"
            stages.append(
                {
                    "segment": segment,
                    "status": stage_status,
                    "shortlist": formats,
                    "selected_format": selected_format,
                    "selected": selected,
                    "report": str(report_path),
                    "report_sha256": _sha256(report_path),
                    "dual_report": str(dual_path),
                    "dual_report_sha256": _sha256(dual_path),
                }
            )
        summary = {
            "schema_version": 1,
            "status": "complete_ready_for_one_final_mixed_qat",
            "activation": "qsilu_pq_a8",
            "dependency": {
                "plan": str(DEPENDENCY_PLAN),
                "plan_sha256": DEPENDENCY_PLAN_SHA256,
            },
            "independent": {
                "cells": len(phase_a_rows),
                "report": str(PHASE_A_REPORT),
                "report_sha256": _sha256(PHASE_A_REPORT),
                "dual_report": str(PHASE_A_DUAL),
                "dual_report_sha256": _sha256(PHASE_A_DUAL),
                "shortlists": shortlists,
                "results": phase_a_rows,
            },
            "progressive_order": [segment for segment, _ in SEGMENT_ROUTES],
            "base_policy": "fixed_sd4_head_three",
            "locked_assignments": [
                {"route_id": route_id, "format": dict(format_spec)}
                for route_id, format_spec in locked
            ],
            "stages": stages,
            "next_training": {
                "maximum_candidates": 1,
                "arms": ["matched_sham", "qat"],
                "epochs_per_arm": 4,
                "patience": 5,
                "activation": "qsilu_pq_a8",
                "only_if_final_policy_is_green_or_recover": True,
                "formal_training": False,
            },
            "formal_validation": False,
        }
        _atomic_json(QUEUE_ROOT / "summary.json", summary)
        _transition(
            "complete_ready_for_one_final_mixed_qat",
            completed_jobs=completed_jobs,
            locked_assignments=len(locked),
            summary=str(QUEUE_ROOT / "summary.json"),
        )
        return summary
    except Exception as error:
        _transition(
            "error",
            completed_jobs=completed_jobs,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V35 qSiLU mixed-layer successor")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--execute-reviewed-queue", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_reviewed_queue:
        parser.error("V35 execution requires --execute-reviewed-queue")
    result = run_queue(device_index=args.device, poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
