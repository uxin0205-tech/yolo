"""qSiLU layer-sensitivity and structured-weight PTQ queue for Tuesday."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_PLAN = (
    PROJECT_ROOT
    / "configs/experiments/v11-qsilu-special-format-route-sensitivity-v1.yaml"
)
BASE_PLAN_SHA256 = "d6dbac10f1fb62f96eef53654e1ae81bd50dda61a1430dcd4fd3a2e2e7af127b"
CPU_PROFILE = (
    PROJECT_ROOT / "artifacts/reports/v34-qsilu-structured-148-layer-cpu-v1.json"
)
CPU_PROFILE_SHA256 = "28280630afb4d11ef75cdbd71032da40fa9a8d7cf6a29d78e8e1017ff13dbb77"
QUEUE_ROOT = PROJECT_ROOT / "artifacts/queues/v34-qsilu-tuesday-quick-v1"
LAYER_ROUTE_MANIFEST = (
    QUEUE_ROOT / "generated/v34-qsilu-layer-sensitivity-routes-v1.yaml"
)
GENERATED_PLAN = (
    QUEUE_ROOT / "generated/v34-qsilu-layer-and-structured-ptq-v1.yaml"
)
PTQ_REPORT = (
    PROJECT_ROOT / "artifacts/reports/v34-qsilu-layer-and-structured-ptq-v1.json"
)
DUAL_REPORT = (
    PROJECT_ROOT
    / "artifacts/reports/v34-qsilu-layer-and-structured-ptq-dual-v1.json"
)

FIXED_SD4_PROFILE_ID = (
    "fixed-sd4-per_output_channel-optimal_scaled_codebook"
)
SEGMENT_REGIONS = {
    "backbone": frozenset(
        {"backbone_early", "backbone_deep", "backbone_attention_safe"}
    ),
    "neck": frozenset({"neck", "neck_attention_safe", "masf"}),
    "head": frozenset(
        {
            "detect_one2one_tower",
            "detect_one2one_predictor",
            "pose_one2one_tower",
            "pose_one2one_predictor",
        }
    ),
}
SENSITIVITY_LEVELS = ("low", "median", "high")

HEAD_ROUTES = (
    "fixed-sd4-neck-attention",
    "fixed-sd4-detect-tower",
    "fixed-sd4-detect-predictor",
)
POSE_SAFE_ROUTE = "paper-twn-safe"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _write_same_or_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite drifted artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _path_routes(
    route_ids: Sequence[str], format_payload: Mapping[str, object]
) -> list[dict[str, object]]:
    return [
        {"route_id": route_id, "format": dict(format_payload)}
        for route_id in route_ids
    ]


def build_layer_sensitivity_selection() -> list[dict[str, object]]:
    """Select low/median/high SD4-error layers in backbone, neck and head."""

    if _sha256(CPU_PROFILE) != CPU_PROFILE_SHA256:
        raise RuntimeError("V34 CPU profile SHA-256 drifted")
    profile = json.loads(CPU_PROFILE.read_text(encoding="utf-8"))
    parent = profile.get("parent")
    if (
        profile.get("kind") != "full35_static_weight_format_analysis"
        or profile.get("profile") != "structured"
        or profile.get("gpu_used") is not False
        or profile.get("formal_training") is not False
        or not isinstance(parent, dict)
        or parent.get("policy_id") != "qsilu_pq--lsq-plus-a8"
    ):
        raise ValueError("V34 CPU profile contract differs")
    measurements = profile.get("measurements")
    if not isinstance(measurements, list):
        raise TypeError("V34 CPU profile measurements must be a list")
    rows: list[dict[str, object]] = []
    for raw in measurements:
        if not isinstance(raw, dict):
            raise TypeError("V34 CPU profile row must be a mapping")
        if raw.get("view") != "deployment" or raw.get("format_id") != FIXED_SD4_PROFILE_ID:
            continue
        numeric = raw.get("numeric")
        if not isinstance(numeric, dict):
            raise TypeError("V34 CPU profile numeric record must be a mapping")
        rows.append(
            {
                "path": str(raw["path"]),
                "region": str(raw["region"]),
                "normalized_rmse": float(numeric["normalized_rmse"]),
            }
        )
    if len(rows) != 148 or len({str(row["path"]) for row in rows}) != 148:
        raise ValueError("V34 fixed-SD4 deployment profile must cover 148 unique layers")
    known_regions = set().union(*SEGMENT_REGIONS.values())
    if {str(row["region"]) for row in rows} != known_regions:
        raise ValueError("V34 fixed-SD4 deployment regions differ")

    selected: list[dict[str, object]] = []
    for segment, regions in SEGMENT_REGIONS.items():
        ranked = sorted(
            (row for row in rows if row["region"] in regions),
            key=lambda row: (float(row["normalized_rmse"]), str(row["path"])),
        )
        indexes = (0, len(ranked) // 2, len(ranked) - 1)
        for level, rank_index in zip(SENSITIVITY_LEVELS, indexes, strict=True):
            row = ranked[rank_index]
            selected.append(
                {
                    **row,
                    "segment": segment,
                    "sensitivity_level": level,
                    "rank_index_zero_based": rank_index,
                    "segment_layer_count": len(ranked),
                    "route_id": f"layer-{segment}-{level}",
                }
            )
    if len(selected) != 9 or len({str(row["path"]) for row in selected}) != 9:
        raise RuntimeError("V34 layer sensitivity selection must contain 9 unique layers")
    return selected


def build_layer_route_manifest_payload() -> dict[str, object]:
    selection = build_layer_sensitivity_selection()
    payload: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": "v34-qsilu-layer-sensitivity-routes-v1",
        "date": "2026-09-06",
        "execution_authorized": False,
        "map_validation_run": False,
        "source_profile": {
            "path": str(CPU_PROFILE.relative_to(PROJECT_ROOT)),
            "sha256": CPU_PROFILE_SHA256,
            "view": "deployment",
            "format_id": FIXED_SD4_PROFILE_ID,
        },
        "selection_method": {
            "scope": "all_148_deployment_weight_layers",
            "segments": {key: sorted(value) for key, value in SEGMENT_REGIONS.items()},
            "ranks": ["minimum_nrmse", "upper_median_nrmse", "maximum_nrmse"],
            "gpu_candidates": 9,
            "selection_claim": False,
        },
        "selection_records": selection,
    }
    for row in selection:
        payload[str(row["route_id"])] = [str(row["path"])]
    return payload


def _layer_manifest_text() -> str:
    return yaml.safe_dump(
        build_layer_route_manifest_payload(), allow_unicode=True, sort_keys=False
    )


def build_quick_plan_payload() -> dict[str, Any]:
    """Build 9 single-layer probes followed by 6 structured route candidates."""

    if _sha256(BASE_PLAN) != BASE_PLAN_SHA256:
        raise RuntimeError("V34 base plan SHA-256 drifted")
    selection = build_layer_sensitivity_selection()
    layer_manifest_sha256 = _text_sha256(_layer_manifest_text())
    payload = yaml.safe_load(BASE_PLAN.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("V34 base plan must be a mapping")
    payload["plan_id"] = "v34-qsilu-layer-and-structured-ptq-v1"
    payload["date"] = "2026-09-06"
    payload["metric_contract_id"] = (
        "full35-coco-val-bbat5-search-val-bittrue-dual-v1"
    )
    authorization = payload["execution_authorization"]
    layer_candidates = [
        (
            f"layer-sensitivity-{row['segment']}-{row['sensitivity_level']}-fixed-sd4",
            (str(row["route_id"]),),
            {"family": "fixed_sd4", "scale_method": "optimal_scaled_codebook"},
        )
        for row in selection
    ]
    structured_candidates = [
        (
            "fixed-sd4-head-three",
            HEAD_ROUTES,
            {"family": "fixed_sd4", "scale_method": "optimal_scaled_codebook"},
        ),
        (
            "exact-ternary-head-three",
            HEAD_ROUTES,
            {
                "family": "exact_scaled_ternary",
                "scale_method": "optimal_scaled_codebook",
            },
        ),
        (
            "twn-v3-head-three",
            HEAD_ROUTES,
            {"family": "twn_filterwise", "threshold_multiplier": 0.75},
        ),
        (
            "paper-twn-v2-head-three",
            HEAD_ROUTES,
            {"family": "paper_twn", "threshold_multiplier": 0.7},
        ),
        (
            "exact-ternary-pose-safe",
            (POSE_SAFE_ROUTE,),
            {
                "family": "exact_scaled_ternary",
                "scale_method": "optimal_scaled_codebook",
            },
        ),
        (
            "twn-v3-pose-safe",
            (POSE_SAFE_ROUTE,),
            {"family": "twn_filterwise", "threshold_multiplier": 0.75},
        ),
    ]
    candidates = [*layer_candidates, *structured_candidates]
    layer_route_ids = [str(row["route_id"]) for row in selection]
    group_route_ids = [*HEAD_ROUTES, POSE_SAFE_ROUTE]
    authorization.update(
        {
            "authorization_id": "user-2026-09-06-tuesday-quick-structured-ptq",
            "candidate_ids": [item[0] for item in candidates],
            "route_ids": [*layer_route_ids, *group_route_ids],
            "training": False,
            "stop_after": (
                "nine_single_layer_then_six_structured_ptq_then_dual_regate"
            ),
        }
    )
    base_routes = payload["routes"]
    if not isinstance(base_routes, dict):
        raise TypeError("V34 base routes must be a mapping")
    layer_manifest_record = {
        "path": str(LAYER_ROUTE_MANIFEST.relative_to(PROJECT_ROOT)),
        "sha256": layer_manifest_sha256,
    }
    payload["routes"] = {
        **{
            route_id: {
                "manifest": dict(layer_manifest_record),
                "list_key": route_id,
            }
            for route_id in layer_route_ids
        },
        **{route_id: base_routes[route_id] for route_id in group_route_ids},
    }
    payload["candidates"] = [
        {
            "candidate_id": candidate_id,
            "region_defaults": [],
            "path_routes": _path_routes(route_ids, format_payload),
        }
        for candidate_id, route_ids, format_payload in candidates
    ]
    payload["quick_queue_provenance"] = {
        "base_plan": str(BASE_PLAN),
        "base_plan_sha256": BASE_PLAN_SHA256,
        "cpu_profile": str(CPU_PROFILE),
        "cpu_profile_sha256": CPU_PROFILE_SHA256,
        "candidate_count": len(candidates),
        "single_layer_candidate_count": len(layer_candidates),
        "structured_route_candidate_count": len(structured_candidates),
        "layer_route_manifest": str(LAYER_ROUTE_MANIFEST.relative_to(PROJECT_ROOT)),
        "layer_route_manifest_sha256": layer_manifest_sha256,
        "layer_selection": selection,
        "formal_training": False,
        "formal_validation": False,
    }
    return payload


def materialize_quick_plan(path: Path = GENERATED_PLAN) -> Full35MixedPolicySearchPlan:
    _write_same_or_new(LAYER_ROUTE_MANIFEST, _layer_manifest_text())
    payload = build_quick_plan_payload()
    encoded = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    _write_same_or_new(path, encoded)
    return Full35MixedPolicySearchPlan.from_yaml(path)


def _transition(kind: str, **values: object) -> None:
    _atomic_json(
        QUEUE_ROOT / "execution-status.json",
        {
            "schema_version": 1,
            "kind": kind,
            "time_unix": time.time(),
            "values": values,
        },
    )


def run_queue(device_index: int) -> dict[str, object]:
    plan = materialize_quick_plan()
    try:
        _transition(
            "ptq_started",
            stage="layer_sensitivity_nine_then_structured_six",
            plan=str(plan.config_path),
            plan_sha256=plan.config_sha256,
            candidates=len(plan.candidates),
        )
        return_code = run_mixed_policy_search(
            plan=plan,
            output=PTQ_REPORT,
            device_index=device_index,
            resume=PTQ_REPORT.exists(),
        )
        if return_code:
            raise RuntimeError(f"V34 PTQ returned {return_code}")
        report_sha256 = _sha256(PTQ_REPORT)
        dual = regate_candidate_report(
            source_report=PTQ_REPORT,
            expected_source_sha256=report_sha256,
            metric_contract_id=plan.metric_contract_id,
        )
        encoded_dual = json.dumps(
            dual, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        _write_same_or_new(DUAL_REPORT, encoded_dual)
        decisions = {
            candidate_id: record["decision"]
            for candidate_id, record in dual["candidates"].items()
        }
        result = {
            "status": "ptq_complete_ready_for_lsq_sd4_selection",
            "ptq_report": str(PTQ_REPORT),
            "ptq_report_sha256": report_sha256,
            "dual_report": str(DUAL_REPORT),
            "dual_report_sha256": _sha256(DUAL_REPORT),
            "decisions": decisions,
        }
        _transition("ptq_complete_ready_for_lsq_sd4_selection", **result)
        return result
    except Exception as error:
        _transition(
            "error",
            stage="layer_sensitivity_nine_then_structured_six",
            error_type=type(error).__name__,
            message=str(error),
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V34 Tuesday quick PTQ queue")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--execute-reviewed-queue", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_reviewed_queue:
        parser.error("V34 execution requires --execute-reviewed-queue")
    result = run_queue(args.device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
