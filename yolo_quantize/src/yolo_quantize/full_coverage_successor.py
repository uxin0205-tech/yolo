"""qSiLU successor that assigns a weight format to every deployment layer."""

from __future__ import annotations

import gc
import hashlib
import json
import time
import copy
import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .weight_formats import WeightAnalysisPlan

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENT_MANIFEST = PROJECT_ROOT / "artifacts/manifests/v35-qsilu-mixed-11path-epoch3-locked-parent-v1.yaml"
CPU_PROFILE = PROJECT_ROOT / "artifacts/reports/v36-qsilu-v35-parent-148-layer-9format-cpu-v1.json"
V35_BASE_PLAN = PROJECT_ROOT / "artifacts/queues/v35-qsilu-mixed-layer-successor-v1/generated/v35-cumulative-head-v1.yaml"
V35_BASE_PLAN_SHA256 = "c40a4ff33585d1460b538f3df911c94074a6347f4a10c8b00b68dc1eafc82a2c"
V35_BASE_QAT_PLAN = (
    PROJECT_ROOT
    / "artifacts/queues/v35-qsilu-mixed-layer-successor-v1/short-qat-mixed-final-v1/generated/qat-plan.yaml"
)
V35_BASE_QAT_PLAN_SHA256 = "28d5887dd9cc61b1cecbebc1249b9c7aef5c65cd8782dfd9931e34726f0c9a84"
V35_SHAM_COMPLETION = (
    PROJECT_ROOT
    / "artifacts/runs/qat/v35-qsilu-mixed-final-short-v1/"
    "v35-qsilu-mixed-lssd4-w6-w4-short-qat-v1-sham-seed1/qat-experiment.json"
)
V35_SHAM_METRICS = (
    PROJECT_ROOT
    / "artifacts/runs/qat/v35-qsilu-mixed-final-short-v1/"
    "v35-qsilu-mixed-lssd4-w6-w4-short-qat-v1-sham-seed1/validation/"
    "epoch-0003/bittrue/metrics.json"
)
QUEUE_ROOT = PROJECT_ROOT / "artifacts/queues/v36-qsilu-full-coverage-progressive-v1"
CONTINUATION_QAT_ROOT = QUEUE_ROOT / "short-qat-full-coverage-v3"
ALL_REGIONS = (
    "backbone_early",
    "backbone_deep",
    "backbone_attention_safe",
    "neck",
    "masf",
    "neck_attention_safe",
    "detect_one2one_tower",
    "detect_one2one_predictor",
    "pose_one2one_tower",
    "pose_one2one_predictor",
)
INHERITED_ROUTE_IDS = (
    "fixed-sd4-neck-attention",
    "fixed-sd4-detect-tower",
    "fixed-sd4-detect-predictor",
    "layer-backbone-high",
    "layer-neck-high",
    "layer-head-high",
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value

CPU_FORMAT_IDS = {
    "w8": "uniform-w8-per_output_channel-optimal_scaled_codebook",
    "w7": "uniform-w7-per_output_channel-optimal_scaled_codebook",
    "w6": "uniform-w6-per_output_channel-optimal_scaled_codebook",
    "w5": "uniform-w5-per_output_channel-optimal_scaled_codebook",
    "w4": "uniform-w4-per_output_channel-optimal_scaled_codebook",
    "fixed-sd4": "fixed-sd4-per_output_channel-optimal_scaled_codebook",
    "exact-ternary": "exact-scaled-ternary-per_tensor",
    "twn-v3": "twn-v3-0.75-filterwise",
    "paper-twn-v2": "paper-twn-layerwise",
}

SPECIAL_FORMAT_ORDER = (
    "fixed-sd4",
    "exact-ternary",
    "twn-v3",
    "paper-twn-v2",
)
UNIFORM_FORMAT_ORDER = ("w6", "w5", "w7", "w4")

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

W8_FORMAT = {
    "family": "uniform",
    "bits": 8,
    "scale_method": "optimal_scaled_codebook",
}

FORMAT_PAYLOADS = {
    "fixed-sd4": {
        "family": "fixed_sd4",
        "scale_method": "optimal_scaled_codebook",
    },
    "exact-ternary": {
        "family": "exact_scaled_ternary",
        "scale_method": "optimal_scaled_codebook",
    },
    "twn-v3": {"family": "twn_filterwise", "threshold_multiplier": 0.75},
    "paper-twn-v2": {"family": "paper_twn", "threshold_multiplier": 0.7},
    "w7": {"family": "uniform", "bits": 7, "scale_method": "optimal_scaled_codebook"},
    "w6": {"family": "uniform", "bits": 6, "scale_method": "optimal_scaled_codebook"},
    "w5": {"family": "uniform", "bits": 5, "scale_method": "optimal_scaled_codebook"},
    "w4": {"family": "uniform", "bits": 4, "scale_method": "optimal_scaled_codebook"},
}


def build_cpu_analysis_plan() -> WeightAnalysisPlan:
    """Return the all-layer, deployment-view CPU sensitivity matrix."""

    return WeightAnalysisPlan(
        view_names=("deployment",),
        uniform_bits=(8, 7, 6, 5, 4),
        uniform_granularities=("per_output_channel",),
        uniform_scale_methods=("optimal_scaled_codebook",),
        fixed_sd4_granularities=("per_output_channel",),
        fixed_sd4_scale_methods=("optimal_scaled_codebook",),
        include_fixed_sd4=True,
        include_paper_twn=True,
        include_filterwise_twn=True,
        include_exact_scaled_ternary=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite drifted artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _write_new_text(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite drifted artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_status_payload(
    status: str,
    *,
    current_index: int | None,
    current_candidate: str | None,
    current_arm: str | None,
    completed_jobs: int,
    error: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return the stable low-token status schema consumed by the monitor."""

    if not status.strip() or completed_jobs < 0:
        raise ValueError("queue status and completed_jobs must be valid")
    return {
        "schema_version": 1,
        "status": status,
        "current_index": current_index,
        "current_candidate": current_candidate,
        "current_arm": current_arm,
        "completed_jobs": completed_jobs,
        "error": None if error is None else dict(error),
        "time_unix": time.time(),
    }


def build_cpu_profile_payload(
    *,
    parent: Mapping[str, object],
    summary: Mapping[str, object],
    measurements: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Create a self-describing CPU-only profile; metrics remain ranking-only."""

    return {
        "schema_version": 1,
        "kind": "full35_v36_static_weight_format_analysis",
        "status": "completed",
        "gpu_used": False,
        "formal_training": False,
        "formal_validation": False,
        "selection_claim": False,
        "parent": dict(parent),
        "analysis_contract": {
            "scope": "all_148_deployment_weight_paths",
            "view": "deployment",
            "formats": list(CPU_FORMAT_IDS),
            "exact_solver": "cvpr2021_optimal_scaled_codebook_event_sweep",
            "cpu_metrics_are_ranking_only": True,
        },
        "summary": dict(summary),
        "measurements": [dict(row) for row in measurements],
    }


def prepare_cpu_profile(
    *,
    parent_manifest: Path = PARENT_MANIFEST,
    output: Path = CPU_PROFILE,
) -> dict[str, object]:
    """Reconstruct the V35 learned deployment graph and profile all 148 paths on CPU."""

    from .progressive_preparation import LockedQATParentSpec
    from .qat_runtime import Full35QATRuntime
    from .weight_formats import WeightFormatAnalyzer

    parent = LockedQATParentSpec.from_yaml(parent_manifest)
    runtime = Full35QATRuntime.from_yaml(parent.plan_path)
    started = time.perf_counter()
    loaded = runtime.load_deployment_parent(
        parent.inference_checkpoint,
        checkpoint_sha256=parent.inference_sha256,
        full_resume_sha256=parent.full_resume_sha256,
        epoch=parent.selected_epoch,
    )
    devices = sorted({tensor.device.type for tensor in loaded.model.state_dict().values()})
    if devices != ["cpu"]:
        raise RuntimeError(f"V36 CPU profile escaped CPU: {devices}")
    analysis = WeightFormatAnalyzer().analyze_deployment(
        loaded.model,
        build_cpu_analysis_plan(),
    )
    measurements = [item.to_dict() for item in analysis.measurements]
    summary = summarize_cpu_measurements(
        measurements,
        expected_paths=parent.deployment_modules,
    )
    payload = build_cpu_profile_payload(
        parent={
            "manifest": str(parent.config_path),
            "manifest_sha256": parent.config_sha256,
            "parent_id": parent.parent_id,
            "selected_epoch": parent.selected_epoch,
            "activation": parent.activation_name,
            "activation_bits": parent.activation_bits,
            "inference_checkpoint": str(parent.inference_checkpoint),
            "inference_checkpoint_sha256": parent.inference_sha256,
            "full_resume_checkpoint": str(parent.full_resume_checkpoint),
            "full_resume_checkpoint_sha256": parent.full_resume_sha256,
            "deployment_modules": parent.deployment_modules,
            "deployment_weight_elements": parent.deployment_weight_elements,
        },
        summary=summary,
        measurements=measurements,
    )
    payload["view_catalogs"] = analysis.view_catalogs
    payload["elapsed_seconds"] = time.perf_counter() - started
    _write_new_json(output, payload)
    result = {
        "status": "completed",
        "profile": str(output.resolve()),
        "profile_sha256": _sha256(output),
        "deployment_paths": summary["coverage"]["deployment_paths"],
        "measurements": summary["coverage"]["measurements"],
        "gpu_used": False,
    }
    del analysis, loaded
    gc.collect()
    return result


def _segment(region: str) -> str:
    matches = tuple(
        segment for segment, regions in SEGMENT_REGIONS.items() if region in regions
    )
    if len(matches) != 1:
        raise ValueError(f"weight region does not map to one segment: {region}")
    return matches[0]


def build_safe_cohort_routes(
    rows: Sequence[Mapping[str, object]],
    *,
    inherited_paths: set[str],
    format_ids: Sequence[str] = SPECIAL_FORMAT_ORDER,
    maximum_paths_per_cohort: int = 4,
) -> dict[str, list[str]]:
    """Rank each special format per segment and return disjoint parent-safe cohorts."""

    if maximum_paths_per_cohort < 1:
        raise ValueError("maximum_paths_per_cohort must be positive")
    routes: dict[str, list[str]] = {}
    for segment in SEGMENT_REGIONS:
        segment_rows = [
            row
            for row in rows
            if _segment(str(row["region"])) == segment
            and str(row["path"]) not in inherited_paths
        ]
        if not segment_rows:
            raise ValueError(f"no unclaimed layer remains in segment: {segment}")
        for format_id in format_ids:
            ranked: list[tuple[float, str]] = []
            for row in segment_rows:
                formats = row.get("formats")
                if not isinstance(formats, Mapping) or format_id not in formats:
                    raise ValueError(
                        f"CPU evidence is missing {format_id} for {row['path']}"
                    )
                numeric = formats[format_id]
                if not isinstance(numeric, Mapping):
                    raise TypeError("CPU format evidence must be a mapping")
                ranked.append(
                    (float(numeric["normalized_rmse"]), str(row["path"]))
                )
            ranked.sort()
            routes[f"special-{segment}-{format_id}"] = [
                path for _, path in ranked[:maximum_paths_per_cohort]
            ]
    return routes


def summarize_cpu_measurements(
    measurements: Sequence[Mapping[str, object]],
    *,
    expected_paths: int = 148,
) -> dict[str, object]:
    """Normalize analyzer rows into one independent nine-format record per path."""

    cpu_to_short = {value: key for key, value in CPU_FORMAT_IDS.items()}
    grouped: dict[str, dict[str, object]] = {}
    for measurement in measurements:
        if measurement.get("view") != "deployment":
            continue
        cpu_format_id = str(measurement.get("format_id"))
        short_id = cpu_to_short.get(cpu_format_id)
        if short_id is None:
            continue
        path = str(measurement["path"])
        region = str(measurement["region"])
        row = grouped.setdefault(
            path,
            {
                "path": path,
                "region": region,
                "segment": _segment(region),
                "elements": int(measurement["elements"]),
                "formats": {},
            },
        )
        if row["region"] != region or row["elements"] != int(
            measurement["elements"]
        ):
            raise ValueError(f"CPU layer identity drifted: {path}")
        formats = row["formats"]
        if not isinstance(formats, dict) or short_id in formats:
            raise ValueError(f"duplicate CPU format evidence: {path}/{short_id}")
        numeric = measurement.get("numeric")
        if not isinstance(numeric, Mapping):
            raise TypeError("CPU numeric evidence must be a mapping")
        formats[short_id] = {
            "cpu_format_id": cpu_format_id,
            "normalized_rmse": float(numeric["normalized_rmse"]),
            "cosine": float(numeric["cosine"]),
            "packed_bytes": int(measurement["code_bytes"])
            + int(measurement["metadata_bytes"]),
        }

    if len(grouped) != expected_paths:
        raise ValueError(
            f"CPU profile path coverage drifted: {len(grouped)} != {expected_paths}"
        )
    expected_formats = set(CPU_FORMAT_IDS)
    rows = sorted(grouped.values(), key=lambda row: str(row["path"]))
    for row in rows:
        formats = row["formats"]
        if not isinstance(formats, dict) or set(formats) != expected_formats:
            raise ValueError(f"CPU profile format coverage drifted: {row['path']}")
        row["formats"] = {
            format_id: formats[format_id] for format_id in CPU_FORMAT_IDS
        }
    return {
        "coverage": {
            "deployment_paths": len(rows),
            "formats_per_path": len(CPU_FORMAT_IDS),
            "measurements": len(rows) * len(CPU_FORMAT_IDS),
        },
        "path_rankings": rows,
    }


def build_phase_candidates(
    rows: Sequence[Mapping[str, object]],
    *,
    phase: str,
    regions: Sequence[str],
    inherited_routes: Sequence[tuple[str, Mapping[str, object]]] = (),
    inherited_paths: set[str] | None = None,
    cohort_size: int = 4,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Build one-candidate-per-format cohorts while keeping W8 full coverage."""

    if phase not in {"special", "uniform"}:
        raise ValueError("phase must be special or uniform")
    if cohort_size < 1:
        raise ValueError("cohort_size must be positive")
    order = SPECIAL_FORMAT_ORDER if phase == "special" else UNIFORM_FORMAT_ORDER
    inherited = inherited_paths or set()
    routes: dict[str, list[str]] = {}
    candidates: list[dict[str, Any]] = []
    region_set = set(regions)
    for segment, segment_regions in SEGMENT_REGIONS.items():
        segment_rows = [
            row
            for row in rows
            if str(row["region"]) in region_set
            and str(row["region"]) in segment_regions
            and str(row["path"]) not in inherited
        ]
        if not segment_rows:
            continue
        for format_id in order:
            format_payload = FORMAT_PAYLOADS[format_id]
            ranked: list[tuple[float, str]] = []
            for row in segment_rows:
                formats = row.get("formats")
                if not isinstance(formats, Mapping) or format_id not in formats:
                    raise ValueError(
                        f"CPU evidence is missing {format_id} for {row['path']}"
                    )
                numeric = formats[format_id]
                if not isinstance(numeric, Mapping):
                    raise TypeError("CPU format evidence must be a mapping")
                ranked.append((float(numeric["normalized_rmse"]), str(row["path"])))
            ranked.sort()
            route_id = f"{phase}-{segment}-{format_id}"
            routes[route_id] = [path for _, path in ranked[:cohort_size]]
            candidates.append(
                full_coverage_candidate(
                    candidate_id=route_id,
                    regions=regions,
                    path_routes=(
                        *inherited_routes,
                        (route_id, format_payload),
                    ),
                )
            )
    if not candidates:
        raise ValueError(f"no {phase} candidates were built")
    return routes, candidates


def build_phase_plan_payload(
    base_payload: Mapping[str, object],
    *,
    plan_id: str,
    phase: str,
    candidates: Sequence[Mapping[str, object]],
    new_routes: Mapping[str, Sequence[str]],
    route_manifest_path: str,
    route_manifest_sha256: str,
    cpu_profile_path: str,
    cpu_profile_sha256: str,
) -> dict[str, Any]:
    """Clone the reviewed V35 plan and pin a full-coverage phase."""

    if phase not in {"special", "uniform", "final"}:
        raise ValueError("phase must be special, uniform, or final")
    if not plan_id.strip() or not candidates:
        raise ValueError("phase plan requires an id and candidates")
    payload = copy.deepcopy(dict(base_payload))
    payload["plan_id"] = plan_id
    payload["date"] = "2026-09-06"
    payload["candidates"] = [dict(candidate) for candidate in candidates]
    base_routes = payload.get("routes")
    if not isinstance(base_routes, dict):
        raise TypeError("V35 base routes must be a mapping")
    route_map = copy.deepcopy(base_routes)
    for route_id in new_routes:
        route_map[route_id] = {
            "manifest": {
                "path": route_manifest_path,
                "sha256": route_manifest_sha256,
            },
            "list_key": route_id,
        }
    used_route_ids: list[str] = []
    for candidate in candidates:
        raw_routes = candidate.get("path_routes")
        if not isinstance(raw_routes, list):
            raise TypeError("phase candidate path_routes must be a list")
        for route in raw_routes:
            if not isinstance(route, Mapping):
                raise TypeError("phase candidate route must be a mapping")
            route_id = str(route["route_id"])
            if route_id not in route_map:
                raise ValueError(f"phase candidate references unknown route: {route_id}")
            if route_id not in used_route_ids:
                used_route_ids.append(route_id)
    payload["routes"] = {route_id: route_map[route_id] for route_id in used_route_ids}
    payload["execution_authorization"] = {
        "authorization_id": f"user-2026-09-06-v36-{phase}",
        "validation_roles": ["candidate"],
        "candidate_ids": [str(candidate["candidate_id"]) for candidate in candidates],
        "route_ids": used_route_ids,
        "training": False,
        "stop_after": f"{phase}_dual_regate",
    }
    payload["formal_training"] = False
    payload["formal_validation"] = False
    payload["successor_provenance"] = {
        "phase": phase,
        "parent": "v35-qsilu-a8-mixed-11path-qat-epoch3",
        "activation": "qsilu_pq_a8",
        "cpu_profile": {
            "path": cpu_profile_path,
            "sha256": cpu_profile_sha256,
        },
        "full_coverage": True,
        "all_deployment_paths_default_to": "w8",
        "special_formats_before_uniform": True,
        "map50_max_drop": 0.015,
        "map50_95_max_drop": 0.04,
        "formal_training": False,
        "formal_validation": False,
    }
    return payload


def build_route_manifest_payload(
    *,
    manifest_id: str,
    routes: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Build a route manifest that is safe to pin into a mixed-policy plan."""

    if not manifest_id.strip():
        raise ValueError("route manifest requires an id")
    payload: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "date": "2026-09-06",
        "execution_authorized": False,
        "map_validation_run": False,
        "selection_claim": False,
    }
    for route_id, paths in routes.items():
        encoded = [str(path) for path in paths]
        if not route_id.strip() or not encoded or len(set(encoded)) != len(encoded):
            raise ValueError(f"invalid route manifest entry: {route_id}")
        payload[route_id] = encoded
    return payload


def materialize_phase_plan(
    *,
    base_payload: Mapping[str, object],
    plan_path: Path,
    route_manifest_path: Path,
    phase: str,
    candidates: Sequence[Mapping[str, object]],
    new_routes: Mapping[str, Sequence[str]],
    cpu_profile_path: Path,
) -> Any:
    """Write a phase's route/plan artifacts and parse them through the strict seam."""

    route_payload = build_route_manifest_payload(
        manifest_id=f"v36-{phase}-routes-v1",
        routes=new_routes,
    )
    route_text = yaml.safe_dump(route_payload, allow_unicode=True, sort_keys=False)
    _write_new_text(route_manifest_path, route_text)
    payload = build_phase_plan_payload(
        base_payload,
        plan_id=f"v36-qsilu-full-coverage-{phase}-ptq-v1",
        phase=phase,
        candidates=candidates,
        new_routes=new_routes,
        route_manifest_path=str(route_manifest_path.resolve().relative_to(PROJECT_ROOT)),
        route_manifest_sha256=_sha256(route_manifest_path),
        cpu_profile_path=str(cpu_profile_path.resolve().relative_to(PROJECT_ROOT)),
        cpu_profile_sha256=_sha256(cpu_profile_path),
    )
    _write_new_text(
        plan_path,
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
    )
    from .mixed_policy_search import Full35MixedPolicySearchPlan

    return Full35MixedPolicySearchPlan.from_yaml(plan_path)


def select_phase_routes(
    *,
    phase: str,
    report: Mapping[str, object],
    dual: Mapping[str, object],
    candidate_formats: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Select one green/recover route per segment using accuracy then bytes."""

    if phase not in {"special", "uniform"}:
        raise ValueError("phase must be special or uniform")
    raw_results = report.get("results")
    raw_dual = dual.get("candidates")
    if not isinstance(raw_results, Mapping) or not isinstance(raw_dual, Mapping):
        raise TypeError("phase report and dual candidates must be mappings")
    grouped: dict[str, list[dict[str, object]]] = {
        segment: [] for segment in SEGMENT_REGIONS
    }
    for candidate_id, raw_gate in raw_dual.items():
        candidate_id = str(candidate_id)
        prefix = f"{phase}-"
        if not candidate_id.startswith(prefix):
            continue
        remainder = candidate_id.removeprefix(prefix)
        segment, separator, _ = remainder.partition("-")
        if not separator or segment not in grouped:
            continue
        gate = raw_gate
        result = raw_results.get(candidate_id)
        if not isinstance(gate, Mapping) or not isinstance(result, Mapping):
            raise TypeError(f"phase candidate evidence is malformed: {candidate_id}")
        if gate.get("decision") not in {"green", "recover"}:
            continue
        weight = result.get("weight_quantization")
        if not isinstance(weight, Mapping) or not isinstance(weight.get("formats"), list):
            raise ValueError(f"phase candidate has no weight evidence: {candidate_id}")
        packed_bytes = 0
        for raw_format in weight["formats"]:
            if not isinstance(raw_format, Mapping):
                raise TypeError("phase format evidence must be a mapping")
            numeric = raw_format.get("numeric")
            if not isinstance(numeric, Mapping):
                raise TypeError("phase numeric evidence must be a mapping")
            packed_bytes += int(numeric["weight_code_bytes"]) + int(
                numeric["scale_bytes"]
            )
        if candidate_id not in candidate_formats:
            raise ValueError(f"missing format payload for {candidate_id}")
        grouped[segment].append(
            {
                "route_id": candidate_id,
                "format": dict(candidate_formats[candidate_id]),
                "decision": str(gate["decision"]),
                "worst_map50_delta": float(
                    gate.get("worst_total_delta", gate.get("worst_map50_delta", -1.0))
                ),
                "worst_map50_95_delta": float(
                    gate.get(
                        "worst_total_map50_95_delta",
                        gate.get("worst_map50_95_delta", -1.0),
                    )
                ),
                "packed_bytes": packed_bytes,
            }
        )
    selected: list[dict[str, object]] = []
    for segment in SEGMENT_REGIONS:
        options = grouped[segment]
        if not options:
            continue
        options.sort(
            key=lambda row: (
                -float(row["worst_map50_delta"]),
                -float(row["worst_map50_95_delta"]),
                int(row["packed_bytes"]),
                str(row["route_id"]),
            )
        )
        selected.append(options[0])
    return selected


def _load_base_payload() -> dict[str, Any]:
    if _sha256(V35_BASE_PLAN) != V35_BASE_PLAN_SHA256:
        raise RuntimeError("V35 base plan SHA-256 drifted")
    return _mapping(
        yaml.safe_load(V35_BASE_PLAN.read_text(encoding="utf-8")),
        "V35 base plan",
    )


def _load_profile_rows(profile_path: Path = CPU_PROFILE) -> list[dict[str, object]]:
    payload = _mapping(
        json.loads(profile_path.read_text(encoding="utf-8")),
        "V36 CPU profile",
    )
    if (
        payload.get("status") != "completed"
        or payload.get("gpu_used") is not False
        or payload.get("formal_training") is not False
        or payload.get("selection_claim") is not False
    ):
        raise ValueError("V36 CPU profile is not a completed ranking-only artifact")
    parent = _mapping(payload.get("parent"), "V36 profile parent")
    if parent.get("parent_id") != "v35-qsilu-a8-mixed-11path-qat-epoch3":
        raise ValueError("V36 CPU profile parent differs from locked V35 parent")
    summary = _mapping(payload.get("summary"), "V36 profile summary")
    rows = summary.get("path_rankings")
    if not isinstance(rows, list) or len(rows) != 148:
        raise ValueError("V36 CPU profile must contain 148 path rankings")
    return [_mapping(row, "V36 path ranking") for row in rows]


def _inherited_routes(
    base_payload: Mapping[str, object],
) -> tuple[list[tuple[str, Mapping[str, object]]], set[str]]:
    candidates = base_payload.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError("V35 base candidates must be a list")
    final = next(
        (
            _mapping(candidate, "V35 final candidate")
            for candidate in candidates
            if _mapping(candidate, "V35 candidate").get("candidate_id")
            == "cumulative-head-w4"
        ),
        None,
    )
    if final is None:
        raise ValueError("V35 base final candidate is missing")
    route_formats = {
        str(_mapping(route, "V35 inherited route")["route_id"]): _mapping(
            route, "V35 inherited route"
        )["format"]
        for route in _mapping(final, "V35 final candidate")["path_routes"]
    }
    raw_routes = _mapping(base_payload.get("routes"), "V35 base routes")
    inherited: list[tuple[str, Mapping[str, object]]] = []
    paths: set[str] = set()
    for route_id in INHERITED_ROUTE_IDS:
        if route_id not in route_formats or route_id not in raw_routes:
            raise ValueError(f"V35 inherited route is missing: {route_id}")
        route = _mapping(raw_routes[route_id], f"V35 route {route_id}")
        manifest = _mapping(route["manifest"], f"V35 route manifest {route_id}")
        manifest_path = Path(str(manifest["path"]))
        manifest_path = (
            manifest_path.resolve()
            if manifest_path.is_absolute()
            else (PROJECT_ROOT / manifest_path).resolve()
        )
        if _sha256(manifest_path) != str(manifest["sha256"]):
            raise RuntimeError(f"V35 inherited route SHA-256 drifted: {route_id}")
        manifest_payload = _mapping(
            yaml.safe_load(manifest_path.read_text(encoding="utf-8")),
            f"V35 route manifest payload {route_id}",
        )
        encoded_paths = manifest_payload.get(str(route["list_key"]))
        if not isinstance(encoded_paths, list) or not encoded_paths:
            raise ValueError(f"V35 inherited route list is empty: {route_id}")
        route_paths = [str(path) for path in encoded_paths]
        if paths.intersection(route_paths):
            raise ValueError("V35 inherited routes overlap")
        paths.update(route_paths)
        inherited.append((route_id, _mapping(route_formats[route_id], "V35 format")))
    if len(paths) != 11:
        raise ValueError(f"V35 inherited policy must contain 11 paths, got {len(paths)}")
    return inherited, paths


def _transition(
    status: str,
    *,
    current_index: int | None,
    current_candidate: str | None,
    current_arm: str | None,
    completed_jobs: int,
    error: Mapping[str, object] | None = None,
    **extra: object,
) -> None:
    payload = build_status_payload(
        status,
        current_index=current_index,
        current_candidate=current_candidate,
        current_arm=current_arm,
        completed_jobs=completed_jobs,
        error=error,
    )
    payload["values"] = extra
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    with (QUEUE_ROOT / "execution-events.jsonl").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    _atomic_json(QUEUE_ROOT / "execution-status.json", payload)


def _wait_for_gpu(*, device_index: int, poll_seconds: int, completed_jobs: int) -> None:
    from .progressive_queue import _foreign_gpu_pids

    while True:
        foreign = _foreign_gpu_pids(device_index)
        if not foreign:
            return
        _transition(
            "waiting_for_gpu",
            current_index=None,
            current_candidate=None,
            current_arm=None,
            completed_jobs=completed_jobs,
            foreign_pids=list(foreign),
            poll_seconds=poll_seconds,
        )
        time.sleep(poll_seconds)


def _run_ptq_phase(
    *,
    base_payload: Mapping[str, object],
    phase: str,
    candidates: Sequence[Mapping[str, object]],
    new_routes: Mapping[str, Sequence[str]],
    cpu_profile_path: Path,
    plan_path: Path,
    route_manifest_path: Path,
    report_path: Path,
    dual_path: Path,
    device_index: int,
    poll_seconds: int,
    completed_jobs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = materialize_phase_plan(
        base_payload=base_payload,
        plan_path=plan_path,
        route_manifest_path=route_manifest_path,
        phase=phase,
        candidates=candidates,
        new_routes=new_routes,
        cpu_profile_path=cpu_profile_path,
    )
    report: dict[str, Any] | None = None
    if report_path.is_file():
        candidate = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "PTQ report")
        if candidate.get("status") == "completed":
            report = candidate
    if report is None:
        _wait_for_gpu(
            device_index=device_index,
            poll_seconds=poll_seconds,
            completed_jobs=completed_jobs,
        )
        _transition(
            f"{phase}_started",
            current_index={"special": 1, "uniform": 2, "final": 3}[phase],
            current_candidate=None,
            current_arm="ptq",
            completed_jobs=completed_jobs,
            candidate_count=len(candidates),
            plan=str(plan.config_path),
        )
        from .mixed_policy_search import run_mixed_policy_search

        code = run_mixed_policy_search(
            plan=plan,
            output=report_path,
            device_index=device_index,
            resume=report_path.exists(),
        )
        if code:
            raise RuntimeError(f"V36 {phase} PTQ returned {code}")
        report = _mapping(
            json.loads(report_path.read_text(encoding="utf-8")),
            f"V36 {phase} PTQ report",
        )
    if report.get("status") != "completed":
        raise RuntimeError(f"V36 {phase} PTQ report is incomplete")
    from .dual_metric_regate import regate_candidate_report

    dual = regate_candidate_report(
        source_report=report_path,
        expected_source_sha256=_sha256(report_path),
        metric_contract_id=plan.metric_contract_id,
    )
    _write_new_json(dual_path, dual)
    _transition(
        f"{phase}_complete",
        current_index={"special": 1, "uniform": 2, "final": 3}[phase],
        current_candidate=None,
        current_arm="ptq",
        completed_jobs=completed_jobs + 1,
        report=str(report_path),
        dual_report=str(dual_path),
    )
    return report, dual


def full_coverage_candidate(
    *,
    candidate_id: str,
    regions: Sequence[str],
    path_routes: Sequence[tuple[str, Mapping[str, object]]] = (),
) -> dict[str, Any]:
    """Build one policy with W8 defaults and explicit lower-bit path overrides."""

    if not candidate_id.strip() or not regions or len(set(regions)) != len(regions):
        raise ValueError("candidate identity and unique regions are required")
    return {
        "candidate_id": candidate_id,
        "region_defaults": [
            {"region": region, "format": dict(W8_FORMAT)} for region in regions
        ],
        "path_routes": [
            {"route_id": route_id, "format": dict(format_spec)}
            for route_id, format_spec in path_routes
        ],
    }


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _qat_format_payload(format_spec: Mapping[str, object]) -> dict[str, object]:
    """Translate a PTQ route format into the strict QAT format vocabulary."""

    family = str(format_spec.get("family", ""))
    if family == "fixed_sd4":
        return {
            "family": "ls_sd4",
            "scale_method": str(
                format_spec.get("scale_method", "optimal_scaled_codebook")
            ),
        }
    if family == "uniform":
        bits = int(format_spec["bits"])
        if bits not in {4, 5, 6, 7, 8}:
            raise ValueError(f"unsupported uniform QAT bits: {bits}")
        return {
            "family": "uniform",
            "bits": bits,
            "scale_method": str(
                format_spec.get("scale_method", "optimal_scaled_codebook")
            ),
        }
    if family in {"paper_twn", "twn_filterwise", "exact_scaled_ternary"}:
        result = dict(format_spec)
        if family == "exact_scaled_ternary":
            result.setdefault("scale_method", "optimal_scaled_codebook")
        return result
    raise ValueError(f"unsupported QAT route format: {family}")


def _resolve_route_paths(
    base_payload: Mapping[str, object],
    route_ids: Sequence[str],
    new_routes: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Resolve route manifests once and verify their pinned hashes."""

    raw_routes = _mapping(base_payload.get("routes"), "base route map")
    resolved: dict[str, list[str]] = {}
    for route_id in route_ids:
        route_id = str(route_id)
        if route_id in new_routes:
            paths = [str(path) for path in new_routes[route_id]]
            if not paths or len(paths) != len(set(paths)):
                raise ValueError(f"new route is empty or overlaps itself: {route_id}")
            resolved[route_id] = paths
            continue
        raw_route = _mapping(raw_routes.get(route_id), f"route {route_id}")
        manifest = _mapping(raw_route.get("manifest"), f"route manifest {route_id}")
        path = Path(str(manifest["path"]))
        path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if _sha256(path) != str(manifest["sha256"]):
            raise RuntimeError(f"route manifest SHA-256 drifted: {route_id}")
        payload = _mapping(
            yaml.safe_load(path.read_text(encoding="utf-8")),
            f"route manifest payload {route_id}",
        )
        paths = payload.get(str(raw_route["list_key"]))
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"route list is empty: {route_id}")
        resolved[route_id] = [str(value) for value in paths]
    all_paths = [path for paths in resolved.values() for path in paths]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("selected route paths overlap")
    return resolved


def _build_qat_assignments(
    *,
    rows: Sequence[Mapping[str, object]],
    route_specs: Sequence[tuple[str, Mapping[str, object]]],
    route_paths: Mapping[str, Sequence[str]],
) -> list[dict[str, object]]:
    path_regions = {str(row["path"]): str(row["region"]) for row in rows}
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    routed_paths: set[str] = set()
    for route_id, format_spec in route_specs:
        paths = route_paths.get(route_id)
        if paths is None:
            raise ValueError(f"missing paths for selected route: {route_id}")
        qat_format = _qat_format_payload(format_spec)
        format_key = json.dumps(qat_format, sort_keys=True)
        for path in paths:
            path = str(path)
            if path not in path_regions:
                raise ValueError(f"selected route path missing from CPU profile: {path}")
            if path in routed_paths:
                raise ValueError(f"selected route path appears twice: {path}")
            routed_paths.add(path)
            region = path_regions[path]
            key = (region, format_key)
            entry = grouped.setdefault(
                key,
                {"region": region, "paths": [], "format": qat_format},
            )
            entry["paths"].append(path)  # type: ignore[index]

    # The locked V35 checkpoint has qparams only for its eleven old paths.  Make
    # every remaining W8 site an explicit path override so the warm-start seam
    # can safely reset all newly introduced qparams instead of treating them as
    # an incompatible graph change.
    w8_key = json.dumps(dict(W8_FORMAT), sort_keys=True)
    for path in sorted(set(path_regions) - routed_paths):
        region = path_regions[path]
        key = (region, w8_key)
        entry = grouped.setdefault(
            key,
            {"region": region, "paths": [], "format": dict(W8_FORMAT)},
        )
        entry["paths"].append(path)  # type: ignore[index]
        routed_paths.add(path)

    assignments: list[dict[str, object]] = [
        {"region": region, "paths": [], "format": dict(W8_FORMAT)}
        for region in ALL_REGIONS
    ]
    assignments.extend(grouped.values())
    if len(routed_paths) != len(path_regions):
        raise AssertionError("full deployment path coverage invariant failed")
    if len(routed_paths) != len(set(routed_paths)):
        raise AssertionError("route path uniqueness invariant failed")
    return assignments


def build_continuation_qat_payload(
    *,
    final_plan_path: Path,
    final_report_path: Path,
    final_dual_path: Path,
    candidate_id: str,
    assignments: Sequence[Mapping[str, object]],
    parent_manifest: Path = PARENT_MANIFEST,
) -> dict[str, object]:
    """Create a short QAT-only continuation pinned to the V35 external sham."""

    if _sha256(V35_BASE_QAT_PLAN) != V35_BASE_QAT_PLAN_SHA256:
        raise RuntimeError("V35 QAT plan SHA-256 drifted")
    for path in (final_plan_path, final_report_path, final_dual_path, parent_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = yaml.safe_load(V35_BASE_QAT_PLAN.read_text(encoding="utf-8"))
    payload = _mapping(payload, "V35 QAT payload")
    payload["plan_id"] = "v36-qsilu-full-coverage-short-qat-v1"
    payload["date"] = "2026-09-06"
    payload["execution_authorized"] = True
    payload["formal_validation"] = False
    payload["execution_authorization"] = {
        "authorization_id": "user-2026-09-06-v36-full-coverage-short-qat",
        "arms": ["qat"],
        "scope": "continuation_from_locked_v35_parent_no_new_sham",
        "external_control": {
            "mode": "v35_external_sham_reference",
            "plan": _path_record(V35_BASE_QAT_PLAN),
            "completion": _path_record(V35_SHAM_COMPLETION),
            "metrics": _path_record(V35_SHAM_METRICS),
        },
    }
    sources = _mapping(payload["sources"], "V35 QAT sources")
    sources["weight_plan"] = _path_record(final_plan_path)
    sources["candidate_evidence"] = _path_record(final_report_path)
    sources["candidate_dual_regate"] = _path_record(final_dual_path)
    payload["sources"] = sources
    payload["warm_start"] = {
        "locked_parent": _path_record(parent_manifest),
        "checkpoint_role": "full_resume",
        "state_key": "ema_state",
        "reset_path_override_quantizers": True,
        "optimizer_state": "fresh",
    }
    payload["weight_policy"] = {
        "candidate_id": candidate_id,
        "float_regions": [],
        "assignments": [dict(item) for item in assignments],
    }
    training = _mapping(payload["training"], "V35 QAT training")
    training.update(
        {
            "epochs": 3,
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
    payload["training"] = training
    payload["run_root"] = "artifacts/runs/qat/v36-qsilu-full-coverage-short-v2"
    payload["quick_recovery_provenance"] = {
        "selection": "cpu_sensitive_special_then_uniform_full_coverage",
        "candidate_id": candidate_id,
        "parent_manifest": _path_record(parent_manifest),
        "deployment_paths": 148,
        "quantization_default": "w8",
        "short_qat_epochs": 3,
        "patience": 5,
        "no_new_sham": True,
        "formal_validation": False,
    }
    payload["successor_provenance"] = {
        "parent": "v35-qsilu-a8-mixed-11path-qat-epoch3",
        "activation": "qsilu_pq_a8",
        "scope": "all_148_deployment_weight_paths",
        "special_formats_first": list(SPECIAL_FORMAT_ORDER),
        "uniform_formats": list(UNIFORM_FORMAT_ORDER),
        "map50_max_drop": 0.015,
        "map50_95_max_drop": 0.04,
    }
    return payload


def materialize_continuation_qat_plan(
    *, path: Path, payload: Mapping[str, object]
) -> Any:
    _write_new_text(path, yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False))
    from .qat_plan import Full35QATPlan

    return Full35QATPlan.from_yaml(path)


def _candidate_formats(candidates: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    formats: dict[str, Mapping[str, object]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        routes = candidate.get("path_routes")
        if not isinstance(routes, list) or not routes:
            raise ValueError(f"candidate has no route format: {candidate_id}")
        formats[candidate_id] = _mapping(routes[-1], "candidate route")["format"]  # type: ignore[assignment]
    return formats


def _selected_route_specs(
    selected: Sequence[Mapping[str, object]],
    route_source: Mapping[str, Sequence[str]],
) -> list[tuple[str, Mapping[str, object]]]:
    result: list[tuple[str, Mapping[str, object]]] = []
    for item in selected:
        route_id = str(item["route_id"])
        if route_id not in route_source:
            raise ValueError(f"selected route is absent from source map: {route_id}")
        result.append((route_id, _mapping(item["format"], "selected route format")))
    return result


def run_queue(*, device_index: int = 0, poll_seconds: int = 600) -> dict[str, object]:
    """Execute the reviewed CPU→PTQ→short-QAT successor queue."""

    if poll_seconds < 30:
        raise ValueError("poll_seconds must be at least 30 seconds")
    completed_jobs = 0
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if CPU_PROFILE.is_file():
            rows = _load_profile_rows()
            completed_jobs = 1
        else:
            _transition(
                "cpu_profile_started",
                current_index=0,
                current_candidate=None,
                current_arm="cpu",
                completed_jobs=0,
            )
            prepare_cpu_profile()
            rows = _load_profile_rows()
            completed_jobs = 1
            _transition(
                "cpu_profile_complete",
                current_index=0,
                current_candidate=None,
                current_arm="cpu",
                completed_jobs=completed_jobs,
                profile=str(CPU_PROFILE),
            )

        base_payload = _load_base_payload()
        inherited_specs, inherited_paths = _inherited_routes(base_payload)
        generated = QUEUE_ROOT / "generated"
        reports = PROJECT_ROOT / "artifacts/reports"
        special_routes, special_candidates = build_phase_candidates(
            rows,
            phase="special",
            regions=ALL_REGIONS,
            inherited_routes=inherited_specs,
            inherited_paths=set(inherited_paths),
        )
        special_report = reports / "v36-qsilu-full-coverage-special-v1.json"
        special_dual = reports / "v36-qsilu-full-coverage-special-dual-v1.json"
        special_report_payload, special_dual_payload = _run_ptq_phase(
            base_payload=base_payload,
            phase="special",
            candidates=special_candidates,
            new_routes=special_routes,
            cpu_profile_path=CPU_PROFILE,
            plan_path=generated / "special-plan.yaml",
            route_manifest_path=generated / "special-routes.yaml",
            report_path=special_report,
            dual_path=special_dual,
            device_index=device_index,
            poll_seconds=poll_seconds,
            completed_jobs=completed_jobs,
        )
        completed_jobs += 1
        selected_special = select_phase_routes(
            phase="special",
            report=special_report_payload,
            dual=special_dual_payload,
            candidate_formats=_candidate_formats(special_candidates),
        )
        special_specs = _selected_route_specs(selected_special, special_routes)
        selected_special_map = {route_id: special_routes[route_id] for route_id, _ in special_specs}

        uniform_inherited_specs = [*inherited_specs, *special_specs]
        uniform_inherited_paths = set(inherited_paths)
        for paths in selected_special_map.values():
            uniform_inherited_paths.update(paths)
        uniform_routes, uniform_candidates = build_phase_candidates(
            rows,
            phase="uniform",
            regions=ALL_REGIONS,
            inherited_routes=uniform_inherited_specs,
            inherited_paths=uniform_inherited_paths,
        )
        uniform_new_routes = {**selected_special_map, **uniform_routes}
        uniform_report = reports / "v36-qsilu-full-coverage-uniform-v1.json"
        uniform_dual = reports / "v36-qsilu-full-coverage-uniform-dual-v1.json"
        uniform_report_payload, uniform_dual_payload = _run_ptq_phase(
            base_payload=base_payload,
            phase="uniform",
            candidates=uniform_candidates,
            new_routes=uniform_new_routes,
            cpu_profile_path=CPU_PROFILE,
            plan_path=generated / "uniform-plan.yaml",
            route_manifest_path=generated / "uniform-routes.yaml",
            report_path=uniform_report,
            dual_path=uniform_dual,
            device_index=device_index,
            poll_seconds=poll_seconds,
            completed_jobs=completed_jobs,
        )
        completed_jobs += 1
        selected_uniform = select_phase_routes(
            phase="uniform",
            report=uniform_report_payload,
            dual=uniform_dual_payload,
            candidate_formats=_candidate_formats(uniform_candidates),
        )
        uniform_specs = _selected_route_specs(selected_uniform, uniform_routes)
        final_specs = [*inherited_specs, *special_specs, *uniform_specs]
        final_new_routes = {
            **selected_special_map,
            **{route_id: uniform_routes[route_id] for route_id, _ in uniform_specs},
        }
        final_candidate = full_coverage_candidate(
            candidate_id="final-full-coverage-policy",
            regions=ALL_REGIONS,
            path_routes=final_specs,
        )
        final_report = reports / "v36-qsilu-full-coverage-final-v1.json"
        final_dual = reports / "v36-qsilu-full-coverage-final-dual-v1.json"
        final_report_payload, final_dual_payload = _run_ptq_phase(
            base_payload=base_payload,
            phase="final",
            candidates=[final_candidate],
            new_routes=final_new_routes,
            cpu_profile_path=CPU_PROFILE,
            plan_path=generated / "final-plan.yaml",
            route_manifest_path=generated / "final-routes.yaml",
            report_path=final_report,
            dual_path=final_dual,
            device_index=device_index,
            poll_seconds=poll_seconds,
            completed_jobs=completed_jobs,
        )
        completed_jobs += 1
        final_gate = _mapping(
            _mapping(final_dual_payload.get("candidates"), "final dual candidates").get(
                "final-full-coverage-policy"
            ),
            "final policy dual gate",
        )
        qat_candidate_id = "final-full-coverage-policy"
        qat_specs = final_specs
        qat_new_routes = final_new_routes
        qat_plan_source = generated / "final-plan.yaml"
        qat_report_source = final_report
        qat_dual_source = final_dual
        if final_gate.get("decision") not in {"green", "recover"}:
            recoverable = [
                item
                for item in selected_special
                if item.get("decision") in {"green", "recover"}
            ]
            if not recoverable:
                _transition(
                    "complete_no_qat_candidate",
                    current_index=3,
                    current_candidate="final-full-coverage-policy",
                    current_arm="ptq",
                    completed_jobs=completed_jobs,
                    decision=final_gate.get("decision"),
                )
                return {"status": "complete_no_qat_candidate", "decision": final_gate.get("decision")}
            stage = max(
                recoverable,
                key=lambda item: (
                    float(item.get("worst_map50_delta", -1.0)),
                    float(item.get("worst_map50_95_delta", -1.0)),
                    -int(item.get("packed_bytes", 0)),
                    str(item["route_id"]),
                ),
            )
            qat_candidate_id = str(stage["route_id"])
            qat_specs = [
                *inherited_specs,
                (qat_candidate_id, _mapping(stage["format"], "staged route format")),
            ]
            qat_new_routes = {qat_candidate_id: special_routes[qat_candidate_id]}
            qat_plan_source = generated / "special-plan.yaml"
            qat_report_source = special_report
            qat_dual_source = special_dual
            _transition(
                "ptq_final_rejected_stage_qat_selected",
                current_index=3,
                current_candidate=qat_candidate_id,
                current_arm="ptq",
                completed_jobs=completed_jobs,
                final_decision=final_gate.get("decision"),
                reason="full-policy PTQ rejected; staged recoverable all-layer W8+LS-SD4 candidate",
            )

        route_paths = _resolve_route_paths(
            base_payload,
            [route_id for route_id, _ in qat_specs],
            qat_new_routes,
        )
        qat_assignments = _build_qat_assignments(
            rows=rows,
            route_specs=qat_specs,
            route_paths=route_paths,
        )
        qat_payload = build_continuation_qat_payload(
            final_plan_path=qat_plan_source,
            final_report_path=qat_report_source,
            final_dual_path=qat_dual_source,
            candidate_id=qat_candidate_id,
            assignments=qat_assignments,
        )
        qat_plan_path = CONTINUATION_QAT_ROOT / "generated/qat-plan.yaml"
        qat_runtime = materialize_continuation_qat_plan(path=qat_plan_path, payload=qat_payload)
        from .qat_runtime import Full35QATRuntime

        runtime = Full35QATRuntime.from_yaml(qat_runtime.config_path)
        completion = runtime.plan.run_root / runtime.run_name("qat") / "qat-experiment.json"
        if not completion.is_file():
            _wait_for_gpu(
                device_index=device_index,
                poll_seconds=poll_seconds,
                completed_jobs=completed_jobs,
            )
            _transition(
                "qat_started",
                current_index=4,
                current_candidate=qat_candidate_id,
                current_arm="qat",
                completed_jobs=completed_jobs,
                plan=str(qat_plan_path),
                epochs=runtime.plan.training.epochs,
                patience=runtime.plan.training.patience,
            )
            runtime.run("qat", device_index=device_index)
        completed_jobs += 1
        _transition(
            "complete",
            current_index=4,
            current_candidate=qat_candidate_id,
            current_arm="qat",
            completed_jobs=completed_jobs,
            qat_completion=str(completion),
        )
        return {
            "status": "complete",
            "completed_jobs": completed_jobs,
            "qat_candidate": qat_candidate_id,
            "qat_plan": str(qat_plan_path),
            "qat_completion": str(completion),
        }
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        _transition(
            "error",
            current_index=None,
            current_candidate=None,
            current_arm=None,
            completed_jobs=completed_jobs,
            error=error,
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-reviewed-queue", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if not args.execute_reviewed_queue:
        parser.error("V36 execution requires --execute-reviewed-queue")
    result = run_queue(device_index=args.device, poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = (
    "CPU_FORMAT_IDS",
    "FORMAT_PAYLOADS",
    "SEGMENT_REGIONS",
    "SPECIAL_FORMAT_ORDER",
    "UNIFORM_FORMAT_ORDER",
    "W8_FORMAT",
    "build_cpu_analysis_plan",
    "build_cpu_profile_payload",
    "build_phase_candidates",
    "build_phase_plan_payload",
    "build_route_manifest_payload",
    "build_safe_cohort_routes",
    "full_coverage_candidate",
    "summarize_cpu_measurements",
    "prepare_cpu_profile",
    "materialize_phase_plan",
    "select_phase_routes",
    "build_continuation_qat_payload",
    "materialize_continuation_qat_plan",
    "build_status_payload",
    "run_queue",
    "main",
)


if __name__ == "__main__":
    raise SystemExit(main())
