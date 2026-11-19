from __future__ import annotations

from pathlib import Path

from yolo_quantize.full_coverage_successor import (
    CPU_FORMAT_IDS,
    FORMAT_PAYLOADS,
    SPECIAL_FORMAT_ORDER,
    UNIFORM_FORMAT_ORDER,
    W8_FORMAT,
    build_phase_candidates,
    build_phase_plan_payload,
    build_cpu_analysis_plan,
    build_cpu_profile_payload,
    build_safe_cohort_routes,
    full_coverage_candidate,
    select_phase_routes,
    build_status_payload,
    summarize_cpu_measurements,
    _build_qat_assignments,
    build_continuation_qat_payload,
)


def _row(path: str, region: str, values: dict[str, float]) -> dict[str, object]:
    return {
        "path": path,
        "region": region,
        "formats": {
            format_id: {"normalized_rmse": value}
            for format_id, value in values.items()
        },
    }


def test_cpu_matrix_covers_all_requested_formats_in_review_order() -> None:
    plan = build_cpu_analysis_plan()

    assert plan.view_names == ("deployment",)
    assert plan.uniform_bits == (8, 7, 6, 5, 4)
    assert plan.uniform_scale_methods == ("optimal_scaled_codebook",)
    assert plan.include_fixed_sd4 is True
    assert plan.include_exact_scaled_ternary is True
    assert plan.include_filterwise_twn is True
    assert plan.include_paper_twn is True
    assert SPECIAL_FORMAT_ORDER == (
        "fixed-sd4",
        "exact-ternary",
        "twn-v3",
        "paper-twn-v2",
    )
    assert UNIFORM_FORMAT_ORDER == ("w6", "w5", "w7", "w4")


def test_safe_cohorts_are_layer_ranked_and_exclude_the_v35_parent_paths() -> None:
    rows = [
        _row("backbone.a", "backbone_early", {"fixed-sd4": 0.20, "twn-v3": 0.50}),
        _row("backbone.b", "backbone_deep", {"fixed-sd4": 0.10, "twn-v3": 0.30}),
        _row("backbone.locked", "backbone_deep", {"fixed-sd4": 0.01, "twn-v3": 0.01}),
        _row("neck.a", "neck", {"fixed-sd4": 0.11, "twn-v3": 0.41}),
        _row("neck.b", "masf", {"fixed-sd4": 0.12, "twn-v3": 0.31}),
        _row("head.a", "detect_one2one_tower", {"fixed-sd4": 0.13, "twn-v3": 0.33}),
        _row("head.b", "pose_one2one_tower", {"fixed-sd4": 0.09, "twn-v3": 0.43}),
    ]

    routes = build_safe_cohort_routes(
        rows,
        inherited_paths={"backbone.locked"},
        format_ids=("fixed-sd4", "twn-v3"),
        maximum_paths_per_cohort=1,
    )

    assert routes["special-backbone-fixed-sd4"] == ["backbone.b"]
    assert routes["special-backbone-twn-v3"] == ["backbone.b"]
    assert routes["special-neck-fixed-sd4"] == ["neck.a"]
    assert routes["special-neck-twn-v3"] == ["neck.b"]
    assert routes["special-head-fixed-sd4"] == ["head.b"]
    assert routes["special-head-twn-v3"] == ["head.a"]
    assert all(
        "backbone.locked" not in paths for paths in routes.values()
    )


def test_candidate_quantizes_every_region_with_w8_then_applies_path_overrides() -> None:
    candidate = full_coverage_candidate(
        candidate_id="anchor-plus-special",
        regions=("backbone_early", "neck", "pose_one2one_tower"),
        path_routes=(
            ("v35-ls-sd4", {"family": "fixed_sd4", "scale_method": "optimal_scaled_codebook"}),
            ("special-backbone-twn-v3", {"family": "twn_filterwise", "threshold_multiplier": 0.75}),
        ),
    )

    assert [item["region"] for item in candidate["region_defaults"]] == [
        "backbone_early",
        "neck",
        "pose_one2one_tower",
    ]
    assert all(
        item["format"]
        == {
            "family": "uniform",
            "bits": 8,
            "scale_method": "optimal_scaled_codebook",
        }
        for item in candidate["region_defaults"]
    )
    assert [item["route_id"] for item in candidate["path_routes"]] == [
        "v35-ls-sd4",
        "special-backbone-twn-v3",
    ]


def test_cpu_summary_keeps_independent_evidence_for_every_layer_and_format() -> None:
    measurements = []
    for path, region in (
        ("backbone.a", "backbone_early"),
        ("neck.a", "neck"),
    ):
        for index, cpu_format_id in enumerate(CPU_FORMAT_IDS.values(), start=1):
            measurements.append(
                {
                    "view": "deployment",
                    "path": path,
                    "region": region,
                    "format_id": cpu_format_id,
                    "elements": 64,
                    "code_bytes": index * 10,
                    "metadata_bytes": 4,
                    "numeric": {
                        "normalized_rmse": index / 100,
                        "cosine": 1.0 - index / 1000,
                    },
                }
            )

    summary = summarize_cpu_measurements(measurements, expected_paths=2)

    assert summary["coverage"] == {
        "deployment_paths": 2,
        "formats_per_path": 9,
        "measurements": 18,
    }
    assert [row["path"] for row in summary["path_rankings"]] == [
        "backbone.a",
        "neck.a",
    ]
    assert list(summary["path_rankings"][0]["formats"]) == list(CPU_FORMAT_IDS)
    assert summary["path_rankings"][0]["formats"]["w8"]["packed_bytes"] == 14


def test_special_phase_candidates_preserve_full_w8_coverage_and_route_order() -> None:
    rows = [
        _row("backbone.a", "backbone_early", {"fixed-sd4": 0.20, "exact-ternary": 0.30, "twn-v3": 0.40, "paper-twn-v2": 0.50}),
        _row("backbone.b", "backbone_deep", {"fixed-sd4": 0.10, "exact-ternary": 0.20, "twn-v3": 0.30, "paper-twn-v2": 0.40}),
        _row("neck.a", "neck", {"fixed-sd4": 0.11, "exact-ternary": 0.21, "twn-v3": 0.31, "paper-twn-v2": 0.41}),
        _row("head.a", "detect_one2one_tower", {"fixed-sd4": 0.12, "exact-ternary": 0.22, "twn-v3": 0.32, "paper-twn-v2": 0.42}),
    ]

    routes, candidates = build_phase_candidates(
        rows,
        phase="special",
        regions=("backbone_early", "backbone_deep", "neck", "detect_one2one_tower"),
        inherited_routes=(("v35-ls-sd4", {"family": "fixed_sd4", "scale_method": "optimal_scaled_codebook"}),),
        inherited_paths={"backbone.locked"},
        cohort_size=1,
    )

    assert list(routes) == [
        "special-backbone-fixed-sd4",
        "special-backbone-exact-ternary",
        "special-backbone-twn-v3",
        "special-backbone-paper-twn-v2",
        "special-neck-fixed-sd4",
        "special-neck-exact-ternary",
        "special-neck-twn-v3",
        "special-neck-paper-twn-v2",
        "special-head-fixed-sd4",
        "special-head-exact-ternary",
        "special-head-twn-v3",
        "special-head-paper-twn-v2",
    ]
    assert len(candidates) == 12
    assert all(len(item["region_defaults"]) == 4 for item in candidates)
    assert all(
        item["region_defaults"][0]["format"] == W8_FORMAT
        for item in candidates
    )
    assert all(
        item["path_routes"][-1]["route_id"].startswith("special-")
        for item in candidates
    )
    assert all("backbone.locked" not in route_paths for route_paths in routes.values())


def test_cpu_profile_payload_declares_ranking_only_and_no_gpu_or_training() -> None:
    summary = {
        "coverage": {
            "deployment_paths": 148,
            "formats_per_path": 9,
            "measurements": 1332,
        },
        "path_rankings": [],
    }

    payload = build_cpu_profile_payload(
        parent={
            "parent_id": "v35-parent",
            "manifest_sha256": "a" * 64,
            "selected_epoch": 3,
            "activation": "qsilu_pq",
        },
        summary=summary,
        measurements=[],
    )

    assert payload["status"] == "completed"
    assert payload["gpu_used"] is False
    assert payload["formal_training"] is False
    assert payload["selection_claim"] is False
    assert payload["analysis_contract"]["scope"] == "all_148_deployment_weight_paths"
    assert payload["summary"]["coverage"]["formats_per_path"] == 9


def test_phase_plan_payload_pins_candidate_and_route_authorization() -> None:
    base = {
        "plan_id": "base",
        "date": "old",
        "routes": {
            "v35-ls-sd4": {"manifest": {"path": "old.yaml", "sha256": "a" * 64}, "list_key": "old"},
        },
        "execution_authorization": {
            "authorization_id": "old-auth",
            "validation_roles": ["candidate"],
            "candidate_ids": [],
            "route_ids": [],
            "training": False,
        },
    }
    candidates = [
        {
            "candidate_id": "special-backbone-fixed-sd4",
            "region_defaults": [],
            "path_routes": [
                {"route_id": "v35-ls-sd4", "format": dict(FORMAT_PAYLOADS["fixed-sd4"])},
                {"route_id": "special-backbone-fixed-sd4", "format": dict(FORMAT_PAYLOADS["fixed-sd4"])},
            ],
        }
    ]

    payload = build_phase_plan_payload(
        base,
        plan_id="v36-special",
        phase="special",
        candidates=candidates,
        new_routes={
            "special-backbone-fixed-sd4": ["graph.model.0.conv"],
        },
        route_manifest_path="artifacts/queues/v36/special-routes.yaml",
        route_manifest_sha256="b" * 64,
        cpu_profile_path="artifacts/reports/v36-cpu.json",
        cpu_profile_sha256="c" * 64,
    )

    assert payload["plan_id"] == "v36-special"
    assert payload["successor_provenance"]["phase"] == "special"
    assert payload["execution_authorization"]["candidate_ids"] == [
        "special-backbone-fixed-sd4"
    ]
    assert payload["routes"]["special-backbone-fixed-sd4"]["list_key"] == (
        "special-backbone-fixed-sd4"
    )


def test_select_phase_routes_prefers_green_accuracy_then_smaller_packed_policy() -> None:
    report = {
        "results": {
            "special-backbone-fixed-sd4": {
                "weight_quantization": {
                    "formats": [{"numeric": {"weight_code_bytes": 100, "scale_bytes": 20}}]
                }
            },
            "special-backbone-exact-ternary": {
                "weight_quantization": {
                    "formats": [{"numeric": {"weight_code_bytes": 40, "scale_bytes": 20}}]
                }
            },
        }
    }
    dual = {
        "candidates": {
            "special-backbone-fixed-sd4": {
                "decision": "green",
                "worst_total_delta": -0.010,
                "worst_total_map50_95_delta": -0.020,
            },
            "special-backbone-exact-ternary": {
                "decision": "green",
                "worst_total_delta": -0.010,
                "worst_total_map50_95_delta": -0.020,
            },
        }
    }

    selected = select_phase_routes(
        phase="special",
        report=report,
        dual=dual,
        candidate_formats={
            "special-backbone-fixed-sd4": FORMAT_PAYLOADS["fixed-sd4"],
            "special-backbone-exact-ternary": FORMAT_PAYLOADS["exact-ternary"],
        },
    )

    assert selected[0]["route_id"] == "special-backbone-exact-ternary"
    assert selected[0]["format"] == FORMAT_PAYLOADS["exact-ternary"]


def test_status_payload_exposes_only_compact_blocking_monitor_fields() -> None:
    payload = build_status_payload(
        "uniform_started",
        current_index=2,
        current_candidate="uniform-neck-w5",
        current_arm=None,
        completed_jobs=3,
        error=None,
    )

    assert payload["status"] == "uniform_started"
    assert payload["current_index"] == 2
    assert payload["current_candidate"] == "uniform-neck-w5"
    assert payload["current_arm"] is None
    assert payload["completed_jobs"] == 3
    assert payload["error"] is None


def test_qat_assignments_keep_all_regions_w8_and_group_route_formats() -> None:
    rows = [
        {"path": "backbone.a", "region": "backbone_early"},
        {"path": "neck.a", "region": "neck"},
    ]
    assignments = _build_qat_assignments(
        rows=rows,
        route_specs=[
            ("r1", {"family": "fixed_sd4", "scale_method": "optimal_scaled_codebook"}),
            ("r2", {"family": "uniform", "bits": 5, "scale_method": "optimal_scaled_codebook"}),
        ],
        route_paths={"r1": ["backbone.a"], "r2": ["neck.a"]},
    )

    assert [item["region"] for item in assignments[:10]] == [
        "backbone_early", "backbone_deep", "backbone_attention_safe", "neck",
        "masf", "neck_attention_safe", "detect_one2one_tower",
        "detect_one2one_predictor", "pose_one2one_tower", "pose_one2one_predictor",
    ]
    assert assignments[0]["format"] == W8_FORMAT
    assert assignments[10]["format"]["family"] == "ls_sd4"
    assert assignments[11]["format"]["bits"] == 5


def test_continuation_qat_payload_is_short_and_external_controlled() -> None:
    from yolo_quantize.full_coverage_successor import (
        V35_BASE_QAT_PLAN, V35_SHAM_COMPLETION, V35_SHAM_METRICS,
        PARENT_MANIFEST,
    )
    payload = build_continuation_qat_payload(
        final_plan_path=V35_BASE_QAT_PLAN,
        final_report_path=Path("artifacts/reports/v35-cumulative-head-v1.json"),
        final_dual_path=Path("artifacts/reports/v35-cumulative-head-dual-v1.json"),
        candidate_id="cumulative-head-w4",
        assignments=[{"region": "backbone_early", "paths": [], "format": dict(W8_FORMAT)}],
        parent_manifest=PARENT_MANIFEST,
    )
    authorization = payload["execution_authorization"]
    assert authorization["arms"] == ["qat"]
    assert authorization["external_control"]["completion"]["path"] == str(V35_SHAM_COMPLETION)
    assert authorization["external_control"]["metrics"]["path"] == str(V35_SHAM_METRICS)
    assert payload["training"]["epochs"] == 3
    assert payload["training"]["patience"] == 5
    assert payload["run_root"] == "artifacts/runs/qat/v36-qsilu-full-coverage-short-v2"
    assert payload["quick_recovery_provenance"]["no_new_sham"] is True
