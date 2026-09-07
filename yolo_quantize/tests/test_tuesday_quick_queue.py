from __future__ import annotations

from yolo_quantize.tuesday_quick_queue import (
    build_layer_route_manifest_payload,
    build_layer_sensitivity_selection,
    build_quick_plan_payload,
)


def test_quick_plan_starts_with_nine_single_layer_candidates() -> None:
    payload = build_quick_plan_payload()
    candidates = payload["candidates"]

    assert [item["candidate_id"] for item in candidates[:9]] == [
        "layer-sensitivity-backbone-low-fixed-sd4",
        "layer-sensitivity-backbone-median-fixed-sd4",
        "layer-sensitivity-backbone-high-fixed-sd4",
        "layer-sensitivity-neck-low-fixed-sd4",
        "layer-sensitivity-neck-median-fixed-sd4",
        "layer-sensitivity-neck-high-fixed-sd4",
        "layer-sensitivity-head-low-fixed-sd4",
        "layer-sensitivity-head-median-fixed-sd4",
        "layer-sensitivity-head-high-fixed-sd4",
    ]
    assert [item["candidate_id"] for item in candidates[9:]] == [
        "fixed-sd4-head-three",
        "exact-ternary-head-three",
        "twn-v3-head-three",
        "paper-twn-v2-head-three",
        "exact-ternary-pose-safe",
        "twn-v3-pose-safe",
    ]
    assert all(item["region_defaults"] == [] for item in candidates)
    assert all(len(item["path_routes"]) == 1 for item in candidates[:9])
    assert payload["activation"]["policy_id"] == "qsilu_pq--lsq-plus-a8"
    assert payload["execution_authorization"]["training"] is False


def test_layer_selection_covers_all_segments_and_cpu_error_levels() -> None:
    selected = build_layer_sensitivity_selection()
    manifest = build_layer_route_manifest_payload()

    assert len(selected) == 9
    for segment in ("backbone", "neck", "head"):
        rows = [row for row in selected if row["segment"] == segment]
        assert [row["sensitivity_level"] for row in rows] == [
            "low",
            "median",
            "high",
        ]
        assert [row["normalized_rmse"] for row in rows] == sorted(
            row["normalized_rmse"] for row in rows
        )
    for row in selected:
        assert manifest[row["route_id"]] == [row["path"]]


def test_quick_plan_pins_structured_format_semantics() -> None:
    payload = build_quick_plan_payload()
    candidates = {
        item["candidate_id"]: item for item in payload["candidates"]
    }

    fixed = candidates["fixed-sd4-head-three"]["path_routes"][0]["format"]
    exact = candidates["exact-ternary-head-three"]["path_routes"][0]["format"]
    filterwise = candidates["twn-v3-head-three"]["path_routes"][0]["format"]
    paper = candidates["paper-twn-v2-head-three"]["path_routes"][0]["format"]
    assert fixed == {
        "family": "fixed_sd4",
        "scale_method": "optimal_scaled_codebook",
    }
    assert exact == {
        "family": "exact_scaled_ternary",
        "scale_method": "optimal_scaled_codebook",
    }
    assert filterwise == {
        "family": "twn_filterwise",
        "threshold_multiplier": 0.75,
    }
    assert paper == {"family": "paper_twn", "threshold_multiplier": 0.7}
