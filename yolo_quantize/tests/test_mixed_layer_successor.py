from __future__ import annotations

from pathlib import Path

from yolo_quantize.mixed_layer_successor import (
    BASE_HEAD_ROUTES,
    FORMAT_SPECS,
    _materialize,
    build_cumulative_plan_payload,
    build_independent_plan_payload,
)


def test_independent_matrix_is_three_layers_by_nine_formats() -> None:
    payload = build_independent_plan_payload()
    candidates = payload["candidates"]

    assert len(FORMAT_SPECS) == 9
    assert len(candidates) == 27
    assert all(len(item["path_routes"]) == 1 for item in candidates)
    assert [item["candidate_id"] for item in candidates[:9]] == [
        f"independent-backbone-{format_id}" for format_id, _ in FORMAT_SPECS
    ]
    assert payload["activation"]["policy_id"] == "qsilu_pq--lsq-plus-a8"
    assert payload["execution_authorization"]["training"] is False


def test_cumulative_plan_keeps_head_base_and_allows_heterogeneous_formats() -> None:
    payload = build_cumulative_plan_payload(
        segment="neck",
        locked=(("layer-backbone-high", {"family": "uniform", "bits": 6, "scale_method": "optimal_scaled_codebook"}),),
        format_ids=("w5", "fixed-sd4"),
    )
    candidates = payload["candidates"]

    assert len(candidates) == 2
    for candidate in candidates:
        routes = candidate["path_routes"]
        assert [item["route_id"] for item in routes[:3]] == list(BASE_HEAD_ROUTES)
        assert routes[3]["route_id"] == "layer-backbone-high"
        assert routes[4]["route_id"] == "layer-neck-high"
    assert candidates[0]["path_routes"][3]["format"]["bits"] == 6
    assert candidates[0]["path_routes"][4]["format"]["bits"] == 5


def test_independent_plan_round_trips_strict_parser(tmp_path: Path) -> None:
    parsed = _materialize(
        tmp_path / "v35-independent.yaml", build_independent_plan_payload()
    )

    assert len(parsed.candidates) == 27
    assert parsed.matched_policy_id == "qsilu_pq--lsq-plus-a8"
