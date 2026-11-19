from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_quantize.search_race import Full35SearchRacePlan, main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v5_search_race_pins_five_diagnostic_survivors_and_map50_gate() -> None:
    plan = Full35SearchRacePlan.from_yaml(
        PROJECT_ROOT / "configs/experiments/v5-qsilu-backbone-early-bit-search-v1.yaml"
    )

    assert plan.execution_authorization_id == (
        "user-2026-09-03-autonomous-v5-stage1-search"
    )
    assert tuple(cell.bits for cell in plan.cells) == (7, 6, 5, 4, 4)
    assert tuple(cell.scale_method for cell in plan.cells)[-1] == (
        "optimal_scaled_codebook"
    )
    assert plan.gate_spec.metric_family == "map50"
    assert plan.gate_spec.total_max_drop == 0.015
    assert plan.accuracy_tolerance == 0.002


def test_v5_search_race_list_only_never_requires_cuda(capsys) -> None:
    result = main(
        [
            "--plan",
            str(
                PROJECT_ROOT
                / "configs/experiments/v5-qsilu-backbone-early-bit-search-v1.yaml"
            ),
            "--list-only",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["candidate_count"] == 5
    assert payload["gates"]["metric_family"] == "map50"


def test_v5_search_race_requires_explicit_execution_acknowledgement(capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--plan",
                str(
                    PROJECT_ROOT
                    / "configs/experiments/v5-qsilu-backbone-early-bit-search-v1.yaml"
                ),
            ]
        )

    assert "--execute-reviewed-plan" in capsys.readouterr().err


def test_v18_poly_shift_w7_exact_race_covers_all_ten_regions() -> None:
    plan = Full35SearchRacePlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v18-poly-shift-ten-regions-w7-exact-search-v1.yaml"
    )

    assert plan.execution_authorization_id == (
        "user-2026-09-03-autonomous-poly-shift-stage3-w7-exact-search"
    )
    assert tuple(cell.region for cell in plan.cells) == plan.weight_study.regions
    assert tuple(cell.bits for cell in plan.cells) == (7,) * 10
    assert tuple(cell.scale_method for cell in plan.cells) == (
        "optimal_scaled_codebook",
    ) * 10
    assert plan.gate_spec.total_max_drop == 0.015


def test_v20_poly_shift_w6_exact_race_only_covers_dual_green_w7_regions() -> None:
    plan = Full35SearchRacePlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v20-poly-shift-eight-regions-w6-exact-search-v1.yaml"
    )

    assert plan.execution_authorization_id == (
        "user-2026-09-03-autonomous-poly-shift-stage4-w6-exact-search"
    )
    assert tuple(cell.region for cell in plan.cells) == (
        "backbone_deep",
        "neck",
        "masf",
        "neck_attention_safe",
        "detect_one2one_tower",
        "detect_one2one_predictor",
        "pose_one2one_tower",
        "pose_one2one_predictor",
    )
    assert tuple(cell.bits for cell in plan.cells) == (6,) * 8
    assert tuple(cell.scale_method for cell in plan.cells) == (
        "optimal_scaled_codebook",
    ) * 8
    assert plan.gate_spec.total_max_drop == 0.015


def test_v21_poly_shift_w5_exact_race_only_covers_dual_green_w6_regions() -> None:
    plan = Full35SearchRacePlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v21-poly-shift-seven-regions-w5-exact-search-v1.yaml"
    )

    assert plan.execution_authorization_id == (
        "user-2026-09-03-autonomous-poly-shift-stage5-w5-exact-search"
    )
    assert tuple(cell.region for cell in plan.cells) == (
        "neck",
        "masf",
        "neck_attention_safe",
        "detect_one2one_tower",
        "detect_one2one_predictor",
        "pose_one2one_tower",
        "pose_one2one_predictor",
    )
    assert tuple(cell.bits for cell in plan.cells) == (5,) * 7
    assert {cell.scale_method for cell in plan.cells} == {
        "optimal_scaled_codebook"
    }


def test_v22_poly_shift_w4_exact_race_only_covers_dual_green_w5_regions() -> None:
    plan = Full35SearchRacePlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v22-poly-shift-five-regions-w4-exact-search-v1.yaml"
    )

    assert plan.execution_authorization_id == (
        "user-2026-09-03-autonomous-poly-shift-stage6-w4-exact-search"
    )
    assert tuple(cell.region for cell in plan.cells) == (
        "masf",
        "neck_attention_safe",
        "detect_one2one_predictor",
        "pose_one2one_tower",
        "pose_one2one_predictor",
    )
    assert tuple(cell.bits for cell in plan.cells) == (4,) * 5
    assert {cell.scale_method for cell in plan.cells} == {
        "optimal_scaled_codebook"
    }
    assert plan.gate_spec.total_max_drop == 0.015
