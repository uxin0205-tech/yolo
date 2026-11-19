from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from yolo_quantize import WeightSensitivityStudy
from yolo_quantize.weight_sensitivity import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_study_expands_deterministic_ptq_cells_from_reviewed_plan() -> None:
    study = WeightSensitivityStudy.from_yaml(
        PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v1.yaml"
    )

    cells = study.cells(
        activations=("poly_quality",),
        regions=("backbone_early", "neck"),
        bits=(8, 4),
        scale_method="mse",
    )

    assert tuple(cell.cell_id for cell in cells) == (
        "poly_quality--lsq-plus-a8--backbone_early--w8-mse",
        "poly_quality--lsq-plus-a8--backbone_early--w4-mse",
        "poly_quality--lsq-plus-a8--neck--w8-mse",
        "poly_quality--lsq-plus-a8--neck--w4-mse",
    )
    assert cells[0].parent.sha256 == (
        "eedd48007345bc8dfd2179819b932bdf1dd40b25a654a7eb82c53796885ac968"
    )
    assert cells[0].formal_training is False


def test_study_rejects_a_new_silu_ptq_run() -> None:
    study = WeightSensitivityStudy.from_yaml(
        PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v1.yaml"
    )

    with pytest.raises(ValueError, match="activation selection"):
        study.cells(activations=("silu",))


def test_cli_lists_selected_cells_without_loading_the_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "--plan",
            str(
                PROJECT_ROOT
                / "configs/experiments/weight-region-sensitivity-plan-v1.yaml"
            ),
            "--activations",
            "poly_quality",
            "--regions",
            "backbone_early",
            "--bits",
            "8,4",
            "--list-only",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["cell_count"] == 2
    assert [cell["weight_bits"] for cell in payload["cells"]] == [8, 4]


def test_active_v2_plan_uses_correct_catalog_and_excludes_historical_parent() -> None:
    study = WeightSensitivityStudy.from_yaml(
        PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v2.yaml"
    )

    assert tuple(parent.activation for parent in study.parents) == (
        "qsilu_pq",
        "hardswish",
        "poly_shift",
    )
    assert study.expected_catalog["totals"] == {
        "modules": 152,
        "weight_elements": 22_702_912,
        "deployment_modules": 148,
        "deployment_weight_elements": 22_571_840,
        "training_only_modules": 0,
        "training_only_weight_elements": 0,
        "protected_modules": 4,
        "protected_weight_elements": 131_072,
    }
    assert study.expected_master_catalog["totals"]["training_only_modules"] == 99
    assert study.expected_catalog["deployment_regions"]["pose_one2one_predictor"] == {
        "modules": 9,
        "weight_elements": 3456,
    }
    assert study.execution_authorized is False
    assert study.graph_view == "bn_folded_deployment"
    assert study.diagnostic_manifest == (
        PROJECT_ROOT / "artifacts/manifests/full35-diagnostic-cal32-probe64-v1.json"
    )
    assert study.diagnostic_manifest_sha256 == (
        "b0eb2a068aa515caa5ea22f2781dc68225971224cae2324d17a6be1e2d31eef2"
    )
    assert study.diagnostic_probe_per_task == 1
    assert study.load_diagnostic_manifest(verify_files=False).sha256 == (
        study.diagnostic_manifest_sha256
    )


def test_active_plan_rejects_a_manifest_not_pinned_by_its_own_contract(
    tmp_path: Path,
) -> None:
    source = PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v2.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["diagnostic"]["manifest"] = str(
        PROJECT_ROOT / "artifacts/manifests/full35-diagnostic-cal32-probe64-v1.json"
    )
    payload["diagnostic"]["manifest_sha256"] = "0" * 64
    plan_path = tmp_path / "digest-mismatch.yaml"
    plan_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    study = WeightSensitivityStudy.from_yaml(plan_path)

    with pytest.raises(RuntimeError, match="not pinned"):
        study.load_diagnostic_manifest(verify_files=False)


def test_cli_refuses_execution_when_plan_is_not_authorized(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--plan",
                str(
                    PROJECT_ROOT
                    / "configs/experiments/weight-region-sensitivity-plan-v2.yaml"
                ),
                "--activations",
                "qsilu_pq",
                "--regions",
                "backbone_early",
                "--bits",
                "8",
                "--execute-reviewed-plan",
            ]
        )

    assert "execution_authorized=false" in capsys.readouterr().err


def test_authorized_plan_only_releases_explicit_cells(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v2.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["execution_authorized"] = True
    payload["execution_authorization"] = {
        "authorization_id": "user-2026-09-03-first-gpu-bridge",
        "cells": [
            "qsilu_pq--lsq-plus-a8--backbone_early--w8-mse_grid_v1",
        ],
    }
    plan_path = tmp_path / "authorized.yaml"
    plan_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    study = WeightSensitivityStudy.from_yaml(plan_path)

    authorized = study.cells(
        activations=("qsilu_pq",),
        regions=("backbone_early",),
        bits=(8,),
        scale_method="mse_grid_v1",
    )
    study.require_execution_authorized(authorized)

    unauthorized = study.cells(
        activations=("qsilu_pq",),
        regions=("backbone_deep",),
        bits=(8,),
        scale_method="mse_grid_v1",
    )
    with pytest.raises(RuntimeError, match="outside the reviewed authorization"):
        study.require_execution_authorized(unauthorized)


def test_active_v3_plan_authorizes_only_the_first_gpu_bridge() -> None:
    study = WeightSensitivityStudy.from_yaml(
        PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v3.yaml"
    )

    assert study.execution_authorized is True
    assert study.execution_authorization_id == ("user-2026-09-03-first-gpu-bridge")
    assert study.authorized_cell_ids == (
        "qsilu_pq--lsq-plus-a8--backbone_early--w8-mse_grid_v1",
    )


def test_v4_plan_authorizes_only_the_qsilu_bit_boundary_diagnostics() -> None:
    study = WeightSensitivityStudy.from_yaml(
        PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v4.yaml"
    )

    assert study.execution_authorization_id == ("user-2026-09-03-autonomous-v5-stage1")
    assert study.authorized_cell_ids == (
        "qsilu_pq--lsq-plus-a8--backbone_early--w7-mse_grid_v1",
        "qsilu_pq--lsq-plus-a8--backbone_early--w6-mse_grid_v1",
        "qsilu_pq--lsq-plus-a8--backbone_early--w5-mse_grid_v1",
        "qsilu_pq--lsq-plus-a8--backbone_early--w4-mse_grid_v1",
        "qsilu_pq--lsq-plus-a8--backbone_early--w4-optimal_scaled_codebook",
    )


def test_v8_plan_resolves_heterogeneous_authorized_cells_in_declared_order() -> None:
    study = WeightSensitivityStudy.from_yaml(
        PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v8.yaml"
    )

    cells = study.authorized_cells()

    assert tuple((cell.region, cell.bits, cell.scale_method) for cell in cells) == (
        ("backbone_early", 7, "optimal_scaled_codebook"),
        ("backbone_deep", 6, "optimal_scaled_codebook"),
        ("detect_one2one_tower", 6, "optimal_scaled_codebook"),
        ("neck", 5, "optimal_scaled_codebook"),
        ("masf", 5, "optimal_scaled_codebook"),
        ("neck_attention_safe", 5, "optimal_scaled_codebook"),
        ("detect_one2one_predictor", 5, "optimal_scaled_codebook"),
        ("pose_one2one_tower", 5, "optimal_scaled_codebook"),
        ("pose_one2one_predictor", 5, "optimal_scaled_codebook"),
    )
    assert tuple(cell.cell_id for cell in cells) == study.authorized_cell_ids


def test_cli_lists_exact_heterogeneous_authorized_cells(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "--plan",
            str(
                PROJECT_ROOT
                / "configs/experiments/weight-region-sensitivity-plan-v8.yaml"
            ),
            "--authorized-cells",
            "--list-only",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["cell_count"] == 9
    assert payload["cells"][0]["cell_id"].endswith(
        "backbone_early--w7-optimal_scaled_codebook"
    )
    assert payload["cells"][-1]["cell_id"].endswith(
        "pose_one2one_predictor--w5-optimal_scaled_codebook"
    )


def test_v15_poly_shift_plan_authorizes_all_ten_w7_exact_diagnostics() -> None:
    study = WeightSensitivityStudy.from_yaml(
        PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v15.yaml"
    )

    cells = study.authorized_cells()

    assert study.execution_authorization_id == (
        "user-2026-09-03-autonomous-poly-shift-stage3-w7-exact"
    )
    assert tuple(cell.parent.activation for cell in cells) == ("poly_shift",) * 10
    assert tuple(cell.region for cell in cells) == study.regions
    assert tuple(cell.bits for cell in cells) == (7,) * 10
    assert tuple(cell.scale_method for cell in cells) == (
        "optimal_scaled_codebook",
    ) * 10


def test_v16_poly_shift_plan_only_promotes_eight_dual_green_regions_to_w6() -> None:
    study = WeightSensitivityStudy.from_yaml(
        PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v16.yaml"
    )

    cells = study.authorized_cells()

    assert tuple(cell.region for cell in cells) == (
        "backbone_deep",
        "neck",
        "masf",
        "neck_attention_safe",
        "detect_one2one_tower",
        "detect_one2one_predictor",
        "pose_one2one_tower",
        "pose_one2one_predictor",
    )
    assert tuple(cell.bits for cell in cells) == (6,) * 8
    assert {cell.scale_method for cell in cells} == {"optimal_scaled_codebook"}


def test_v17_poly_shift_plan_only_promotes_seven_dual_green_regions_to_w5() -> None:
    study = WeightSensitivityStudy.from_yaml(
        PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v17.yaml"
    )

    cells = study.authorized_cells()

    assert tuple(cell.region for cell in cells) == (
        "neck",
        "masf",
        "neck_attention_safe",
        "detect_one2one_tower",
        "detect_one2one_predictor",
        "pose_one2one_tower",
        "pose_one2one_predictor",
    )
    assert tuple(cell.bits for cell in cells) == (5,) * 7
    assert {cell.scale_method for cell in cells} == {"optimal_scaled_codebook"}


def test_v18_poly_shift_plan_only_promotes_five_dual_green_regions_to_w4() -> None:
    study = WeightSensitivityStudy.from_yaml(
        PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v18.yaml"
    )

    cells = study.authorized_cells()

    assert tuple(cell.region for cell in cells) == (
        "masf",
        "neck_attention_safe",
        "detect_one2one_predictor",
        "pose_one2one_tower",
        "pose_one2one_predictor",
    )
    assert tuple(cell.bits for cell in cells) == (4,) * 5
    assert {cell.scale_method for cell in cells} == {"optimal_scaled_codebook"}


def test_cli_rejects_cartesian_filters_with_authorized_cell_whitelist(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--plan",
                str(
                    PROJECT_ROOT
                    / "configs/experiments/weight-region-sensitivity-plan-v8.yaml"
                ),
                "--authorized-cells",
                "--bits",
                "4",
                "--list-only",
            ]
        )

    assert "cannot be combined" in capsys.readouterr().err


def test_module_list_only_cli_does_not_warn_about_a_duplicate_import() -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "yolo_quantize.weight_sensitivity",
            "--activations",
            "qsilu_pq",
            "--regions",
            "backbone_early",
            "--bits",
            "8",
            "--list-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "RuntimeWarning" not in completed.stderr
    assert json.loads(completed.stdout)["cell_count"] == 1
