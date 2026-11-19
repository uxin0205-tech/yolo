from __future__ import annotations

from pathlib import Path

import yaml

from yolo_quantize import QSiLUPQProfile
from yolo_quantize.weight_quantization import UniformWeightSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_qsilu_lsq_plus_policy_separates_function_and_output_quantization() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs/activation/qsilu-pq-lsq-plus.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert tuple(payload["activation"]["knots"]) == QSiLUPQProfile().knots
    assert payload["activation_function_bittrue"]["q_format"] == {
        "total_bits": 16,
        "fraction_bits": 10,
    }
    assert (
        payload["activation_function_bittrue"]["rounding_mode"]
        == "legacy_signed_half_away_from_zero"
    )
    output = payload["activation_output_quantizer"]
    assert output["candidate_bits"] == [3, 4, 5, 6, 7, 8]
    assert output["rounding_mode"] == "nearest_even"
    assert output["family"] == "lsq_plus"


def test_v3_plan_uses_hardswish_and_excludes_poly_quality_from_active_matrix() -> None:
    plan = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/full35-quantization-plan-v3.yaml"
        ).read_text(encoding="utf-8")
    )
    prepared = yaml.safe_load(
        (PROJECT_ROOT / "configs/experiments/v4-plus-prepared-plan-v3.yaml").read_text(
            encoding="utf-8"
        )
    )

    active = plan["activation_policies"]["active_uniform_a8"]
    assert [item["activation"] for item in active] == [
        "qsilu_pq",
        "hardswish",
        "poly_shift",
    ]
    excluded = plan["activation_policies"]["historical_excluded"]
    assert excluded["poly_quality"]["future_execution"] is False
    assert prepared["v4_bridge"]["axes"]["activation_parents"] == [
        "qsilu_pq",
        "hardswish",
        "poly_shift",
    ]
    assert prepared["v4_bridge"]["cells"] == 15
    assert prepared["v5_isolated_regions"]["cells"] == 150
    assert prepared["execution_authorized"] is False


def test_v3_plan_keeps_q3_regional_hardswish_separate_from_uniform_parent() -> None:
    plan = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/full35-quantization-plan-v3.yaml"
        ).read_text(encoding="utf-8")
    )
    q3 = plan["activation_policies"]["q3_regional_hardswish"]

    assert q3["base_policy_id"] == "qsilu_pq--lsq-plus-a8"
    assert q3["candidate_activation"] == "hardswish"
    assert q3["priority_single_regions"] == [
        "neck_attention",
        "masf",
        "backbone_attention",
    ]
    assert q3["multi_region_status"] == "not_measured_must_validate"
    assert q3["source_report_sha256"] == (
        "8c495d71a7d4e50db7dd29676dfc8919a88cac3e362ab3f7d4737428c9b41d77"
    )
    assert q3["source_summary_sha256"] == (
        "1e3b553484cd9eaceafd19603292804a9d3a55e74a5e5fcbb8db7edc33c08316"
    )
    assert q3["quantization_project_cpu_parity_status"] == (
        "complete_three_single_regions"
    )
    assert set(q3["cpu_parity_manifests"]) == {
        "neck_attention",
        "masf",
        "backbone_attention",
    }


def test_v4_plan_records_one_scoped_gpu_cell_without_authorizing_expansion() -> None:
    plan = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/full35-quantization-plan-v4.yaml"
        ).read_text(encoding="utf-8")
    )
    prepared = yaml.safe_load(
        (PROJECT_ROOT / "configs/experiments/v4-plus-prepared-plan-v4.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert plan["supersedes"] == "full35-activation-aware-quantization-v3"
    assert plan["active_weight_runner"] == (
        "configs/experiments/weight-region-sensitivity-plan-v2.yaml"
    )
    authorization = plan["authorization"]
    assert authorization["cpu_contract_implementation"] is True
    assert authorization["gpu_validation"] is False
    assert authorization["qat"] is False
    assert authorization["formal_training"] is False
    assert authorization["export"] is False
    assert authorization["scoped_gpu_validation_completed"] == {
        "cell_id": "qsilu_pq--lsq-plus-a8--backbone_early--w8-mse_grid_v1",
        "decision": "green",
        "expansion_authorized": False,
    }
    assert plan["scale_methods"]["historical"] == ["max", "mse_grid_v1"]
    assert plan["scale_methods"]["strong_baseline"] == ("optimal_scaled_codebook")
    assert plan["data"]["diagnostic_manifest"]["runner_binding"] == (
        "implemented_verify_all_file_evidence"
    )
    assert plan["data"]["diagnostic_manifest"]["sha256"] == (
        "b0eb2a068aa515caa5ea22f2781dc68225971224cae2324d17a6be1e2d31eef2"
    )
    assert plan["sd4"]["bittrue_encoding"]["status"] == "implemented_cpu_tested"
    assert plan["qat_graph_contract"]["status"] == "p0_unresolved"
    assert plan["integer_boundary_contract"]["status"] == (
        "cpu_reference_and_first_a8_calibration_complete_boundary_lowering_pending"
    )
    assert (
        plan["integer_boundary_contract"]["first_bridge_completed"]["search_gate"]
        == "green"
    )
    assert plan["integer_boundary_contract"]["native_integer_kernel_claimed"] is False
    assert plan["data"]["bbat5"]["selection_validation"].endswith(
        "configs/pose-search.yaml"
    )
    assert plan["data"]["bbat5"]["formal_validation"].endswith("configs/pose.yaml")
    assert plan["data"]["bbat5"]["new_split_created"] is False

    assert prepared["execution_authorized"] is False
    assert (
        "bind_runner_to_fixed_calibration_and_probe_manifest"
        in (prepared["p0_before_gpu"]["completed"])
    )
    assert (
        "bind_runner_to_fixed_calibration_and_probe_manifest"
        not in (prepared["p0_before_gpu"]["pending"])
    )
    assert prepared["p0_before_gpu"]["pending"] == []
    assert (
        "implement_integer_boundary_reference_contract"
        in prepared["p0_before_gpu"]["completed"]
    )
    assert prepared["p0_before_gpu"]["next_requires_gpu"] == [
        "measure_join_saturation_and_layer_output_error",
    ]
    assert prepared["v4_uniform_bridge"]["completed_cells"][0]["search_gate"] == "green"
    assert "resolve_fold_aware_qat_contract" in prepared["before_qat"]["pending"]
    assert prepared["before_w4_and_special_formats"]["pending"] == []
    assert prepared["special_format_stage"]["stable_static_candidates"] == 36
    assert prepared["special_format_stage"]["promotion_authorized"] is False
    assert prepared["v5_universe"]["static_cells"] == 150
    assert prepared["v5_universe"]["all_150_full_validation"] is False
    assert prepared["evaluation_tiers"]["formal"]["maximum_finalists"] == 6
    assert prepared["qat_funnel"]["s15"]["maximum_policies"] == 8
    assert prepared["qat_funnel"]["d60"]["maximum_policies"] == 2


def test_active_uniform_ranges_match_the_implemented_two_complement_codes() -> None:
    plan = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/weight-region-sensitivity-plan-v2.yaml"
        ).read_text(encoding="utf-8")
    )
    formats = plan["uniform_weight_formats"]

    assert formats["zero_point"] == 0
    assert formats["signed_integer_range"] == "full_twos_complement"
    for bits in formats["bits"]:
        spec = UniformWeightSpec(bits=bits)
        assert formats["ranges"][f"w{bits}"] == [spec.qmin, spec.qmax]
