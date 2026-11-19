from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from yolo_quantize import (
    ExactTernaryWeightSpec,
    FilterwiseTWNWeightSpec,
    FixedSD4WeightSpec,
    Full35WeightRegionCatalog,
    UniformWeightSpec,
    WeightQuantizationAdapter,
)
from yolo_quantize.mixed_policy_search import (
    Full35MixedPolicySearchPlan,
    MixedWeightPolicyCandidate,
    PathFormatRoute,
    PinnedPathRoute,
    RegionFormatDefault,
    main,
    parse_weight_format_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _MixedPolicyWeights(nn.Module):
    def add_weight(self, path: str) -> nn.Conv2d:
        current = self
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in current._modules:
                current.add_module(part, nn.Module())
            current = current._modules[part]
        module = nn.Conv2d(1, 1, (1, 4), bias=False)
        current.add_module(parts[-1], module)
        return module


def test_candidate_compiles_region_defaults_and_cross_region_path_route() -> None:
    model = _MixedPolicyWeights()
    early_default = model.add_weight("graph.model.0.conv")
    early_override = model.add_weight("graph.model.1.conv")
    neck_default = model.add_weight("graph.model.17.conv")
    neck_override = model.add_weight("graph.model.16.cv1.conv")
    with torch.no_grad():
        for module in (early_default, early_override, neck_default, neck_override):
            module.weight.copy_(
                torch.tensor([-1.0, -0.6, 0.6, 1.0]).reshape_as(module.weight)
            )
    originals = tuple(
        module.weight.detach().clone()
        for module in (early_default, early_override, neck_default, neck_override)
    )
    catalog = Full35WeightRegionCatalog.inspect(model)
    candidate = MixedWeightPolicyCandidate(
        candidate_id="w8-with-sd4-route",
        region_defaults=(
            RegionFormatDefault("backbone_early", UniformWeightSpec(8, "max")),
            RegionFormatDefault("neck", UniformWeightSpec(8, "max")),
        ),
        path_routes=(
            PathFormatRoute(
                route_id="fixed-sd4-v3",
                paths=("graph.model.1.conv", "graph.model.16.cv1.conv"),
                spec=FixedSD4WeightSpec(),
            ),
        ),
    )

    assignments = candidate.compile(catalog)

    assert tuple(
        (item.region, item.spec.format_id, item.paths) for item in assignments
    ) == (
        ("backbone_early", "w8", ()),
        ("neck", "w8", ()),
        ("backbone_early", "fixed-sd4", ("graph.model.1.conv",)),
        ("neck", "fixed-sd4", ("graph.model.16.cv1.conv",)),
    )
    with WeightQuantizationAdapter().quantized_policy(
        model,
        catalog=catalog,
        assignments=assignments,
    ) as applied:
        assert tuple(item.format_id for item in applied.applied) == (
            "w8",
            "w8",
            "fixed-sd4",
            "fixed-sd4",
        )
        assert applied.quantized_modules == 4

    assert all(
        torch.equal(module.weight, original)
        for module, original in zip(
            (early_default, early_override, neck_default, neck_override),
            originals,
            strict=True,
        )
    )


def test_pinned_route_loads_exact_sd4_v3_candidates_by_hash() -> None:
    route = PinnedPathRoute.from_yaml(
        route_id="fixed-sd4-v3",
        path=PROJECT_ROOT / "artifacts/manifests/fixed-sd4-routing-candidates-v3.yaml",
        expected_sha256=(
            "afbefae48e3bafd87a2842da205c3f7a5b5d081f5351a9caf480ecacb1ed98f4"
        ),
        list_key="stable_candidates",
    )

    assert route.route_id == "fixed-sd4-v3"
    assert len(route.paths) == 36
    assert route.paths[0] == "graph.model.1.conv"
    assert route.paths[-1] == "graph.model.9.cv2.conv"
    assert route.source_execution_authorized is False
    assert route.map_validation_run is False


def test_weight_format_parser_supports_reviewed_uniform_sd4_and_ternary_formats() -> (
    None
):
    uniform = parse_weight_format_spec(
        {
            "family": "uniform",
            "bits": 7,
            "scale_method": "optimal_scaled_codebook",
        }
    )
    fixed_sd4 = parse_weight_format_spec(
        {"family": "fixed_sd4", "scale_method": "optimal_scaled_codebook"}
    )
    paper_twn = parse_weight_format_spec(
        {"family": "paper_twn", "threshold_multiplier": 0.7}
    )
    exact_ternary = parse_weight_format_spec(
        {
            "family": "exact_scaled_ternary",
            "scale_method": "optimal_scaled_codebook",
        }
    )
    filterwise_twn = parse_weight_format_spec(
        {"family": "twn_filterwise", "threshold_multiplier": 0.75}
    )

    assert isinstance(uniform, UniformWeightSpec)
    assert (uniform.bits, uniform.scale_method) == (
        7,
        "optimal_scaled_codebook",
    )
    assert isinstance(fixed_sd4, FixedSD4WeightSpec)
    assert fixed_sd4.bits == 4
    assert paper_twn.format_id == "paper-twn"
    assert paper_twn.bits == 2
    assert isinstance(exact_ternary, ExactTernaryWeightSpec)
    assert exact_ternary.format_id == "exact-scaled-ternary"
    assert isinstance(filterwise_twn, FilterwiseTWNWeightSpec)
    assert filterwise_twn.format_id == "twn-v3-0.75-filterwise"


def test_weight_format_parser_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported mixed-policy weight family"):
        parse_weight_format_spec({"family": "ls_sd4"})
    with pytest.raises(ValueError, match="exactly"):
        parse_weight_format_spec(
            {"family": "uniform", "bits": 8, "scale_method": "max", "x": 1}
        )


def test_reviewed_mixed_plan_pins_routes_comparisons_and_total_gate() -> None:
    plan = Full35MixedPolicySearchPlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v10-qsilu-mixed-weight-policy-search-v1.yaml"
    )

    assert plan.gate_spec.total_max_drop == 0.015
    assert plan.gate_spec.metric_family == "map50"
    assert plan.activation_policy.policy_id == "qsilu_pq--lsq-plus-a8"
    assert tuple(candidate.candidate_id for candidate in plan.candidates) == (
        "qsilu-balanced-pose-w5",
        "qsilu-balanced-lower-safe",
        "qsilu-all-w8-fixed-sd4-v3",
    )
    assert tuple(item.candidate_id for item in plan.comparison_candidates) == (
        "accuracy-a05-add-pose-tower-w8",
        "compression-c04-add-pose-tower-w8",
        "accuracy-a09-add-neck-w8",
    )
    assert len(plan.pinned_routes["fixed-sd4-v3"].paths) == 36
    assert plan.formal_training is False
    assert plan.formal_validation is False


def test_poly_shift_frontier_plan_covers_accuracy_to_boundary_policies() -> None:
    plan = Full35MixedPolicySearchPlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v23-poly-shift-mixed-boundary-search-v1.yaml"
    )

    assert plan.activation_policy.policy_id == "poly_shift--lsq-plus-a8"
    assert tuple(candidate.candidate_id for candidate in plan.candidates) == (
        "polyshift-frontier-small-w4",
        "polyshift-frontier-heads",
        "polyshift-nine-region-quality",
        "polyshift-nine-region-balanced",
        "polyshift-nine-region-boundary",
    )
    boundary = plan.candidates[-1]
    assert {
        default.region: default.spec.bits for default in boundary.region_defaults
    } == {
        "backbone_early": 8,
        "backbone_deep": 7,
        "neck": 6,
        "masf": 4,
        "neck_attention_safe": 4,
        "detect_one2one_tower": 6,
        "detect_one2one_predictor": 5,
        "pose_one2one_tower": 4,
        "pose_one2one_predictor": 4,
    }
    assert tuple(item.candidate_id for item in plan.comparison_candidates) == (
        "compression-c08-add-predictors-w8",
    )
    assert plan.gate_spec.total_max_drop == 0.015


def test_poly_shift_special_format_plan_pairs_sd4_and_twn_with_w4_controls() -> None:
    plan = Full35MixedPolicySearchPlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v24-poly-shift-special-format-route-search-v1.yaml"
    )

    assert plan.activation_policy.policy_id == "poly_shift--lsq-plus-a8"
    assert len(plan.candidates) == 16
    assert {
        route_id: len(route.paths) for route_id, route in plan.pinned_routes.items()
    } == {
        "fixed-sd4-backbone-early": 4,
        "fixed-sd4-backbone-deep": 4,
        "fixed-sd4-neck": 20,
        "fixed-sd4-neck-attention": 1,
        "fixed-sd4-detect-tower": 6,
        "fixed-sd4-detect-predictor": 1,
        "paper-twn-safe": 3,
        "paper-twn-balanced": 7,
    }
    for index in range(0, 12, 2):
        sd4 = plan.candidates[index]
        w4 = plan.candidates[index + 1]
        assert sd4.path_routes[0].paths == w4.path_routes[0].paths
        assert sd4.path_routes[0].spec.format_id == "fixed-sd4"
        assert w4.path_routes[0].spec.format_id == "w4"
    for index in (12, 14):
        ternary = plan.candidates[index]
        w4 = plan.candidates[index + 1]
        assert ternary.path_routes[0].paths == w4.path_routes[0].paths
        assert ternary.path_routes[0].spec.format_id == "paper-twn"
        assert w4.path_routes[0].spec.format_id == "w4"


def test_poly_shift_coupled_special_routes_keep_single_and_combined_attribution() -> (
    None
):
    plan = Full35MixedPolicySearchPlan.from_yaml(
        PROJECT_ROOT
        / "configs/experiments/v27-poly-shift-coupled-special-route-search-v1.yaml"
    )

    assert plan.activation_policy.policy_id == "poly_shift--lsq-plus-a8"
    assert tuple(candidate.candidate_id for candidate in plan.candidates) == (
        "polyshift-all9-w8-plus-w4-neck-attention",
        "polyshift-all9-w8-plus-sd4-detect-predictor",
        "polyshift-all9-w8-plus-sd4-detect-tower",
        "polyshift-all9-w8-plus-w4-neck-attention-sd4-predictor",
        "polyshift-all9-w8-plus-sd4-detect-head",
        "polyshift-all9-w8-plus-w4-neck-attention-sd4-tower",
        "polyshift-all9-w8-plus-three-special-routes",
    )
    assert {
        route_id: len(route.paths) for route_id, route in plan.pinned_routes.items()
    } == {
        "w4-neck-attention": 1,
        "fixed-sd4-detect-tower": 6,
        "fixed-sd4-detect-predictor": 1,
    }
    assert all(len(candidate.region_defaults) == 9 for candidate in plan.candidates)
    assert [route.spec.format_id for route in plan.candidates[-1].path_routes] == [
        "w4",
        "fixed-sd4",
        "fixed-sd4",
    ]
    assert tuple(item.candidate_id for item in plan.comparison_candidates) == (
        "compression-c08-add-predictors-w8",
    )


def test_mixed_policy_list_only_never_requires_cuda(capsys) -> None:
    result = main(
        [
            "--plan",
            str(
                PROJECT_ROOT
                / "configs/experiments/v10-qsilu-mixed-weight-policy-search-v1.yaml"
            ),
            "--list-only",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert len(payload["candidates"]) == 3
    assert payload["gates"]["total_drop"] == 0.015
    assert payload["formal_training"] is False


def test_mixed_policy_execution_requires_explicit_acknowledgement(capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--plan",
                str(
                    PROJECT_ROOT
                    / "configs/experiments/v10-qsilu-mixed-weight-policy-search-v1.yaml"
                ),
            ]
        )

    assert "--execute-reviewed-plan" in capsys.readouterr().err
