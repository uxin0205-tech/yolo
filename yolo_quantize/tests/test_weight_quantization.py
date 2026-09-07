from __future__ import annotations

import pytest
import torch
from torch import nn

from yolo_quantize import (
    ExactTernaryWeightSpec,
    FilterwiseTWNWeightSpec,
    FixedSD4WeightSpec,
    Full35ActivationAdapter,
    Full35ActivationPolicy,
    Full35WeightRegionCatalog,
    PaperTWNWeightSpec,
    UniformWeightSpec,
    WeightQuantizationAdapter,
    WeightRegionAssignment,
)


class _RepresentativeFull35Weights(nn.Module):
    def add_weight(self, path: str, shape: tuple[int, ...]) -> nn.Conv2d:
        current = self
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in current._modules:
                current.add_module(part, nn.Module())
            current = current._modules[part]
        out_channels, in_channels, height, width = shape
        module = nn.Conv2d(
            in_channels,
            out_channels,
            (height, width),
            bias=False,
        )
        current.add_module(parts[-1], module)
        return module


def test_catalog_classifies_representative_full35_weight_paths() -> None:
    expected = {
        "graph.model.0.conv": "backbone_early",
        "graph.model.5.conv": "backbone_deep",
        "graph.model.10.m.0.attn.qkv.v.conv": "backbone_attention_safe",
        "graph.model.16.cv1.conv": "neck",
        "graph.model.16.p3_masf.context.dw3.conv": "masf",
        "graph.model.22.m.0.1.ffn.0.conv": "neck_attention_safe",
        "graph.model.23.detect_head.one2one_cv2.0.0.conv": ("detect_one2one_tower"),
        "graph.model.23.detect_head.one2one_cv3.0.2": ("detect_one2one_predictor"),
        "graph.model.23.pose_head.one2one_cv4.0.0.conv": ("pose_one2one_tower"),
        "graph.model.23.pose_head.one2one_cv4_kpts.0": ("pose_one2one_predictor"),
    }
    model = _RepresentativeFull35Weights()
    for path in expected:
        model.add_weight(path, (4, 4, 1, 1))

    catalog = Full35WeightRegionCatalog.inspect(model)

    assert {site.path: site.region for site in catalog.deployment_sites} == expected


def test_catalog_separates_training_only_and_binary_qk_paths() -> None:
    model = _RepresentativeFull35Weights()
    model.add_weight("graph.model.0.conv", (4, 3, 1, 1))
    model.add_weight("graph.model.10.m.0.attn.qkv.q.conv", (4, 4, 1, 1))
    model.add_weight("graph.model.23.detect_head.cv2.0.0.conv", (4, 4, 1, 1))

    catalog = Full35WeightRegionCatalog.inspect(model)

    assert tuple(site.path for site in catalog.deployment_sites) == (
        "graph.model.0.conv",
    )
    assert tuple(site.path for site in catalog.protected_sites) == (
        "graph.model.10.m.0.attn.qkv.q.conv",
    )
    assert tuple(site.path for site in catalog.training_only_sites) == (
        "graph.model.23.detect_head.cv2.0.0.conv",
    )
    assert catalog.deployment_weight_elements == 12


def test_catalog_keeps_pose_sigma_predictors_training_only() -> None:
    model = _RepresentativeFull35Weights()
    model.add_weight(
        "graph.model.23.pose_head.one2one_cv4_kpts.0",
        (6, 4, 1, 1),
    )
    model.add_weight(
        "graph.model.23.pose_head.one2one_cv4_sigma.0",
        (2, 4, 1, 1),
    )

    catalog = Full35WeightRegionCatalog.inspect(model)

    assert {site.path: (site.status, site.region) for site in catalog.sites} == {
        "graph.model.23.pose_head.one2one_cv4_kpts.0": (
            "deployment",
            "pose_one2one_predictor",
        ),
        "graph.model.23.pose_head.one2one_cv4_sigma.0": (
            "training_only",
            "pose_sigma",
        ),
    }


def test_adapter_quantizes_only_one_region_and_restores_fp32_weights() -> None:
    model = _RepresentativeFull35Weights()
    target = model.add_weight("graph.model.0.conv", (2, 1, 1, 4))
    protected = model.add_weight(
        "graph.model.10.m.0.attn.qkv.q.conv",
        (2, 1, 1, 4),
    )
    with torch.no_grad():
        target.weight.copy_(
            torch.tensor([[-1.0, -0.6, 0.0, 0.6], [-2.0, -0.9, 0.0, 0.9]]).reshape_as(
                target.weight
            )
        )
    target_before = target.weight.detach().clone()
    protected_before = protected.weight.detach().clone()
    catalog = Full35WeightRegionCatalog.inspect(model)

    with WeightQuantizationAdapter().quantized(
        model,
        catalog=catalog,
        region="backbone_early",
        spec=UniformWeightSpec(bits=4, scale_method="max"),
    ) as applied:
        expected = torch.tensor(
            [
                [-1.0, -4.0 / 7.0, 0.0, 4.0 / 7.0],
                [-2.0, -6.0 / 7.0, 0.0, 6.0 / 7.0],
            ]
        ).reshape_as(target.weight)
        assert torch.allclose(target.weight, expected)
        assert torch.equal(protected.weight, protected_before)
        assert applied.quantized_modules == 1
        assert applied.weight_elements == 8
        assert applied.format_id == "w4"
        assert applied.numeric["code_min"] == -7
        assert applied.numeric["code_max"] == 4

    assert torch.equal(target.weight, target_before)
    assert torch.equal(protected.weight, protected_before)


def test_catalog_matches_the_accepted_full35_weight_contract() -> None:
    built = Full35ActivationAdapter().build(
        Full35ActivationPolicy(activation="poly_quality", bits=8)
    )

    summary = Full35WeightRegionCatalog.inspect(built.model).summary()

    assert summary["totals"] == {
        "modules": 251,
        "weight_elements": 26_451_392,
        "deployment_modules": 148,
        "deployment_weight_elements": 22_571_840,
        "training_only_modules": 99,
        "training_only_weight_elements": 3_748_480,
        "protected_modules": 4,
        "protected_weight_elements": 131_072,
    }
    assert summary["deployment_regions"] == {
        "backbone_early": {"modules": 21, "weight_elements": 1_218_240},
        "backbone_deep": {"modules": 22, "weight_elements": 8_126_464},
        "backbone_attention_safe": {
            "modules": 7,
            "weight_elements": 919_808,
        },
        "neck": {"modules": 33, "weight_elements": 8_142_848},
        "masf": {"modules": 3, "weight_elements": 74_240},
        "neck_attention_safe": {"modules": 5, "weight_elements": 395_520},
        "detect_one2one_tower": {
            "modules": 18,
            "weight_elements": 1_390_592,
        },
        "detect_one2one_predictor": {
            "modules": 6,
            "weight_elements": 62_208,
        },
        "pose_one2one_tower": {
            "modules": 24,
            "weight_elements": 2_238_464,
        },
        "pose_one2one_predictor": {
            "modules": 9,
            "weight_elements": 3_456,
        },
    }


def test_mse_scale_search_improves_outlier_heavy_w4_reconstruction() -> None:
    model = _RepresentativeFull35Weights()
    target = model.add_weight("graph.model.0.conv", (1, 1, 1, 64))
    with torch.no_grad():
        target.weight.copy_(
            torch.tensor([10.0, *([1.0] * 63)]).reshape_as(target.weight)
        )
    catalog = Full35WeightRegionCatalog.inspect(model)
    adapter = WeightQuantizationAdapter()

    with adapter.quantized(
        model,
        catalog=catalog,
        region="backbone_early",
        spec=UniformWeightSpec(bits=4, scale_method="max"),
    ) as maximum:
        maximum_mse = float(maximum.numeric["mse"])

    with adapter.quantized(
        model,
        catalog=catalog,
        region="backbone_early",
        spec=UniformWeightSpec(bits=4, scale_method="mse"),
    ) as searched:
        searched_mse = float(searched.numeric["mse"])
        searched_scale = float(searched.numeric["scale_maximum"])

    assert searched_mse < maximum_mse
    assert searched_scale < 10.0 / 7.0
    assert torch.equal(
        target.weight,
        torch.tensor([10.0, *([1.0] * 63)]).reshape_as(target.weight),
    )


def test_adapter_reports_finite_metrics_for_an_all_zero_weight_group() -> None:
    model = _RepresentativeFull35Weights()
    target = model.add_weight("graph.model.0.conv", (2, 1, 1, 4))
    with torch.no_grad():
        target.weight.zero_()
    catalog = Full35WeightRegionCatalog.inspect(model)

    with WeightQuantizationAdapter().quantized(
        model,
        catalog=catalog,
        region="backbone_early",
        spec=UniformWeightSpec(
            bits=4,
            scale_method="optimal_scaled_codebook",
        ),
    ) as applied:
        assert applied.numeric["mse"] == 0.0
        assert applied.numeric["sqnr_db"] == 0.0
        assert applied.numeric["occupied_codes"] == 1
        assert applied.numeric["code_min"] == 0
        assert applied.numeric["code_max"] == 0


def test_adapter_applies_fixed_sd4_with_exact_scaled_codebook_and_restores() -> None:
    model = _RepresentativeFull35Weights()
    target = model.add_weight("graph.model.0.conv", (1, 1, 1, 7))
    with torch.no_grad():
        target.weight.copy_(
            torch.tensor([-1.0, -0.5, -0.125, 0.0, 0.125, 0.5, 1.0]).reshape_as(
                target.weight
            )
        )
    original = target.weight.detach().clone()
    catalog = Full35WeightRegionCatalog.inspect(model)

    with WeightQuantizationAdapter().quantized(
        model,
        catalog=catalog,
        region="backbone_early",
        spec=FixedSD4WeightSpec(scale_method="optimal_scaled_codebook"),
    ) as applied:
        assert torch.equal(target.weight, original)
        assert applied.format_id == "fixed-sd4"
        assert applied.numeric["encoded_bits"] == 4
        assert applied.numeric["logical_code_count"] == 15
        assert applied.numeric["weight_code_bytes"] == 4

    assert torch.equal(target.weight, original)


def test_adapter_applies_paper_twn_as_an_explicit_ternary_control() -> None:
    model = _RepresentativeFull35Weights()
    target = model.add_weight("graph.model.0.conv", (1, 1, 1, 5))
    with torch.no_grad():
        target.weight.copy_(
            torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).reshape_as(target.weight)
        )
    original = target.weight.detach().clone()
    catalog = Full35WeightRegionCatalog.inspect(model)

    with WeightQuantizationAdapter().quantized(
        model,
        catalog=catalog,
        region="backbone_early",
        spec=PaperTWNWeightSpec(),
    ) as applied:
        assert torch.equal(
            target.weight,
            torch.tensor([-0.75, -0.75, 0.0, 0.75, 0.75]).reshape_as(target.weight),
        )
        assert applied.format_id == "paper-twn"
        assert applied.numeric["encoded_bits"] == 2
        assert applied.numeric["logical_code_count"] == 3
        assert applied.numeric["zero_ratio"] == pytest.approx(0.2)

    assert torch.equal(target.weight, original)


def test_adapter_applies_filterwise_twn_v3_with_one_alpha_per_filter() -> None:
    model = _RepresentativeFull35Weights()
    target = model.add_weight("graph.model.0.conv", (2, 1, 1, 4))
    with torch.no_grad():
        target.weight.copy_(
            torch.tensor([[-1.0, -0.2, 0.2, 1.0], [-4.0, -2.0, 2.0, 4.0]]).reshape_as(
                target.weight
            )
        )
    original = target.weight.detach().clone()
    catalog = Full35WeightRegionCatalog.inspect(model)

    with WeightQuantizationAdapter().quantized(
        model,
        catalog=catalog,
        region="backbone_early",
        spec=FilterwiseTWNWeightSpec(threshold_multiplier=0.75),
    ) as applied:
        assert torch.equal(
            target.weight,
            torch.tensor([[-1.0, 0.0, 0.0, 1.0], [-4.0, 0.0, 0.0, 4.0]]).reshape_as(
                target.weight
            ),
        )
        assert applied.format_id == "twn-v3-0.75-filterwise"
        assert applied.numeric["scale_count"] == 2
        assert applied.numeric["logical_code_count"] == 3
        assert applied.numeric["zero_ratio"] == pytest.approx(0.5)

    assert torch.equal(target.weight, original)


def test_adapter_applies_exact_scaled_ternary_as_a_distinct_ptq_control() -> None:
    model = _RepresentativeFull35Weights()
    target = model.add_weight("graph.model.0.conv", (1, 1, 1, 5))
    with torch.no_grad():
        target.weight.copy_(
            torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).reshape_as(target.weight)
        )
    original = target.weight.detach().clone()
    catalog = Full35WeightRegionCatalog.inspect(model)

    with WeightQuantizationAdapter().quantized(
        model,
        catalog=catalog,
        region="backbone_early",
        spec=ExactTernaryWeightSpec(),
    ) as applied:
        assert torch.allclose(
            target.weight,
            torch.tensor([-0.75, -0.75, 0.0, 0.75, 0.75]).reshape_as(target.weight),
        )
        assert applied.format_id == "exact-scaled-ternary"
        assert applied.numeric["scale_count"] == 1
        assert applied.numeric["encoded_bits"] == 2
        assert applied.numeric["logical_code_count"] == 3

    assert torch.equal(target.weight, original)


def test_adapter_applies_multiple_disjoint_regions_as_one_reversible_policy() -> None:
    model = _RepresentativeFull35Weights()
    backbone = model.add_weight("graph.model.0.conv", (1, 1, 1, 4))
    neck = model.add_weight("graph.model.16.cv1.conv", (1, 1, 1, 4))
    protected = model.add_weight("graph.model.10.m.0.attn.qkv.q.conv", (1, 1, 1, 4))
    with torch.no_grad():
        backbone.weight.copy_(
            torch.tensor([-1.0, -0.4, 0.4, 1.0]).reshape_as(backbone.weight)
        )
        neck.weight.copy_(torch.tensor([-2.0, -0.7, 0.7, 2.0]).reshape_as(neck.weight))
    originals = {
        "backbone": backbone.weight.detach().clone(),
        "neck": neck.weight.detach().clone(),
        "protected": protected.weight.detach().clone(),
    }
    catalog = Full35WeightRegionCatalog.inspect(model)

    with WeightQuantizationAdapter().quantized_policy(
        model,
        catalog=catalog,
        assignments=(
            WeightRegionAssignment("backbone_early", UniformWeightSpec(4, "max")),
            WeightRegionAssignment("neck", UniformWeightSpec(8, "max")),
        ),
    ) as policy:
        assert policy.regions == ("backbone_early", "neck")
        assert policy.quantized_modules == 2
        assert policy.weight_elements == 8
        assert policy.weight_code_bytes == 6
        assert policy.scale_bytes == 8
        assert not torch.equal(backbone.weight, originals["backbone"])
        assert not torch.equal(neck.weight, originals["neck"])
        assert torch.equal(protected.weight, originals["protected"])

    assert torch.equal(backbone.weight, originals["backbone"])
    assert torch.equal(neck.weight, originals["neck"])
    assert torch.equal(protected.weight, originals["protected"])


def test_policy_can_partition_one_region_into_disjoint_layer_formats() -> None:
    model = _RepresentativeFull35Weights()
    first = model.add_weight("graph.model.0.conv", (1, 1, 1, 4))
    second = model.add_weight("graph.model.1.conv", (1, 1, 1, 4))
    with torch.no_grad():
        first.weight.copy_(
            torch.tensor([-1.0, -0.6, 0.6, 1.0]).reshape_as(first.weight)
        )
        second.weight.copy_(
            torch.tensor([-2.0, -0.7, 0.7, 2.0]).reshape_as(second.weight)
        )
    originals = {
        "first": first.weight.detach().clone(),
        "second": second.weight.detach().clone(),
    }
    catalog = Full35WeightRegionCatalog.inspect(model)

    with WeightQuantizationAdapter().quantized_policy(
        model,
        catalog=catalog,
        assignments=(
            WeightRegionAssignment(
                "backbone_early",
                UniformWeightSpec(4, "max"),
                paths=("graph.model.0.conv",),
            ),
            WeightRegionAssignment(
                "backbone_early",
                UniformWeightSpec(8, "max"),
                paths=("graph.model.1.conv",),
            ),
        ),
    ) as policy:
        assert policy.quantized_modules == 2
        assert policy.weight_elements == 8
        assert policy.weight_code_bytes == 6
        assert policy.assignment_records == (
            {
                "region": "backbone_early",
                "selector": {
                    "kind": "paths",
                    "paths": ["graph.model.0.conv"],
                },
                "format": {
                    "format_id": "w4",
                    "encoded_bits": 4,
                    "scale_method": "max",
                },
            },
            {
                "region": "backbone_early",
                "selector": {
                    "kind": "paths",
                    "paths": ["graph.model.1.conv"],
                },
                "format": {
                    "format_id": "w8",
                    "encoded_bits": 8,
                    "scale_method": "max",
                },
            },
        )
        assert torch.allclose(
            first.weight,
            torch.tensor([-1.0, -4.0 / 7.0, 4.0 / 7.0, 1.0]).reshape_as(first.weight),
        )
        assert torch.allclose(
            second.weight,
            torch.tensor([-2.0, -88.0 / 127.0, 88.0 / 127.0, 2.0]).reshape_as(
                second.weight
            ),
        )

    assert torch.equal(first.weight, originals["first"])
    assert torch.equal(second.weight, originals["second"])


def test_layer_policy_rejects_protected_path_before_mutating_weights() -> None:
    model = _RepresentativeFull35Weights()
    deployment = model.add_weight(
        "graph.model.10.m.0.attn.qkv.v.conv",
        (1, 1, 1, 4),
    )
    protected = model.add_weight(
        "graph.model.10.m.0.attn.qkv.q.conv",
        (1, 1, 1, 4),
    )
    originals = {
        "deployment": deployment.weight.detach().clone(),
        "protected": protected.weight.detach().clone(),
    }
    catalog = Full35WeightRegionCatalog.inspect(model)

    with (
        pytest.raises(ValueError, match="not deployment"),
        WeightQuantizationAdapter().quantized_policy(
            model,
            catalog=catalog,
            assignments=(
                WeightRegionAssignment(
                    "backbone_attention_safe",
                    FixedSD4WeightSpec(),
                    paths=("graph.model.10.m.0.attn.qkv.q.conv",),
                ),
            ),
        ),
    ):
        pass

    assert torch.equal(deployment.weight, originals["deployment"])
    assert torch.equal(protected.weight, originals["protected"])


def test_region_default_can_be_overridden_by_one_layer_format() -> None:
    model = _RepresentativeFull35Weights()
    default_layer = model.add_weight("graph.model.0.conv", (1, 1, 1, 4))
    override_layer = model.add_weight("graph.model.1.conv", (1, 1, 1, 4))
    with torch.no_grad():
        default_layer.weight.copy_(
            torch.tensor([-1.0, -0.6, 0.6, 1.0]).reshape_as(default_layer.weight)
        )
        override_layer.weight.copy_(
            torch.tensor([-1.0, -0.5, 0.0, 1.0]).reshape_as(override_layer.weight)
        )
    originals = (
        default_layer.weight.detach().clone(),
        override_layer.weight.detach().clone(),
    )
    catalog = Full35WeightRegionCatalog.inspect(model)

    with WeightQuantizationAdapter().quantized_policy(
        model,
        catalog=catalog,
        assignments=(
            WeightRegionAssignment("backbone_early", UniformWeightSpec(8, "max")),
            WeightRegionAssignment(
                "backbone_early",
                FixedSD4WeightSpec(),
                paths=("graph.model.1.conv",),
            ),
        ),
    ) as policy:
        assert tuple(item.format_id for item in policy.applied) == ("w8", "fixed-sd4")
        assert tuple(site.path for site in policy.applied[0].sites) == (
            "graph.model.0.conv",
        )
        assert tuple(site.path for site in policy.applied[1].sites) == (
            "graph.model.1.conv",
        )
        assert not torch.equal(default_layer.weight, originals[0])
        assert torch.equal(override_layer.weight, originals[1])

    assert torch.equal(default_layer.weight, originals[0])
    assert torch.equal(override_layer.weight, originals[1])


def test_multi_region_policy_rejects_duplicates_before_mutating_weights() -> None:
    model = _RepresentativeFull35Weights()
    target = model.add_weight("graph.model.0.conv", (1, 1, 1, 4))
    original = target.weight.detach().clone()
    catalog = Full35WeightRegionCatalog.inspect(model)
    duplicate = (
        WeightRegionAssignment("backbone_early", UniformWeightSpec(8)),
        WeightRegionAssignment("backbone_early", UniformWeightSpec(4)),
    )

    with (
        pytest.raises(ValueError, match="unique"),
        WeightQuantizationAdapter().quantized_policy(
            model, catalog=catalog, assignments=duplicate
        ),
    ):
        pass

    assert torch.equal(target.weight, original)
