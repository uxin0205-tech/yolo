from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from yolo_quantize.qat_weights import (
    FoldedQATWeightAdapter,
    ProgressiveQuantizationSchedule,
    QATConv2d,
    QATLinear,
    TrainableWeightFakeQuantizer,
)
from yolo_quantize.weight_quantization import (
    ExactTernaryWeightSpec,
    FilterwiseTWNWeightSpec,
    FixedSD4WeightSpec,
    Full35WeightRegionCatalog,
    PaperTWNWeightSpec,
    UniformWeightSpec,
    WeightRegionAssignment,
)


class _ToyFoldedGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graph = nn.Module()
        self.graph.model = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU()),
                nn.Sequential(nn.Flatten(), nn.Linear(16, 3)),
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.graph.model[0](value)
        return self.graph.model[1](value)


def _toy_catalog(model: nn.Module) -> Full35WeightRegionCatalog:
    catalog = Full35WeightRegionCatalog.inspect(model)
    assert {site.region for site in catalog.deployment_sites} == {"backbone_early"}
    return catalog


@pytest.mark.parametrize(
    "spec",
    [
        UniformWeightSpec(bits=7, scale_method="optimal_scaled_codebook"),
        FixedSD4WeightSpec(scale_method="optimal_scaled_codebook"),
        PaperTWNWeightSpec(threshold_multiplier=0.7),
        ExactTernaryWeightSpec(),
        FilterwiseTWNWeightSpec(threshold_multiplier=0.75),
    ],
)
def test_trainable_fake_quantizer_has_finite_master_and_scale_gradients(spec) -> None:
    weight = torch.linspace(-1.0, 1.0, 48).reshape(4, 3, 2, 2).requires_grad_()
    quantizer = TrainableWeightFakeQuantizer.from_weight(weight.detach(), spec)
    quantizer.set_blend_ratio(1.0)

    output = quantizer(weight)
    output.square().mean().backward()

    assert output.shape == weight.shape
    assert torch.isfinite(output).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert quantizer.scale_parameter.grad is not None
    assert torch.isfinite(quantizer.scale_parameter.grad).all()


def test_progressive_quantization_schedule_reaches_full_strength() -> None:
    schedule = ProgressiveQuantizationSchedule(start_epoch=1, full_epoch=4)

    assert schedule.ratio(0) == 0.0
    assert schedule.ratio(1) == 0.0
    assert schedule.ratio(2) == pytest.approx(1.0 / 3.0)
    assert schedule.ratio(3) == pytest.approx(2.0 / 3.0)
    assert schedule.ratio(4) == 1.0
    assert schedule.ratio(100) == 1.0


@pytest.mark.parametrize(
    ("spec", "maximum_levels"),
    [
        (FixedSD4WeightSpec(), 15),
        (PaperTWNWeightSpec(), 3),
        (ExactTernaryWeightSpec(), 3),
        (FilterwiseTWNWeightSpec(), 3),
    ],
)
def test_special_format_forward_uses_only_its_declared_codebook(
    spec,
    maximum_levels: int,
) -> None:
    weight = torch.linspace(-1.0, 1.0, 100).reshape(4, 25)
    quantizer = TrainableWeightFakeQuantizer.from_weight(weight, spec)
    quantizer.set_blend_ratio(1.0)
    output = quantizer(weight)
    scales = quantizer.scale.detach().reshape(-1, 1)
    normalized = output.detach() / scales

    assert torch.unique(normalized).numel() <= maximum_levels
    if isinstance(spec, FixedSD4WeightSpec):
        expected = {
            -1.0,
            -0.5,
            -0.25,
            -0.125,
            -0.0625,
            -0.03125,
            -0.015625,
            0.0,
            0.015625,
            0.03125,
            0.0625,
            0.125,
            0.25,
            0.5,
            1.0,
        }
    else:
        expected = {-1.0, 0.0, 1.0}
    assert set(torch.unique(normalized).tolist()) <= expected


@pytest.mark.parametrize(
    ("spec", "expected_scale_count"),
    [
        (ExactTernaryWeightSpec(), 1),
        (FilterwiseTWNWeightSpec(), 4),
    ],
)
def test_ternary_qat_scale_granularity_matches_the_declared_format(
    spec,
    expected_scale_count: int,
) -> None:
    weight = torch.linspace(-1.0, 1.0, 100).reshape(4, 25)

    quantizer = TrainableWeightFakeQuantizer.from_weight(weight, spec)

    assert quantizer.scale.numel() == expected_scale_count


def test_qat_adapter_refuses_unfolded_batch_norm() -> None:
    model = _ToyFoldedGraph()
    model.graph.model[0].add_module("bn", nn.BatchNorm2d(4))
    catalog = _toy_catalog(model)

    with pytest.raises(ValueError, match="BN-folded"):
        FoldedQATWeightAdapter().apply(
            model,
            catalog=catalog,
            assignments=(
                WeightRegionAssignment(
                    region="backbone_early",
                    spec=UniformWeightSpec(bits=8),
                ),
            ),
        )


def test_qat_adapter_preserves_master_keys_and_materializes_plain_modules() -> None:
    torch.manual_seed(7)
    model = _ToyFoldedGraph()
    original = copy.deepcopy(model.state_dict())
    catalog = _toy_catalog(model)
    applied = FoldedQATWeightAdapter().apply(
        model,
        catalog=catalog,
        assignments=(
            WeightRegionAssignment(
                region="backbone_early",
                spec=UniformWeightSpec(bits=6),
            ),
        ),
    )

    assert applied.quantized_modules == 2
    assert isinstance(model.get_submodule("graph.model.0.0"), QATConv2d)
    assert isinstance(model.get_submodule("graph.model.1.1"), QATLinear)
    assert torch.equal(
        model.state_dict()["graph.model.0.0.weight"], original["graph.model.0.0.weight"]
    )
    assert torch.equal(
        model.state_dict()["graph.model.1.1.weight"], original["graph.model.1.1.weight"]
    )

    applied.set_blend_ratio(1.0)
    materialized = applied.materialize(clone_model=True)
    assert isinstance(materialized.get_submodule("graph.model.0.0"), nn.Conv2d)
    assert not isinstance(materialized.get_submodule("graph.model.0.0"), QATConv2d)
    assert isinstance(materialized.get_submodule("graph.model.1.1"), nn.Linear)
    assert not isinstance(materialized.get_submodule("graph.model.1.1"), QATLinear)
    assert not any("weight_quantizer" in name for name in materialized.state_dict())

    rebound = applied.rebind(copy.deepcopy(model))
    rebound.set_blend_ratio(0.5)
    assert rebound.blend_ratio == 0.5
    assert applied.blend_ratio == 1.0

    sample = torch.randn(2, 3, 2, 2)
    model.eval()
    materialized.eval()
    assert torch.allclose(model(sample), materialized(sample), atol=1e-6, rtol=0.0)


def test_qat_adapter_supports_disjoint_default_and_path_override() -> None:
    model = _ToyFoldedGraph()
    catalog = _toy_catalog(model)
    applied = FoldedQATWeightAdapter().apply(
        model,
        catalog=catalog,
        assignments=(
            WeightRegionAssignment(
                region="backbone_early",
                spec=UniformWeightSpec(bits=8),
            ),
            WeightRegionAssignment(
                region="backbone_early",
                paths=("graph.model.1.1",),
                spec=FixedSD4WeightSpec(),
            ),
        ),
    )

    records = {record["path"]: record for record in applied.site_records}
    assert records["graph.model.0.0"]["format_id"] == "w8"
    assert records["graph.model.1.1"]["format_id"] == "fixed-sd4"
    assert records["graph.model.1.1"]["encoded_bits"] == 4


def test_matched_sham_materializes_plain_fp_weights_at_ratio_zero() -> None:
    model = _ToyFoldedGraph()
    original = copy.deepcopy(model.state_dict())
    applied = FoldedQATWeightAdapter().apply(
        model,
        catalog=_toy_catalog(model),
        assignments=(
            WeightRegionAssignment(
                region="backbone_early",
                spec=UniformWeightSpec(bits=7),
            ),
        ),
    )

    materialized = applied.materialize(
        clone_model=True,
        expected_blend_ratio=0.0,
    )

    assert not any("weight_quantizer" in name for name in materialized.state_dict())
    assert torch.equal(
        materialized.state_dict()["graph.model.0.0.weight"],
        original["graph.model.0.0.weight"],
    )


@pytest.mark.parametrize(
    "spec",
    [
        UniformWeightSpec(bits=8, scale_method="mse_grid_v1"),
        FixedSD4WeightSpec(),
        PaperTWNWeightSpec(),
        ExactTernaryWeightSpec(),
        FilterwiseTWNWeightSpec(),
    ],
)
def test_trainable_quantizer_reinitializes_scale_from_warm_started_weight(spec) -> None:
    initial = torch.linspace(-0.1, 0.1, 48).reshape(4, 3, 2, 2)
    warm_started = torch.linspace(-1.5, 0.75, 48).reshape(4, 3, 2, 2)
    quantizer = TrainableWeightFakeQuantizer.from_weight(initial, spec)
    expected = TrainableWeightFakeQuantizer.from_weight(warm_started, spec)
    quantizer.set_blend_ratio(1.0)

    quantizer.reinitialize_from_weight(warm_started)

    assert torch.allclose(quantizer.scale, expected.scale, atol=1e-7, rtol=0.0)
    assert quantizer.blend_ratio == 0.0
