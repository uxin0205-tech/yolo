from __future__ import annotations

import math

from torch import nn

from yolo_quantize.qat_schedule import (
    QATTrainabilityController,
    QuantizationEpochController,
)
from yolo_quantize.qat_weights import (
    FoldedQATWeightAdapter,
    ProgressiveQuantizationSchedule,
)
from yolo_quantize.weight_quantization import (
    Full35WeightRegionCatalog,
    UniformWeightSpec,
    WeightRegionAssignment,
)


class _ToyGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.graph = nn.Module()
        self.graph.model = nn.ModuleList([nn.Sequential(nn.Conv2d(3, 4, 1))])


def _policy(model: nn.Module):
    catalog = Full35WeightRegionCatalog.inspect(model)
    return FoldedQATWeightAdapter().apply(
        model,
        catalog=catalog,
        assignments=(
            WeightRegionAssignment(
                region="backbone_early",
                spec=UniformWeightSpec(bits=8),
            ),
        ),
    )


def test_quantization_epoch_controller_applies_progressive_ratio() -> None:
    policy = _policy(_ToyGraph())
    controller = QuantizationEpochController(
        policy=policy,
        schedule=ProgressiveQuantizationSchedule(start_epoch=0, full_epoch=2),
        sham=False,
    )

    assert controller.begin_epoch(0).blend_ratio == 0.0
    assert controller.begin_epoch(1).blend_ratio == 0.5
    assert controller.begin_epoch(2).blend_ratio == 1.0
    assert policy.blend_ratio == 1.0


def test_quantization_epoch_controller_accepts_float32_schedule_ratio() -> None:
    policy = _policy(_ToyGraph())
    controller = QuantizationEpochController(
        policy=policy,
        schedule=ProgressiveQuantizationSchedule(start_epoch=0, full_epoch=5),
        sham=False,
    )

    state = controller.begin_epoch(1)

    assert state.blend_ratio == 0.2
    assert math.isclose(policy.blend_ratio, 0.2, rel_tol=0.0, abs_tol=1e-7)


def test_quantization_epoch_controller_keeps_matched_sham_at_fp() -> None:
    policy = _policy(_ToyGraph())
    controller = QuantizationEpochController(
        policy=policy,
        schedule=ProgressiveQuantizationSchedule(start_epoch=0, full_epoch=2),
        sham=True,
    )

    assert controller.begin_epoch(0).blend_ratio == 0.0
    assert controller.begin_epoch(10).blend_ratio == 0.0
    assert policy.blend_ratio == 0.0


def test_five_epoch_continuation_has_four_full_quantization_epochs():
    model = _ToyGraph()
    policy = _policy(model)
    quantization = QuantizationEpochController(
        policy=policy,
        schedule=ProgressiveQuantizationSchedule(start_epoch=0, full_epoch=1),
    )
    trainability = QATTrainabilityController(scale_only_epochs=1)
    ratios, frozen = [], []
    for epoch in range(5):
        ratios.append(quantization.begin_epoch(epoch).blend_ratio)
        frozen.append(trainability.apply(model, epoch=epoch).scale_only)
    assert ratios == [0, 1, 1, 1, 1]
    assert frozen == [True, False, False, False, False]


def test_fixed_weight_scale_stays_frozen_after_stage_reenable():
    model = _ToyGraph()
    model.graph.model[0].add_module("second", nn.Conv2d(4, 4, 1))
    _policy(model)
    controller = QATTrainabilityController(
        scale_only_epochs=1, frozen_weight_scale_paths=("graph.model.0.0",)
    )
    for epoch in range(5):
        model.requires_grad_(True)
        controller.apply(model, epoch=epoch)
        assert not model.graph.model[0][
            0
        ].weight_quantizer.scale_parameter.requires_grad
        assert model.graph.model[
            0
        ].second.weight_quantizer.scale_parameter.requires_grad


def test_scale_only_epochs_freeze_model_but_keep_quantizers_trainable() -> None:
    model = _ToyGraph()
    policy = _policy(model)
    controller = QATTrainabilityController(scale_only_epochs=2)

    scale_only = controller.apply(model, epoch=1)
    joint = controller.apply(model, epoch=2)

    assert scale_only.scale_only is True
    assert scale_only.model_trainable_parameters == 0
    assert scale_only.quantizer_trainable_parameters == 1
    assert joint.scale_only is False
    assert joint.model_trainable_parameters == 2
    assert joint.quantizer_trainable_parameters == 1
    assert policy.quantized_modules == 1
