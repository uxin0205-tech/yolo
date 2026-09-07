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
