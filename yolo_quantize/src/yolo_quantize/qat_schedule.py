"""Epoch-boundary controls for progressive weight QAT and matched shams."""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import nn

from .qat_optimizer import is_quantizer_parameter
from .qat_weights import (
    AppliedFoldedQATWeightPolicy,
    ProgressiveQuantizationSchedule,
)


@dataclass(frozen=True)
class QuantizationEpochState:
    """One auditable epoch-boundary fake-quant setting."""

    epoch: int
    blend_ratio: float
    sham: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "epoch": self.epoch,
            "blend_ratio": self.blend_ratio,
            "sham": self.sham,
        }


@dataclass(frozen=True)
class QuantizationEpochController:
    """Apply exactly one progressive schedule or its architecture-matched sham."""

    policy: AppliedFoldedQATWeightPolicy
    schedule: ProgressiveQuantizationSchedule
    sham: bool = False

    def begin_epoch(self, epoch: int) -> QuantizationEpochState:
        ratio = 0.0 if self.sham else self.schedule.ratio(epoch)
        self.policy.set_blend_ratio(ratio)
        if not math.isclose(
            self.policy.blend_ratio,
            ratio,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise RuntimeError("QAT weight modules did not accept a common blend ratio")
        return QuantizationEpochState(
            epoch=epoch,
            blend_ratio=ratio,
            sham=self.sham,
        )


@dataclass(frozen=True)
class QATTrainabilityState:
    """Parameter-tensor counts after applying one epoch's trainability phase."""

    epoch: int
    scale_only: bool
    model_trainable_parameters: int
    quantizer_trainable_parameters: int

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "epoch": self.epoch,
            "scale_only": self.scale_only,
            "model_trainable_parameters": self.model_trainable_parameters,
            "quantizer_trainable_parameters": self.quantizer_trainable_parameters,
        }


class QATTrainabilityController:
    """Freeze FP model tensors during scale-only epochs, then restore stage policy."""

    def __init__(
        self, *, scale_only_epochs: int, frozen_weight_scale_paths: tuple[str, ...] = ()
    ) -> None:
        if scale_only_epochs < 0:
            raise ValueError("scale_only_epochs cannot be negative")
        self.scale_only_epochs = int(scale_only_epochs)
        self.frozen_weight_scale_names = {
            f"{p}.weight_quantizer._scale_unconstrained"
            for p in frozen_weight_scale_paths
        }
        self._stage_trainability: dict[str, bool] | None = None

    def apply(self, model: nn.Module, *, epoch: int) -> QATTrainabilityState:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        parameters = dict(model.named_parameters())
        if not self.frozen_weight_scale_names.issubset(parameters):
            raise ValueError("fixed weight scale path is absent from model")
        if self._stage_trainability is None:
            self._stage_trainability = {
                name: parameter.requires_grad for name, parameter in parameters.items()
            }
        elif set(parameters) != set(self._stage_trainability):
            raise RuntimeError("QAT graph parameters changed between epochs")
        scale_only = epoch < self.scale_only_epochs
        for name, parameter in parameters.items():
            stage_trainable = self._stage_trainability[name]
            parameter.requires_grad_(
                stage_trainable
                and name not in self.frozen_weight_scale_names
                and (not scale_only or is_quantizer_parameter(name))
            )
        model_count = sum(
            parameter.requires_grad and not is_quantizer_parameter(name)
            for name, parameter in parameters.items()
        )
        quantizer_count = sum(
            parameter.requires_grad and is_quantizer_parameter(name)
            for name, parameter in parameters.items()
        )
        if scale_only and quantizer_count == 0:
            raise RuntimeError("scale-only QAT has no trainable quantizer parameters")
        return QATTrainabilityState(
            epoch=epoch,
            scale_only=scale_only,
            model_trainable_parameters=model_count,
            quantizer_trainable_parameters=quantizer_count,
        )


__all__ = (
    "QATTrainabilityController",
    "QATTrainabilityState",
    "QuantizationEpochController",
    "QuantizationEpochState",
)
