"""Independent Pose26 baselines with tunable MASF/attention hardware seams."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.utils.torch_utils import unwrap_model

from . import _baselines_impl as _impl

_IMMUTABLE_PARAMETER_PARTS = (
    ".attn.qkv.q.",
    ".attn.qkv.k.",
    ".attn.score.gamma",
)
_IMMUTABLE_BUFFER_PARTS = (
    ".attn.score.fixed_coefficients",
    ".attn.score.calibration_",
    ".attn.normalize.knots",
    ".attn.normalize.values",
)


@dataclass
class _PoseHardwareGuard:
    state: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model: nn.Module) -> "_PoseHardwareGuard":
        selected: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if any(part in name for part in _IMMUTABLE_PARAMETER_PARTS):
                parameter.requires_grad_(False)
                selected[name] = parameter.detach().cpu().clone()
        for name, buffer in model.named_buffers():
            if any(part in name for part in _IMMUTABLE_BUFFER_PARTS):
                selected[name] = buffer.detach().cpu().clone()
        for suffix in (
            "score.fixed_coefficients",
            "normalize.knots",
            "normalize.values",
            "score.gamma",
        ):
            matches = [name for name in selected if name.endswith(suffix)]
            if len(matches) != 2:
                raise ValueError(
                    f"expected two immutable Pose attention states ending {suffix}, got {matches}"
                )
        return cls(selected)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(self.state)

    def assert_unchanged(self, model: nn.Module) -> None:
        current: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if any(part in name for part in _IMMUTABLE_PARAMETER_PARTS):
                current[name] = parameter.detach().cpu()
                if parameter.requires_grad:
                    raise AssertionError(f"immutable Pose parameter became trainable: {name}")
        for name, buffer in model.named_buffers():
            if any(part in name for part in _IMMUTABLE_BUFFER_PARTS):
                current[name] = buffer.detach().cpu()
        if set(current) != set(self.state):
            raise AssertionError("Pose hardware-contract state paths changed")
        changed = [
            name for name in current if not torch.equal(current[name], self.state[name])
        ]
        if changed:
            raise AssertionError(f"Pose hardware-contract state changed: {changed[:20]}")


class MaterializedPoseTrainer(_impl.MaterializedPoseTrainer):
    """Allow V/PE/projection/bias and MASF tuning while locking bit-true state."""

    def build_optimizer(self, model, *args, **kwargs):
        self._inherited_guard = _PoseHardwareGuard.capture(unwrap_model(model))
        return PoseTrainer.build_optimizer(self, model, *args, **kwargs)

    def _model_train(self) -> None:
        # Preserve native head/shared BN behavior for standalone Pose baselines.
        PoseTrainer._model_train(self)


_impl.MaterializedPoseTrainer = MaterializedPoseTrainer
PoseBaselineReport = _impl.PoseBaselineReport
train_pose_baseline = _impl.train_pose_baseline

__all__ = (
    "MaterializedPoseTrainer",
    "PoseBaselineReport",
    "train_pose_baseline",
)

