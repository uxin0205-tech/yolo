"""Fail-closed guard for immutable XNOR/PWL/PoT and shared-BN state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .fusion_model import GraphSharedDualHeadModel

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


def _is_head_path(name: str) -> bool:
    return ".detect_head." in name or ".pose_head." in name


def _immutable_state(
    model: GraphSharedDualHeadModel,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    selected: dict[str, torch.Tensor] = {}
    shared_bn_paths: list[str] = []
    for name, parameter in model.named_parameters():
        if any(part in name for part in _IMMUTABLE_PARAMETER_PARTS):
            selected[name] = parameter
    for name, buffer in model.named_buffers():
        if any(part in name for part in _IMMUTABLE_BUFFER_PARTS):
            selected[name] = buffer
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if _is_head_path(module_name):
            continue
        shared_bn_paths.append(module_name)
        for state_name in ("running_mean", "running_var", "num_batches_tracked"):
            value = getattr(module, state_name, None)
            if isinstance(value, torch.Tensor):
                selected[f"{module_name}.{state_name}"] = value
    return selected, tuple(shared_bn_paths)


@dataclass
class HardwareContractGuard:
    """Snapshot only the state that must remain bit-true compatible."""

    state: dict[str, torch.Tensor]
    shared_bn_paths: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        model: GraphSharedDualHeadModel,
    ) -> "HardwareContractGuard":
        selected, shared_bn_paths = _immutable_state(model)
        required_suffixes = (
            "score.fixed_coefficients",
            "normalize.knots",
            "normalize.values",
            "score.gamma",
        )
        for suffix in required_suffixes:
            matches = [name for name in selected if name.endswith(suffix)]
            if len(matches) != 2:
                raise ValueError(
                    f"expected two immutable attention states ending {suffix}, got {matches}"
                )
        if not shared_bn_paths:
            raise ValueError("shared graph contains no BatchNorm modules")
        for name, parameter in model.named_parameters():
            if any(part in name for part in _IMMUTABLE_PARAMETER_PARTS):
                parameter.requires_grad_(False)
        return cls(
            state={
                name: value.detach().cpu().clone()
                for name, value in selected.items()
            },
            shared_bn_paths=shared_bn_paths,
        )

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(self.state)

    def assert_unchanged(
        self,
        model: GraphSharedDualHeadModel,
    ) -> None:
        selected, shared_bn_paths = _immutable_state(model)
        if shared_bn_paths != self.shared_bn_paths:
            raise AssertionError(
                "shared BatchNorm module paths changed: "
                f"{shared_bn_paths} != {self.shared_bn_paths}"
            )
        if set(selected) != set(self.state):
            missing = sorted(set(self.state) - set(selected))
            unexpected = sorted(set(selected) - set(self.state))
            raise AssertionError(
                f"hardware state keys changed; missing={missing}, unexpected={unexpected}"
            )
        changed = [
            name
            for name, value in selected.items()
            if not torch.equal(value.detach().cpu(), self.state[name])
        ]
        if changed:
            raise AssertionError(
                f"hardware-contract state changed: {changed[:20]}"
            )
        modules = dict(model.named_modules())
        training_bn = [
            path for path in self.shared_bn_paths if modules[path].training
        ]
        if training_bn:
            raise AssertionError(
                f"shared BatchNorm entered training mode: {training_bn[:20]}"
            )
        unfrozen = [
            name
            for name, parameter in model.named_parameters()
            if any(part in name for part in _IMMUTABLE_PARAMETER_PARTS)
            and parameter.requires_grad
        ]
        if unfrozen:
            raise AssertionError(
                f"hardware-contract parameters became trainable: {unfrozen}"
            )
