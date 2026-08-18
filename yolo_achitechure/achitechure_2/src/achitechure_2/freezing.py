"""Immutable MASF and attention scope enforcement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

FROZEN_CLASS_NAMES = frozenset({"HardwareFriendlyAttention", "P3MASFFull35", "P3MASFPartial75"})


def frozen_modules(model: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    selected: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if type(module).__name__ in FROZEN_CLASS_NAMES and not any(
            name == parent or name.startswith(parent + ".") for parent, _ in selected
        ):
            selected.append((name, module))
    paths = tuple(name for name, _ in selected)
    expected_attention = ("model.10.m.0.attn", "model.22.m.0.1.attn")
    if tuple(name for name in paths if name.endswith("attn")) != expected_attention:
        raise ValueError(f"expected frozen attention paths {expected_attention}, got {paths}")
    if "model.16.p3_masf" not in paths:
        raise ValueError("P3 MASF is missing from frozen scope")
    return tuple(selected)


def apply_frozen_scope(model: nn.Module) -> tuple[str, ...]:
    """Train the detector while freezing MASF/attention parameters and state."""

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    modules = frozen_modules(model)
    for _, module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return tuple(name for name, _ in modules)


def enforce_frozen_eval(model: nn.Module) -> None:
    for _, module in frozen_modules(model):
        module.eval()


@dataclass
class FrozenStateGuard:
    """Snapshot frozen parameters and BN buffers and reject any drift."""

    paths: tuple[str, ...]
    state: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model: nn.Module) -> FrozenStateGuard:
        paths = apply_frozen_scope(model)
        state: dict[str, torch.Tensor] = {}
        for path, module in frozen_modules(model):
            for name, value in module.state_dict().items():
                state[f"{path}.{name}"] = value.detach().cpu().clone()
        return cls(paths, state)

    def assert_unchanged(self, model: nn.Module) -> None:
        current: dict[str, torch.Tensor] = {}
        for path, module in frozen_modules(model):
            for name, value in module.state_dict().items():
                current[f"{path}.{name}"] = value.detach().cpu()
            if module.training:
                raise AssertionError(f"frozen module entered training mode: {path}")
        if current.keys() != self.state.keys():
            raise AssertionError("frozen state keys changed")
        changed = [name for name, value in current.items() if not torch.equal(value, self.state[name])]
        if changed:
            raise AssertionError(f"frozen parameters or buffers changed: {changed[:10]}")
