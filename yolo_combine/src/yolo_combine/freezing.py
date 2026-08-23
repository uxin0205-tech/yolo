"""Immutable scope for the accepted attention and MASF feature modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

INHERITED_CLASS_NAMES = frozenset(
    {"HardwareFriendlyAttention", "P3MASFFull35", "P3MASFPartial75"}
)
EXPECTED_SUFFIXES = frozenset(
    {"10.m.0.attn", "16.p3_masf", "22.m.0.1.attn"}
)


def inherited_modules(model: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    """Find the three selected feature modules in a task graph or shared trunk."""

    selected = tuple(
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ in INHERITED_CLASS_NAMES
    )
    suffixes = {
        suffix
        for suffix in EXPECTED_SUFFIXES
        if any(name.endswith(suffix) for name, _ in selected)
    }
    if suffixes != EXPECTED_SUFFIXES or len(selected) != 3:
        paths = tuple(name for name, _ in selected)
        raise ValueError(f"expected exactly three inherited modules {sorted(EXPECTED_SUFFIXES)}, got {paths}")
    return selected


def freeze_inherited(model: nn.Module) -> tuple[str, ...]:
    """Freeze parameters and training-time buffers without changing other layers."""

    modules = inherited_modules(model)
    for _, module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return tuple(name for name, _ in modules)


def enforce_inherited_eval(model: nn.Module) -> None:
    for _, module in inherited_modules(model):
        module.eval()


@dataclass
class InheritedFreezeGuard:
    """Reject any parameter, buffer, or mode drift in the inherited feature scope."""

    paths: tuple[str, ...]
    state: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model: nn.Module) -> InheritedFreezeGuard:
        paths = freeze_inherited(model)
        state: dict[str, torch.Tensor] = {}
        for path, module in inherited_modules(model):
            for name, value in module.state_dict().items():
                state[f"{path}.{name}"] = value.detach().cpu().clone()
        return cls(paths=paths, state=state)

    def assert_unchanged(self, model: nn.Module) -> None:
        current: dict[str, torch.Tensor] = {}
        for path, module in inherited_modules(model):
            if module.training:
                raise AssertionError(f"inherited module entered training mode: {path}")
            for name, value in module.state_dict().items():
                current[f"{path}.{name}"] = value.detach().cpu()
        if current.keys() != self.state.keys():
            raise AssertionError("inherited module state keys changed")
        changed = [
            name
            for name, value in current.items()
            if not torch.equal(value, self.state[name])
        ]
        if changed:
            raise AssertionError(f"inherited parameters or buffers changed: {changed[:10]}")
