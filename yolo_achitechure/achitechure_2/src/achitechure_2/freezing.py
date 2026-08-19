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


def apply_frozen_scope(model: nn.Module, *, reset_trainable: bool = True) -> tuple[str, ...]:
    """Freeze MASF/attention while optionally preserving a stage's broader freeze."""

    if reset_trainable:
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

    @classmethod
    def capture_preserving_stage(cls, model: nn.Module) -> FrozenStateGuard:
        paths = apply_frozen_scope(model, reset_trainable=False)
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


def apply_stage_freeze(model: nn.Module, stage: str) -> tuple[int, ...]:
    """Apply the explicit Pose stage scope before the permanent inherited freeze."""

    stage = stage.upper()
    if stage == "P1":
        indices = tuple(range(23))
    elif stage == "P2":
        indices = tuple(range(11))
    elif stage in {"P0", "P3", "P4", "D0", "D1", "D2", "Q2"}:
        indices = ()
    else:
        raise ValueError(f"unknown freeze stage {stage}")
    layers = getattr(model, "model", None)
    if not isinstance(layers, nn.Sequential):
        raise TypeError("stage freeze requires an Ultralytics Sequential graph")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for index in indices:
        layers[index].eval()
        for parameter in layers[index].parameters():
            parameter.requires_grad_(False)
    apply_frozen_scope(model, reset_trainable=False)
    return indices


def enforce_stage_eval(model: nn.Module, stage: str) -> None:
    """Keep stage-frozen BN/dropout buffers out of train mode."""

    indices = (
        tuple(range(23)) if stage.upper() == "P1" else (tuple(range(11)) if stage.upper() == "P2" else ())
    )
    layers = getattr(model, "model", None)
    if not isinstance(layers, nn.Sequential):
        raise TypeError("stage eval requires an Ultralytics Sequential graph")
    for index in indices:
        layers[index].eval()


@dataclass
class StageFrozenStateGuard:
    """Reject parameter/buffer or mode drift in P1/P2 frozen layers."""

    stage: str
    indices: tuple[int, ...]
    state: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model: nn.Module, stage: str) -> StageFrozenStateGuard:
        indices = apply_stage_freeze(model, stage)
        layers = model.model
        state: dict[str, torch.Tensor] = {}
        for index in indices:
            for name, value in layers[index].state_dict().items():
                state[f"model.{index}.{name}"] = value.detach().cpu().clone()
        return cls(stage.upper(), indices, state)

    def assert_unchanged(self, model: nn.Module) -> None:
        layers = model.model
        current: dict[str, torch.Tensor] = {}
        for index in self.indices:
            if layers[index].training:
                raise AssertionError(f"stage-frozen layer entered training mode: {index}")
            for name, value in layers[index].state_dict().items():
                current[f"model.{index}.{name}"] = value.detach().cpu()
        if current.keys() != self.state.keys():
            raise AssertionError("stage-frozen state keys changed")
        changed = [name for name, value in current.items() if not torch.equal(value, self.state[name])]
        if changed:
            raise AssertionError(f"stage-frozen parameters or buffers changed: {changed[:10]}")
