"""Staged unfreezing, immutable frozen-state checks, and MuSGD groups."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn
from ultralytics.optim import MuSGD

from .model import ATTENTION_PATHS, inspect_yolo26_graph


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    epochs: int
    learning_rates: dict[str, float]
    lrf: float
    warmup_epochs: float
    patience: int
    cosine: bool
    momentum: float = 0.948
    weight_decay: float = 0.00027


@dataclass(frozen=True)
class ScopeReport:
    phase: str
    trainable_parameters: int
    total_parameters: int
    trainable_names: tuple[str, ...]


PHASES = {
    "a1": PhaseSpec("a1", 5, {"masf": 1.0e-3}, 1.0, 0.5, 100, False),
    "a2": PhaseSpec("a2", 10, {"masf": 3.8e-4}, 0.5, 0.0, 4, True),
    "b": PhaseSpec("b", 10, {"masf": 3.8e-4, "neck_detect": 1.9e-4}, 0.5, 1.0, 4, True),
    "c": PhaseSpec(
        "c",
        55,
        {"masf": 3.8e-4, "neck_detect": 1.9e-4, "backbone": 3.8e-5, "attention": 5.0e-6},
        0.1,
        1.0,
        8,
        True,
    ),
}


def tuning_phase(masf_lr: float) -> PhaseSpec:
    """Return one 10-epoch winner continuation with Phase-C LR ratios."""

    if masf_lr not in {2.0e-4, 3.8e-4, 6.0e-4}:
        raise ValueError("winner tuning LR must be one of 0.0002, 0.00038, or 0.0006")
    scale = masf_lr / PHASES["c"].learning_rates["masf"]
    learning_rates = {name: value * scale for name, value in PHASES["c"].learning_rates.items()}
    return PhaseSpec("tune", 10, learning_rates, 0.1, 1.0, 5, True)


def _is_attention(name: str) -> bool:
    return any(name.startswith(f"{path}.") for path in ATTENTION_PATHS)


def parameter_role(name: str) -> str:
    """Map one graph parameter to exactly one discriminative-LR role."""

    if ".p3_masf." in name:
        return "masf"
    if _is_attention(name):
        return "attention"
    match = re.match(r"^model\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"parameter is outside the YOLO graph: {name}")
    return "backbone" if int(match.group(1)) <= 10 else "neck_detect"


def apply_phase_scope(model: nn.Module, phase: str) -> ScopeReport:
    """Apply the exact Phase A/B/C trainable scope to a validated graph."""

    phase = phase.lower()
    if phase not in {*PHASES, "tune"}:
        raise ValueError(f"unknown phase {phase!r}")
    inspect_yolo26_graph(model)
    for name, parameter in model.named_parameters():
        role = parameter_role(name)
        if phase in {"a1", "a2"}:
            trainable = role == "masf"
        elif phase == "b":
            trainable = role in {"masf", "neck_detect"} and not _is_attention(name)
        else:
            trainable = True
        parameter.requires_grad_(trainable)
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    return ScopeReport(
        phase=phase,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_names=names,
    )


def enforce_frozen_modules_eval(model: nn.Module) -> None:
    """Freeze train/eval-sensitive state without changing Detect's training output contract."""

    for module in model.modules():
        parameters = tuple(module.parameters())
        is_frozen = not any(parameter.requires_grad for parameter in parameters)
        is_attention = module.__class__.__name__ == "HardwareFriendlyAttention"
        if is_frozen and (isinstance(module, nn.modules.batchnorm._BatchNorm) or is_attention):
            module.eval()


def _normalization_parameter_ids(model: nn.Module) -> set[int]:
    return {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for parameter in module.parameters(recurse=False)
    }


def build_phase_optimizer(model: nn.Module, phase: PhaseSpec) -> MuSGD:
    """Build exact MuSGD role/decay groups without frozen parameters."""

    expected_roles = set(phase.learning_rates)
    norm_ids = _normalization_parameter_ids(model)
    buckets: dict[tuple[str, bool, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = parameter_role(name)
        if role not in expected_roles:
            raise RuntimeError(f"phase {phase.name} has no LR for trainable role {role}: {name}")
        use_muon = parameter.ndim >= 2
        decay = use_muon and id(parameter) not in norm_ids and not name.endswith(".bias")
        buckets.setdefault((role, decay, use_muon), []).append(parameter)
    actual_roles = {role for role, _, _ in buckets}
    if actual_roles != expected_roles:
        raise RuntimeError(f"empty optimizer role(s): expected {expected_roles}, got {actual_roles}")
    groups = []
    for (role, decay, use_muon), parameters in buckets.items():
        lr = phase.learning_rates[role]
        groups.append(
            {
                "params": parameters,
                "lr": lr,
                "initial_lr": lr,
                "momentum": phase.momentum,
                "nesterov": True,
                "weight_decay": phase.weight_decay if decay else 0.0,
                "use_muon": use_muon,
                "role": role,
            }
        )
    optimizer = MuSGD(groups, muon=0.2, sgd=1.0)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        raise RuntimeError("optimizer parameter identities do not equal the trainable scope")
    return optimizer


def snapshot_frozen_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy every frozen parameter and buffer owned by a frozen module."""

    snapshot = {
        f"parameter:{name}": value.detach().clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    modules = dict(model.named_modules())
    for name, value in model.named_buffers():
        owner_name = name.rsplit(".", 1)[0] if "." in name else ""
        owner = modules[owner_name]
        direct_parameters = tuple(owner.parameters(recurse=False))
        if not direct_parameters or not any(parameter.requires_grad for parameter in direct_parameters):
            snapshot[f"buffer:{name}"] = value.detach().clone()
    return snapshot


def assert_frozen_state_unchanged(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    """Raise with exact tensor names if a frozen parameter or buffer changed."""

    current = {
        **{f"parameter:{name}": value for name, value in model.named_parameters()},
        **{f"buffer:{name}": value for name, value in model.named_buffers()},
    }
    changed = [name for name, expected in snapshot.items() if not torch.equal(current[name], expected)]
    if changed:
        raise AssertionError(f"frozen state changed: {changed[:10]}")
