"""Joint-training stages, parameter ownership, BN policy, and optimizer groups."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import torch
from torch import nn
from ultralytics.optim import MuSGD

from .fusion_model import GraphSharedDualHeadModel

SemanticRole = Literal[
    "backbone",
    "neck",
    "masf",
    "attention",
    "detect_head",
    "pose_head",
]
OptimizerName = Literal["AdamW", "MuSGD"]
TaskMode = Literal["pose", "joint"]

_ROLES: tuple[SemanticRole, ...] = (
    "backbone",
    "neck",
    "masf",
    "attention",
    "detect_head",
    "pose_head",
)
_LAYER_PATTERN = re.compile(r"^graph\.model\.(\d+)\.")
_HARDWARE_FROZEN_PARTS = (
    ".attn.qkv.q.",
    ".attn.qkv.k.",
    ".attn.score.gamma",
)


@dataclass(frozen=True)
class JointStage:
    """One accepted J-stage with explicit trainable scope and learning rates."""

    name: Literal["j0", "j1", "j2", "j3"]
    task_mode: TaskMode
    epochs: int
    patience: int
    backbone_start_layer: int
    tune_attention: bool
    learning_rates: Mapping[SemanticRole, float]
    warmup_epochs: int = 3

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("stage epochs must be positive")
        if self.patience < 0:
            raise ValueError("stage patience cannot be negative")
        if not 0 <= self.backbone_start_layer <= 11:
            raise ValueError("backbone_start_layer must be in [0, 11]")
        if set(self.learning_rates) != set(_ROLES):
            raise ValueError(
                f"learning-rate roles must be exactly {_ROLES}, got {tuple(self.learning_rates)}"
            )
        if any(value < 0 for value in self.learning_rates.values()):
            raise ValueError("learning rates cannot be negative")


JOINT_STAGES: Mapping[str, JointStage] = MappingProxyType(
    {
        "j0": JointStage(
            name="j0",
            task_mode="pose",
            epochs=8,
            patience=0,
            backbone_start_layer=11,
            tune_attention=False,
            learning_rates=MappingProxyType(
                {
                    "backbone": 0.0,
                    "neck": 0.0,
                    "masf": 0.0,
                    "attention": 0.0,
                    "detect_head": 0.0,
                    "pose_head": 2.0e-4,
                }
            ),
            warmup_epochs=1,
        ),
        "j1": JointStage(
            name="j1",
            task_mode="joint",
            epochs=20,
            patience=8,
            backbone_start_layer=11,
            tune_attention=False,
            learning_rates=MappingProxyType(
                {
                    "backbone": 0.0,
                    "neck": 7.5e-5,
                    "masf": 0.0,
                    "attention": 0.0,
                    "detect_head": 2.0e-4,
                    "pose_head": 2.0e-4,
                }
            ),
            warmup_epochs=1,
        ),
        "j2": JointStage(
            name="j2",
            task_mode="joint",
            epochs=80,
            patience=17,
            backbone_start_layer=9,
            tune_attention=False,
            learning_rates=MappingProxyType(
                {
                    "backbone": 1.5e-5,
                    "neck": 7.5e-5,
                    "masf": 1.5e-4,
                    "attention": 0.0,
                    "detect_head": 2.0e-4,
                    "pose_head": 2.0e-4,
                }
            ),
            warmup_epochs=1,
        ),
        "j3": JointStage(
            name="j3",
            task_mode="joint",
            epochs=20,
            patience=5,
            backbone_start_layer=0,
            tune_attention=True,
            learning_rates=MappingProxyType(
                {
                    "backbone": 3.8e-6,
                    "neck": 1.9e-5,
                    "masf": 3.8e-5,
                    "attention": 5.0e-7,
                    "detect_head": 5.0e-5,
                    "pose_head": 5.0e-5,
                }
            ),
        ),
    }
)


@dataclass(frozen=True)
class StageApplicationReport:
    stage: str
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    hardware_frozen_names: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int
    shared_bn_frozen: int
    head_bn_training: int


@dataclass(frozen=True)
class OptimizerBuildReport:
    optimizer: OptimizerName
    group_names: tuple[str, ...]
    semantic_roles: tuple[SemanticRole, ...]
    parameter_names: tuple[str, ...]
    duplicate_parameter_names: tuple[str, ...]
    parameter_count: int
    weight_decay: float


@dataclass(frozen=True)
class _ParameterPolicy:
    name: str
    parameter: nn.Parameter
    role: SemanticRole
    layer: int
    hardware_frozen: bool
    decay: bool


def _layer_index(name: str) -> int:
    match = _LAYER_PATTERN.match(name)
    if match is None:
        raise ValueError(f"parameter is outside the shared graph: {name}")
    return int(match.group(1))


def _classify(name: str, parameter: nn.Parameter) -> _ParameterPolicy:
    layer = _layer_index(name)
    if ".detect_head." in name:
        role: SemanticRole = "detect_head"
    elif ".pose_head." in name:
        role = "pose_head"
    elif ".p3_masf." in name:
        role = "masf"
    elif ".attn." in name:
        role = "attention"
    elif layer <= 10:
        role = "backbone"
    elif layer <= 22:
        role = "neck"
    else:
        raise ValueError(f"unclassified layer {layer} parameter: {name}")
    hardware_frozen = any(part in name for part in _HARDWARE_FROZEN_PARTS)
    decay = (
        parameter.ndim > 1
        and not name.endswith(".bias")
        and ".bias.table_" not in name
    )
    return _ParameterPolicy(
        name=name,
        parameter=parameter,
        role=role,
        layer=layer,
        hardware_frozen=hardware_frozen,
        decay=decay,
    )


def _parameter_policies(
    model: GraphSharedDualHeadModel,
) -> tuple[_ParameterPolicy, ...]:
    occurrences: dict[int, list[str]] = {}
    raw: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        occurrences.setdefault(id(parameter), []).append(name)
        raw.append((name, parameter))
    duplicates = {
        identifier: names
        for identifier, names in occurrences.items()
        if len(names) > 1
    }
    if duplicates:
        rendered = tuple(tuple(names) for names in duplicates.values())
        raise ValueError(f"Parameter objects are registered more than once: {rendered}")
    return tuple(_classify(name, parameter) for name, parameter in raw)


def _is_trainable(policy: _ParameterPolicy, stage: JointStage) -> bool:
    if policy.hardware_frozen:
        return False
    if stage.learning_rates[policy.role] <= 0:
        return False
    if policy.role in {"detect_head", "pose_head", "neck", "masf"}:
        return True
    if policy.role == "attention":
        return stage.tune_attention
    if policy.role == "backbone":
        return policy.layer >= stage.backbone_start_layer
    raise AssertionError(policy.role)


def _apply_bn_modes(
    model: GraphSharedDualHeadModel,
    stage: JointStage,
) -> tuple[int, int]:
    shared = 0
    heads = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if ".detect_head." in name:
            trainable = stage.learning_rates["detect_head"] > 0
            module.train(trainable)
            heads += int(trainable)
        elif ".pose_head." in name:
            trainable = stage.learning_rates["pose_head"] > 0
            module.train(trainable)
            heads += int(trainable)
        else:
            module.eval()
            shared += 1
    return shared, heads


def apply_stage(
    model: GraphSharedDualHeadModel,
    stage: JointStage,
) -> StageApplicationReport:
    """Reapply trainability and BN modes after every model.train() call."""

    model.train(True)
    policies = _parameter_policies(model)
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    hardware_names: list[str] = []
    trainable_parameters = 0
    frozen_parameters = 0
    for policy in policies:
        trainable = _is_trainable(policy, stage)
        policy.parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(policy.name)
            trainable_parameters += policy.parameter.numel()
        else:
            frozen_names.append(policy.name)
            frozen_parameters += policy.parameter.numel()
            if policy.hardware_frozen:
                hardware_names.append(policy.name)
    shared_bn, head_bn = _apply_bn_modes(model, stage)
    return StageApplicationReport(
        stage=stage.name,
        trainable_names=tuple(trainable_names),
        frozen_names=tuple(frozen_names),
        hardware_frozen_names=tuple(hardware_names),
        trainable_parameters=trainable_parameters,
        frozen_parameters=frozen_parameters,
        shared_bn_frozen=shared_bn,
        head_bn_training=head_bn,
    )


def build_joint_optimizer(
    model: GraphSharedDualHeadModel,
    stage: JointStage,
    *,
    optimizer_name: OptimizerName = "AdamW",
    weight_decay: float = 0.00027,
    beta1: float = 0.948,
    beta2: float = 0.999,
) -> tuple[torch.optim.Optimizer, OptimizerBuildReport]:
    """Build persistent semantic groups without Ultralytics nbs rescaling."""

    if optimizer_name not in {"AdamW", "MuSGD"}:
        raise ValueError("optimizer_name must be AdamW or MuSGD")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("optimizer beta/momentum values must be in [0, 1)")
    apply_stage(model, stage)
    policies = tuple(policy for policy in _parameter_policies(model) if not policy.hardware_frozen)
    grouped: dict[tuple[SemanticRole, bool], list[_ParameterPolicy]] = {}
    for policy in policies:
        grouped.setdefault((policy.role, policy.decay), []).append(policy)
    parameter_groups: list[dict[str, object]] = []
    for role in _ROLES:
        for decay in (True, False):
            selected = grouped.get((role, decay), [])
            if not selected:
                continue
            group_name = f"{role}.{'decay' if decay else 'no_decay'}"
            group: dict[str, object] = {
                "params": [policy.parameter for policy in selected],
                "param_names": tuple(policy.name for policy in selected),
                "group_name": group_name,
                "role": role,
                "lr": float(stage.learning_rates[role]),
                "weight_decay": float(weight_decay if decay else 0.0),
            }
            if optimizer_name == "MuSGD":
                group.update(
                    momentum=beta1,
                    nesterov=True,
                    use_muon=decay,
                )
            parameter_groups.append(group)

    identifiers = [
        id(parameter)
        for group in parameter_groups
        for parameter in group["params"]  # type: ignore[index,union-attr]
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("the same Parameter object appears in multiple optimizer groups")
    if optimizer_name == "AdamW":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=(beta1, beta2),
        )
    else:
        optimizer = MuSGD(parameter_groups, muon=0.2, sgd=1.0)
    report = OptimizerBuildReport(
        optimizer=optimizer_name,
        group_names=tuple(str(group["group_name"]) for group in parameter_groups),
        semantic_roles=tuple(
            role for role in _ROLES if any(group["role"] == role for group in parameter_groups)
        ),
        parameter_names=tuple(policy.name for policy in policies),
        duplicate_parameter_names=(),
        parameter_count=sum(policy.parameter.numel() for policy in policies),
        weight_decay=weight_decay,
    )
    return optimizer, report


def update_optimizer_stage(
    optimizer: torch.optim.Optimizer,
    model: GraphSharedDualHeadModel,
    stage: JointStage,
) -> StageApplicationReport:
    """Move an existing optimizer to the next stage without discarding state."""

    report = apply_stage(model, stage)
    for group in optimizer.param_groups:
        role = group.get("role")
        if role not in _ROLES:
            raise ValueError(f"optimizer group has unknown semantic role: {role!r}")
        group["lr"] = float(stage.learning_rates[role])
    return report
