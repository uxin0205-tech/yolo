"""最終訓練使用的 fail-closed parameter 與 buffer scopes。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from torch import nn

from .attention import HardwareFriendlyAttention
from .integration import YOLO26M_ATTENTION_PATHS, set_progressive_epoch
from .run_config import TRAINABLE_SCOPES


@dataclass(frozen=True)
class TrainableSummary:
    scope: str
    trainable_parameters: int
    total_parameters: int
    trainable_names: tuple[str, ...]


def _allowed_parameter(name: str, scope: str) -> bool:
    if name.startswith("bias.table_"):
        return True
    if scope == "bias_only":
        return False
    if scope == "qk_recovery":
        return name.startswith(("qkv.q.", "qkv.k."))
    if scope == "attention_refinement":
        return not name.startswith(("score.", "normalize."))
    return False


def _is_fixed_attention_parameter(name: str) -> bool:
    return any(name.startswith((f"{site}.score.", f"{site}.normalize.")) for site in YOLO26M_ATTENTION_PATHS)


def learning_rate_group(name: str) -> str:
    """把一個 YOLO26m parameter 分到不重疊的 recovery LR group。"""

    if any(name.startswith(f"{site}.") for site in YOLO26M_ATTENTION_PATHS):
        return "attention"
    if name.startswith(("model.10.", "model.22.m.0.1.")):
        return "adjacent_block"
    match = re.match(r"^model\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"parameter is outside the YOLO26m model graph: {name}")
    index = int(match.group(1))
    if 11 <= index <= 23:
        return "neck_detect"
    if 0 <= index <= 10:
        return "backbone"
    raise ValueError(f"unexpected YOLO26m layer index in parameter: {name}")


def _allowed_recovery_parameter(name: str, scope: str) -> bool:
    if _is_fixed_attention_parameter(name):
        return False
    group = learning_rate_group(name)
    allowed = {
        "block_recovery": {"attention", "adjacent_block"},
        "neck_recovery": {"attention", "adjacent_block", "neck_detect"},
        "backbone_last_recovery": {"attention", "adjacent_block", "neck_detect"},
        "full_model_recovery": {"attention", "adjacent_block", "neck_detect", "backbone"},
    }[scope]
    if scope == "backbone_last_recovery" and group == "backbone":
        match = re.match(r"^model\.(\d+)\.", name)
        return match is not None and 7 <= int(match.group(1)) <= 10
    return group in allowed


def apply_trainable_scope(model: nn.Module, scope: str) -> TrainableSummary:
    """只啟用 final-training scope 指定的 parameters。"""

    scope = scope.lower().replace("-", "_")
    if scope not in TRAINABLE_SCOPES:
        raise ValueError(f"unknown trainable scope: {scope}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    sites = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, HardwareFriendlyAttention)
    }
    if set(sites) != set(YOLO26M_ATTENTION_PATHS):
        raise ValueError(f"expected exactly {list(YOLO26M_ATTENTION_PATHS)}, found {sorted(sites)}")
    recovery_scope = scope.endswith("_recovery") and scope != "qk_recovery"
    if recovery_scope:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(_allowed_recovery_parameter(name, scope))
    else:
        for module in sites.values():
            for name, parameter in module.named_parameters():
                parameter.requires_grad_(_allowed_parameter(name, scope))
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if any(name.endswith("score.gamma") or ".normalize." in name for name in names):
        raise AssertionError("dead score/PWL state entered optimizer scope")
    return TrainableSummary(
        scope=scope,
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_names=names,
    )


def enforce_frozen_batchnorm(model: nn.Module, scope: str) -> None:
    """把允許 Attention paths 之外的所有 BN 設為 eval mode。"""

    if scope in {"block_recovery", "neck_recovery", "backbone_last_recovery", "full_model_recovery"}:
        # Recovery 可訓練 BN affine parameters，但所有 running statistics 維持 immutable。
        allowed = ()
    elif scope == "bias_only":
        allowed: tuple[str, ...] = ()
    elif scope == "qk_recovery":
        allowed = tuple(f"{site}.qkv.{part}." for site in YOLO26M_ATTENTION_PATHS for part in ("q", "k"))
    elif scope == "attention_refinement":
        allowed = tuple(f"{site}." for site in YOLO26M_ATTENTION_PATHS)
    else:
        raise ValueError(f"unknown trainable scope: {scope}")
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not name.startswith(allowed):
            module.eval()


def sync_progressive_epoch(live_model: nn.Module, ema_model: nn.Module, epoch: int) -> None:
    """明確複製 integer epoch state；EMA arithmetic 只會更新 floating state。"""

    set_progressive_epoch(live_model, epoch)
    set_progressive_epoch(ema_model, epoch)
