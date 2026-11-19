"""共用 macro engine；本輪僅改可訓參數與明確分組 LR。"""

import torch
from stage7.ops import (
    MacroEngine,
    ProbeOptimizer,
    fixed_hashes,
    loss_for_recipe,
    macros,
    restore_rng,
    rng_state,
)

from .model import role


def optimizer_for(model, recipe):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = role(name, model.stage8_manifest["layers"])
            if group == "fixed":
                raise ValueError("fixed 參數誤啟用")
            groups.setdefault((group, parameter.ndim >= 2), []).append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": params,
                "lr": recipe["lrs"][name],
                "weight_decay": recipe["weight_decay"] if decay else 0.0,
                "group_name": f"{name}-{int(decay)}",
            }
            for (name, decay), params in groups.items()
        ],
        betas=tuple(recipe["betas"]),
    )


def gradient_groups(model):
    result = {"shared": [], "detect_head": [], "pose_head": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = role(name, model.stage8_manifest["layers"])
            result[group if group in ("detect_head", "pose_head") else "shared"].append(
                parameter
            )
    return result


__all__ = [
    "MacroEngine",
    "ProbeOptimizer",
    "fixed_hashes",
    "gradient_groups",
    "loss_for_recipe",
    "macros",
    "optimizer_for",
    "restore_rng",
    "rng_state",
]
