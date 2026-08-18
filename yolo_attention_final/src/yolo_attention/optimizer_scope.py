"""Optimizer filtering independent of Ultralytics grouping policy."""

from __future__ import annotations

from typing import Any

from torch import nn

from .scopes import learning_rate_group


def restrict_optimizer_to_trainable(optimizer: Any, model: nn.Module) -> None:
    """Remove every frozen parameter and prove exact identity-set equality."""

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    for group in optimizer.param_groups:
        group["params"] = [parameter for parameter in group["params"] if parameter.requires_grad]
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        missing = len(expected - actual)
        unexpected = len(actual - expected)
        raise RuntimeError(f"optimizer scope mismatch: missing={missing}, unexpected={unexpected}")


def apply_layerwise_learning_rates(
    optimizer: Any, model: nn.Module, learning_rates: dict[str, float] | None
) -> dict[str, int]:
    """Split Ultralytics decay groups by recovery layer while preserving all group options."""

    if not learning_rates:
        return {}
    names = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    split_groups: list[dict[str, Any]] = []
    counts = {name: 0 for name in learning_rates}
    for original in optimizer.param_groups:
        buckets: dict[str, list[nn.Parameter]] = {}
        for parameter in original["params"]:
            name = names.get(id(parameter))
            if name is None:
                raise RuntimeError("optimizer contains a parameter outside the trainable name map")
            group_name = learning_rate_group(name)
            if group_name not in learning_rates:
                raise RuntimeError(f"missing learning rate for trainable group {group_name!r}: {name}")
            buckets.setdefault(group_name, []).append(parameter)
            counts[group_name] += parameter.numel()
        for group_name, parameters in buckets.items():
            group = dict(original)
            group["params"] = parameters
            group["lr"] = float(learning_rates[group_name])
            if "initial_lr" in group:
                group["initial_lr"] = float(learning_rates[group_name])
            group["layer_group"] = group_name
            split_groups.append(group)
    if not split_groups or any(count == 0 for count in counts.values()):
        raise RuntimeError(f"empty layerwise optimizer group: {counts}")
    optimizer.param_groups[:] = split_groups
    return counts
