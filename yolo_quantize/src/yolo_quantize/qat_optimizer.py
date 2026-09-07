"""Explicit optimizer-group separation for model and quantizer parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

_QUANTIZER_PATH_PARTS = (".weight_quantizer.", ".output_quantizer.")


def is_quantizer_parameter(name: str) -> bool:
    """Return whether a named parameter belongs to a weight/output quantizer."""

    padded = f".{name}."
    return any(part in padded for part in _QUANTIZER_PATH_PARTS)


@dataclass(frozen=True)
class QuantizerOptimizerGroupReport:
    """Manifest of a deterministic pre-scheduler optimizer group split."""

    qparam_lr_ratio: float
    model_parameters: int
    quantizer_parameters: int
    model_groups: tuple[str, ...]
    quantizer_groups: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "qparam_lr_ratio": self.qparam_lr_ratio,
            "model_parameters": self.model_parameters,
            "quantizer_parameters": self.quantizer_parameters,
            "model_groups": list(self.model_groups),
            "quantizer_groups": list(self.quantizer_groups),
        }


def split_quantizer_parameter_groups(
    optimizer: torch.optim.Optimizer,
    *,
    qparam_lr_ratio: float,
) -> QuantizerOptimizerGroupReport:
    """Split qparams before the first step and apply a separate no-decay LR."""

    if not math.isfinite(qparam_lr_ratio) or qparam_lr_ratio <= 0:
        raise ValueError("qparam_lr_ratio must be positive finite")
    if optimizer.state:
        raise RuntimeError("optimizer groups must be split before optimizer state exists")

    model_parameters = 0
    quantizer_parameters = 0
    model_groups: list[str] = []
    quantizer_groups: list[str] = []
    additions: list[dict[str, Any]] = []
    for index, group in enumerate(tuple(optimizer.param_groups)):
        parameters = tuple(group["params"])
        raw_names = group.get("param_names")
        if not isinstance(raw_names, (tuple, list)) or len(raw_names) != len(parameters):
            raise ValueError(
                f"optimizer group {index} has no aligned param_names manifest"
            )
        names = tuple(str(name) for name in raw_names)
        model_pairs = tuple(
            (name, parameter)
            for name, parameter in zip(names, parameters, strict=True)
            if not is_quantizer_parameter(name)
        )
        quantizer_pairs = tuple(
            (name, parameter)
            for name, parameter in zip(names, parameters, strict=True)
            if is_quantizer_parameter(name)
        )
        group_name = str(group.get("group_name", index))
        if model_pairs:
            group["params"] = [parameter for _, parameter in model_pairs]
            group["param_names"] = tuple(name for name, _ in model_pairs)
            model_parameters += len(model_pairs)
            model_groups.append(group_name)
        if quantizer_pairs:
            quantizer_parameters += len(quantizer_pairs)
            qname = f"{group_name}.qparam"
            quantizer_groups.append(qname)
            qgroup = {
                key: value
                for key, value in group.items()
                if key not in {"params", "param_names", "group_name"}
            }
            qgroup.update(
                params=[parameter for _, parameter in quantizer_pairs],
                param_names=tuple(name for name, _ in quantizer_pairs),
                group_name=qname,
                lr=float(group["lr"]) * qparam_lr_ratio,
                weight_decay=0.0,
            )
            if "use_muon" in qgroup:
                qgroup["use_muon"] = False
            if model_pairs:
                additions.append(qgroup)
            else:
                group.clear()
                group.update(qgroup)

    for group in additions:
        optimizer.add_param_group(group)
    if quantizer_parameters == 0:
        raise ValueError("optimizer contains no activation or weight quantizer parameters")
    return QuantizerOptimizerGroupReport(
        qparam_lr_ratio=float(qparam_lr_ratio),
        model_parameters=model_parameters,
        quantizer_parameters=quantizer_parameters,
        model_groups=tuple(model_groups),
        quantizer_groups=tuple(quantizer_groups),
    )


__all__ = (
    "QuantizerOptimizerGroupReport",
    "is_quantizer_parameter",
    "split_quantizer_parameter_groups",
)
