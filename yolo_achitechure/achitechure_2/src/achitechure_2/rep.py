"""Numerical deployment-fusion check for conditional R1."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from ultralytics.nn.modules.conv import RepConv


@dataclass(frozen=True)
class RepFuseReport:
    max_abs_diff: float
    tolerance: float
    passed: bool


def assert_rep_fuse(model: nn.Module, sample: torch.Tensor, tolerance: float = 1e-4) -> RepFuseReport:
    """Check every R1 RepConv branch without conflating unrelated global fusion."""

    evaluated = copy.deepcopy(model).eval()
    rep_convs = tuple(
        (name, module) for name, module in evaluated.named_modules() if isinstance(module, RepConv)
    )
    if not rep_convs:
        raise TypeError("R1 model contains no RepConv modules")
    captured: dict[str, torch.Tensor] = {}
    hooks = [
        module.register_forward_pre_hook(
            lambda _module, inputs, path=name: captured.setdefault(path, inputs[0].detach().clone())
        )
        for name, module in rep_convs
    ]
    with torch.no_grad():
        evaluated(sample)
    for hook in hooks:
        hook.remove()

    differences: list[float] = []
    with torch.no_grad():
        for name, module in rep_convs:
            local = copy.deepcopy(module).eval()
            value = captured[name]
            reference = local(value)
            local.fuse_convs()
            local.forward = local.forward_fuse
            actual = local(value)
            differences.append(float((reference.float() - actual.float()).abs().max()))
    difference = max(differences)
    report = RepFuseReport(difference, tolerance, difference <= tolerance)
    if not report.passed:
        raise AssertionError(f"R1 fuse max_abs_diff {difference:.8g} exceeds {tolerance}")
    return report
