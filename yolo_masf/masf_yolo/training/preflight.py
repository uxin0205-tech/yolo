"""Real loss/backward and common-batch preflight gates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from masf_yolo.contracts import PHASE1_VARIANTS

from .resume import NonFiniteLossError


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def run_finite_loss_batch(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> float:
    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    result = model(_to_device(batch, device))
    components = result[0] if isinstance(result, tuple) else result
    if not isinstance(components, Tensor) or components.numel() < 1:
        raise NonFiniteLossError("model did not return loss components")
    if not torch.isfinite(components).all():
        raise NonFiniteLossError(f"non-finite loss components: {components.detach().cpu().tolist()}")
    loss = components.sum()
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
        raise NonFiniteLossError("backward produced no gradients")
    return float(loss.detach().cpu())


def run_optimizer_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    amp: bool = True,
) -> float:
    """Run the formal SGD/momentum allocation path used by batch probing."""
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.937)
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        result = model(_to_device(batch, device))
        components = result[0] if isinstance(result, tuple) else result
        if not isinstance(components, Tensor) or components.numel() < 1:
            raise NonFiniteLossError("model did not return loss components")
        if not torch.isfinite(components).all():
            raise NonFiniteLossError(
                f"non-finite loss components: {components.detach().cpu().tolist()}"
            )
        loss = components.sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().cpu())


def probe_common_batch(
    probe: Callable[[str, int], bool],
    candidates: tuple[int, ...] = (16, 8, 4, 2, 1),
) -> int:
    for batch in candidates:
        results = [probe(variant, batch) for variant in PHASE1_VARIANTS]
        if all(results):
            return batch
    raise RuntimeError("no common batch size works for all Phase 1 variants")
