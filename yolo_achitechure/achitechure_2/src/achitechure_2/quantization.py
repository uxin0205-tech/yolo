"""W8A8 Conv fake-quant simulation; this is not an INT8 deployment backend."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)

SIMULATION_ONLY = True
EXCLUDED_ROOT_CLASSES = frozenset({"HardwareFriendlyAttention"})
CUSTOM_NAME_MARKERS = ("Binary", "PWL", "Piecewise", "Softmax")


def _activation_fake_quant() -> FakeQuantize:
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )


def _weight_fake_quant() -> FakeQuantize:
    return FakeQuantize(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        reduce_range=False,
    )


class Conv2dSimulationAdapter(nn.Module):
    """Apply PyTorch observer/fake-quant primitives around one Conv2d."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv
        self.activation_fake_quant = _activation_fake_quant()
        self.weight_fake_quant = _weight_fake_quant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_fake_quant(x)
        weight = self.weight_fake_quant(self.conv.weight)
        return self.conv._conv_forward(x, weight, self.conv.bias)


@dataclass(frozen=True)
class QuantScope:
    simulation_only: bool
    quantized_nodes: tuple[str, ...]
    unquantized_nodes: tuple[str, ...]
    weight_scheme: str = "per_channel_symmetric_int8"
    activation_scheme: str = "per_tensor_affine_int8"


def make_fused_reference(model: nn.Module) -> nn.Module:
    """Create the Q0 eval/fused deployment-reference graph."""

    fused = copy.deepcopy(model).eval()
    fuse = getattr(fused, "fuse", None)
    if not callable(fuse):
        raise TypeError("model does not expose the Ultralytics fuse() method")
    fuse(verbose=False)
    return fused


def prepare_w8a8_simulation(model: nn.Module) -> tuple[nn.Module, QuantScope]:
    """Deep-copy a model and wrap Conv2d nodes outside custom attention roots."""

    prepared = copy.deepcopy(model)
    quantized: list[str] = []
    unquantized: list[str] = []

    def recurse(module: nn.Module, prefix: str) -> None:
        for name, child in tuple(module.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if type(child).__name__ in EXCLUDED_ROOT_CLASSES:
                unquantized.append(path)
                for subpath, submodule in child.named_modules():
                    if subpath and any(marker in type(submodule).__name__ for marker in CUSTOM_NAME_MARKERS):
                        unquantized.append(f"{path}.{subpath}")
                continue
            if isinstance(child, nn.Conv2d):
                setattr(module, name, Conv2dSimulationAdapter(child))
                quantized.append(path)
            else:
                recurse(child, path)

    recurse(prepared, "")
    return prepared, QuantScope(SIMULATION_ONLY, tuple(quantized), tuple(sorted(set(unquantized))))


def calibrate_w8a8(model: nn.Module, batches: Any, *, max_batches: int | None = None) -> int:
    """Collect activation ranges, then enable frozen fake-quant for Q1 evaluation."""

    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    model.eval()
    set_fake_quant(model, enabled=False)
    set_observers(model, enabled=True)
    count = 0
    with torch.inference_mode():
        for value in batches:
            if max_batches is not None and count >= max_batches:
                break
            model(value)
            count += 1
    if count == 0:
        raise ValueError("calibration requires at least one batch")
    set_observers(model, enabled=False)
    set_fake_quant(model, enabled=True)
    return count


def set_observers(model: nn.Module, *, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, FakeQuantize):
            if enabled:
                module.enable_observer()
            else:
                module.disable_observer()


def set_fake_quant(model: nn.Module, *, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, FakeQuantize):
            if enabled:
                module.enable_fake_quant()
            else:
                module.disable_fake_quant()


def configure_qat_epoch(model: nn.Module, epoch: int) -> bool:
    """Update observers for epochs 1-3, then freeze ranges."""

    if epoch < 1:
        raise ValueError("epoch is one-based")
    observers_enabled = epoch <= 3
    set_observers(model, enabled=observers_enabled)
    set_fake_quant(model, enabled=True)
    return observers_enabled


@dataclass(frozen=True)
class RobustnessReport:
    simulation_only: bool
    q0_map50_95: float
    q1_map50_95: float
    q2_map50_95: float
    ptq_gap: float
    qat_gap: float
    qat_recovery: float
    verdict: str


def robustness_report(q0: float, q1: float, q2: float) -> RobustnessReport:
    qat_gap = q0 - q2
    if qat_gap <= 0.005:
        verdict = "robust"
    elif qat_gap <= 0.008:
        verdict = "acceptable_but_sensitive"
    else:
        verdict = "quantization_sensitive"
    return RobustnessReport(True, q0, q1, q2, q0 - q1, qat_gap, q2 - q1, verdict)


def quant_scope_dict(scope: QuantScope) -> dict[str, Any]:
    return asdict(scope)
