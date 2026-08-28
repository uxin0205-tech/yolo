"""W8A8 Conv fake-quant simulation; this is not an INT8 deployment backend."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
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


def configure_qat_lite_step(
    model: nn.Module,
    step: int,
    *,
    observer_update_steps: int = 50,
) -> bool:
    """QAT-lite 使用 one-based step；短暫更新 observer 後固定量化範圍。"""

    if step < 1:
        raise ValueError("step is one-based")
    if observer_update_steps < 1:
        raise ValueError("observer_update_steps must be positive")
    observers_enabled = step <= observer_update_steps
    set_observers(model, enabled=observers_enabled)
    set_fake_quant(model, enabled=True)
    return observers_enabled


def require_quantization_stage(
    candidate_id: str,
    stage: str,
    *,
    user_approved: Iterable[str] = (),
    gpu_authorized: bool = False,
) -> str:
    """Fail closed：C0 預設 eligible，其餘候選與 Q2 需要額外授權。"""

    normalized_stage = stage.upper()
    if normalized_stage not in {"Q0", "Q1", "Q2"}:
        raise ValueError(f"未知量化 stage：{stage}")
    approved = set(user_approved)
    if candidate_id != "C0" and candidate_id not in approved:
        raise PermissionError(
            f"{candidate_id} 尚未取得使用者核准；需先檢視適用階段的 Float 結果並明確核准"
        )
    if normalized_stage == "Q2" and not gpu_authorized:
        raise PermissionError("Q2 正式 QAT 需要使用者明確 GPU 長訓練授權")
    return "allowed"


@dataclass(frozen=True)
class QuantizationGapReport:
    simulation_only: bool
    q0: dict[str, float]
    q1: dict[str, float]
    q2: dict[str, float]
    q1_drop: dict[str, float]
    q2_drop: dict[str, float]
    qat_recovery: dict[str, float]
    selection_status: str
    accepted: None


def quantization_gap_report(
    *,
    q0: Mapping[str, float],
    q1: Mapping[str, float],
    q2: Mapping[str, float],
) -> QuantizationGapReport:
    """逐指標呈現 Q0/Q1/Q2 差距，不用固定 drop 自動接受量化。"""

    if not q0 or set(q0) != set(q1) or set(q0) != set(q2):
        raise ValueError("Q0/Q1/Q2 必須提供相同且非空的 metric keys")
    normalized = {
        stage: {name: float(value) for name, value in values.items()}
        for stage, values in (("q0", q0), ("q1", q1), ("q2", q2))
    }
    invalid = {
        f"{stage}.{name}": value
        for stage, values in normalized.items()
        for name, value in values.items()
        if not 0 <= value <= 1
    }
    if invalid:
        raise ValueError(f"量化精度指標必須介於 [0,1]：{invalid}")
    return QuantizationGapReport(
        simulation_only=True,
        q0=normalized["q0"],
        q1=normalized["q1"],
        q2=normalized["q2"],
        q1_drop={
            name: normalized["q0"][name] - normalized["q1"][name]
            for name in normalized["q0"]
        },
        q2_drop={
            name: normalized["q0"][name] - normalized["q2"][name]
            for name in normalized["q0"]
        },
        qat_recovery={
            name: normalized["q2"][name] - normalized["q1"][name]
            for name in normalized["q0"]
        },
        selection_status="pending_user_decision",
        accepted=None,
    )


def quant_scope_dict(scope: QuantScope) -> dict[str, Any]:
    return asdict(scope)
