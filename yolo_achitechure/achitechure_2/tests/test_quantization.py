from __future__ import annotations

import pytest
import torch
from torch import nn

from achitechure_2.graph import inspect_graph
from achitechure_2.quantization import (
    Conv2dSimulationAdapter,
    calibrate_w8a8,
    configure_qat_epoch,
    prepare_w8a8_simulation,
    robustness_report,
)


class HardwareFriendlyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.binary = nn.Conv2d(4, 4, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.binary(value)


class QuantToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.p3_masf = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())
        self.attn = HardwareFriendlyAttention()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.attn(self.p3_masf(self.conv(value)))


def test_quantized_yolo_graph_remains_inspectable(toy_parent) -> None:
    prepared, _ = prepare_w8a8_simulation(toy_parent)
    assert inspect_graph(prepared).masf_variant == "full35"


def test_quant_scope_includes_masf_conv_and_excludes_attention() -> None:
    prepared, scope = prepare_w8a8_simulation(QuantToy())
    assert scope.simulation_only
    assert scope.quantized_nodes == ("conv", "p3_masf.0")
    assert scope.unquantized_nodes == ("attn",)
    assert isinstance(prepared.conv, Conv2dSimulationAdapter)
    assert isinstance(prepared.p3_masf[0], Conv2dSimulationAdapter)
    assert isinstance(prepared.attn.binary, nn.Conv2d)


def test_ptq_calibration_freezes_ranges_and_enables_fake_quant() -> None:
    prepared, _ = prepare_w8a8_simulation(QuantToy())
    assert calibrate_w8a8(prepared, [torch.randn(1, 3, 8, 8)], max_batches=1) == 1
    assert int(prepared.conv.activation_fake_quant.observer_enabled[0]) == 0
    assert int(prepared.conv.activation_fake_quant.fake_quant_enabled[0]) == 1


def test_observer_freeze_int8_ranges_and_finite_gradients() -> None:
    prepared, _ = prepare_w8a8_simulation(QuantToy())
    assert configure_qat_epoch(prepared, 3)
    value = torch.randn(2, 3, 8, 8, requires_grad=True)
    loss = prepared(value).square().mean()
    loss.backward()
    assert not configure_qat_epoch(prepared, 4)
    assert torch.isfinite(loss)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert prepared.conv.weight_fake_quant.quant_min == -128
    assert prepared.conv.weight_fake_quant.quant_max == 127


def test_q0_q1_q2_gap_report_is_simulation_only() -> None:
    report = robustness_report(0.5, 0.488, 0.494)
    assert report.simulation_only
    assert report.ptq_gap == pytest.approx(0.012)
    assert report.qat_gap == pytest.approx(0.006)
    assert report.qat_recovery == pytest.approx(0.006)
    assert report.verdict == "acceptable_but_sensitive"
