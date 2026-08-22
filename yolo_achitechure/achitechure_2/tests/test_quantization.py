from __future__ import annotations

import pytest
import torch
from torch import nn

from achitechure_2.quantization import (
    Conv2dSimulationAdapter,
    calibrate_w8a8,
    configure_qat_epoch,
    prepare_w8a8_simulation,
    quantization_gap_report,
    require_quantization_stage,
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


def test_combined_contract_survives_cpu_w8a8_plumbing(combined_parent) -> None:
    prepared, scope = prepare_w8a8_simulation(combined_parent)
    output = prepared(torch.randn(1, 3, 16, 16), tasks="both")

    assert scope.simulation_only
    assert set(output) == {"detect", "pose"}
    assert output["detect"].shape == (1, 80, 16, 16)
    assert output["pose"].shape == (1, 8, 16, 16)
    assert isinstance(combined_parent.stem, nn.Conv2d)
    assert isinstance(prepared.stem, Conv2dSimulationAdapter)


def test_quant_scope_includes_masf_conv_and_excludes_custom_attention() -> None:
    prepared, scope = prepare_w8a8_simulation(QuantToy())
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


def test_cpu_qat_plumbing_has_int8_ranges_and_finite_gradients() -> None:
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


def test_quantization_stage_is_fail_closed_and_q2_needs_gpu_permission() -> None:
    assert require_quantization_stage("C0", "Q1") == "allowed"
    with pytest.raises(PermissionError, match="使用者"):
        require_quantization_stage("C1", "Q1")
    assert (
        require_quantization_stage("C1", "Q1", user_approved=("C1",))
        == "allowed"
    )
    with pytest.raises(PermissionError, match="GPU"):
        require_quantization_stage("C0", "Q2")


def test_q0_q1_q2_gap_is_descriptive_and_never_auto_accepts() -> None:
    report = quantization_gap_report(
        q0={"coco_box_map50_95": 0.50, "bbat5_keypoint_map50_95": 0.48},
        q1={"coco_box_map50_95": 0.49, "bbat5_keypoint_map50_95": 0.46},
        q2={"coco_box_map50_95": 0.497, "bbat5_keypoint_map50_95": 0.475},
    )
    assert report.simulation_only
    assert report.q1_drop["coco_box_map50_95"] == pytest.approx(0.01)
    assert report.q2_drop["bbat5_keypoint_map50_95"] == pytest.approx(0.005)
    assert report.qat_recovery["bbat5_keypoint_map50_95"] == pytest.approx(0.015)
    assert report.selection_status == "pending_user_decision"
    assert report.accepted is None
