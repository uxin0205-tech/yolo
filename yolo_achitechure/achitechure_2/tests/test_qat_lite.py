from __future__ import annotations

import pytest
import torch
from torch import nn

from achitechure_2.qat_lite import (
    QAT_LITE_OBSERVER_UPDATE_STEPS,
    QAT_LITE_OPTIMIZER_STEPS,
    QAT_LITE_VALIDATION_INTERVAL,
    qat_lite_gap_report,
    require_qat_lite_stage,
)
from achitechure_2.quantization import (
    configure_qat_lite_step,
    prepare_w8a8_simulation,
)


class QuantToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


def test_qat_lite_budget_is_small_fixed_and_observers_freeze() -> None:
    assert QAT_LITE_OPTIMIZER_STEPS == 200
    assert QAT_LITE_OBSERVER_UPDATE_STEPS == 50
    assert QAT_LITE_VALIDATION_INTERVAL == 50
    prepared, _ = prepare_w8a8_simulation(QuantToy())
    assert configure_qat_lite_step(prepared, 50, observer_update_steps=50)
    assert not configure_qat_lite_step(prepared, 51, observer_update_steps=50)
    assert int(prepared.conv.activation_fake_quant.observer_enabled[0]) == 0
    assert int(prepared.conv.activation_fake_quant.fake_quant_enabled[0]) == 1


def test_qat_lite_is_fail_closed_for_candidate_and_gpu_authorization() -> None:
    with pytest.raises(PermissionError, match="GPU"):
        require_qat_lite_stage("C0")
    with pytest.raises(PermissionError, match="使用者"):
        require_qat_lite_stage("C1", gpu_authorized=True)
    assert (
        require_qat_lite_stage(
            "C1",
            user_approved=("C1",),
            gpu_authorized=True,
        )
        == "allowed"
    )


def test_qat_lite_gap_reports_recovery_without_automatic_acceptance() -> None:
    report = qat_lite_gap_report(
        q0={"map50_95": 0.50, "macro_f1": 0.60},
        q1={"map50_95": 0.47, "macro_f1": 0.57},
        q2_lite={"map50_95": 0.49, "macro_f1": 0.59},
    )
    assert report.ptq_drop["map50_95"] == pytest.approx(0.03)
    assert report.qat_lite_drop["map50_95"] == pytest.approx(0.01)
    assert report.qat_lite_recovery["macro_f1"] == pytest.approx(0.02)
    assert report.selection_status == "screening_only_pending_user_decision"
    assert report.accepted is None
