from __future__ import annotations

import torch
from torch import nn

from yolo_quantize.activation_adapter import (
    ActivationOutputAdapter,
    ActivationOutputPolicy,
)
from yolo_quantize.qat_calibration import calibrate_activation_outputs


class _ToyTaskGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor, *, task: str) -> torch.Tensor:
        multiplier = 1.0 if task == "detect" else 2.0
        return self.activation(self.linear(value) * multiplier)


def test_calibration_freezes_all_activation_observers_from_both_tasks() -> None:
    model = _ToyTaskGraph()
    applied = ActivationOutputAdapter().apply(
        model,
        module_paths=("activation",),
        policy=ActivationOutputPolicy(policy_id="toy-a8", bits=8),
        clone_model=False,
    )

    report = calibrate_activation_outputs(
        applied,
        samples={
            "detect": (torch.ones(2, 3),),
            "pose": (torch.full((2, 3), 2.0),),
        },
        device=torch.device("cpu"),
    )

    assert report.samples == {"detect": 1, "pose": 1}
    assert report.quantizers == 1
    assert report.valid_observers == 1
    assert report.invalid_observers == ()
    assert report.observed_minimum < report.observed_maximum
    assert applied.mode == "fake_quant"
