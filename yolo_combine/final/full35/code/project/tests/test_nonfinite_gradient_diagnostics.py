from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import nn

from yolo_combine.contracts import Task
from yolo_combine.joint_loss import MacroStepEngine, TaskLossResult


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(1.0))
        self.detect_head = nn.Parameter(torch.tensor(2.0))
        self.pose_head = nn.Parameter(torch.tensor(3.0))


class _NonFiniteRouter:
    def __init__(self, model: _TinyModel) -> None:
        self.model = model

    def loss_for(
        self,
        task: Task | str,
        batch: Mapping[str, torch.Tensor],
    ) -> TaskLossResult:
        selected = Task(task)
        head = (
            self.model.detect_head
            if selected is Task.DETECT
            else self.model.pose_head
        )
        batch_size = int(batch["img"].shape[0])
        raw = (self.model.shared + head) * float("inf")
        return TaskLossResult(
            task=selected,
            raw_total=raw,
            components=torch.ones(1),
            actual_batch_size=batch_size,
        )

    def advance_epoch(self) -> None:
        return None


def _batch(size: int) -> dict[str, torch.Tensor]:
    return {"img": torch.zeros(size, 1)}


def test_nonfinite_gradient_error_names_parameters_and_amp_scale() -> None:
    model = _TinyModel()
    engine = MacroStepEngine(
        model=model,
        losses=_NonFiniteRouter(model),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        gradient_groups={
            "shared": (model.shared,),
            "detect_head": (model.detect_head,),
            "pose_head": (model.pose_head,),
        },
    )

    with pytest.raises(FloatingPointError) as captured:
        engine.run(
            detect_batches=(_batch(2),),
            pose_batches=(_batch(1),),
        )

    message = str(captured.value)
    assert "scale=1.0" in message
    assert "shared" in message
    assert "detect_head" in message
    assert "pose_head" in message
    assert "non_finite=" in message
