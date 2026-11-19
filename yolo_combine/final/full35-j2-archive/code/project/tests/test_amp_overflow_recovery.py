from __future__ import annotations

from collections.abc import Mapping

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


class _FiniteRouter:
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
        per_image = (self.model.shared + head).square()
        return TaskLossResult(
            task=selected,
            raw_total=per_image * batch_size,
            components=per_image.detach().reshape(1),
            actual_batch_size=batch_size,
        )

    def advance_epoch(self) -> None:
        return None


class _OneOverflowScaler:
    """CPU fake matching GradScaler skip/backoff semantics for one attempt."""

    def __init__(self) -> None:
        self.current_scale = 2.0
        self.overflow = True
        self.skipped_steps = 0

    def is_enabled(self) -> bool:
        return True

    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs * (float("inf") if self.overflow else self.current_scale)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        if self.overflow:
            return
        for parameter in _MODEL.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(self.current_scale)

    def step(self, optimizer: torch.optim.Optimizer):
        if self.overflow:
            self.skipped_steps += 1
            return None
        return optimizer.step()

    def update(self) -> None:
        if self.overflow:
            self.current_scale *= 0.5
            self.overflow = False

    def get_scale(self) -> float:
        return self.current_scale


def _batch(size: int) -> dict[str, torch.Tensor]:
    return {"img": torch.zeros(size, 1)}


_MODEL = _TinyModel()


def test_amp_overflow_retries_same_macro_before_step() -> None:
    model = _MODEL
    model.zero_grad(set_to_none=True)
    scaler = _OneOverflowScaler()
    engine = MacroStepEngine(
        model=model,
        losses=_FiniteRouter(model),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        gradient_groups={
            "shared": (model.shared,),
            "detect_head": (model.detect_head,),
            "pose_head": (model.pose_head,),
        },
        scaler=scaler,
        max_amp_retries=2,
    )

    report = engine.run(
        detect_batches=(_batch(2),),
        pose_batches=(_batch(1),),
    )

    assert scaler.skipped_steps == 1
    assert report.amp_overflow_retries == 1
    assert report.amp_scale == 1.0
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )
