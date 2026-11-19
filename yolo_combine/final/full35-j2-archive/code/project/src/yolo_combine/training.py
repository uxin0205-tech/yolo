"""Task-aware loss routing and memory-safe joint optimizer steps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.utils.loss import E2ELoss, PoseLoss26

from .contracts import Task
from .models import SharedDualHeadModel

Batch = Mapping[str, torch.Tensor]


class _CriterionModelView:
    """Present only the attributes Ultralytics loss constructors require."""

    def __init__(self, owner: SharedDualHeadModel, task: Task, args: Any) -> None:
        self._owner = owner
        self.model = (owner.head_for(task),)
        self.args = args
        self.class_weights = None

    def parameters(self) -> Iterable[nn.Parameter]:
        return self._owner.parameters()


@dataclass(frozen=True)
class TaskLoss:
    total: torch.Tensor
    items: torch.Tensor


@dataclass(frozen=True)
class OptimizerStepReport:
    total: float
    detect: float
    pose: float
    detect_items: tuple[float, ...]
    pose_items: tuple[float, ...]
    detect_microbatches: int
    pose_microbatches: int


class TaskLossRouter:
    """Hide two Ultralytics criteria behind one missing-label-safe interface."""

    def __init__(
        self,
        model: SharedDualHeadModel,
        *,
        epochs: int,
        imgsz: int = 640,
        detect_overrides: Mapping[str, Any] | None = None,
        pose_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        self.model = model
        detect_args = {"task": "detect", "epochs": epochs, "imgsz": imgsz, **dict(detect_overrides or {})}
        pose_args = {"task": "pose", "epochs": epochs, "imgsz": imgsz, **dict(pose_overrides or {})}
        self.detect_args = get_cfg(DEFAULT_CFG, overrides=detect_args)
        self.pose_args = get_cfg(DEFAULT_CFG, overrides=pose_args)
        self.device = next(model.parameters()).device
        self.criteria = {
            Task.DETECT: E2ELoss(_CriterionModelView(model, Task.DETECT, self.detect_args)),
            Task.POSE: E2ELoss(_CriterionModelView(model, Task.POSE, self.pose_args), PoseLoss26),
        }

    def loss_for(self, task: Task | str, batch: Batch) -> TaskLoss:
        """Compute exactly one task loss; no other head or annotation is consulted."""

        selected = Task(task)
        if "img" not in batch:
            raise KeyError("batch is missing img")
        current_device = next(self.model.parameters()).device
        if current_device != self.device:
            raise RuntimeError(
                "TaskLossRouter was built before the model reached its final device; "
                "move the model first, then rebuild the router"
            )
        if batch["img"].device != self.device:
            raise RuntimeError(
                f"{selected.value} batch is on {batch['img'].device}, expected {self.device}"
            )
        predictions = self.model(batch["img"], tasks=selected)[selected.value]
        loss_items, detached_items = self.criteria[selected](predictions, dict(batch))
        total = loss_items.sum()
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite {selected.value} loss: {loss_items.detach().cpu().tolist()}")
        return TaskLoss(total=total, items=detached_items)

    def update_schedule(self) -> None:
        """Advance both YOLO26 one-to-many/one-to-one schedules once per optimizer step."""

        for criterion in self.criteria.values():
            criterion.update()

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        detect_batches: Sequence[Batch],
        pose_batches: Sequence[Batch],
        task_weights: Mapping[Task | str, float] | None = None,
    ) -> OptimizerStepReport:
        """Run one optimizer update; each task is averaged across its microbatches first."""

        if not detect_batches or not pose_batches:
            raise ValueError("a joint optimizer step requires at least one Detect and one Pose microbatch")
        weights = {Task.DETECT: 1.0, Task.POSE: 1.0}
        for task, value in (task_weights or {}).items():
            normalized = Task(task)
            if value < 0:
                raise ValueError("task weights must be non-negative")
            weights[normalized] = float(value)
        optimizer.zero_grad(set_to_none=True)
        totals: dict[Task, list[torch.Tensor]] = {Task.DETECT: [], Task.POSE: []}
        items: dict[Task, list[torch.Tensor]] = {Task.DETECT: [], Task.POSE: []}
        batches_by_task = {Task.DETECT: detect_batches, Task.POSE: pose_batches}
        try:
            for task in (Task.DETECT, Task.POSE):
                scale = weights[task] / len(batches_by_task[task])
                for batch in batches_by_task[task]:
                    result = self.loss_for(task, batch)
                    totals[task].append(result.total.detach())
                    items[task].append(result.items.detach())
                    (result.total * scale).backward()
            optimizer.step()
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            raise
        self.update_schedule()
        means = {task: torch.stack(values).mean() for task, values in totals.items()}
        item_means = {task: torch.stack(values).mean(0) for task, values in items.items()}
        weighted_total = weights[Task.DETECT] * means[Task.DETECT] + weights[Task.POSE] * means[Task.POSE]
        return OptimizerStepReport(
            total=float(weighted_total.cpu()),
            detect=float(means[Task.DETECT].cpu()),
            pose=float(means[Task.POSE].cpu()),
            detect_items=tuple(float(value) for value in item_means[Task.DETECT].cpu()),
            pose_items=tuple(float(value) for value in item_means[Task.POSE].cpu()),
            detect_microbatches=len(detect_batches),
            pose_microbatches=len(pose_batches),
        )
