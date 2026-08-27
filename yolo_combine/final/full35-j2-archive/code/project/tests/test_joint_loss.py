from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import nn

from yolo_combine.contracts import Task
from yolo_combine.joint_loss import (
    JointMacroPlan,
    MacroStepEngine,
    TaskLossResult,
)


def test_two_to_one_macro_uses_per_image_task_means_and_reference_scale() -> None:
    plan = JointMacroPlan.from_batch_sizes(
        detect_batch_sizes=(128, 128),
        pose_batch_sizes=(16,),
        reference_batch_size=64,
        detect_weight=1.0,
        pose_weight=1.0,
    )

    assert plan.detect_images == 256
    assert plan.pose_images == 16
    assert plan.detect_backward_scale == pytest.approx(0.125)
    assert plan.pose_backward_scale == pytest.approx(2.0)
    assert plan.joint_mean(detect_raw_total=512.0, pose_raw_total=160.0) == pytest.approx(6.0)
    assert plan.backward_total(detect_raw_total=512.0, pose_raw_total=160.0) == pytest.approx(384.0)


def test_partial_final_macro_recomputes_scales_from_actual_batch_sizes() -> None:
    plan = JointMacroPlan.from_batch_sizes(
        detect_batch_sizes=(17,),
        pose_batch_sizes=(7,),
        reference_batch_size=64,
        detect_weight=1.0,
        pose_weight=1.0,
    )

    assert plan.detect_backward_scale == pytest.approx(32 / 17)
    assert plan.pose_backward_scale == pytest.approx(32 / 7)
    assert plan.joint_mean(detect_raw_total=34.0, pose_raw_total=70.0) == pytest.approx(6.0)


def test_pose_only_macro_ignores_inactive_detect_weight() -> None:
    plan = JointMacroPlan.from_batch_sizes(
        detect_batch_sizes=(),
        pose_batch_sizes=(16,),
        reference_batch_size=64,
        detect_weight=1.0,
        pose_weight=0.25,
    )

    assert plan.detect_images == 0
    assert plan.pose_images == 16
    assert plan.detect_backward_scale == 0.0
    assert plan.pose_backward_scale == pytest.approx(4.0)
    assert plan.joint_mean(detect_raw_total=0.0, pose_raw_total=256.0) == 16.0
    assert plan.backward_total(detect_raw_total=0.0, pose_raw_total=256.0) == 1024.0


class TinyJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(1.0))
        self.detect_head = nn.Parameter(torch.tensor(2.0))
        self.pose_head = nn.Parameter(torch.tensor(3.0))


class TinyLossRouter:
    def __init__(self, model: TinyJointModel) -> None:
        self.model = model
        self.epoch_updates = 0

    def loss_for(
        self,
        task: Task | str,
        batch: Mapping[str, torch.Tensor],
    ) -> TaskLossResult:
        selected = Task(task)
        batch_size = int(batch["img"].shape[0])
        head = (
            self.model.detect_head
            if selected is Task.DETECT
            else self.model.pose_head
        )
        per_image = (self.model.shared + head).square()
        raw = per_image * batch_size
        return TaskLossResult(
            task=selected,
            raw_total=raw,
            components=per_image.detach().reshape(1),
            actual_batch_size=batch_size,
        )

    def advance_epoch(self) -> None:
        self.epoch_updates += 1


def _batch(size: int) -> dict[str, torch.Tensor]:
    return {"img": torch.zeros(size, 1)}


def test_task_loss_routing_and_macro_gradient_presence() -> None:
    model = TinyJointModel()
    router = TinyLossRouter(model)

    detect_loss = router.loss_for(Task.DETECT, _batch(3))
    detect_loss.mean_total.backward()
    assert model.shared.grad is not None
    assert model.detect_head.grad is not None
    assert model.pose_head.grad is None
    model.zero_grad(set_to_none=True)

    pose_loss = router.loss_for(Task.POSE, _batch(2))
    pose_loss.mean_total.backward()
    assert model.shared.grad is not None
    assert model.detect_head.grad is None
    assert model.pose_head.grad is not None
    model.zero_grad(set_to_none=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    engine = MacroStepEngine(
        model=model,
        losses=router,
        optimizer=optimizer,
        reference_batch_size=64,
        task_weights={Task.DETECT: 1.0, Task.POSE: 1.0},
        gradient_groups={
            "shared": (model.shared,),
            "detect_head": (model.detect_head,),
            "pose_head": (model.pose_head,),
        },
        max_grad_norm=10_000.0,
    )
    report = engine.run(
        detect_batches=(_batch(4), _batch(2)),
        pose_batches=(_batch(3),),
        record_gradient_statistics=True,
    )

    assert report.detect_images == 6
    assert report.pose_images == 3
    assert report.detect_mean_loss == pytest.approx(9.0)
    assert report.pose_mean_loss == pytest.approx(16.0)
    assert report.joint_mean_loss == pytest.approx(12.5)
    assert report.loss_for_backward == pytest.approx(800.0)
    assert report.gradient_presence == {
        "shared": True,
        "detect_head": True,
        "pose_head": True,
    }
    assert report.gradient_statistics is not None
    assert report.gradient_statistics.detect_norm > 0
    assert report.gradient_statistics.pose_norm > 0
    assert -1.0 <= report.gradient_statistics.cosine_similarity <= 1.0
    assert router.epoch_updates == 0
    engine.advance_epoch()
    assert router.epoch_updates == 1


def test_per_image_loss_is_stable_when_batch_size_changes() -> None:
    model = TinyJointModel()
    router = TinyLossRouter(model)

    small = router.loss_for(Task.DETECT, _batch(2))
    large = router.loss_for(Task.DETECT, _batch(11))

    assert small.raw_total.item() / 2 == pytest.approx(large.raw_total.item() / 11)
    assert small.mean_total.item() == pytest.approx(large.mean_total.item())


def test_pose_only_macro_never_routes_gradient_to_detect_head() -> None:
    model = TinyJointModel()
    router = TinyLossRouter(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    engine = MacroStepEngine(
        model=model,
        losses=router,
        optimizer=optimizer,
        reference_batch_size=64,
        task_weights={Task.DETECT: 1.0, Task.POSE: 0.25},
        gradient_groups={
            "shared": (model.shared,),
            "detect_head": (model.detect_head,),
            "pose_head": (model.pose_head,),
        },
        max_grad_norm=10_000.0,
    )

    report = engine.run(
        detect_batches=(),
        pose_batches=(_batch(5),),
        record_gradient_statistics=True,
    )

    assert report.detect_mean_loss == 0.0
    assert report.pose_mean_loss == pytest.approx(16.0)
    assert report.joint_mean_loss == pytest.approx(16.0)
    assert report.loss_for_backward == pytest.approx(1024.0)
    assert report.gradient_presence == {
        "shared": True,
        "detect_head": False,
        "pose_head": True,
    }
    assert report.gradient_statistics is None
