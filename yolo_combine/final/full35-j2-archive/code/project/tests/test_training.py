from __future__ import annotations

from types import MethodType

import pytest
import torch
from torch import nn

from yolo_combine.contracts import Task
from yolo_combine.source import SourceBundle
from yolo_combine.training import TaskLoss, TaskLossRouter


class ScheduleCounter:
    def __init__(self) -> None:
        self.updates = 0

    def update(self) -> None:
        self.updates += 1


def test_two_to_one_detect_losses_are_averaged_before_joint_backward():
    parameter = nn.Parameter(torch.tensor(1.0))
    router = object.__new__(TaskLossRouter)
    router.criteria = {
        Task.DETECT: ScheduleCounter(),
        Task.POSE: ScheduleCounter(),
    }

    def fake_loss_for(self, task, batch):
        coefficient = batch["coefficient"]
        value = parameter * coefficient
        return TaskLoss(total=value, items=coefficient[None])

    router.loss_for = MethodType(fake_loss_for, router)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    report = router.optimizer_step(
        optimizer,
        detect_batches=[
            {"coefficient": torch.tensor(2.0)},
            {"coefficient": torch.tensor(4.0)},
        ],
        pose_batches=[{"coefficient": torch.tensor(10.0)}],
    )

    assert parameter.item() == pytest.approx(-0.3)
    assert report.total == pytest.approx(13.0)
    assert report.detect == pytest.approx(3.0)
    assert report.pose == pytest.approx(10.0)
    assert report.detect_microbatches == 2
    assert report.pose_microbatches == 1
    assert router.criteria[Task.DETECT].updates == 1
    assert router.criteria[Task.POSE].updates == 1


def _detect_batch(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, 128, 128, device=device),
        "batch_idx": torch.zeros(1, dtype=torch.long, device=device),
        "cls": torch.zeros(1, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]], device=device),
    }


def _pose_batch(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, 128, 128, device=device),
        "batch_idx": torch.zeros(1, dtype=torch.long, device=device),
        "cls": torch.ones(1, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.2]], device=device),
        "keypoints": torch.tensor(
            [[[0.4, 0.5, 2.0], [0.6, 0.5, 2.0]]],
            device=device,
        ),
    }


@pytest.mark.integration
@pytest.mark.gpu
def test_real_full35_joint_optimizer_step(
    source_bundle: SourceBundle,
    cuda_device: torch.device,
):
    model = source_bundle.build_shared().to(cuda_device).train()
    router = TaskLossRouter(model, epochs=1, imgsz=128)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    report = router.optimizer_step(
        optimizer,
        detect_batches=[_detect_batch(cuda_device), _detect_batch(cuda_device)],
        pose_batches=[_pose_batch(cuda_device)],
    )

    assert report.total > 0
    assert report.detect > 0
    assert report.pose > 0
    assert len(report.detect_items) == 3
    assert len(report.pose_items) == 6
    assert report.detect_microbatches == 2
    assert report.pose_microbatches == 1
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.trunk.parameters()
    )
