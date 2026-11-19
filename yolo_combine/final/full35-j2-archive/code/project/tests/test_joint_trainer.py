from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from yolo_combine.contracts import Task
from yolo_combine.experiment_log import ExperimentLogger
from yolo_combine.joint_trainer import (
    EpochTrainingReport,
    JointEpochRunner,
    PoseEpochRunner,
    StageWarmupCosineScheduler,
)


def _optimizer() -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.AdamW(
        [
            {
                "params": [parameter],
                "param_names": ("value",),
                "group_name": "backbone.decay",
                "role": "backbone",
                "lr": 1e-3,
            }
        ]
    )


def test_stage_scheduler_warms_then_cosines_and_round_trips() -> None:
    optimizer = _optimizer()
    scheduler = StageWarmupCosineScheduler(
        optimizer,
        stage="j1",
        epochs=4,
        steps_per_epoch=2,
        warmup_epochs=1,
        warmup_start_factor=0.1,
        final_lr_factor=0.5,
    )

    observed = []
    for _ in range(5):
        observed.append(scheduler.prepare_step()["backbone.decay"])
        scheduler.advance()

    assert observed[0] == pytest.approx(0.00055)
    assert observed[1] == pytest.approx(0.001)
    assert observed[2] == pytest.approx(0.001)
    assert observed[-1] < observed[2]
    state = scheduler.state_dict()

    replacement = StageWarmupCosineScheduler(
        optimizer,
        stage="j1",
        epochs=4,
        steps_per_epoch=2,
        warmup_epochs=1,
        warmup_start_factor=0.1,
        final_lr_factor=0.5,
    )
    replacement.load_state_dict(state)
    assert replacement.state_dict() == state
    assert replacement.prepare_step() == scheduler.prepare_step()


class _FakeEngine:
    def __init__(self) -> None:
        self.calls = 0
        self.epochs = 0
        self.epoch_tasks = []

    def run(self, *, detect_batches, pose_batches, record_gradient_statistics):
        self.calls += 1
        detect_images = sum(int(batch["img"].shape[0]) for batch in detect_batches)
        pose_images = sum(int(batch["img"].shape[0]) for batch in pose_batches)
        statistics = (
            type("Stats", (), {"detect_norm": 1.0, "pose_norm": 2.0, "cosine_similarity": -0.25})()
            if record_gradient_statistics
            else None
        )
        return type(
            "Report",
            (),
            {
                "detect_mean_loss": 3.0,
                "pose_mean_loss": 4.0,
                "joint_mean_loss": 3.5 if detect_batches else 4.0,
                "loss_for_backward": 224.0,
                "detect_components": (1.0, 2.0),
                "pose_components": (1.0, 2.0, 3.0),
                "detect_batch_sizes": tuple(int(batch["img"].shape[0]) for batch in detect_batches),
                "pose_batch_sizes": tuple(int(batch["img"].shape[0]) for batch in pose_batches),
                "detect_images": detect_images,
                "pose_images": pose_images,
                "gradient_presence": {
                    "shared": bool(detect_batches or pose_batches),
                    "detect_head": bool(detect_batches),
                    "pose_head": bool(pose_batches),
                },
                "gradient_statistics": statistics,
                "clipped_gradient_norm": 5.0,
                "amp_scale": 1024.0,
                "amp_overflow_retries": 1,
            },
        )()

    def advance_epoch(self, tasks=None) -> None:
        self.epochs += 1
        self.epoch_tasks.append(None if tasks is None else tuple(tasks))


class _Dataset(list):
    pass


class _Loader:
    def __init__(self, sizes: tuple[int, ...]) -> None:
        self.batches = [
            {"img": torch.zeros(size, 3, 8, 8)} for size in sizes
        ]
        self.dataset = _Dataset(range(sum(sizes)))

    def __iter__(self):
        return iter(self.batches)


class _FakeScheduler:
    def __init__(self) -> None:
        self.prepared = 0
        self.advanced = 0

    def prepare_step(self) -> Mapping[str, float]:
        self.prepared += 1
        return {"head": 1e-3}

    def advance(self) -> None:
        self.advanced += 1


def test_epoch_runner_uses_two_to_one_tail_logs_and_advances_loss_once(
    tmp_path: Path,
) -> None:
    engine = _FakeEngine()
    scheduler = _FakeScheduler()
    logger = ExperimentLogger(tmp_path / "logs", tensorboard="off")
    mode_calls: list[int] = []
    guard_calls: list[int] = []
    runner = JointEpochRunner(
        engine=engine,
        detect_loader=_Loader((4, 4, 1)),
        pose_loader=_Loader((2,)),
        scheduler=scheduler,
        logger=logger,
        apply_training_mode=lambda: mode_calls.append(1),
        assert_hardware_contract=lambda: guard_calls.append(1),
        detect_batches_per_macro=2,
        gradient_statistics_interval=2,
    )

    report = runner.run_epoch(epoch=0, global_macro_step=0, stage="j1")
    logger.close()

    assert isinstance(report, EpochTrainingReport)
    assert report.macros == 2
    assert report.detect_batches == 3
    assert report.pose_batches == 2
    assert report.detect_images == 9
    assert report.pose_images == 4
    assert report.pose_dataset_passes == pytest.approx(2.0)
    assert report.next_global_macro_step == 2
    assert engine.calls == 2
    assert engine.epochs == 1
    assert scheduler.prepared == scheduler.advanced == 2
    assert mode_calls == [1]
    assert guard_calls == [1]
    lines = (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
    assert len(lines) == 3  # two macro records plus one epoch record


def test_pose_epoch_runner_uses_only_pose_batches_and_updates_pose_criterion(
    tmp_path: Path,
) -> None:
    engine = _FakeEngine()
    scheduler = _FakeScheduler()
    logger = ExperimentLogger(tmp_path / "pose-logs", tensorboard="off")
    mode_calls: list[int] = []
    guard_calls: list[int] = []
    runner = PoseEpochRunner(
        engine=engine,
        pose_loader=_Loader((2, 3)),
        scheduler=scheduler,
        logger=logger,
        apply_training_mode=lambda: mode_calls.append(1),
        assert_hardware_contract=lambda: guard_calls.append(1),
    )

    report = runner.run_epoch(epoch=0, global_macro_step=7, stage="j0")
    logger.close()

    assert report.macros == 2
    assert report.detect_batches == 0
    assert report.pose_batches == 2
    assert report.detect_images == 0
    assert report.pose_images == 5
    assert report.detect_dataset_passes == 0.0
    assert report.pose_dataset_passes == pytest.approx(1.0)
    assert report.detect_mean_loss == 0.0
    assert report.pose_mean_loss == pytest.approx(4.0)
    assert report.joint_mean_loss == pytest.approx(4.0)
    assert report.next_global_macro_step == 9
    assert engine.calls == 2
    assert engine.epoch_tasks == [(Task.POSE,)]
    assert scheduler.prepared == scheduler.advanced == 2
    assert mode_calls == [1]
    assert guard_calls == [1]
    lines = (tmp_path / "pose-logs" / "events.jsonl").read_text().splitlines()
    assert len(lines) == 3
