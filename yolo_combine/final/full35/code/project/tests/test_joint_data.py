from __future__ import annotations

from pathlib import Path

import pytest
import torch

from yolo_combine.contracts import Task
from yolo_combine.joint_data import (
    JointEpochScheduler,
    TaskLoaderSettings,
    validate_canonical_pose_source,
)


def _batch(size: int) -> dict[str, torch.Tensor]:
    return {"img": torch.zeros(size, 3, 32, 32)}


def test_task_loader_defaults_keep_detect_and_pose_augmentation_separate() -> None:
    detect = TaskLoaderSettings.for_detect(batch_size=128, workers=4)
    pose = TaskLoaderSettings.for_pose(batch_size=16, workers=8)

    assert detect.task is Task.DETECT
    assert detect.batch_size == 128
    assert detect.augmentation["mosaic"] == 0.0
    assert detect.augmentation["fliplr"] == 0.5
    assert pose.task is Task.POSE
    assert pose.batch_size == 16
    assert pose.augmentation["mosaic"] == 0.0
    assert pose.augmentation["fliplr"] == 0.0
    assert pose.augmentation["mixup"] == 0.0
    assert pose.augmentation["cutmix"] == 0.0
    assert pose.augmentation["copy_paste"] == 0.0


def test_joint_epoch_is_detect_primary_and_cycles_pose_without_dropping_tail() -> None:
    detect_batches = [_batch(size) for size in (4, 4, 4, 4, 1)]
    pose_batches = [_batch(size) for size in (2, 2)]
    scheduler = JointEpochScheduler(
        detect_loader=detect_batches,
        pose_loader=pose_batches,
        detect_batches_per_macro=2,
    )

    macros = list(scheduler)
    report = scheduler.report()

    assert [len(macro.detect_batches) for macro in macros] == [2, 2, 1]
    assert [macro.detect_images for macro in macros] == [8, 8, 1]
    assert [macro.pose_images for macro in macros] == [2, 2, 2]
    assert report.detect_batches == 5
    assert report.pose_batches == 3
    assert report.detect_images == 17
    assert report.pose_images == 6
    assert report.detect_dataset_passes == pytest.approx(1.0)
    assert report.pose_dataset_passes == pytest.approx(1.5)
    assert report.pose_wraps == 1


def test_canonical_pose_guard_accepts_only_registry_or_verified_runtime_view(
    tmp_path: Path,
) -> None:
    registry = Path("/home/uxin/yolo/configs/datasets/bbat5-v1.yaml")
    canonical = Path(
        "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml"
    )
    report = validate_canonical_pose_source(canonical, registry=registry)
    assert report.dataset_id == "bbat5-v1"
    assert report.train_images == 5964
    assert report.val_images == 683
    assert report.source_kind == "canonical"

    historical = Path("/home/uxin/yolo/original/pose/dataset/data.yaml")
    with pytest.raises(ValueError, match="canonical"):
        validate_canonical_pose_source(historical, registry=registry)
