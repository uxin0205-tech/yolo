"""Ultralytics 8.4.90 dataset adapters for task-separated joint training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from .contracts import Task
from .models import SharedDualHeadModel

Mode = Literal["train", "val"]


@dataclass(frozen=True)
class TaskLoader:
    task: Task
    data: dict[str, Any]
    dataset: Any
    loader: Any
    device: torch.device

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Match the official Detect/Pose trainer tensor preprocessing."""

        processed: dict[str, Any] = {}
        for key, value in batch.items():
            processed[key] = (
                value.to(self.device, non_blocking=self.device.type == "cuda")
                if isinstance(value, torch.Tensor)
                else value
            )
        image = processed.get("img")
        if not isinstance(image, torch.Tensor):
            raise TypeError("Ultralytics batch contains no image tensor")
        processed["img"] = image.float() / 255
        return processed


@dataclass(frozen=True)
class JointLoaders:
    detect: TaskLoader
    pose: TaskLoader


class CyclingLoader:
    """Cycle a finite epoch loader while exposing how often it wrapped."""

    def __init__(self, task_loader: TaskLoader) -> None:
        self.task_loader = task_loader
        self._iterator = iter(task_loader.loader)
        self.wraps = 0

    def next(self) -> dict[str, Any]:
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.wraps += 1
            self._iterator = iter(self.task_loader.loader)
            batch = next(self._iterator)
        return self.task_loader.preprocess(batch)


def _validate_task_schema(model: SharedDualHeadModel, task: Task, data: dict[str, Any]) -> None:
    head = model.head_for(task)
    dataset_nc = int(data["nc"])
    if dataset_nc != int(head.nc):
        raise ValueError(f"{task.value} dataset nc={dataset_nc}, but head nc={int(head.nc)}")
    expected_names = model.detect_names if task is Task.DETECT else model.pose_names
    if dict(data["names"]) != expected_names:
        raise ValueError(f"{task.value} class names {data['names']} do not match head {expected_names}")
    if task is Task.POSE:
        actual_shape = tuple(int(value) for value in data.get("kpt_shape", ()))
        expected_shape = tuple(int(value) for value in model.pose_head.kpt_shape)
        if actual_shape != expected_shape:
            raise ValueError(f"Pose kpt_shape={actual_shape}, expected {expected_shape}")


def build_task_loader(
    model: SharedDualHeadModel,
    *,
    task: Task | str,
    data_yaml: str | Path,
    device: torch.device,
    batch_size: int,
    imgsz: int,
    workers: int = 0,
    mode: Mode = "train",
    fraction: float = 1.0,
    seed: int = 0,
) -> TaskLoader:
    """Build one loader and fail before training if its schema targets the wrong head."""

    selected = Task(task)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32")
    if workers < 0:
        raise ValueError("workers cannot be negative")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    yaml_path = Path(data_yaml).expanduser().resolve()
    data = check_det_dataset(str(yaml_path), autodownload=False)
    _validate_task_schema(model, selected, data)
    args = get_cfg(
        DEFAULT_CFG,
        overrides={
            "task": selected.value,
            "data": str(yaml_path),
            "batch": batch_size,
            "imgsz": imgsz,
            "workers": workers,
            "fraction": fraction,
            "seed": seed,
            "cache": False,
            "rect": mode == "val",
        },
    )
    head = model.head_for(selected)
    stride = max(int(head.stride.max().item()), 32)
    split = "train" if mode == "train" else "val"
    dataset = build_yolo_dataset(
        args,
        data[split],
        batch_size,
        data,
        mode=mode,
        rect=mode == "val",
        stride=stride,
        fraction=fraction if mode == "train" else 1.0,
    )
    loader = build_dataloader(
        dataset,
        batch=batch_size,
        workers=workers,
        shuffle=mode == "train",
        rank=-1,
        drop_last=False,
    )
    return TaskLoader(task=selected, data=data, dataset=dataset, loader=loader, device=device)


def build_joint_train_loaders(
    model: SharedDualHeadModel,
    *,
    detect_yaml: str | Path,
    pose_yaml: str | Path,
    device: torch.device,
    batch_size: int,
    imgsz: int,
    workers: int = 0,
    fraction: float = 1.0,
    seed: int = 0,
) -> JointLoaders:
    common = {
        "model": model,
        "device": device,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "workers": workers,
        "fraction": fraction,
        "seed": seed,
    }
    detect = build_task_loader(task=Task.DETECT, data_yaml=detect_yaml, **common)
    pose = build_task_loader(task=Task.POSE, data_yaml=pose_yaml, **common)
    return JointLoaders(detect=detect, pose=pose)
