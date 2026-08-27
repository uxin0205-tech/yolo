"""Materialize official task models from a shared F1 state for validation/export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import torch
from ultralytics.nn.modules.head import Detect, Pose26
from ultralytics.nn.tasks import DetectionModel, PoseModel

from .contracts import Task
from .models import SharedDualHeadModel
from .source import CheckpointKind, SourceBundle

TaskModel = TypeVar("TaskModel", DetectionModel, PoseModel)


@dataclass(frozen=True)
class TaskMaterializationReport:
    task: Task
    compatible_tensors: int
    missing_tensors: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_tensors and not self.shape_mismatches


@dataclass(frozen=True)
class MaterializedTaskModels:
    detect: DetectionModel
    pose: PoseModel
    detect_report: TaskMaterializationReport
    pose_report: TaskMaterializationReport


def _shared_name(target_name: str, *, head_index: int, task: Task) -> str | None:
    parts = target_name.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    layer_index = int(parts[1])
    remainder = ".".join(parts[2:])
    if layer_index < head_index:
        return f"trunk.layers.{layer_index}.{remainder}"
    if layer_index == head_index:
        prefix = "detect_head" if task is Task.DETECT else "pose_head"
        return f"{prefix}.{remainder}"
    return None


def materialize_task_model(
    shared: SharedDualHeadModel,
    target: TaskModel,
    task: Task | str,
) -> tuple[TaskModel, TaskMaterializationReport]:
    """Copy one shared trunk and one selected head into an official Ultralytics graph."""

    selected = Task(task)
    head_index = len(target.model) - 1
    target_head = target.model[-1]
    if selected is Task.DETECT:
        if not isinstance(target, DetectionModel):
            raise TypeError("Detect materialization requires DetectionModel")
        if not isinstance(target_head, Detect) or isinstance(target_head, Pose26):
            raise TypeError("Detect target must end in a non-Pose Detect head")
        if int(target_head.nc) != int(shared.detect_head.nc):
            raise ValueError("Detect target class count does not match shared head")
    else:
        if not isinstance(target, PoseModel) or not isinstance(target_head, Pose26):
            raise TypeError("Pose materialization requires PoseModel ending in Pose26")
        if int(target_head.nc) != int(shared.pose_head.nc):
            raise ValueError("Pose target class count does not match shared head")
        if tuple(target_head.kpt_shape) != tuple(shared.pose_head.kpt_shape):
            raise ValueError("Pose target keypoint shape does not match shared head")
    expected_inputs = tuple(int(index) for index in shared.trunk.output_indices)
    if tuple(int(index) for index in target_head.f) != expected_inputs:
        raise ValueError("task target consumes different feature layers")

    shared_state = shared.state_dict()
    target_state = target.state_dict()
    mapping = {
        name: _shared_name(name, head_index=head_index, task=selected)
        for name in target_state
    }
    missing = tuple(
        name
        for name, source_name in mapping.items()
        if source_name is None or source_name not in shared_state
    )
    mismatched = tuple(
        name
        for name, source_name in mapping.items()
        if source_name in shared_state
        and target_state[name].shape != shared_state[source_name].shape
    )
    compatible = tuple(
        name
        for name, source_name in mapping.items()
        if source_name in shared_state
        and target_state[name].shape == shared_state[source_name].shape
    )
    report = TaskMaterializationReport(
        task=selected,
        compatible_tensors=len(compatible),
        missing_tensors=missing,
        shape_mismatches=mismatched,
    )
    if not report.complete or len(compatible) != len(target_state):
        return target, report
    with torch.no_grad():
        for name in compatible:
            target_state[name].copy_(shared_state[mapping[name]])
    target.load_state_dict(target_state, strict=True)
    target.names = dict(shared.detect_names if selected is Task.DETECT else shared.pose_names)
    target.requires_grad_(False)
    target.eval()
    return target, report


def build_validation_models(
    shared: SharedDualHeadModel,
    source: SourceBundle,
    *,
    kind: CheckpointKind = "float",
) -> MaterializedTaskModels:
    """Create official task graphs suitable for Ultralytics validators."""

    templates = source.build_task_models(kind)
    detect, detect_report = materialize_task_model(shared, templates.detect, Task.DETECT)
    pose, pose_report = materialize_task_model(shared, templates.pose, Task.POSE)
    if not detect_report.complete or not pose_report.complete:
        raise RuntimeError(
            f"incomplete validation materialization: detect={detect_report}, pose={pose_report}"
        )
    return MaterializedTaskModels(
        detect=detect,
        pose=pose,
        detect_report=detect_report,
        pose_report=pose_report,
    )
