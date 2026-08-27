"""Materialize official Detect/Pose26 validators from graph-shared EMA state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import torch
from ultralytics.nn.modules.head import Detect, Pose26
from ultralytics.nn.tasks import DetectionModel, PoseModel

from .contracts import Task
from .fusion_model import GraphSharedDualHeadModel
from .source import CheckpointKind, SourceBundle

TaskModel = TypeVar("TaskModel", DetectionModel, PoseModel)


@dataclass(frozen=True)
class GraphMaterializationReport:
    task: Task
    compatible_tensors: int
    missing_tensors: tuple[str, ...]
    unexpected_tensors: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not (
            self.missing_tensors
            or self.unexpected_tensors
            or self.shape_mismatches
        )


@dataclass(frozen=True)
class GraphValidationModels:
    detect: DetectionModel
    pose: PoseModel
    detect_report: GraphMaterializationReport
    pose_report: GraphMaterializationReport
    checkpoint_kind: CheckpointKind


def _source_name(
    target_name: str,
    *,
    head_index: int,
    task: Task,
) -> str | None:
    parts = target_name.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    layer = int(parts[1])
    remainder = ".".join(parts[2:])
    if layer < head_index:
        return f"graph.model.{layer}.{remainder}"
    if layer == head_index:
        head = "detect_head" if task is Task.DETECT else "pose_head"
        return f"graph.model.{layer}.{head}.{remainder}"
    return None


def materialize_graph_task_model(
    shared: GraphSharedDualHeadModel,
    target: TaskModel,
    task: Task | str,
) -> tuple[TaskModel, GraphMaterializationReport]:
    """Copy by explicit names; state-dict insertion order is never consulted."""

    selected = Task(task)
    head_index = len(target.model) - 1
    target_head = target.model[-1]
    if selected is Task.DETECT:
        if not isinstance(target, DetectionModel):
            raise TypeError("Detect materialization requires DetectionModel")
        if not isinstance(target_head, Detect) or isinstance(target_head, Pose26):
            raise TypeError("Detect target must end in non-Pose Detect")
        shared_head = shared.detect_head
    else:
        if not isinstance(target, PoseModel) or not isinstance(target_head, Pose26):
            raise TypeError("Pose materialization requires PoseModel ending Pose26")
        shared_head = shared.pose_head
    if int(target_head.nc) != int(shared_head.nc):
        raise ValueError(
            f"{selected.value} class count changed: {target_head.nc} != {shared_head.nc}"
        )
    if selected is Task.POSE and tuple(target_head.kpt_shape) != tuple(shared_head.kpt_shape):
        raise ValueError(
            f"Pose kpt_shape changed: {target_head.kpt_shape} != {shared_head.kpt_shape}"
        )
    if tuple(int(index) for index in target_head.f) != tuple(shared.prediction.f):
        raise ValueError("target head consumes different P3/P4/P5 layers")

    shared_state = shared.state_dict()
    target_state = target.state_dict()
    mapping = {
        name: _source_name(name, head_index=head_index, task=selected)
        for name in target_state
    }
    missing = tuple(
        name
        for name, source_name in mapping.items()
        if source_name is None or source_name not in shared_state
    )
    shape_mismatches = tuple(
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
    used_sources = {mapping[name] for name in compatible}
    expected_sources = {
        name
        for name in shared_state
        if name.startswith("graph.model.")
        and (
            not name.startswith(f"graph.model.{head_index}.")
            or name.startswith(
                f"graph.model.{head_index}."
                f"{'detect_head' if selected is Task.DETECT else 'pose_head'}."
            )
        )
    }
    unexpected = tuple(sorted(expected_sources - used_sources))
    report = GraphMaterializationReport(
        task=selected,
        compatible_tensors=len(compatible),
        missing_tensors=missing,
        unexpected_tensors=unexpected,
        shape_mismatches=shape_mismatches,
    )
    if not report.complete or len(compatible) != len(target_state):
        return target, report
    with torch.no_grad():
        for name in compatible:
            source_name = mapping[name]
            if source_name is None:
                raise AssertionError(name)
            target_state[name].copy_(shared_state[source_name])
    target.load_state_dict(target_state, strict=True)
    target.names = dict(
        shared.detect_names if selected is Task.DETECT else shared.pose_names
    )
    target.requires_grad_(False)
    target.eval()
    return target, report


def build_graph_validation_models(
    shared: GraphSharedDualHeadModel,
    source: SourceBundle,
    *,
    kind: CheckpointKind = "bittrue",
) -> GraphValidationModels:
    """Build temporary official graphs; the live training model stays shared."""

    templates = source.build_task_models(kind)
    detect, detect_report = materialize_graph_task_model(
        shared,
        templates.detect,
        Task.DETECT,
    )
    pose, pose_report = materialize_graph_task_model(
        shared,
        templates.pose,
        Task.POSE,
    )
    if not detect_report.complete or not pose_report.complete:
        raise RuntimeError(
            "incomplete graph materialization: "
            f"detect={detect_report}, pose={pose_report}"
        )
    return GraphValidationModels(
        detect=detect,
        pose=pose,
        detect_report=detect_report,
        pose_report=pose_report,
        checkpoint_kind=kind,
    )
