"""Materialize official task graphs while preserving Bit-True PWL tables."""

from __future__ import annotations

from typing import TypeVar

import torch
from ultralytics.nn.modules.head import Detect, Pose26
from ultralytics.nn.tasks import DetectionModel, PoseModel

from . import _graph_materialize_impl as _impl
from .contracts import Task
from .fusion_model import GraphSharedDualHeadModel
from .source import CheckpointKind, SourceBundle

GraphMaterializationReport = _impl.GraphMaterializationReport
GraphValidationModels = _impl.GraphValidationModels
TaskModel = TypeVar("TaskModel", DetectionModel, PoseModel)

_TARGET_BITTRUE_SUFFIX = ".attn.normalize.endpoint_table"
_SOURCE_FLOAT_SUFFIXES = (
    ".attn.normalize.knots",
    ".attn.normalize.values",
)


def materialize_graph_task_model(
    shared: GraphSharedDualHeadModel,
    target: TaskModel,
    task: Task | str,
) -> tuple[TaskModel, GraphMaterializationReport]:
    """Copy common named state and retain only deterministic Bit-True endpoints."""

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
    if (
        selected is Task.POSE
        and tuple(target_head.kpt_shape) != tuple(shared_head.kpt_shape)
    ):
        raise ValueError(
            f"Pose kpt_shape changed: {target_head.kpt_shape} "
            f"!= {shared_head.kpt_shape}"
        )
    if tuple(int(index) for index in target_head.f) != tuple(shared.prediction.f):
        raise ValueError("target head consumes different P3/P4/P5 layers")

    shared_state = shared.state_dict()
    target_state = target.state_dict()
    mapping = {
        name: _impl._source_name(name, head_index=head_index, task=selected)
        for name in target_state
    }
    preserved_target = {
        name
        for name, source_name in mapping.items()
        if source_name not in shared_state
        and name.endswith(_TARGET_BITTRUE_SUFFIX)
    }
    missing = tuple(
        name
        for name, source_name in mapping.items()
        if (
            (source_name is None or source_name not in shared_state)
            and name not in preserved_target
        )
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
    task_head = "detect_head" if selected is Task.DETECT else "pose_head"
    expected_sources = {
        name
        for name in shared_state
        if name.startswith("graph.model.")
        and (
            not name.startswith(f"graph.model.{head_index}.")
            or name.startswith(f"graph.model.{head_index}.{task_head}.")
        )
    }
    ignored_float = {
        name
        for name in expected_sources
        if any(name.endswith(suffix) for suffix in _SOURCE_FLOAT_SUFFIXES)
        and preserved_target
    }
    unexpected = tuple(sorted(expected_sources - used_sources - ignored_float))
    report = GraphMaterializationReport(
        task=selected,
        compatible_tensors=len(compatible) + len(preserved_target),
        missing_tensors=missing,
        unexpected_tensors=unexpected,
        shape_mismatches=shape_mismatches,
    )
    if (
        not report.complete
        or len(compatible) + len(preserved_target) != len(target_state)
    ):
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
    """Build temporary official graphs; live training remains one shared model."""

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


__all__ = (
    "GraphMaterializationReport",
    "GraphValidationModels",
    "build_graph_validation_models",
    "materialize_graph_task_model",
)
