"""Ultralytics-graph-preserving YOLO26 Detect/Pose fusion model.

The public seam is :class:`GraphSharedDualHeadModel`. It keeps the complete
Ultralytics DetectionModel graph and replaces only its final prediction module
with a dual-head wrapper. Consequently skip connections, Concat nodes, saved
from-indices, and graph profiling metadata remain owned by Ultralytics.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from ultralytics.nn.modules.head import Detect, Pose26
from ultralytics.nn.tasks import DetectionModel, PoseModel

from .contracts import Task, normalize_tasks


class GraphCompatibilityError(ValueError):
    """Raised with every discovered shared-graph mismatch."""


@dataclass(frozen=True)
class TaskPairAudit:
    """Complete compatibility contract for one Detect/Pose source pair."""

    compatible: bool
    shared_layers: int
    differences: tuple[str, ...]
    head_inputs: tuple[int, ...]
    feature_channels: tuple[int, ...]
    strides: tuple[float, ...]
    reg_max: int
    end2end: bool
    detect_nc: int
    pose_nc: int
    detect_names: dict[int, str]
    pose_names: dict[int, str]
    pose_kpt_shape: tuple[int, int]
    pose_flow_module: str

    def require_compatible(self) -> None:
        if self.compatible:
            return
        rendered = "\n".join(f"- {difference}" for difference in self.differences)
        raise GraphCompatibilityError(
            "Detect/Pose shared graph is incompatible; fusion was not attempted:\n"
            f"{rendered}"
        )


@dataclass(frozen=True)
class AssemblyReport:
    """Observable evidence that assembly removed exactly one duplicate trunk."""

    audit: TaskPairAudit
    independent_parameters: int
    shared_parameters: int
    parameter_reduction_fraction: float
    loaded_shared_tensors: int
    loaded_detect_head_tensors: int
    loaded_pose_head_tensors: int
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()
    shape_mismatches: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not (
            self.missing_keys or self.unexpected_keys or self.shape_mismatches
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.audit.compatible,
            "shared_layers": self.audit.shared_layers,
            "differences": list(self.audit.differences),
            "head_inputs": list(self.audit.head_inputs),
            "feature_channels": list(self.audit.feature_channels),
            "strides": list(self.audit.strides),
            "reg_max": self.audit.reg_max,
            "end2end": self.audit.end2end,
            "detect_nc": self.audit.detect_nc,
            "pose_nc": self.audit.pose_nc,
            "detect_names": self.audit.detect_names,
            "pose_names": self.audit.pose_names,
            "pose_kpt_shape": list(self.audit.pose_kpt_shape),
            "pose_flow_module": self.audit.pose_flow_module,
            "independent_parameters": self.independent_parameters,
            "shared_parameters": self.shared_parameters,
            "parameter_reduction_fraction": self.parameter_reduction_fraction,
            "loaded_shared_tensors": self.loaded_shared_tensors,
            "loaded_detect_head_tensors": self.loaded_detect_head_tensors,
            "loaded_pose_head_tensors": self.loaded_pose_head_tensors,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "shape_mismatches": list(self.shape_mismatches),
        }


def _normalized_from(module: nn.Module) -> int | tuple[int, ...]:
    value = getattr(module, "f", -1)
    if isinstance(value, int):
        return value
    return tuple(int(index) for index in value)


def _tensor_signature(
    tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (name, tuple(int(size) for size in tensor.shape), str(tensor.dtype))
        for name, tensor in tensors
    )


def _repeat_hint(module: nn.Module) -> int:
    return len(module) if isinstance(module, nn.Sequential) else 1


def _head_input_channels(head: Detect) -> tuple[int, ...]:
    channels: list[int] = []
    for level, branch in enumerate(head.cv2):
        convolution = next(
            (module for module in branch.modules() if isinstance(module, nn.Conv2d)),
            None,
        )
        if convolution is None:
            raise GraphCompatibilityError(
                f"head cv2[{level}] contains no Conv2d input projection"
            )
        channels.append(int(convolution.in_channels))
    return tuple(channels)


def _qualified_type(module: nn.Module) -> str:
    kind = type(module)
    return f"{kind.__module__}.{kind.__qualname__}"


def audit_task_pair(
    detect_model: DetectionModel,
    pose_model: PoseModel,
) -> TaskPairAudit:
    """Compare every pre-head graph fact and the YOLO26 task-head contract."""

    if not isinstance(detect_model, DetectionModel):
        raise TypeError("detect_model must be an Ultralytics DetectionModel")
    if not isinstance(pose_model, PoseModel):
        raise TypeError("pose_model must be an Ultralytics PoseModel")
    detect_layers = detect_model.model
    pose_layers = pose_model.model
    differences: list[str] = []
    if len(detect_layers) != len(pose_layers):
        differences.append(
            f"top-level layer count: detect={len(detect_layers)}, pose={len(pose_layers)}"
        )
    shared_layers = max(min(len(detect_layers), len(pose_layers)) - 1, 0)
    for index in range(shared_layers):
        detect_layer = detect_layers[index]
        pose_layer = pose_layers[index]
        fields = {
            "module_type": (_qualified_type(detect_layer), _qualified_type(pose_layer)),
            "from": (_normalized_from(detect_layer), _normalized_from(pose_layer)),
            "index": (
                int(getattr(detect_layer, "i", -1)),
                int(getattr(pose_layer, "i", -1)),
            ),
            "repeat": (_repeat_hint(detect_layer), _repeat_hint(pose_layer)),
            "parameter_count": (
                sum(parameter.numel() for parameter in detect_layer.parameters()),
                sum(parameter.numel() for parameter in pose_layer.parameters()),
            ),
            "parameters": (
                _tensor_signature(detect_layer.named_parameters()),
                _tensor_signature(pose_layer.named_parameters()),
            ),
            "buffers": (
                _tensor_signature(detect_layer.named_buffers()),
                _tensor_signature(pose_layer.named_buffers()),
            ),
        }
        for field, (detect_value, pose_value) in fields.items():
            if detect_value != pose_value:
                differences.append(
                    f"layer[{index}].{field}: detect={detect_value!r}, pose={pose_value!r}"
                )

    detect_head = detect_layers[-1] if len(detect_layers) else None
    pose_head = pose_layers[-1] if len(pose_layers) else None
    if not isinstance(detect_head, Detect) or isinstance(detect_head, Pose26):
        differences.append(
            "detect final module must be a non-Pose Detect head, got "
            f"{type(detect_head).__name__}"
        )
    if not isinstance(pose_head, Pose26):
        differences.append(
            f"pose final module must be Pose26, got {type(pose_head).__name__}"
        )
    if not isinstance(detect_head, Detect) or not isinstance(pose_head, Pose26):
        return TaskPairAudit(
            compatible=False,
            shared_layers=shared_layers,
            differences=tuple(differences),
            head_inputs=(),
            feature_channels=(),
            strides=(),
            reg_max=-1,
            end2end=False,
            detect_nc=-1,
            pose_nc=-1,
            detect_names=dict(getattr(detect_model, "names", {})),
            pose_names=dict(getattr(pose_model, "names", {})),
            pose_kpt_shape=(-1, -1),
            pose_flow_module="missing",
        )

    detect_inputs = tuple(int(index) for index in detect_head.f)
    pose_inputs = tuple(int(index) for index in pose_head.f)
    if detect_inputs != pose_inputs:
        differences.append(
            f"head from-indices: detect={detect_inputs}, pose={pose_inputs}"
        )
    detect_channels = _head_input_channels(detect_head)
    pose_channels = _head_input_channels(pose_head)
    if detect_channels != pose_channels:
        differences.append(
            f"P3/P4/P5 channels: detect={detect_channels}, pose={pose_channels}"
        )
    detect_strides = tuple(float(value) for value in detect_head.stride.tolist())
    pose_strides = tuple(float(value) for value in pose_head.stride.tolist())
    if detect_strides != pose_strides:
        differences.append(
            f"head strides: detect={detect_strides}, pose={pose_strides}"
        )
    if int(detect_head.reg_max) != int(pose_head.reg_max):
        differences.append(
            f"reg_max: detect={detect_head.reg_max}, pose={pose_head.reg_max}"
        )
    if bool(detect_head.end2end) != bool(pose_head.end2end):
        differences.append(
            f"end2end: detect={detect_head.end2end}, pose={pose_head.end2end}"
        )
    flow = getattr(pose_head, "flow_model", None)
    if flow is None:
        differences.append("Pose26 flow_model/RLE module is missing")

    return TaskPairAudit(
        compatible=not differences,
        shared_layers=shared_layers,
        differences=tuple(differences),
        head_inputs=detect_inputs,
        feature_channels=detect_channels,
        strides=detect_strides,
        reg_max=int(detect_head.reg_max),
        end2end=bool(detect_head.end2end),
        detect_nc=int(detect_head.nc),
        pose_nc=int(pose_head.nc),
        detect_names=dict(detect_model.names),
        pose_names=dict(pose_model.names),
        pose_kpt_shape=tuple(int(value) for value in pose_head.kpt_shape),
        pose_flow_module=type(flow).__name__ if flow is not None else "missing",
    )


_ACTIVE_TASKS: ContextVar[frozenset[Task] | None] = ContextVar(
    "yolo_combine_active_tasks",
    default=None,
)


class DualHeadPredictionModule(nn.Module):
    """Final graph module that routes one P3/P4/P5 list to one or both heads."""

    def __init__(self, detect_head: Detect, pose_head: Pose26) -> None:
        super().__init__()
        if isinstance(detect_head, Pose26):
            raise TypeError("detect_head cannot be Pose26")
        detect_inputs = tuple(int(index) for index in detect_head.f)
        pose_inputs = tuple(int(index) for index in pose_head.f)
        if detect_inputs != pose_inputs:
            raise ValueError(
                f"head from-index mismatch: {detect_inputs} != {pose_inputs}"
            )
        self.detect_head = detect_head
        self.pose_head = pose_head
        self.f = list(detect_inputs)
        self.i = int(detect_head.i)
        self.type = str(detect_head.type)
        self.detect_type = str(detect_head.type)
        self.pose_type = str(pose_head.type)
        self.np = sum(parameter.numel() for parameter in self.parameters())

    @contextmanager
    def selecting(
        self,
        tasks: frozenset[Task],
    ) -> Iterator[None]:
        token = _ACTIVE_TASKS.set(tasks)
        try:
            yield
        finally:
            _ACTIVE_TASKS.reset(token)

    def forward(self, features: Sequence[torch.Tensor]) -> dict[str, Any]:
        if len(features) != 3:
            raise ValueError(f"expected P3/P4/P5 feature list, got {len(features)}")
        selected = _ACTIVE_TASKS.get() or normalize_tasks("both")
        outputs: dict[str, Any] = {}
        if Task.DETECT in selected:
            outputs[Task.DETECT.value] = self.detect_head(list(features))
        if Task.POSE in selected:
            outputs[Task.POSE.value] = self.pose_head(list(features))
        return outputs


class GraphSharedDualHeadModel(nn.Module):
    """One Ultralytics graph with one shared trunk and two task-specific heads."""

    model_kind = "graph_shared_dual_head"

    def __init__(
        self,
        graph: DetectionModel,
        *,
        detect_names: dict[int, str],
        pose_names: dict[int, str],
    ) -> None:
        super().__init__()
        if not isinstance(graph.model[-1], DualHeadPredictionModule):
            raise TypeError("graph final module must be DualHeadPredictionModule")
        self.graph = graph
        self.detect_names = dict(detect_names)
        self.pose_names = dict(pose_names)

    @property
    def prediction(self) -> DualHeadPredictionModule:
        prediction = self.graph.model[-1]
        if not isinstance(prediction, DualHeadPredictionModule):
            raise RuntimeError("shared graph final module was replaced unexpectedly")
        return prediction

    @property
    def detect_head(self) -> Detect:
        return self.prediction.detect_head

    @property
    def pose_head(self) -> Pose26:
        return self.prediction.pose_head

    @property
    def trunk_layers(self) -> tuple[nn.Module, ...]:
        return tuple(self.graph.model[:-1])

    def head_for(self, task: Task | str) -> Detect | Pose26:
        selected = Task(task)
        return self.detect_head if selected is Task.DETECT else self.pose_head

    def forward(
        self,
        images: torch.Tensor,
        task: Task | str | Iterable[Task | str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(images, torch.Tensor):
            raise TypeError("GraphSharedDualHeadModel forward expects an image tensor")
        selected = normalize_tasks(task)
        with self.prediction.selecting(selected):
            outputs = self.graph.predict(images)
        if not isinstance(outputs, dict):
            raise RuntimeError(
                f"dual-head graph returned {type(outputs).__name__}, expected dict"
            )
        expected = {
            name
            for name in (Task.DETECT.value, Task.POSE.value)
            if Task(name) in selected
        }
        if set(outputs) != expected:
            raise RuntimeError(
                f"dual-head output keys {set(outputs)} do not match {expected}"
            )
        return outputs

    def contract(self) -> dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "shared_layers": len(self.trunk_layers),
            "head_inputs": list(self.prediction.f),
            "feature_channels": list(_head_input_channels(self.detect_head)),
            "strides": [float(value) for value in self.detect_head.stride.tolist()],
            "reg_max": int(self.detect_head.reg_max),
            "end2end": bool(self.detect_head.end2end),
            "detect_nc": int(self.detect_head.nc),
            "pose_nc": int(self.pose_head.nc),
            "kpt_shape": list(self.pose_head.kpt_shape),
            "detect_names": self.detect_names,
            "pose_names": self.pose_names,
            "pose_flow_module": type(self.pose_head.flow_model).__name__,
        }


def assemble_graph_shared_model(
    detect_model: DetectionModel,
    pose_model: PoseModel,
) -> tuple[GraphSharedDualHeadModel, AssemblyReport]:
    """Consume a compatible task pair and replace only Detect graph layer 23."""

    audit = audit_task_pair(detect_model, pose_model)
    audit.require_compatible()
    independent_parameters = sum(
        parameter.numel()
        for task_model in (detect_model, pose_model)
        for parameter in task_model.parameters()
    )
    detect_head = detect_model.model[-1]
    pose_head = pose_model.model[-1]
    if not isinstance(detect_head, Detect) or isinstance(detect_head, Pose26):
        raise TypeError("Detect graph no longer ends in a Detect head")
    if not isinstance(pose_head, Pose26):
        raise TypeError("Pose graph no longer ends in Pose26")
    loaded_shared_tensors = sum(
        len(layer.state_dict()) for layer in detect_model.model[:-1]
    )
    loaded_detect_head_tensors = len(detect_head.state_dict())
    loaded_pose_head_tensors = len(pose_head.state_dict())
    detect_names = dict(detect_model.names)
    pose_names = dict(pose_model.names)
    detect_model.model[-1] = DualHeadPredictionModule(detect_head, pose_head)
    model = GraphSharedDualHeadModel(
        detect_model,
        detect_names=detect_names,
        pose_names=pose_names,
    )
    shared_parameters = sum(parameter.numel() for parameter in model.parameters())
    report = AssemblyReport(
        audit=audit,
        independent_parameters=independent_parameters,
        shared_parameters=shared_parameters,
        parameter_reduction_fraction=(
            (independent_parameters - shared_parameters) / independent_parameters
        ),
        loaded_shared_tensors=loaded_shared_tensors,
        loaded_detect_head_tensors=loaded_detect_head_tensors,
        loaded_pose_head_tensors=loaded_pose_head_tensors,
    )
    if not report.complete:
        raise RuntimeError(f"incomplete dual-head assembly: {report}")
    return model, report
