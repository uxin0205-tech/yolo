"""Routed and truly shared YOLO26 Detect/Pose model implementations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn
from ultralytics.nn.modules.head import Detect, Pose26
from ultralytics.nn.tasks import DetectionModel, PoseModel

from .contracts import Task, normalize_tasks


class SharedFeatureTrunk(nn.Module):
    """Execute the Ultralytics graph through its last shared layer and return head features."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        *,
        save_indices: Iterable[int],
        output_indices: Sequence[int],
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("shared trunk requires at least one layer")
        self.layers = nn.ModuleList(layers)
        self.output_indices = tuple(int(index) for index in output_indices)
        if len(self.output_indices) != 3:
            raise ValueError(f"expected P3/P4/P5 output indices, got {self.output_indices}")
        if max(self.output_indices) >= len(self.layers):
            raise ValueError("a requested head input is outside the shared trunk")
        self.save_indices = frozenset(int(index) for index in save_indices) | frozenset(self.output_indices)
        for index, layer in enumerate(self.layers):
            if int(getattr(layer, "i", -1)) != index:
                raise ValueError(f"shared layer index mismatch at {index}: {getattr(layer, 'i', None)}")

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the original skip graph exactly once and expose P3/P4/P5."""

        saved: list[torch.Tensor | None] = [None] * len(self.layers)
        current: Any = images
        for layer in self.layers:
            source = getattr(layer, "f", -1)
            if source != -1:
                if isinstance(source, int):
                    current = saved[source]
                else:
                    current = [current if index == -1 else saved[index] for index in source]
                if current is None or (isinstance(current, list) and any(value is None for value in current)):
                    raise RuntimeError(f"missing saved input for shared layer {layer.i} from {source}")
            current = layer(current)
            saved[layer.i] = current if layer.i in self.save_indices else None
        features = tuple(saved[index] for index in self.output_indices)
        if any(feature is None for feature in features):
            raise RuntimeError("shared trunk did not produce every head feature")
        return features  # type: ignore[return-value]


class RoutedDualModel(nn.Module):
    """F0.5: two complete models behind the same task interface as F1."""

    model_kind = "routed_dual"

    def __init__(self, detect_model: DetectionModel, pose_model: PoseModel) -> None:
        super().__init__()
        if not isinstance(detect_model.model[-1], Detect) or isinstance(detect_model.model[-1], Pose26):
            raise TypeError("detect_model must end in a Detect head")
        if not isinstance(pose_model.model[-1], Pose26):
            raise TypeError("pose_model must end in a Pose26 head")
        self.detect_model = detect_model
        self.pose_model = pose_model

    def forward(
        self,
        images: torch.Tensor,
        tasks: Task | str | Iterable[Task | str] | None = None,
    ) -> dict[str, Any]:
        selected = normalize_tasks(tasks)
        outputs: dict[str, Any] = {}
        if Task.DETECT in selected:
            outputs[Task.DETECT.value] = self.detect_model(images)
        if Task.POSE in selected:
            outputs[Task.POSE.value] = self.pose_model(images)
        return outputs

    def contract(self) -> dict[str, Any]:
        """Return the task interface facts needed to validate an F0.5 checkpoint."""

        detect_head = self.detect_model.model[-1]
        pose_head = self.pose_model.model[-1]
        return {
            "model_kind": self.model_kind,
            "head_inputs": [int(index) for index in detect_head.f],
            "detect_nc": int(detect_head.nc),
            "pose_nc": int(pose_head.nc),
            "kpt_shape": list(pose_head.kpt_shape),
            "detect_names": dict(self.detect_model.names),
            "pose_names": dict(self.pose_model.names),
        }


class SharedDualHeadModel(nn.Module):
    """F1: one live Full35 trunk feeding independent Detect and Pose26 heads."""

    model_kind = "shared_dual_head"

    def __init__(
        self,
        trunk: SharedFeatureTrunk,
        detect_head: Detect,
        pose_head: Pose26,
        *,
        detect_names: dict[int, str],
        pose_names: dict[int, str],
    ) -> None:
        super().__init__()
        if isinstance(detect_head, Pose26):
            raise TypeError("detect_head cannot be a Pose26 head")
        if tuple(int(index) for index in detect_head.f) != trunk.output_indices:
            raise ValueError("Detect head inputs do not match shared trunk outputs")
        if tuple(int(index) for index in pose_head.f) != trunk.output_indices:
            raise ValueError("Pose head inputs do not match shared trunk outputs")
        self.trunk = trunk
        self.detect_head = detect_head
        self.pose_head = pose_head
        self.detect_names = dict(detect_names)
        self.pose_names = dict(pose_names)

    @classmethod
    def from_task_models(
        cls,
        detect_model: DetectionModel,
        pose_model: PoseModel,
    ) -> "SharedDualHeadModel":
        """Consume two compatible task models while retaining only the Detect trunk."""

        detect_layers = detect_model.model
        pose_layers = pose_model.model
        if len(detect_layers) != len(pose_layers):
            raise ValueError("Detect and Pose graphs have different layer counts")
        detect_head = detect_layers[-1]
        pose_head = pose_layers[-1]
        if not isinstance(detect_head, Detect) or isinstance(detect_head, Pose26):
            raise TypeError("source Detect graph does not end in Detect")
        if not isinstance(pose_head, Pose26):
            raise TypeError("source Pose graph does not end in Pose26")
        head_inputs = tuple(int(index) for index in detect_head.f)
        if tuple(int(index) for index in pose_head.f) != head_inputs:
            raise ValueError("Detect and Pose heads consume different feature layers")
        trunk = SharedFeatureTrunk(
            list(detect_layers[:-1]),
            save_indices=detect_model.save,
            output_indices=head_inputs,
        )
        return cls(
            trunk,
            detect_head,
            pose_head,
            detect_names=dict(detect_model.names),
            pose_names=dict(pose_model.names),
        )

    def head_for(self, task: Task | str) -> Detect | Pose26:
        normalized = Task(task)
        return self.detect_head if normalized is Task.DETECT else self.pose_head

    def forward(
        self,
        images: torch.Tensor,
        tasks: Task | str | Iterable[Task | str] | None = None,
    ) -> dict[str, Any]:
        selected = normalize_tasks(tasks)
        features = self.trunk(images)
        outputs: dict[str, Any] = {}
        if Task.DETECT in selected:
            outputs[Task.DETECT.value] = self.detect_head(list(features))
        if Task.POSE in selected:
            outputs[Task.POSE.value] = self.pose_head(list(features))
        return outputs

    def contract(self) -> dict[str, Any]:
        """Return the stable facts required to rebuild and validate a checkpoint."""

        return {
            "model_kind": self.model_kind,
            "head_inputs": list(self.trunk.output_indices),
            "detect_nc": int(self.detect_head.nc),
            "pose_nc": int(self.pose_head.nc),
            "kpt_shape": list(self.pose_head.kpt_shape),
            "detect_names": self.detect_names,
            "pose_names": self.pose_names,
        }
