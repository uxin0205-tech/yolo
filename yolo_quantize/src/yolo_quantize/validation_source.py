"""Activation-aware, BN-folded task templates for official validation."""

from __future__ import annotations

import copy
from typing import Any

from torch import nn

from .activation_adapter import (
    AppliedActivationQuantization,
    _ObservedQuantizedActivation,
    _set_submodule,
)


class Full35DeploymentValidationSource:
    """Adapt a Full35 SourceBundle to one calibrated deployment policy."""

    def __init__(
        self,
        source: Any,
        applied: AppliedActivationQuantization,
    ) -> None:
        self._source = source
        self._applied = applied

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    @staticmethod
    def _target_path(path: str, task: str) -> str | None:
        detect_prefix = "graph.model.23.detect_head."
        pose_prefix = "graph.model.23.pose_head."
        if path.startswith(detect_prefix):
            return (
                "model.23." + path.removeprefix(detect_prefix)
                if task == "detect"
                else None
            )
        if path.startswith(pose_prefix):
            return (
                "model.23." + path.removeprefix(pose_prefix) if task == "pose" else None
            )
        if path.startswith("graph.model."):
            return path.removeprefix("graph.")
        raise ValueError(f"Full35 activation path is outside the graph: {path}")

    def _prepare_task_model(self, model: nn.Module, task: str) -> None:
        copied = 0
        for source_path in self._applied.wrapped_paths:
            target_path = self._target_path(source_path, task)
            if target_path is None:
                continue
            source_wrapper = self._applied.model.get_submodule(source_path)
            if not isinstance(source_wrapper, _ObservedQuantizedActivation):
                raise TypeError(
                    f"calibrated activation path became stale: {source_path}"
                )
            try:
                target_activation = model.get_submodule(target_path)
            except AttributeError as error:
                raise RuntimeError(
                    f"task activation path is missing: {task}:{target_path}"
                ) from error
            if tuple(target_activation.children()):
                raise RuntimeError(
                    f"task activation path is not a leaf: {task}:{target_path}"
                )
            replacement = copy.deepcopy(source_wrapper).cpu()
            _set_submodule(model, target_path, replacement)
            copied += 1
        if copied == 0:
            raise RuntimeError(f"no activation wrappers projected for task {task}")
        fuse = getattr(model, "fuse", None)
        if not callable(fuse):
            raise TypeError(f"{task} validation model exposes no fuse()")
        fuse(verbose=False)
        retained_bn = sum(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in model.modules()
        )
        if retained_bn:
            raise RuntimeError(
                f"{task} deployment validation model retains {retained_bn} BatchNorm"
            )

    def build_task_models(self, *args: Any, **kwargs: Any) -> Any:
        """Build task templates matching the calibrated deployment graph."""

        built = self._source.build_task_models(*args, **kwargs)
        self._prepare_task_model(built.detect, "detect")
        self._prepare_task_model(built.pose, "pose")
        return built
