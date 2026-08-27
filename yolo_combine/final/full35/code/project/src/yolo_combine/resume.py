"""Stable public seam for exact-resume checkpointing."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from . import _resume_impl as _impl


def _to_cpu_preserving_keys(value: Any) -> Any:
    """Move tensors to CPU without changing optimizer integer state keys."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_cpu_preserving_keys(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        converted: dict[Any, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool, type(None))):
                raise TypeError(
                    "checkpoint mapping key is not safely serializable: "
                    f"{type(key).__name__}"
                )
            converted[key] = _to_cpu_preserving_keys(item)
        return converted
    if isinstance(value, tuple):
        return tuple(_to_cpu_preserving_keys(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu_preserving_keys(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(
        f"checkpoint value is not safely serializable: {type(value).__name__}"
    )


# Functions in the implementation resolve this global at call time. Installing
# the converter here keeps the external interface compact while preserving
# optimizer state IDs exactly.
_impl._to_cpu = _to_cpu_preserving_keys

TRAINING_SCHEMA_VERSION = _impl.TRAINING_SCHEMA_VERSION
INFERENCE_SCHEMA_VERSION = _impl.INFERENCE_SCHEMA_VERSION
TrainingProgress = _impl.TrainingProgress
SavedSnapshot = _impl.SavedSnapshot
RestoredSnapshot = _impl.RestoredSnapshot
save_training_snapshot = _impl.save_training_snapshot
load_training_snapshot = _impl.load_training_snapshot
save_inference_weights = _impl.save_inference_weights

__all__ = (
    "INFERENCE_SCHEMA_VERSION",
    "RestoredSnapshot",
    "SavedSnapshot",
    "TRAINING_SCHEMA_VERSION",
    "TrainingProgress",
    "load_training_snapshot",
    "save_inference_weights",
    "save_training_snapshot",
)
