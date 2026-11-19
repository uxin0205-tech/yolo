"""Stable executable seam for the formal single-GPU joint trainer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import _formal_training_impl as _impl
from .resume import save_training_snapshot as _save_training_snapshot
from .xnor import XNORExecutionConfig as _XNORExecutionConfig


def _execution_config(*, backend: str = "bool_tiled", token_tile: int = 32):
    """Translate the experiment label to the runtime backend identifier."""

    if backend == "tiled_exact":
        backend = "bool_tiled"
    return _XNORExecutionConfig(backend=backend, token_tile=token_tile)


def _portable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_portable(item) for item in value)
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value


def _portable_snapshot(destination, **kwargs):
    for key in ("resolved_config", "provenance", "loader_state", "best_state"):
        kwargs[key] = _portable(kwargs[key])
    return _save_training_snapshot(destination, **kwargs)


# The implementation resolves these names at call time. Keeping translation at
# this seam leaves the formal algorithm readable while retaining one canonical
# runtime backend name and fully portable checkpoint metadata.
_impl.XNORExecutionConfig = _execution_config
_impl.save_training_snapshot = _portable_snapshot

FormalJointTrainingSession = _impl.FormalJointTrainingSession
FormalRunReport = _impl.FormalRunReport
reseed_loader_for_epoch = _impl.reseed_loader_for_epoch
seed_everything = _impl.seed_everything

__all__ = (
    "FormalJointTrainingSession",
    "FormalRunReport",
    "reseed_loader_for_epoch",
    "seed_everything",
)

