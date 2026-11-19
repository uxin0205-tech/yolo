"""Portable epoch-boundary checkpoints with exact training continuation state."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import ultralytics
from torch import nn

TRAINING_SCHEMA_VERSION = 2
INFERENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainingProgress:
    stage: str
    next_epoch: int
    global_macro_step: int
    joint_epochs_completed: int

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage cannot be empty")
        if min(
            self.next_epoch,
            self.global_macro_step,
            self.joint_epochs_completed,
        ) < 0:
            raise ValueError("training progress counters cannot be negative")


@dataclass(frozen=True)
class SavedSnapshot:
    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True)
class RestoredSnapshot:
    path: Path
    progress: TrainingProgress
    resolved_config: dict[str, Any]
    provenance: dict[str, Any]
    loader_state: dict[str, Any]
    best_state: dict[str, Any]


def _model_contract(model: nn.Module) -> dict[str, Any]:
    contract = getattr(model, "contract", None)
    if not callable(contract):
        raise TypeError("checkpoint model must expose contract()")
    payload = contract()
    if not isinstance(payload, dict) or "model_kind" not in payload:
        raise TypeError("model contract must contain model_kind")
    return payload


def _ema_model(ema: Any) -> nn.Module:
    candidate = getattr(ema, "ema", None)
    if not isinstance(candidate, nn.Module):
        raise TypeError("EMA object must expose an nn.Module as .ema")
    return candidate


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_cpu(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(f"checkpoint value is not safely serializable: {type(value).__name__}")


def _rng_state() -> dict[str, Any]:
    bit_generator, state, position, has_gauss, cached_gaussian = np.random.get_state()
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(bit_generator),
            "state": torch.from_numpy(state.copy()),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [],
    }
    if torch.cuda.is_initialized():
        payload["torch_cuda"] = [
            state.detach().cpu().clone()
            for state in torch.cuda.get_rng_state_all()
        ]
    return payload


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    numpy_tensor = numpy_state["state"]
    if not isinstance(numpy_tensor, torch.Tensor):
        raise ValueError("NumPy RNG state tensor is missing")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_tensor.cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_initialized():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is not initialized"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from saved RNG state"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _optimizer_manifest(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], ...]:
    manifest: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        names = group.get("param_names")
        if not isinstance(names, (tuple, list)):
            raise ValueError(
                f"optimizer group {index} has no param_names manifest"
            )
        if len(names) != len(group["params"]):
            raise ValueError(
                f"optimizer group {index} parameter/name count mismatch"
            )
        manifest.append(
            {
                "index": index,
                "group_name": str(group.get("group_name", "")),
                "role": str(group.get("role", "")),
                "param_names": tuple(str(name) for name in names),
            }
        )
    return tuple(manifest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_training_snapshot(
    destination: str | Path,
    *,
    model: nn.Module,
    ema: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    criteria: Any,
    progress: TrainingProgress,
    resolved_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    loader_state: Mapping[str, Any],
    best_state: Mapping[str, Any],
) -> SavedSnapshot:
    """Save at an optimizer/epoch boundary; pending gradients are forbidden."""

    pending = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if pending:
        raise ValueError(
            f"exact snapshot refuses pending gradients: {pending[:20]}"
        )
    target = Path(destination).expanduser().resolve()
    ema_model = _ema_model(ema)
    criterion_state = criteria.state_dict()
    payload = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "checkpoint_kind": "full_resume",
        "contract": _to_cpu(_model_contract(model)),
        "model_state": _to_cpu(model.state_dict()),
        "ema_state": _to_cpu(ema_model.state_dict()),
        "ema_updates": int(getattr(ema, "updates", 0)),
        "optimizer_state": _to_cpu(optimizer.state_dict()),
        "optimizer_manifest": _to_cpu(_optimizer_manifest(optimizer)),
        "scheduler_state": _to_cpu(scheduler.state_dict()),
        "scaler_state": _to_cpu(scaler.state_dict()),
        "criteria_state": _to_cpu(criterion_state),
        "progress": _to_cpu(progress),
        "resolved_config": _to_cpu(resolved_config),
        "provenance": _to_cpu(provenance),
        "loader_state": _to_cpu(loader_state),
        "best_state": _to_cpu(best_state),
        "rng": _rng_state(),
        "environment": {
            "torch": str(torch.__version__),
            "ultralytics": str(ultralytics.__version__),
            "python_checkpoint_schema": TRAINING_SCHEMA_VERSION,
        },
    }
    _atomic_save(payload, target)
    return SavedSnapshot(
        path=target,
        sha256=_sha256(target),
        bytes=target.stat().st_size,
    )


def _mapping(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return value


def load_training_snapshot(
    checkpoint: str | Path,
    *,
    model: nn.Module,
    ema: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    criteria: Any,
    restore_rng: bool = True,
) -> RestoredSnapshot:
    """Restore without changing stage, optimizer groups, or model contract."""

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != TRAINING_SCHEMA_VERSION
        or payload.get("checkpoint_kind") != "full_resume"
    ):
        raise ValueError("unsupported or malformed full-resume checkpoint")
    expected_contract = _model_contract(model)
    if payload.get("contract") != expected_contract:
        raise ValueError(
            f"checkpoint contract mismatch: {payload.get('contract')} != {expected_contract}"
        )
    current_manifest = _optimizer_manifest(optimizer)
    saved_manifest = payload.get("optimizer_manifest")
    if tuple(saved_manifest or ()) != current_manifest:
        raise ValueError(
            "optimizer parameter manifest changed; this is a fine-tune "
            "initialization, not an exact resume"
        )
    model.load_state_dict(_mapping(payload, "model_state"), strict=True)
    ema_model = _ema_model(ema)
    ema_model.load_state_dict(_mapping(payload, "ema_state"), strict=True)
    ema.updates = int(payload.get("ema_updates", 0))
    optimizer.load_state_dict(_mapping(payload, "optimizer_state"))
    scheduler.load_state_dict(_mapping(payload, "scheduler_state"))
    scaler.load_state_dict(_mapping(payload, "scaler_state"))
    criteria.load_state_dict(_mapping(payload, "criteria_state"))
    progress_payload = _mapping(payload, "progress")
    progress = TrainingProgress(
        stage=str(progress_payload["stage"]),
        next_epoch=int(progress_payload["next_epoch"]),
        global_macro_step=int(progress_payload["global_macro_step"]),
        joint_epochs_completed=int(
            progress_payload["joint_epochs_completed"]
        ),
    )
    if restore_rng:
        _restore_rng(_mapping(payload, "rng"))
    return RestoredSnapshot(
        path=path,
        progress=progress,
        resolved_config=_mapping(payload, "resolved_config"),
        provenance=_mapping(payload, "provenance"),
        loader_state=_mapping(payload, "loader_state"),
        best_state=_mapping(payload, "best_state"),
    )


def save_inference_weights(
    destination: str | Path,
    *,
    model: nn.Module,
    ema: Any | None = None,
    use_ema: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a small state-dict-only artifact, separate from resumable files."""

    selected = _ema_model(ema) if use_ema and ema is not None else model
    target = Path(destination).expanduser().resolve()
    payload = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "checkpoint_kind": "inference_only",
        "contract": _to_cpu(_model_contract(model)),
        "source": "ema" if selected is not model else "live",
        "state_dict": _to_cpu(selected.state_dict()),
        "metadata": _to_cpu(metadata or {}),
    }
    _atomic_save(payload, target)
    return target
