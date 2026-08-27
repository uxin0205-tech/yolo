"""State-dict-only checkpoint contract for combined models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .source import SourceBundle

SCHEMA_VERSION = 1


def _model_contract(model: nn.Module) -> dict[str, Any]:
    contract_method = getattr(model, "contract", None)
    if not callable(contract_method):
        raise TypeError("model must expose contract()")
    contract = contract_method()
    if not isinstance(contract, dict) or "model_kind" not in contract:
        raise TypeError("model contract must be a mapping containing model_kind")
    return contract


def save_checkpoint(
    destination: str | Path,
    model: nn.Module,
    source: SourceBundle,
    *,
    checkpoint_kind: str = "float",
    training: dict[str, Any] | None = None,
) -> Path:
    """Atomically save tensors and rebuild metadata without pickling a model object."""

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract": _model_contract(model),
        "source": source.provenance(checkpoint_kind),
        "training": dict(training or {}),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    }
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def load_checkpoint(
    checkpoint: str | Path,
    model: nn.Module,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a safe tensor checkpoint into a prebuilt model and verify its interface contract."""

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or malformed combined checkpoint")
    expected = _model_contract(model)
    if payload.get("contract") != expected:
        raise ValueError(f"checkpoint contract mismatch: {payload.get('contract')} != {expected}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint contains no state_dict")
    model.load_state_dict(state_dict, strict=strict)
    return {
        "schema_version": payload["schema_version"],
        "contract": payload["contract"],
        "source": payload.get("source", {}),
        "training": payload.get("training", {}),
    }
