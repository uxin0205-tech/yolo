"""Canonical CPU-float32 state-dict checkpoint storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from masf_yolo.contracts import (
    PIPELINE_SCHEMA_VERSION,
    CheckpointManifest,
    sha256_file,
)
from masf_yolo.variants import VariantDefinition

from .io import atomic_write_json


def _canonical_state_dict(model: nn.Module) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for key, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point():
            value = value.float()
        result[key] = value.clone()
    return result


def state_dict_hash(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _manifest_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".manifest.json")


def save_canonical_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    variant: VariantDefinition,
    *,
    data_hash: str,
    config_hash: str,
    environment_hash: str,
) -> CheckpointManifest:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = _canonical_state_dict(model)
    state_hash = state_dict_hash(state_dict)
    metadata = {
        "variant_id": variant.variant_id,
        "variant_hash": variant.config_hash,
        "data_hash": data_hash,
        "config_hash": config_hash,
        "environment_hash": environment_hash,
        "state_dict_hash": state_hash,
    }
    payload = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "metadata": metadata,
        "state_dict": state_dict,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint_path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    manifest = CheckpointManifest(
        variant_id=variant.variant_id,
        variant_hash=variant.config_hash,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=sha256_file(checkpoint_path),
        state_dict_hash=state_hash,
        data_hash=data_hash,
        config_hash=config_hash,
        environment_hash=environment_hash,
    )
    atomic_write_json(_manifest_path(checkpoint_path), manifest.to_dict())
    return manifest


def load_canonical_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    variant: VariantDefinition,
    *,
    expected_data_hash: str | None = None,
    expected_config_hash: str | None = None,
    expected_environment_hash: str | None = None,
) -> CheckpointManifest:
    checkpoint_path = checkpoint_path.resolve()
    manifest = CheckpointManifest.from_dict(
        json.loads(_manifest_path(checkpoint_path).read_text(encoding="utf-8"))
    )
    if manifest.checkpoint_path.resolve() != checkpoint_path:
        raise ValueError("checkpoint path does not match manifest")
    if sha256_file(checkpoint_path) != manifest.checkpoint_hash:
        raise ValueError("checkpoint hash does not match manifest")
    if manifest.variant_id != variant.variant_id or manifest.variant_hash != variant.config_hash:
        raise ValueError("checkpoint variant does not match requested variant")
    expected = {
        "data": (expected_data_hash, manifest.data_hash),
        "config": (expected_config_hash, manifest.config_hash),
        "environment": (expected_environment_hash, manifest.environment_hash),
    }
    for name, (wanted, actual) in expected.items():
        if wanted is not None and wanted != actual:
            raise ValueError(f"checkpoint {name} hash does not match")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    metadata = payload.get("metadata", {})
    for key in ("variant_id", "variant_hash", "data_hash", "config_hash", "environment_hash", "state_dict_hash"):
        if metadata.get(key) != getattr(manifest, key):
            raise ValueError(f"checkpoint metadata mismatch: {key}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint state_dict is missing")
    if state_dict_hash(state_dict) != manifest.state_dict_hash:
        raise ValueError("checkpoint state-dict hash does not match")
    model.load_state_dict(state_dict, strict=True)
    return manifest
