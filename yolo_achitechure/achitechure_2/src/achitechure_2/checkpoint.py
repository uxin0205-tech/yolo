"""候選 state_dict checkpoint、builder contract 與 lineage。"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .candidate import CandidateBuild
from .config import SPEC_VERSION

SCHEMA_VERSION = 1
LINEAGE_FIELDS = frozenset(
    {
        "spec_version",
        "spec_sha256",
        "handoff_revision",
        "handoff_manifest_sha256",
        "architecture_yaml_sha256",
        "training_yaml_sha256",
        "dataset_yaml_sha256",
        "parent_checkpoint_sha256",
        "candidate_id",
        "resolved_candidate_id",
    }
)
HASH_FIELDS = frozenset(
    {
        "spec_sha256",
        "handoff_manifest_sha256",
        "architecture_yaml_sha256",
        "training_yaml_sha256",
        "dataset_yaml_sha256",
        "parent_checkpoint_sha256",
    }
)


def _model_contract(model: nn.Module) -> dict[str, Any]:
    method = getattr(model, "contract", None)
    if not callable(method):
        raise TypeError("checkpoint model 必須提供 contract()")
    contract = method()
    if not isinstance(contract, dict):
        raise TypeError("model contract 必須是 mapping")
    return contract


def _architecture_contract(model: nn.Module) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in model.state_dict().items()
    }


def _valid_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_lineage(
    lineage: Mapping[str, str],
    report: CandidateBuild,
) -> dict[str, str]:
    payload = dict(lineage)
    if set(payload) != LINEAGE_FIELDS:
        raise ValueError(
            "checkpoint lineage 欄位不完整："
            f"missing={sorted(LINEAGE_FIELDS - set(payload))} "
            f"unknown={sorted(set(payload) - LINEAGE_FIELDS)}"
        )
    if payload["spec_version"] != SPEC_VERSION:
        raise ValueError("checkpoint lineage spec_version 漂移")
    if any(not _valid_hash(payload[name]) for name in HASH_FIELDS):
        raise ValueError("checkpoint lineage 含無效 SHA256")
    if (
        payload["candidate_id"] != report.candidate_id
        or payload["resolved_candidate_id"] != report.resolved_id
    ):
        raise ValueError("checkpoint lineage candidate IDs 與 build report 不一致")
    if not payload["handoff_revision"]:
        raise ValueError("checkpoint lineage 缺少 handoff revision")
    return payload


def save_candidate_checkpoint(
    destination: str | Path,
    model: nn.Module,
    report: CandidateBuild,
    *,
    lineage: Mapping[str, str],
) -> Path:
    """原子寫入 tensors 與重建資訊，不 pickle model instance。"""

    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "builder": "achitechure_2",
        "candidate_id": report.candidate_id,
        "resolved_candidate_id": report.resolved_id,
        "fusion_kind": report.fusion_kind,
        "region_id": report.region_id,
        "changed_fields": list(report.changed_fields),
        "changed_module_paths": list(report.changed_module_paths),
        "model_contract": _model_contract(model),
        "architecture_contract": _architecture_contract(model),
        "lineage": _validate_lineage(lineage, report),
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
    }
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def load_candidate_checkpoint(
    checkpoint: str | Path,
    builder: Callable[[], nn.Module],
    *,
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """由 builder 重建 graph，再驗證 contract 並載入 state_dict。"""

    path = Path(checkpoint).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支援或損壞的 candidate checkpoint")
    if payload.get("builder") != "achitechure_2":
        raise ValueError("checkpoint builder contract mismatch")
    model = builder()
    if payload.get("model_contract") != _model_contract(model):
        raise ValueError("checkpoint model contract mismatch")
    if payload.get("architecture_contract") != _architecture_contract(model):
        raise ValueError("checkpoint architecture contract mismatch")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint 沒有 state_dict mapping")
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as error:
        raise ValueError(f"checkpoint state_dict contract mismatch: {error}") from error
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"state_dict", "architecture_contract", "model_contract"}
    }
    metadata["model_contract"] = payload["model_contract"]
    return model, metadata
