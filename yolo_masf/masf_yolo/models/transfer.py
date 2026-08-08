"""Explicit, auditable weight maps for official and canonical sources."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ShapeMismatch:
    source_key: str
    source_shape: tuple[int, ...]
    destination_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TransferReport:
    matched: dict[str, str]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: dict[str, ShapeMismatch]

    def to_dict(self) -> dict[str, object]:
        return {
            "matched": self.matched,
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "shape_mismatch": {
                key: {
                    "source_key": value.source_key,
                    "source_shape": list(value.source_shape),
                    "destination_shape": list(value.destination_shape),
                }
                for key, value in self.shape_mismatch.items()
            },
        }


_LAYER_KEY = re.compile(r"^model\.(\d+)\.(.+)$")
_DETECT_BRANCH = re.compile(r"^(cv[23])\.(\d+)\.(.+)$")
_OFFICIAL_LAYER_MAP = {
    **{index: index for index in range(17)},
    25: 17,
    27: 19,
    28: 20,
    30: 22,
}


def _official_source_key(destination_key: str) -> str | None:
    match = _LAYER_KEY.match(destination_key)
    if not match:
        return None
    destination_layer = int(match.group(1))
    remainder = match.group(2)
    if destination_layer in _OFFICIAL_LAYER_MAP:
        return f"model.{_OFFICIAL_LAYER_MAP[destination_layer]}.{remainder}"
    if destination_layer != 31:
        return None
    if remainder.startswith("dfl."):
        return f"model.23.{remainder}"
    branch = _DETECT_BRANCH.match(remainder)
    if not branch:
        return None
    tower, destination_branch, suffix = branch.groups()
    branch_index = int(destination_branch)
    if branch_index == 0:
        return None
    return f"model.23.{tower}.{branch_index - 1}.{suffix}"


def _state_dict(source: nn.Module | Mapping[str, Tensor] | Path) -> Mapping[str, Tensor]:
    if isinstance(source, Path):
        from ultralytics import YOLO

        return YOLO(str(source)).model.state_dict()
    if isinstance(source, nn.Module):
        return source.state_dict()
    return source


def _apply_mapping(
    destination: nn.Module,
    source: Mapping[str, Tensor],
    key_mapper,
) -> TransferReport:
    destination_state = destination.state_dict()
    matched: dict[str, str] = {}
    missing: list[str] = []
    mismatches: dict[str, ShapeMismatch] = {}
    consumed_sources: set[str] = set()
    for destination_key, destination_tensor in destination_state.items():
        source_key = key_mapper(destination_key)
        if source_key is None or source_key not in source:
            missing.append(destination_key)
            continue
        source_tensor = source[source_key]
        consumed_sources.add(source_key)
        if source_tensor.shape != destination_tensor.shape:
            mismatches[destination_key] = ShapeMismatch(
                source_key=source_key,
                source_shape=tuple(source_tensor.shape),
                destination_shape=tuple(destination_tensor.shape),
            )
            continue
        destination_state[destination_key] = source_tensor.detach().to(
            device=destination_tensor.device, dtype=destination_tensor.dtype
        ).clone()
        matched[destination_key] = source_key
    destination.load_state_dict(destination_state, strict=True)
    return TransferReport(
        matched=matched,
        missing=tuple(sorted(missing)),
        unexpected=tuple(sorted(set(source) - consumed_sources)),
        shape_mismatch=mismatches,
    )


def transfer_official_weights(
    destination: nn.Module,
    source: nn.Module | Mapping[str, Tensor] | Path,
) -> TransferReport:
    """Map official YOLO11m backbone, shared neck, and P3–P5 Detect tensors."""
    return _apply_mapping(destination, _state_dict(source), _official_source_key)


def transfer_b1_canonical(
    destination: nn.Module,
    source: nn.Module | Mapping[str, Tensor] | Path,
) -> TransferReport:
    """Load same-key B1 tensors while leaving variant-owned MFAM tensors random."""
    return _apply_mapping(destination, _state_dict(source), lambda key: key)
