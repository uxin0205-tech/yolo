"""Explicit, auditable weight maps for official and canonical sources."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from masf_yolo.variants import sp2p_variant_id


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


@dataclass(frozen=True, slots=True)
class SP2PTransferReport:
    selected_partial: str
    shared_keys: tuple[str, ...]
    partial_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_partial": self.selected_partial,
            "shared_keys": list(self.shared_keys),
            "partial_keys": list(self.partial_keys),
        }


def transfer_b0_p3_parent(
    destination: nn.Module,
    source: nn.Module | Mapping[str, Tensor] | Path,
    *,
    slot_prefix: str = "model.16.",
) -> TransferReport:
    """Copy the B0 parent while deliberately leaving the P3 MFAM slot random."""
    source_state = _state_dict(source)

    def key_mapper(destination_key: str) -> str | None:
        if destination_key.startswith(slot_prefix + "1."):
            return None
        if destination_key.startswith(slot_prefix + "0."):
            return destination_key.replace(slot_prefix + "0.", slot_prefix, 1)
        return destination_key

    report = _apply_mapping(destination, source_state, key_mapper)
    # The source's replaced P3 tensors are an explicit, auditable adaptation,
    # not an unexpected key or a shape mismatch.
    skipped = tuple(sorted(key for key in source_state if key.startswith(slot_prefix)))
    return TransferReport(
        matched=report.matched,
        missing=report.missing,
        unexpected=tuple(sorted(set(report.unexpected) - set(skipped))),
        shape_mismatch=report.shape_mismatch,
    )


_LAYER_KEY = re.compile(r"^model\.(\d+)\.(.+)$")
_DETECT_BRANCH = re.compile(r"^(cv[23])\.(\d+)\.(.+)$")
_SELECTIVE_MAIN_BRANCH = re.compile(r"^model\.31\.main_(cv[23])\.(\d+)\.(.+)$")
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
    def source_key(destination_key: str) -> str | None:
        selective = _SELECTIVE_MAIN_BRANCH.match(destination_key)
        if selective:
            tower, branch, suffix = selective.groups()
            return f"model.31.{tower}.{int(branch) + 1}.{suffix}"
        if destination_key.startswith("model.31.ball_cv"):
            return None
        return destination_key

    return _apply_mapping(destination, _state_dict(source), source_key)


def transfer_sp2p_parents(
    destination: nn.Module,
    sp2_source: nn.Module | Mapping[str, Tensor] | Path,
    partial_source: nn.Module | Mapping[str, Tensor] | Path,
    *,
    selected_partial: str,
) -> SP2PTransferReport:
    """Load SP2 shared tensors and only the selected parent's P2 partial-MFAM slot."""
    expected_variant = sp2p_variant_id(selected_partial)
    actual_variant = getattr(destination, "masf_variant", None)
    if actual_variant != expected_variant:
        raise ValueError(
            f"SP2P target variant {actual_variant} does not match selected parent {selected_partial}"
        )

    destination_state = destination.state_dict()
    sp2_state = _state_dict(sp2_source)
    partial_state = _state_dict(partial_source)
    shared_keys = {key for key in destination_state if not key.startswith("model.20.")}
    missing_shared = shared_keys - set(sp2_state)
    if missing_shared:
        raise ValueError(f"SP2P shared keys do not match SP2 parent: {sorted(missing_shared)}")

    slot_keys = {key for key in destination_state if key.startswith("model.20.")}
    source_slot_keys = {key for key in partial_state if key.startswith("model.20.")}
    if not slot_keys or slot_keys != source_slot_keys:
        raise ValueError("SP2P partial slot keys do not match selected parent")

    for key in sorted(shared_keys):
        source_tensor = sp2_state[key]
        if source_tensor.shape != destination_state[key].shape:
            raise ValueError(f"SP2P shared shape mismatch: {key}")
        destination_state[key] = source_tensor.detach().to(
            device=destination_state[key].device,
            dtype=destination_state[key].dtype,
        ).clone()
    for key in sorted(slot_keys):
        source_tensor = partial_state[key]
        if source_tensor.shape != destination_state[key].shape:
            raise ValueError(f"SP2P partial slot shape mismatch: {key}")
        destination_state[key] = source_tensor.detach().to(
            device=destination_state[key].device,
            dtype=destination_state[key].dtype,
        ).clone()

    destination.load_state_dict(destination_state, strict=True)
    return SP2PTransferReport(
        selected_partial=selected_partial,
        shared_keys=tuple(sorted(shared_keys)),
        partial_keys=tuple(sorted(slot_keys)),
    )
