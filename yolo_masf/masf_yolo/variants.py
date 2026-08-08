"""Locked Phase 1 variant definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .contracts import sha256_value


@dataclass(frozen=True, slots=True)
class VariantDefinition:
    variant_id: str
    kernel_branches: tuple[int, ...]
    processed_ratio: float
    p2_slot: str
    p3_slot: str = "identity"

    @property
    def config_hash(self) -> str:
        return sha256_value(asdict(self))


_VARIANTS = {
    "B1": VariantDefinition("B1", (), 0.0, "identity"),
    "M7": VariantDefinition("M7", (3, 5, 7), 1.0, "mfam"),
    "M0": VariantDefinition("M0", (3, 5, 7, 9), 1.0, "mfam"),
    "M1": VariantDefinition("M1", (3, 5), 1.0, "mfam"),
    "M2": VariantDefinition("M2", (3, 5), 0.5, "partial_mfam"),
    "M3": VariantDefinition("M3", (3, 5), 0.25, "partial_mfam"),
}

VARIANTS: Mapping[str, VariantDefinition] = MappingProxyType(_VARIANTS)

CORE_VARIANTS = ("B1", "M0", "M1", "M2", "M3")
PRIORITY_VARIANTS = ("M7",)
TRAINED_VARIANTS = ("B1", "M7", "M0", "M1", "M2", "M3")
EVALUATED_MODELS = ("B0", *TRAINED_VARIANTS)
SELECTION_CANDIDATES = ("M2", "M3")


@dataclass(frozen=True, slots=True)
class PriorityVariantManifest:
    variant_id: str
    kernels: tuple[int, ...]
    processed_ratio: float
    smoke_epochs: int
    formal_epochs: int
    priority_before: str

    @property
    def manifest_hash(self) -> str:
        return sha256_value(asdict(self))


_PRIORITY_KEYS = frozenset(
    {"variant_id", "kernels", "processed_ratio", "smoke_epochs", "formal_epochs", "priority_before"}
)


def load_priority_manifest(path: Path) -> PriorityVariantManifest:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("M7 priority manifest must be a mapping")
    unknown = set(raw) - _PRIORITY_KEYS
    if unknown:
        raise ValueError(f"unknown M7 priority keys: {sorted(unknown)}")
    missing = _PRIORITY_KEYS - set(raw)
    if missing:
        raise ValueError(f"missing M7 priority keys: {sorted(missing)}")
    manifest = PriorityVariantManifest(
        variant_id=str(raw["variant_id"]),
        kernels=tuple(int(kernel) for kernel in raw["kernels"]),
        processed_ratio=float(raw["processed_ratio"]),
        smoke_epochs=int(raw["smoke_epochs"]),
        formal_epochs=int(raw["formal_epochs"]),
        priority_before=str(raw["priority_before"]),
    )
    definition = get_variant(manifest.variant_id)
    expected = (definition.kernel_branches, definition.processed_ratio, 3, 100, "M0")
    actual = (
        manifest.kernels,
        manifest.processed_ratio,
        manifest.smoke_epochs,
        manifest.formal_epochs,
        manifest.priority_before,
    )
    if manifest.variant_id != "M7" or actual != expected:
        raise ValueError("M7 priority manifest does not match the locked variant contract")
    return manifest


def get_variant(variant_id: str) -> VariantDefinition:
    try:
        return VARIANTS[variant_id]
    except KeyError as error:
        raise ValueError(f"unsupported variant: {variant_id}") from error
