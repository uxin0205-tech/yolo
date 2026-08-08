"""Locked Phase 1 variant definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

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
    "M0": VariantDefinition("M0", (3, 5, 7, 9), 1.0, "mfam"),
    "M1": VariantDefinition("M1", (3, 5), 1.0, "mfam"),
    "M2": VariantDefinition("M2", (3, 5), 0.5, "partial_mfam"),
    "M3": VariantDefinition("M3", (3, 5), 0.25, "partial_mfam"),
}

VARIANTS: Mapping[str, VariantDefinition] = MappingProxyType(_VARIANTS)


def get_variant(variant_id: str) -> VariantDefinition:
    try:
        return VARIANTS[variant_id]
    except KeyError as error:
        raise ValueError(f"unsupported variant: {variant_id}") from error
