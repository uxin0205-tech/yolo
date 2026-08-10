"""Immutable contracts for the paper-aligned B1R/P2/P3 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Mapping

import yaml

from ..contracts import CLASS_NAMES, PIPELINE_SCHEMA_VERSION, SPLIT_RATIOS, sha256_value

RETEST_PIPELINE_NAME = "b1r-p2-p3-retest"
P2_VARIANTS = ("PaperFormula-Full", "Lite-35", "Lite-35-F7", "Partial50-35", "Partial25-35")
P3_VARIANTS = P2_VARIANTS
ALL_VARIANTS = tuple(f"P2-{name}" for name in P2_VARIANTS) + tuple(f"P3-{name}" for name in P3_VARIANTS)


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """A named MFAM adaptation with an explicit paper-fidelity boundary."""

    display_name: str
    kernels: tuple[int, ...]
    processed_ratio: float
    formula_version: str = "paper-equations-1-6"

    @property
    def config_hash(self) -> str:
        return sha256_value(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "kernels": list(self.kernels),
            "processed_ratio": self.processed_ratio,
            "formula_version": self.formula_version,
        }


VARIANT_SPECS = {
    "PaperFormula-Full": VariantSpec("PaperFormula-Full", (3, 5, 7, 9), 1.0),
    "Lite-35": VariantSpec("Lite-35", (3, 5), 1.0),
    "Lite-35-F7": VariantSpec("Lite-35-F7", (3, 5, 7), 1.0),
    "Partial50-35": VariantSpec("Partial50-35", (3, 5), 0.5),
    "Partial25-35": VariantSpec("Partial25-35", (3, 5), 0.25),
}


def _reject_unknown(section: str, raw: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {section} keys: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class RetestConfig:
    """Validated YAML mapping; the canonical mapping is retained for hashing."""

    values: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RetestConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("retest config must be a mapping")
        root_allowed = {"schema_version", "pipeline_name", "artifacts_root", "environment", "dataset", "model", "training", "profiling", "pipeline", "variants"}
        _reject_unknown("root", raw, root_allowed)
        if raw.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PIPELINE_SCHEMA_VERSION}")
        if raw.get("pipeline_name") != RETEST_PIPELINE_NAME:
            raise ValueError(f"pipeline_name must be {RETEST_PIPELINE_NAME}")
        required = ("environment", "dataset", "model", "training", "profiling", "pipeline", "variants")
        if any(not isinstance(raw.get(key), Mapping) for key in required[:-1]):
            raise ValueError("environment/dataset/model/training/profiling/pipeline must be mappings")
        variants = raw.get("variants")
        if not isinstance(variants, Mapping) or tuple(variants.get("p2", ())) != P2_VARIANTS or tuple(variants.get("p3", ())) != P3_VARIANTS:
            raise ValueError("p2 and p3 variant order must match the locked five-variant matrix")
        dataset = raw["dataset"]
        _reject_unknown("dataset", dataset, {"source", "locked_artifacts", "split_ratios", "class_names", "seed"})
        if dataset.get("source") != "bbt5-detect-baseline/dataset":
            raise ValueError("dataset source must be bbt5-detect-baseline/dataset")
        if tuple(dataset.get("split_ratios", ())) != SPLIT_RATIOS or tuple(dataset.get("class_names", ())) != CLASS_NAMES:
            raise ValueError("dataset must use the locked 80/10/10 ball/bat contract")
        training = raw["training"]
        _reject_unknown("training", training, {"optimizer", "momentum", "cos_lr", "deterministic", "amp", "nbs", "batch", "seed", "b1_a_epochs", "b1_b_epochs", "direct_epochs", "smoke_epochs", "formal_epochs", "b1_a_lr0", "formal_lr0", "freeze"})
        if training.get("optimizer") != "SGD" or training.get("momentum") != 0.937 or training.get("cos_lr") is not True:
            raise ValueError("training optimizer contract must be SGD/momentum .937/cosine")
        if tuple(training.get("freeze", ())) != tuple(range(11)):
            raise ValueError("B1R-A must freeze backbone indices 0-10")
        values = deepcopy(dict(raw))
        return cls(values)

    @property
    def config_hash(self) -> str:
        return sha256_value(self.values)

    def variant(self, family: str, name: str) -> VariantSpec:
        if family not in {"P2", "P3"}:
            raise ValueError("family must be P2 or P3")
        try:
            return VARIANT_SPECS[name]
        except KeyError as error:
            raise ValueError(f"unsupported MFAM variant: {name}") from error


def load_retest_config(path):
    """Load and validate a retest YAML without changing Phase 1 loader rules."""
    from pathlib import Path

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return RetestConfig.from_mapping(raw)
