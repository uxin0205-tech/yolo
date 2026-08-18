"""Immutable experiment and training configuration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .lite_c3k2 import KernelMode, LiteC3k2Config

TARGET_LAYERS = (6, 8, 13, 19)
P5_ONLY_LAYER = (8,)
PROTECTED_C3K2_LAYERS = (2, 4, 16, 22)
DETECT_INPUTS = (16, 19, 22)
ATTENTION_PATHS = ("model.10.m.0.attn", "model.22.m.0.1.attn")
STRIDES = (8, 16, 32)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    target_layers: tuple[int, ...]
    lite: LiteC3k2Config | None
    base_candidate: str | None = None


CANDIDATES: dict[str, CandidateSpec] = {
    "C0": CandidateSpec("C0", (), None),
    "C1": CandidateSpec("C1", TARGET_LAYERS, LiteC3k2Config(e=0.375)),
    "C2": CandidateSpec("C2", TARGET_LAYERS, LiteC3k2Config(inner_n=1)),
    "C3": CandidateSpec("C3", TARGET_LAYERS, LiteC3k2Config(kernel_mode=KernelMode.K1_K3)),
    "C3-P5": CandidateSpec("C3-P5", P5_ONLY_LAYER, LiteC3k2Config(kernel_mode=KernelMode.K1_K3)),
    "R1": CandidateSpec("R1", TARGET_LAYERS, LiteC3k2Config(inner_n=1, use_rep=True), "C2"),
}


@dataclass(frozen=True)
class TrainingRecipe:
    data: str
    imgsz: int = 640
    batch: int = 16
    seed: int = 0
    optimizer: str = "MuSGD"
    device: str = "0"
    workers: int = 8
    amp: bool = True
    deterministic: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingRecipe:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: payload[key] for key in allowed if key in payload})


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a mapping in {path}")
    return payload


def config_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
