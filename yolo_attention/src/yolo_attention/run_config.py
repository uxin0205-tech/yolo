"""Training resource configuration kept separate from attention variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingRecipe:
    stage: str
    weights: str
    data: str
    epochs: int
    batch: int
    imgsz: int
    device: str
    workers: int
    seed: int
    patience: int
    optimizer: str
    lr0: float
    amp: bool

    def __post_init__(self) -> None:
        valid_stages = {
            "screening",
            "recovery",
            "normalization",
            "bias",
            "scale",
            "bdcn_codebook",
            "q2",
        }
        if self.stage not in valid_stages:
            raise ValueError(f"unknown training stage {self.stage!r}")
        if self.epochs < 1:
            raise ValueError("training recipe epochs must be positive")
        if min(self.batch, self.imgsz) < 1 or self.workers < 0:
            raise ValueError("batch/imgsz must be positive and workers non-negative")
        if self.patience < 0 or self.lr0 <= 0:
            raise ValueError("patience must be non-negative and lr0 positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_ultralytics_args(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("stage")
        data.pop("weights")
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingRecipe:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("training YAML must contain a mapping")
        return cls(**data)

    def to_yaml(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return destination
