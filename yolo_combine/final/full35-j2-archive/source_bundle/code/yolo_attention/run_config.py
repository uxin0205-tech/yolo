"""最終 PWL workflow 使用的 immutable training recipe。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

TRAINABLE_SCOPES = frozenset(
    {
        "bias_only",
        "qk_recovery",
        "attention_refinement",
        "block_recovery",
        "neck_recovery",
        "backbone_last_recovery",
        "full_model_recovery",
    }
)
LAYER_LR_GROUPS = frozenset({"attention", "adjacent_block", "neck_detect", "backbone"})
SCOPE_LR_GROUPS = {
    "block_recovery": frozenset({"attention", "adjacent_block"}),
    "neck_recovery": frozenset({"attention", "adjacent_block", "neck_detect"}),
    "backbone_last_recovery": LAYER_LR_GROUPS,
    "full_model_recovery": LAYER_LR_GROUPS,
}
MAP_SELECTION_METRIC = "metrics/mAP50-95(B)"


@dataclass(frozen=True)
class TrainingRecipe:
    phase: str
    parent: str
    trainable_scope: str
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
    scheduler: str
    lrf: float
    warmup_epochs: float
    warmup_bias_lr: float
    weight_decay: float
    selection_metric: str
    amp: bool
    deterministic: bool
    layer_lrs: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.trainable_scope not in TRAINABLE_SCOPES:
            raise ValueError(f"unknown trainable_scope {self.trainable_scope!r}")
        if not self.phase or not self.parent:
            raise ValueError("phase and parent must be explicit")
        if self.epochs < 1 or min(self.batch, self.imgsz) < 1 or self.workers < 0:
            raise ValueError("invalid epochs/batch/imgsz/workers")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.lr0 <= 0 or self.lrf <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer values")
        if self.scheduler != "constant" or self.lrf != 1.0:
            raise ValueError("short phases require a constant scheduler (lrf=1.0)")
        if self.warmup_epochs != 0 or self.warmup_bias_lr != 0:
            raise ValueError("warmup must be disabled")
        if self.selection_metric != MAP_SELECTION_METRIC:
            raise ValueError(f"selection_metric must be {MAP_SELECTION_METRIC!r}")
        layer_lrs = self.layer_lrs or {}
        if any(name not in LAYER_LR_GROUPS for name in layer_lrs):
            raise ValueError(f"unknown layer LR group: {sorted(set(layer_lrs) - LAYER_LR_GROUPS)}")
        if any(not isinstance(value, (int, float)) or value <= 0 for value in layer_lrs.values()):
            raise ValueError("layer learning rates must be positive numbers")
        expected = SCOPE_LR_GROUPS.get(self.trainable_scope, frozenset())
        if set(layer_lrs) != expected:
            raise ValueError(
                f"scope {self.trainable_scope!r} requires layer_lrs {sorted(expected)}, "
                f"got {sorted(layer_lrs)}"
            )

    @property
    def stage(self) -> str:
        return self.trainable_scope

    def with_seed_and_lr(self, *, seed: int, lr0: float) -> TrainingRecipe:
        return replace(self, seed=seed, lr0=lr0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_ultralytics_args(self) -> dict[str, Any]:
        data = self.to_dict()
        for key in (
            "phase",
            "parent",
            "trainable_scope",
            "weights",
            "scheduler",
            "selection_metric",
            "layer_lrs",
        ):
            data.pop(key)
        data["cos_lr"] = False
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
