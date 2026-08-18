"""Validated common experiment and phase configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .phases import PHASES, PhaseSpec


@dataclass(frozen=True)
class CommonTrainingConfig:
    data: str
    batch: int
    imgsz: int
    device: str
    workers: int
    seed: int
    amp: bool
    deterministic: bool
    optimizer: str
    nbs: int
    end2end: bool
    warmup_momentum: float
    warmup_bias_lr: float
    close_mosaic: int
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    flipud: float
    fliplr: float
    bgr: float
    mosaic: float
    mixup: float
    cutmix: float
    copy_paste: float

    def __post_init__(self) -> None:
        if self.batch < 1 or self.imgsz < 1 or self.workers < 0:
            raise ValueError("batch/imgsz must be positive and workers non-negative")
        if self.optimizer != "MuSGD":
            raise ValueError("formal experiments require optimizer=MuSGD")
        if self.nbs != self.batch:
            raise ValueError("nbs must equal batch so gradient accumulation remains disabled")
        if not self.end2end:
            raise ValueError("end2end=True is mandatory")

    @classmethod
    def from_yaml(cls, path: str | Path) -> CommonTrainingConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("common training YAML must contain a mapping")
        return cls(**payload)

    def to_ultralytics_args(self, phase: PhaseSpec, *, project: Path, name: str) -> dict[str, Any]:
        args = asdict(self)
        args.update(
            epochs=phase.epochs,
            patience=phase.patience,
            lr0=phase.learning_rates["masf"],
            lrf=phase.lrf,
            momentum=phase.momentum,
            weight_decay=phase.weight_decay,
            warmup_epochs=phase.warmup_epochs,
            cos_lr=phase.cosine,
            project=str(project.resolve()),
            name=name,
            exist_ok=False,
            resume=False,
        )
        return args


def load_phase_spec(path: str | Path) -> PhaseSpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("phase YAML must contain a mapping")
    spec = PhaseSpec(**payload)
    expected = PHASES.get(spec.name)
    if expected is not None and spec != expected:
        raise ValueError(f"{spec.name} YAML differs from the fixed experiment contract")
    return spec
