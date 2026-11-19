"""Locked formal stage profiles for architecture-matched Pose26 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PoseStageName = Literal["smoke", "p1", "p2", "p3"]


@dataclass(frozen=True)
class PoseStageSpec:
    name: PoseStageName
    description: str
    epochs: int
    imgsz: int
    batch: int
    workers: int
    fraction: float
    val: bool
    plots: bool
    requires_initial_checkpoint: bool
    overrides: dict[str, Any]

    def trainer_overrides(self) -> dict[str, Any]:
        values = dict(self.overrides)
        freeze = values.get("freeze")
        if isinstance(freeze, tuple):
            values["freeze"] = list(freeze)
        return values

    def validate_transition(self, checkpoint: str | Path | None) -> Path | None:
        if checkpoint is None:
            if self.requires_initial_checkpoint:
                raise ValueError(f"Pose stage {self.name} requires --initial-checkpoint")
            return None
        path = Path(checkpoint).expanduser().resolve()
        if not self.requires_initial_checkpoint:
            raise ValueError(f"Pose stage {self.name} must start fresh")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


_FORMAL_COMMON: dict[str, Any] = {
    "optimizer": "MuSGD",
    "deterministic": True,
    "single_cls": False,
    "rect": False,
    "cos_lr": False,
    "close_mosaic": 10,
    "amp": True,
    "multi_scale": 0.0,
    "compile": False,
    "dropout": 0.0,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "nbs": 64,
    "end2end": True,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 1.0,
    "rle": 1.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "iou": 0.7,
    "max_det": 300,
}


def _formal(
    name: Literal["p1", "p2", "p3"],
    description: str,
    *,
    epochs: int,
    patience: int,
    freeze: int | tuple[int, ...] | None,
    requires_initial_checkpoint: bool,
) -> PoseStageSpec:
    return PoseStageSpec(
        name=name,
        description=description,
        epochs=epochs,
        imgsz=640,
        batch=128,
        workers=8,
        fraction=1.0,
        val=True,
        plots=True,
        requires_initial_checkpoint=requires_initial_checkpoint,
        overrides={**_FORMAL_COMMON, "patience": patience, "freeze": freeze},
    )


POSE_STAGES: dict[PoseStageName, PoseStageSpec] = {
    "smoke": PoseStageSpec(
        name="smoke",
        description="Minimal graph/data/backward/checkpoint validation only.",
        epochs=1,
        imgsz=128,
        batch=2,
        workers=0,
        fraction=0.001,
        val=False,
        plots=False,
        requires_initial_checkpoint=False,
        overrides={
            "optimizer": "SGD",
            "lr0": 1e-4,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "warmup_epochs": 0.0,
            "close_mosaic": 0,
        },
    ),
    "p1": _formal(
        "p1",
        "Train only the fresh Pose26 head.",
        epochs=17,
        patience=0,
        freeze=tuple(range(23)),
        requires_initial_checkpoint=False,
    ),
    "p2": _formal(
        "p2",
        "Load P1 best and train neck plus Pose26 head.",
        epochs=22,
        patience=0,
        freeze=11,
        requires_initial_checkpoint=True,
    ),
    "p3": _formal(
        "p3",
        "Load P2 best and fine-tune all except inherited attention/MASF.",
        epochs=100,
        patience=20,
        freeze=None,
        requires_initial_checkpoint=True,
    ),
}


def pose_stage(name: str) -> PoseStageSpec:
    try:
        return POSE_STAGES[name]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(f"unknown Pose stage {name!r}; expected {tuple(POSE_STAGES)}") from error
