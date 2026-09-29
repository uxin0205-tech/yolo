"""Attention-only QAT fine-tuning profile for the complete experiment matrix."""
from __future__ import annotations

from typing import Any


# BinaryAttention's released command uses FP-checkpoint QAT fine-tuning with
# batch 128, 300 epochs, lr 5e-5, min lr 5e-6 and weight decay 0.02. This plan
# preserves those optimization settings but uses the user-selected 10-epoch
# attention-only adaptation budget for every T/N variant on COCO/640.
PAPER_QAT_TRAINING_ARGS: dict[str, Any] = {
    "epochs": 10,
    "patience": 0,
    "batch": 128,
    "imgsz": 640,
    "save": True,
    "save_period": -1,
    "cache": "disk",
    "workers": 8,
    "optimizer": "AdamW",
    "pretrained": False,
    "seed": 0,
    "deterministic": True,
    "single_cls": False,
    "rect": False,
    "cos_lr": True,
    "close_mosaic": 0,
    "resume": False,
    "amp": True,
    "fraction": 1.0,
    "multi_scale": False,
    "overlap_mask": True,
    "mask_ratio": 4,
    "dropout": 0.0,
    "val": True,
    "split": "val",
    "save_json": False,
    "save_hybrid": False,
    "conf": None,
    "iou": 0.7,
    "max_det": 300,
    "half": False,
    "plots": True,
    "augment": False,
    "agnostic_nms": False,
    "classes": None,
    "retina_masks": False,
    "lr0": 5e-5,
    "lrf": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.02,
    "warmup_epochs": 0.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "nbs": 128,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.9,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "auto_augment": "randaugment",
    "erasing": 0.4,
    "crop_fraction": 1.0,
}

# Compatibility name used by artifact/report code.
FORMAL_TRAINING_ARGS = PAPER_QAT_TRAINING_ARGS

# Kept as explicit constants so reports can distinguish the original source
# checkpoint from the paper-aligned fine-tuning schedule.
BASELINE_EPOCHS = 600
PAPER_REFERENCE_EPOCHS = 300
FORMAL_EPOCHS = 10
PAPER_QAT_LR = 5e-5
PAPER_QAT_MIN_LR = 5e-6


def make_training_overrides(
    *,
    stage: str,
    model: str,
    data: str,
    device: str,
    project: str,
    name: str,
    seed: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Resolve the 10-epoch attention-only formal fine-tuning profile."""

    if stage != "full":
        raise ValueError("BinaryAttention formal training supports only stage='full'")
    overrides = dict(PAPER_QAT_TRAINING_ARGS)
    overrides.update(
        {
            "model": model,
            "data": data,
            "device": device,
            "project": project,
            "name": name,
            "exist_ok": False,
        }
    )
    if seed is not None:
        overrides["seed"] = seed
    if batch is not None:
        overrides["batch"] = batch
    if workers is not None:
        overrides["workers"] = workers
    return overrides
