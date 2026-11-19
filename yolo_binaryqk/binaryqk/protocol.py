"""先核對 optimize 血緣，再執行初篩及選定分支恢復。"""

import json

from .model import ROOT, sha256

REVIEW = ROOT / "yolo_binaryqk/artifacts/stage1/reference-review-v2.json"
COMMON = {
    "version": 2, "batch": 16, "logical_batch": 64, "imgsz": 640, "seed": 0,
    "optimizer": "AdamW", "lr0": 5e-5, "betas": [.937, .999], "weight_decay": .0005,
    "warmup_epochs": 3, "warmup_bias_lr": 0., "linear_final_fraction": .01,
    "gradient_clip": 10., "amp": True, "initial_loss_scale": 1024., "overflow_attempts": 16,
    "dtype": "AMP-FP16; score/softmax/parameters/evaluation-FP32",
    "bn": "all BN running statistics update; frozen affine stays fixed",
    "loss": "native E2ELoss batch-sum; fixed 4 microbatches; retain partial final macro",
    "ema": "native decay; trainable parameters and BN buffers only",
    "normalization": "native softmax; fixed calibrated FP32 score coefficients",
    "selection": "best official bbox AP50_95 including epoch0; strict improvement",
    "resume": "last complete epoch; exact live/EMA/optimizer/scaler; repeat interrupted epoch",
    "reference_review_sha256": None,
}


def recipe_for(phase):
    if phase not in {"screening", "recovery"}:
        raise ValueError(f"unknown phase: {phase}")
    return {**COMMON, "phase": phase, "epochs": 10 if phase == "screening" else 40,
            "patience": 0 if phase == "screening" else 5,
            "scope": "attention" if phase == "screening" else "full",
            "reference_review_sha256": sha256(REVIEW)}


def verify_reference_review():
    review = json.loads(REVIEW.read_text())
    if review["schema"] != "binaryqk-reference-review-v2" or not review["reviewed_before_implementation"]:
        raise ValueError("missing reviewed optimize protocol")
    for name, digest in review["sources_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"reference changed; review applicability before launch: {name}")
    return review


def learning_rate(recipe, epoch, position):
    multiplier = 1 - (1 - recipe["linear_final_fraction"]) * epoch / recipe["epochs"]
    warmup = min(1., (epoch + position) / recipe["warmup_epochs"])
    return recipe["lr0"] * multiplier * warmup


def should_stop(recipe, epoch, best_epoch):
    return bool(recipe["patience"] and epoch - best_epoch >= recipe["patience"])
