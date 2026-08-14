"""Fair training profiles for clean-initializer models."""

from __future__ import annotations

from typing import Any

from ultralytics.cfg import DEFAULT_CFG_DICT

from .contracts import CLEAN_EXPERIMENTS


_HEAD_ONLY_FREEZE = [
    *range(20), *range(21, 31),
    "31.cv2.1", "31.cv2.2", "31.cv2.3",
    "31.cv3.1", "31.cv3.2", "31.cv3.3", "31.dfl",
]


def clean_profile(
    experiment: str,
    *,
    seed: int,
    model: str,
    data: str,
    project: str,
) -> dict[str, Any]:
    if seed not in {42, 43, 44}:
        raise ValueError("clean profiles require seed 42, 43, or 44")
    try:
        spec = CLEAN_EXPERIMENTS[experiment]
    except KeyError as error:
        raise ValueError(f"unsupported clean experiment: {experiment}") from error
    schedule = {
        "direct100": (100, 0.001, None),
        "head20": (20, 0.01, _HEAD_ONLY_FREEZE),
        "full80": (80, 0.001, None),
    }[spec.schedule]
    values = dict(DEFAULT_CFG_DICT)
    values.update({
        "task": "detect", "mode": "train", "model": model, "data": data,
        "project": project, "name": f"{experiment.lower()}-seed{seed}",
        "epochs": schedule[0], "imgsz": 640, "batch": 16, "device": 0,
        "optimizer": "SGD", "lr0": schedule[1], "momentum": 0.937,
        "cos_lr": True, "seed": seed, "deterministic": True, "amp": True,
        "nbs": 64, "freeze": schedule[2],
    })
    return values
