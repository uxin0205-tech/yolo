"""Resolve Phase 1 runs against pinned Ultralytics defaults."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ultralytics.cfg import DEFAULT_CFG_DICT


_COMMON = {
    "task": "detect",
    "mode": "train",
    "data": "artifacts/static-phase1/dataset/data.yaml",
    "imgsz": 640,
    "batch": None,
    "device": 0,
    "optimizer": "SGD",
    "momentum": 0.937,
    "cos_lr": True,
    "seed": 42,
    "deterministic": True,
    "amp": True,
    "nbs": 64,
    "lr0": 0.001,
    "freeze": None,
}


def _resolved(overrides: Mapping[str, Any]) -> dict[str, Any]:
    profile = deepcopy(DEFAULT_CFG_DICT)
    profile.update(_COMMON)
    profile.update(overrides)
    return profile


def formal_profile(
    variant_id: str,
    model_path: str,
    project: str,
    *,
    epochs: int = 100,
) -> dict[str, Any]:
    return _resolved(
        {
            "model": model_path,
            "project": project,
            "name": variant_id.lower(),
            "epochs": epochs,
        }
    )


def b1_a_profile(model_path: str, project: str) -> dict[str, Any]:
    return frozen_stage_profile("B1", model_path, project, name="b1-a")


def frozen_stage_profile(
    variant_id: str,
    model_path: str,
    project: str,
    *,
    name: str,
) -> dict[str, Any]:
    profile = formal_profile(variant_id, model_path, project, epochs=10)
    profile.update(
        {
            "name": name,
            "lr0": 0.01,
            "freeze": list(range(11)),
        }
    )
    return profile


def smoke_profile(variant_id: str, model_path: str, project: str) -> dict[str, Any]:
    return formal_profile(variant_id, model_path, project, epochs=3) | {
        "name": f"{variant_id.lower()}-smoke"
    }


def profile_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    return {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
