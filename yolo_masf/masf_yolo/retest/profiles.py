"""Training profiles for the B1R/P2/P3 retest queue."""

from __future__ import annotations

from typing import Any

from ultralytics.cfg import DEFAULT_CFG_DICT


def _profile(*, model: str, project: str, name: str, epochs: int, lr0: float, freeze: list[int] | None) -> dict[str, Any]:
    values = dict(DEFAULT_CFG_DICT)
    values.update(
        {
            "task": "detect", "mode": "train", "model": model, "project": project,
            "name": name, "epochs": epochs, "imgsz": 640, "batch": 16,
            "device": 0, "optimizer": "SGD", "lr0": lr0, "momentum": 0.937,
            "cos_lr": True, "seed": 42, "deterministic": True, "amp": True,
            "nbs": 64, "freeze": freeze,
        }
    )
    return values


def b1r_a_profile(model: str, project: str) -> dict[str, Any]:
    return _profile(model=model, project=project, name="b1r-a", epochs=10, lr0=0.01, freeze=list(range(11)))


def b1r_b_profile(model: str, project: str) -> dict[str, Any]:
    return _profile(model=model, project=project, name="b1r-b", epochs=90, lr0=0.001, freeze=None)


def direct_profile(model: str, project: str) -> dict[str, Any]:
    return _profile(model=model, project=project, name="direct", epochs=100, lr0=0.001, freeze=None)


def retest_smoke_profile(variant: str, model: str, project: str) -> dict[str, Any]:
    return _profile(model=model, project=project, name=f"{variant.lower()}-smoke", epochs=3, lr0=0.001, freeze=None)


def retest_formal_profile(variant: str, model: str, project: str) -> dict[str, Any]:
    return _profile(model=model, project=project, name=variant.lower(), epochs=100, lr0=0.001, freeze=None)
