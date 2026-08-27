"""Accepted standalone Pose26 stages aligned with the Full35 source recipe."""

from __future__ import annotations

from dataclasses import replace

from . import _pose_stages_public_v1 as _v1

PoseStageName = _v1.PoseStageName
PoseStageSpec = _v1.PoseStageSpec

POSE_STAGES = dict(_v1.POSE_STAGES)
_SOURCE_ALIGNED = {
    "optimizer": "MuSGD",
    "lr0": 0.00038,
    "lrf": 0.5,
    "momentum": 0.948,
    "weight_decay": 0.00027,
    # Ultralytics scales weight decay by batch * accumulate / nbs. The
    # formal physical batch is 128, so nbs must also be 128 to preserve the
    # Full35 source value above instead of silently doubling it.
    "nbs": 128,
    "cos_lr": True,
    "mosaic": 0.0,
    "close_mosaic": 0,
}
_PATIENCE = {"p1": 10, "p2": 12, "p3": 20}
_PHYSICAL_BATCH = {"p1": 128, "p2": 64, "p3": 32}
for _name in ("p1", "p2", "p3"):
    _stage = POSE_STAGES[_name]
    POSE_STAGES[_name] = replace(
        _stage,
        batch=_PHYSICAL_BATCH[_name],
        overrides={
            **_stage.overrides,
            **_SOURCE_ALIGNED,
            "patience": _PATIENCE[_name],
        },
    )


def pose_stage(name: str) -> PoseStageSpec:
    try:
        return POSE_STAGES[name]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(
            f"unknown Pose stage {name!r}; expected {tuple(POSE_STAGES)}"
        ) from error


__all__ = ("POSE_STAGES", "PoseStageName", "PoseStageSpec", "pose_stage")
