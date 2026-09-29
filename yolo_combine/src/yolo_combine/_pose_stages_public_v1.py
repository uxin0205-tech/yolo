"""Accepted standalone Pose26 stage profiles and tuning policy."""

from __future__ import annotations

from dataclasses import replace

from . import _pose_stages_impl as _impl

PoseStageName = _impl.PoseStageName
PoseStageSpec = _impl.PoseStageSpec

POSE_STAGES = dict(_impl.POSE_STAGES)
POSE_STAGES["p2"] = replace(
    POSE_STAGES["p2"],
    description=(
        "Load P1 best; train neck, Pose26 head, MASF and differentiable "
        "attention parts while keeping Q/K and bit-true constants locked."
    ),
)
POSE_STAGES["p3"] = replace(
    POSE_STAGES["p3"],
    description=(
        "Load P2 best; full low-LR fine-tuning including MASF and "
        "differentiable attention parts, excluding Q/K and bit-true constants."
    ),
)


def pose_stage(name: str) -> PoseStageSpec:
    try:
        return POSE_STAGES[name]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(
            f"unknown Pose stage {name!r}; expected {tuple(POSE_STAGES)}"
        ) from error


__all__ = ("POSE_STAGES", "PoseStageName", "PoseStageSpec", "pose_stage")

