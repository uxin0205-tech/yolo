"""Small step-based orchestration for development and smoke validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .loaders import CyclingLoader, JointLoaders
from .training import OptimizerStepReport, TaskLossRouter


@dataclass(frozen=True)
class JointRunReport:
    steps: tuple[OptimizerStepReport, ...]
    detect_loader_wraps: int
    pose_loader_wraps: int
    detect_microbatches: int
    pose_microbatches: int


def run_joint_steps(
    router: TaskLossRouter,
    optimizer: torch.optim.Optimizer,
    loaders: JointLoaders,
    *,
    steps: int,
    detect_per_step: int,
    pose_per_step: int = 1,
) -> JointRunReport:
    """Run an explicit number of updates without pretending task dataset passes are equal."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if detect_per_step < 1 or pose_per_step < 1:
        raise ValueError("each task needs at least one microbatch per optimizer step")
    detect = CyclingLoader(loaders.detect)
    pose = CyclingLoader(loaders.pose)
    reports: list[OptimizerStepReport] = []
    for _ in range(steps):
        reports.append(
            router.optimizer_step(
                optimizer,
                detect_batches=[detect.next() for _ in range(detect_per_step)],
                pose_batches=[pose.next() for _ in range(pose_per_step)],
            )
        )
    return JointRunReport(
        steps=tuple(reports),
        detect_loader_wraps=detect.wraps,
        pose_loader_wraps=pose.wraps,
        detect_microbatches=steps * detect_per_step,
        pose_microbatches=steps * pose_per_step,
    )
