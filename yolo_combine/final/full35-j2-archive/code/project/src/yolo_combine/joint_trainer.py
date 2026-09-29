"""Stable public seam for joint-training scheduling and orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from . import _joint_trainer_impl as _impl
from .plateau import PlateauDecision, PlateauPolicy, PlateauRecovery

EpochTrainingReport = _impl.EpochTrainingReport
JointEpochRunner = _impl.JointEpochRunner
PoseEpochRunner = _impl.PoseEpochRunner


class StageWarmupCosineScheduler(_impl.StageWarmupCosineScheduler):
    """Resume-safe cosine scheduler with optional one-shot J2 recovery."""

    schema_version = 2

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        stage: str,
        epochs: int,
        steps_per_epoch: int,
        warmup_epochs: int = 3,
        warmup_start_factor: float = 0.1,
        final_lr_factor: float = 0.5,
        plateau_policy: PlateauPolicy | None = None,
    ) -> None:
        if plateau_policy is not None and stage != "j2":
            raise ValueError("plateau recovery is only supported for formal J2")
        super().__init__(
            optimizer,
            stage=stage,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs,
            warmup_start_factor=warmup_start_factor,
            final_lr_factor=final_lr_factor,
        )
        self.plateau = (
            PlateauRecovery(plateau_policy)
            if plateau_policy is not None
            else None
        )

    def prepare_step(self) -> dict[str, float]:
        values = super().prepare_step()
        multiplier = self.plateau.lr_multiplier if self.plateau is not None else 1.0
        if multiplier == 1.0:
            return values
        adjusted: dict[str, float] = {}
        for index, group in enumerate(self.optimizer.param_groups):
            name = str(group.get("group_name", index))
            lr = values[name] * multiplier
            group["lr"] = lr
            adjusted[name] = lr
        return adjusted

    def observe_metric(self, score: float) -> PlateauDecision:
        if self.plateau is None:
            raise RuntimeError("this stage has no plateau recovery policy")
        return self.plateau.observe(score)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["plateau"] = (
            self.plateau.state_dict() if self.plateau is not None else None
        )
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        for name in (
            "schema_version",
            "stage",
            "epochs",
            "steps_per_epoch",
            "warmup_epochs",
            "warmup_start_factor",
            "final_lr_factor",
            "total_steps",
            "warmup_steps",
        ):
            if state.get(name) != expected[name]:
                raise ValueError(
                    f"scheduler contract changed at {name}: "
                    f"{state.get(name)!r} != {expected[name]!r}"
                )
        saved_lrs = state.get("base_lrs")
        if not isinstance(saved_lrs, dict) or set(saved_lrs) != set(self.base_lrs):
            raise ValueError("scheduler optimizer group names changed")
        self.base_lrs = {
            str(name): float(value) for name, value in saved_lrs.items()
        }
        saved_plateau = state.get("plateau")
        if self.plateau is None:
            if saved_plateau is not None:
                raise ValueError("checkpoint unexpectedly contains plateau state")
        else:
            if not isinstance(saved_plateau, Mapping):
                raise ValueError("checkpoint is missing J2 plateau state")
            self.plateau.load_state_dict(saved_plateau)
        current = int(state.get("current_step", -1))
        if not 0 <= current <= self.total_steps:
            raise ValueError("scheduler current_step is out of range")
        self.current_step = current


__all__ = (
    "EpochTrainingReport",
    "JointEpochRunner",
    "PoseEpochRunner",
    "PlateauDecision",
    "PlateauPolicy",
    "StageWarmupCosineScheduler",
)
