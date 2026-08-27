"""Deterministic, checkpointable plateau recovery for formal J2 training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PlateauPolicy:
    """Immutable policy for one metric-driven learning-rate recovery."""

    monitor: str
    patience: int
    recovery_after: int
    lr_factor: float
    max_reductions: int
    min_delta: float
    adjust_momentum: bool = False

    def __post_init__(self) -> None:
        if self.monitor != "bittrue_joint_score":
            raise ValueError("plateau monitor must be 'bittrue_joint_score'")
        if self.patience < 1:
            raise ValueError("plateau patience must be positive")
        if not 1 <= self.recovery_after < self.patience:
            raise ValueError("recovery_after must be in [1, patience)")
        if not 0.0 < self.lr_factor < 1.0:
            raise ValueError("plateau lr_factor must be in (0,1)")
        if self.max_reductions != 1:
            raise ValueError("formal J2 supports exactly one LR reduction")
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError("plateau min_delta must be finite and non-negative")
        if self.adjust_momentum:
            raise ValueError("mid-run momentum adjustment is intentionally unsupported")


@dataclass(frozen=True)
class PlateauDecision:
    """Result of observing one completed validation epoch."""

    score: float
    best_score: float
    improved: bool
    stale_epochs: int
    recovery_applied: bool
    reductions: int
    lr_multiplier: float
    should_stop: bool


class PlateauRecovery:
    """Own plateau state without leaking policy logic into the training loop."""

    schema_version = 1

    def __init__(self, policy: PlateauPolicy) -> None:
        self.policy = policy
        self.best_score: float | None = None
        self.stale_epochs = 0
        self.reductions = 0

    @property
    def lr_multiplier(self) -> float:
        return self.policy.lr_factor**self.reductions

    def observe(self, score: float) -> PlateauDecision:
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("plateau score must be finite")
        improved = (
            self.best_score is None
            or score > self.best_score + self.policy.min_delta
        )
        if improved:
            self.best_score = score
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        recovery_applied = (
            not improved
            and self.reductions < self.policy.max_reductions
            and self.stale_epochs == self.policy.recovery_after
        )
        if recovery_applied:
            self.reductions += 1
        if self.best_score is None:  # pragma: no cover - guarded by first observation
            raise AssertionError("plateau best score was not initialized")
        return PlateauDecision(
            score=score,
            best_score=self.best_score,
            improved=improved,
            stale_epochs=self.stale_epochs,
            recovery_applied=recovery_applied,
            reductions=self.reductions,
            lr_multiplier=self.lr_multiplier,
            should_stop=self.stale_epochs >= self.policy.patience,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": asdict(self.policy),
            "best_score": self.best_score,
            "stale_epochs": self.stale_epochs,
            "reductions": self.reductions,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != self.schema_version:
            raise ValueError("plateau state schema changed")
        if state.get("policy") != asdict(self.policy):
            raise ValueError("plateau policy changed across resume")
        best = state.get("best_score")
        if best is not None:
            best = float(best)
            if not math.isfinite(best):
                raise ValueError("plateau best_score must be finite or null")
        stale = int(state.get("stale_epochs", -1))
        reductions = int(state.get("reductions", -1))
        if stale < 0 or stale > self.policy.patience:
            raise ValueError("plateau stale_epochs is out of range")
        if reductions < 0 or reductions > self.policy.max_reductions:
            raise ValueError("plateau reductions is out of range")
        self.best_score = best
        self.stale_epochs = stale
        self.reductions = reductions


__all__ = ("PlateauDecision", "PlateauPolicy", "PlateauRecovery")
