"""Resume-safe stage-local early stopping for formal fusion training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EarlyStopDecision:
    """One observation of a maximized validation score."""

    score: float
    best_score: float
    improved: bool
    stale_epochs: int
    should_stop: bool


class StageEarlyStopping:
    """Track stage-local score stagnation without changing learning rates."""

    schema_version = 1

    def __init__(self, *, stage: str, patience: int, min_delta: float = 0.0) -> None:
        if not stage:
            raise ValueError("early-stop stage cannot be empty")
        if patience < 1:
            raise ValueError("early-stop patience must be positive")
        if min_delta < 0:
            raise ValueError("early-stop min_delta cannot be negative")
        self.stage = stage
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score = -math.inf
        self.stale_epochs = 0

    def observe(self, score: float) -> EarlyStopDecision:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("early-stop score must be finite")
        improved = value > self.best_score + self.min_delta
        if improved:
            self.best_score = value
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return EarlyStopDecision(
            score=value,
            best_score=self.best_score,
            improved=improved,
            stale_epochs=self.stale_epochs,
            should_stop=self.stale_epochs >= self.patience,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "stale_epochs": self.stale_epochs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }
        changed = {
            name: (state.get(name), value)
            for name, value in expected.items()
            if state.get(name) != value
        }
        if changed:
            raise ValueError(f"early-stop contract changed: {changed}")
        best = float(state.get("best_score", math.nan))
        stale = int(state.get("stale_epochs", -1))
        if not math.isfinite(best) or stale < 0:
            raise ValueError("early-stop state is malformed")
        self.best_score = best
        self.stale_epochs = stale
