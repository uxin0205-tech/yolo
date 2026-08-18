"""Deterministic phase gates and final winner policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import mean, pstdev

MAP_TOLERANCE = 0.001
ZERO_TRAIN_MAP = 0.506737
B26_FP_MAP = 0.517998


def _finite(values: Mapping[str, float]) -> dict[str, float]:
    checked = {name: float(value) for name, value in values.items()}
    if not checked or not all(math.isfinite(value) for value in checked.values()):
        raise ValueError("selection metrics must be non-empty and finite")
    return checked


def choose_pilot(metrics: Mapping[float, float]) -> float:
    """Use lower LR inside a strict 0.001 tie; otherwise use higher mAP."""

    values = _finite({str(lr): value for lr, value in metrics.items()})
    if set(metrics) != {1e-5, 5e-6}:
        raise ValueError("pilot requires exactly lr0=1e-5 and lr0=5e-6")
    if abs(values[str(1e-5)] - values[str(5e-6)]) < MAP_TOLERANCE:
        return 5e-6
    return max(metrics, key=metrics.get)


def phase_gate(*, parent_id: str, parent_map: float, child_id: str, child_map: float) -> str:
    values = _finite({parent_id: parent_map, child_id: child_map})
    return parent_id if values[parent_id] - values[child_id] > MAP_TOLERANCE else child_id


@dataclass(frozen=True)
class FinalDecision:
    formal_winner: str
    best_observed: str | None
    seed_mean: float
    seed_std: float
    seed_min: float
    seed_max: float
    baseline_gap: float
    stable_improvement: bool


def choose_final(
    seed_winners: Mapping[str, tuple[str, float]], *, zero_train_id: str = "epoch0-bittrue"
) -> FinalDecision:
    if set(seed_winners) != {"0", "1", "2"}:
        raise ValueError("final selection requires seeds 0, 1, and 2")
    metrics = _finite({seed: item[1] for seed, item in seed_winners.items()})
    average = mean(metrics.values())
    stable = average - ZERO_TRAIN_MAP >= MAP_TOLERANCE
    best_seed = max(seed_winners, key=lambda seed: seed_winners[seed][1])
    best_id = seed_winners[best_seed][0]
    return FinalDecision(
        formal_winner=best_id if stable else zero_train_id,
        best_observed=None if stable or metrics[best_seed] <= ZERO_TRAIN_MAP else best_id,
        seed_mean=average,
        seed_std=pstdev(metrics.values()),
        seed_min=min(metrics.values()),
        seed_max=max(metrics.values()),
        baseline_gap=B26_FP_MAP - average,
        stable_improvement=stable,
    )
