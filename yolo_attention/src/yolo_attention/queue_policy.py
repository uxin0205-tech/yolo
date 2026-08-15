"""Pure, deterministic winner and gate rules for the research queue."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

MAP_TIE = 0.001
N0_LOSS_GATE = 0.01
R1_MAP_LOSS_GATE = 0.002
R1_ROW_ERROR_GATE = 0.01
ARCH_COST_ORDER = ("i-scr", "t5-scr", "h-scr")
D1_COMPLEXITY_ORDER = ("d1-shared-10", "d1-pattn-10", "d1-phead-10")


class SelectionInputError(ValueError):
    pass


@dataclass(frozen=True)
class SelectionDecision:
    winners: tuple[str, ...]
    skipped: tuple[str, ...]
    reason: str
    expand: tuple[str, ...] = ()


def _finite_metric_map(metrics: Mapping[str, object]) -> dict[str, float]:
    if not metrics:
        raise SelectionInputError("selection requires at least one metric")
    validated: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SelectionInputError(f"metric {name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise SelectionInputError(f"metric {name!r} must be finite")
        validated[name] = number
    return validated


def _pick_near_best(metrics: Mapping[str, object], priority: tuple[str, ...]) -> SelectionDecision:
    values = _finite_metric_map(metrics)
    best = max(values.values())
    near = {name for name, value in values.items() if best - value < MAP_TIE - 1e-12}
    winner = next((name for name in priority if name in near), None)
    if winner is None:
        winner = max(values, key=lambda name: values[name])
    skipped = tuple(name for name in values if name != winner)
    return SelectionDecision((winner,), skipped, f"selected {winner} from mAP tie band {MAP_TIE}")


def select_architecture(metrics: Mapping[str, object]) -> SelectionDecision:
    required = set(ARCH_COST_ORDER)
    if set(metrics) != required:
        raise SelectionInputError(f"architecture metrics must contain {sorted(required)}")
    return _pick_near_best(metrics, ARCH_COST_ORDER)


def select_recovery(metrics: Mapping[str, object]) -> SelectionDecision:
    required = {"w-dir", "w-prog"}
    if set(metrics) != required:
        raise SelectionInputError(f"recovery metrics must contain {sorted(required)}")
    return _pick_near_best(metrics, ("w-dir", "w-prog"))


def select_scale(metrics: Mapping[str, object]) -> SelectionDecision:
    required = {"v1-dyn", "v1-shead", "v1-p2"}
    if set(metrics) != required:
        raise SelectionInputError(f"scale metrics must contain {sorted(required)}")
    return _pick_near_best(metrics, ("v1-p2", "v1-shead", "v1-dyn"))


def select_bias(metrics: Mapping[str, object]) -> SelectionDecision:
    required = {"v1-b0", "v1-bd", "v1-br"}
    if set(metrics) != required:
        raise SelectionInputError(f"bias metrics must contain {sorted(required)}")
    return _pick_near_best(metrics, ("v1-br", "v1-bd", "v1-b0"))


def select_d1(
    metrics: Mapping[str, object],
    *,
    five_epoch_metrics: Mapping[str, object],
) -> SelectionDecision:
    required = set(D1_COMPLEXITY_ORDER)
    values = _finite_metric_map(metrics)
    if set(values) != required:
        raise SelectionInputError(f"D1 metrics must contain {sorted(required)}")
    five_required = {name.removesuffix("-10") for name in required}
    five_values = _finite_metric_map(five_epoch_metrics)
    if set(five_values) != five_required:
        raise SelectionInputError(f"D1 five-epoch metrics must contain {sorted(five_required)}")
    decision = _pick_near_best(values, D1_COMPLEXITY_ORDER)
    winner = decision.winners[0]
    ranked = sorted(values.values(), reverse=True)
    top_gap = ranked[0] - ranked[1]
    extension_change = abs(values[winner] - five_values[winner.removesuffix("-10")])
    needs_seed = top_gap < MAP_TIE - 1e-12 or extension_change >= MAP_TIE - 1e-12
    reason = (
        f"{decision.reason}; 5-to-10 change={extension_change:.6f}, "
        f"top-two gap={top_gap:.6f}"
    )
    return SelectionDecision(
        decision.winners,
        decision.skipped,
        reason,
        ("d1-seed1",) if needs_seed else (),
    )


def select_n0(
    metrics: Mapping[str, object],
    *,
    a0_map: float,
    cost_order: tuple[str, ...],
) -> SelectionDecision:
    values = _finite_metric_map(metrics)
    reference = _finite_metric_map({"a0": a0_map})["a0"]
    cost_rank = {name: index for index, name in enumerate(cost_order)}
    if set(values) - set(cost_rank):
        raise SelectionInputError("cost_order must include every N0 candidate")
    eligible = [name for name, value in values.items() if reference - value <= N0_LOSS_GATE + 1e-12]
    eligible.sort(key=lambda name: (-values[name], cost_rank[name]))
    winners = tuple(eligible[:2])
    skipped = tuple(name for name in values if name not in winners)
    return SelectionDecision(winners, skipped, "retained at most two candidates inside 0.01 mAP gate")


def select_normalization(
    metrics: Mapping[str, object],
    profiles: Mapping[str, Mapping[str, object]],
) -> SelectionDecision:
    return select_final(metrics, profiles)


def select_d2(metrics: Mapping[str, object], *, a0_map: float) -> SelectionDecision:
    required = {"d2-fp", "d2-1p", "d2-2p"}
    values = _finite_metric_map(metrics)
    if set(values) != required:
        raise SelectionInputError(f"D2 metrics must contain {sorted(required)}")
    reference = _finite_metric_map({"a0": a0_map})["a0"]
    eligible = {name: value for name, value in values.items() if reference - value <= N0_LOSS_GATE + 1e-12}
    if not eligible:
        raise SelectionInputError("no D2 candidate passed the 0.01 A0 loss gate")
    decision = _pick_near_best(eligible, ("d2-1p", "d2-2p", "d2-fp"))
    skipped = tuple(name for name in values if name not in decision.winners)
    return SelectionDecision(decision.winners, skipped, "selected D2 inside A0 loss gate and mAP tie band")


def select_r_denominator(
    *,
    r0_map: float,
    r1_map: float,
    r1_row_sum_max_error: float,
) -> SelectionDecision:
    values = _finite_metric_map(
        {"r0_map": r0_map, "r1_map": r1_map, "r1_row_sum_max_error": r1_row_sum_max_error}
    )
    map_loss = values["r0_map"] - values["r1_map"]
    misses = map_loss > R1_MAP_LOSS_GATE + 1e-12 or values["r1_row_sum_max_error"] > R1_ROW_ERROR_GATE + 1e-12
    if misses:
        return SelectionDecision((), (), "R1 missed accuracy or row-sum gate", ("r1-newton",))
    return SelectionDecision(("r1-rlut",), ("r0-div", "r2-pshift"), "R1 passed denominator gates")


def select_final(
    metrics: Mapping[str, object],
    profiles: Mapping[str, Mapping[str, object]],
) -> SelectionDecision:
    values = _finite_metric_map(metrics)
    if set(values) != set(profiles):
        raise SelectionInputError("final metrics and profiles must contain the same candidates")
    best = max(values.values())
    eligible = [name for name, value in values.items() if best - value < MAP_TIE - 1e-12]
    costs: dict[str, tuple[float, float]] = {}
    for name in eligible:
        profile = profiles[name]
        for field in ("estimated_memory_traffic", "arithmetic_cost_proxy"):
            if field not in profile:
                raise SelectionInputError(f"profile {name!r} missing {field}")
        validated = _finite_metric_map(
            {
                "estimated_memory_traffic": profile["estimated_memory_traffic"],
                "arithmetic_cost_proxy": profile["arithmetic_cost_proxy"],
            }
        )
        costs[name] = (
            validated["estimated_memory_traffic"],
            validated["arithmetic_cost_proxy"],
        )
    winner = min(eligible, key=lambda name: costs[name])
    skipped = tuple(name for name in values if name != winner)
    return SelectionDecision((winner,), skipped, "selected within mAP band by memory then arithmetic cost")
