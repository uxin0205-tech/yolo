"""Reproducible extension, candidate gating, and C_best selection rules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ExtensionDecision:
    extend: bool
    reason: str
    best_epoch: int
    late_gain: float | None


def should_extend(metrics: Iterable[float], *, best_epoch: int, early_stopped: bool) -> ExtensionDecision:
    """Apply the exact 100→140 epoch gate to one-based epoch metrics."""

    values = tuple(float(value) for value in metrics)
    if early_stopped:
        return ExtensionDecision(False, "formal-a-early-stopped", best_epoch, None)
    if len(values) < 100:
        raise ValueError("extension gate requires 100 epoch metrics")
    earlier_best = max(values[60:80])
    later_best = max(values[80:100])
    gain = later_best - earlier_best
    if 85 <= best_epoch <= 100:
        return ExtensionDecision(True, "best-epoch-in-85-100", best_epoch, gain)
    if gain >= 0.001:
        return ExtensionDecision(True, "rolling-best-gain-at-least-0.001", best_epoch, gain)
    return ExtensionDecision(False, "tail-converged", best_epoch, gain)


@dataclass(frozen=True)
class CandidateMetrics:
    candidate_id: str
    map50_95: float
    latency_ms: float
    gflops: float
    params: int


@dataclass(frozen=True)
class CandidateDecision:
    metrics: CandidateMetrics
    c0_map50_95: float
    drop: float
    decision: Decision
    cost_improvement: float
    eligible: bool
    reason: str


def classify_candidate(candidate: CandidateMetrics, c0: CandidateMetrics) -> CandidateDecision:
    if candidate.candidate_id == "C0":
        return CandidateDecision(candidate, c0.map50_95, 0.0, Decision.PASS, 0.0, False, "reference")
    drop = c0.map50_95 - candidate.map50_95
    latency_gain = 1.0 - candidate.latency_ms / c0.latency_ms
    flop_gain = 1.0 - candidate.gflops / c0.gflops
    cost_gain = max(latency_gain, flop_gain)
    epsilon = 1e-12
    if drop <= 0.005 + epsilon:
        return CandidateDecision(candidate, c0.map50_95, drop, Decision.PASS, cost_gain, True, "drop<=0.005")
    if drop <= 0.008 + epsilon:
        eligible = cost_gain >= 0.08
        reason = "conditional-cost>=8%" if eligible else "conditional-cost<8%"
        return CandidateDecision(
            candidate, c0.map50_95, drop, Decision.CONDITIONAL, cost_gain, eligible, reason
        )
    return CandidateDecision(candidate, c0.map50_95, drop, Decision.REJECT, cost_gain, False, "drop>0.008")


def trigger_c3_p5_fallback(decision: CandidateDecision) -> bool:
    return decision.metrics.candidate_id == "C3" and decision.drop > 0.008


def trigger_r1(decision: CandidateDecision) -> bool:
    return (
        decision.metrics.candidate_id == "C2"
        and decision.decision is Decision.CONDITIONAL
        and decision.cost_improvement >= 0.08
    )


def choose_c_best(decisions: Iterable[CandidateDecision]) -> CandidateDecision | None:
    """Select an eligible single-factor candidate; never fall back to C0."""

    eligible = [item for item in decisions if item.metrics.candidate_id != "C0" and item.eligible]
    if not eligible:
        return None
    rank = {Decision.PASS: 0, Decision.CONDITIONAL: 1, Decision.REJECT: 2}
    return min(
        eligible,
        key=lambda item: (
            rank[item.decision],
            item.drop,
            item.metrics.latency_ms,
            item.metrics.gflops,
            item.metrics.params,
        ),
    )
