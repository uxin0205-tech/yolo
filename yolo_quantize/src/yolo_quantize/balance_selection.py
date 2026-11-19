"""Accuracy-first Pareto selection for deployable quantization candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BalanceCandidate:
    """One fully validated candidate and its comparable storage cost."""

    candidate_id: str
    policy_id: str
    format_id: str
    worst_total_delta: float
    packed_weight_bytes: int
    reference_fp32_bytes: int
    hard_gate_passed: bool

    def __post_init__(self) -> None:
        for name in ("candidate_id", "policy_id", "format_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if not math.isfinite(self.worst_total_delta):
            raise ValueError("worst_total_delta must be finite")
        if not -1.0 <= self.worst_total_delta <= 1.0:
            raise ValueError("worst_total_delta must be in [-1,1]")
        if self.packed_weight_bytes <= 0:
            raise ValueError("packed_weight_bytes must be positive")
        if self.reference_fp32_bytes <= 0:
            raise ValueError("reference_fp32_bytes must be positive")
        if not isinstance(self.hard_gate_passed, bool):
            raise TypeError("hard_gate_passed must be bool")

    @property
    def compression_ratio(self) -> float:
        return self.reference_fp32_bytes / self.packed_weight_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "policy_id": self.policy_id,
            "format_id": self.format_id,
            "worst_total_delta": self.worst_total_delta,
            "packed_weight_bytes": self.packed_weight_bytes,
            "reference_fp32_bytes": self.reference_fp32_bytes,
            "compression_ratio": self.compression_ratio,
            "hard_gate_passed": self.hard_gate_passed,
        }


@dataclass(frozen=True)
class BalanceSelection:
    """Transparent frontier and three non-interchangeable deployment roles."""

    total_max_drop: float
    accuracy_tolerance: float
    eligible_ids: tuple[str, ...]
    rejected_ids: tuple[str, ...]
    pareto_ids: tuple[str, ...]
    accuracy_id: str | None
    balanced_id: str | None
    hardware_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "total_max_drop": self.total_max_drop,
            "accuracy_tolerance": self.accuracy_tolerance,
            "eligible_ids": list(self.eligible_ids),
            "rejected_ids": list(self.rejected_ids),
            "pareto_ids": list(self.pareto_ids),
            "roles": {
                "accuracy": self.accuracy_id,
                "balanced": self.balanced_id,
                "hardware": self.hardware_id,
            },
        }


class QuantizationBalanceSelector:
    """Select the Pareto frontier under an independent hard accuracy budget.

    ``balanced`` is deliberately accuracy-first: candidates must be within
    ``accuracy_tolerance`` of the best worst-case delta before storage decides.
    This avoids hiding a task regression inside a unit-mixed scalar score.
    """

    def __init__(
        self,
        *,
        total_max_drop: float = 0.015,
        accuracy_tolerance: float = 0.002,
    ) -> None:
        if not math.isfinite(total_max_drop) or total_max_drop < 0.0:
            raise ValueError("total_max_drop must be finite and non-negative")
        if not math.isfinite(accuracy_tolerance) or accuracy_tolerance < 0.0:
            raise ValueError("accuracy_tolerance must be finite and non-negative")
        self.total_max_drop = total_max_drop
        self.accuracy_tolerance = accuracy_tolerance

    @staticmethod
    def _dominates(left: BalanceCandidate, right: BalanceCandidate) -> bool:
        no_worse = (
            left.worst_total_delta >= right.worst_total_delta
            and left.packed_weight_bytes <= right.packed_weight_bytes
        )
        strictly_better = (
            left.worst_total_delta > right.worst_total_delta
            or left.packed_weight_bytes < right.packed_weight_bytes
        )
        return no_worse and strictly_better

    def select(self, candidates: tuple[BalanceCandidate, ...]) -> BalanceSelection:
        ids = tuple(candidate.candidate_id for candidate in candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique")

        eligible = tuple(
            candidate
            for candidate in candidates
            if candidate.hard_gate_passed
            and candidate.worst_total_delta >= -self.total_max_drop
        )
        eligible_set = {candidate.candidate_id for candidate in eligible}
        rejected = tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.candidate_id not in eligible_set
        )
        pareto = tuple(
            candidate
            for candidate in eligible
            if not any(
                self._dominates(other, candidate)
                for other in eligible
                if other.candidate_id != candidate.candidate_id
            )
        )

        if not pareto:
            accuracy_id = balanced_id = hardware_id = None
        else:
            accuracy = min(
                pareto,
                key=lambda item: (
                    -item.worst_total_delta,
                    item.packed_weight_bytes,
                    item.candidate_id,
                ),
            )
            hardware = min(
                pareto,
                key=lambda item: (
                    item.packed_weight_bytes,
                    -item.worst_total_delta,
                    item.candidate_id,
                ),
            )
            accuracy_band = tuple(
                item
                for item in pareto
                if item.worst_total_delta
                >= accuracy.worst_total_delta - self.accuracy_tolerance
            )
            balanced = min(
                accuracy_band,
                key=lambda item: (
                    item.packed_weight_bytes,
                    -item.worst_total_delta,
                    item.candidate_id,
                ),
            )
            accuracy_id = accuracy.candidate_id
            balanced_id = balanced.candidate_id
            hardware_id = hardware.candidate_id

        return BalanceSelection(
            total_max_drop=self.total_max_drop,
            accuracy_tolerance=self.accuracy_tolerance,
            eligible_ids=tuple(candidate.candidate_id for candidate in eligible),
            rejected_ids=rejected,
            pareto_ids=tuple(candidate.candidate_id for candidate in pareto),
            accuracy_id=accuracy_id,
            balanced_id=balanced_id,
            hardware_id=hardware_id,
        )
