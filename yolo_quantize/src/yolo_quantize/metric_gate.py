"""Eight-metric Full35 accuracy and matched-sham decision gate."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

FULL35_MAP50_KEYS = (
    "coco/box/map50",
    "coco/person/box/map50",
    "bbat/box/map50",
    "bbat/pose/map50",
    "bbat/ball/box/map50",
    "bbat/bat/box/map50",
    "bbat/ball/pose/map50",
    "bbat/bat/pose/map50",
)

FULL35_MAP50_95_KEYS = (
    "coco/box/map50_95",
    "coco/person/box/map50_95",
    "bbat/box/map50_95",
    "bbat/pose/map50_95",
    "bbat/ball/box/map50_95",
    "bbat/bat/box/map50_95",
    "bbat/ball/pose/map50_95",
    "bbat/bat/pose/map50_95",
)

# The active v5 contract is mAP50.  The named legacy tuple keeps v4 replayable.
FULL35_METRIC_KEYS = FULL35_MAP50_KEYS
_METRIC_FAMILY_KEYS = {
    "map50": FULL35_MAP50_KEYS,
    "map50_95": FULL35_MAP50_95_KEYS,
}


def _metrics(values: Mapping[str, float]) -> dict[str, float]:
    keys = next(
        (
            family_keys
            for family_keys in _METRIC_FAMILY_KEYS.values()
            if set(values) == set(family_keys)
        ),
        None,
    )
    if keys is None:
        raise ValueError(
            "Full35 metrics must contain exactly one supported eight-key family: "
            "map50 or map50_95"
        )
    normalized = {key: float(values[key]) for key in keys}
    invalid = tuple(
        key
        for key, value in normalized.items()
        if not math.isfinite(value) or not 0.0 <= value <= 1.0
    )
    if invalid:
        raise ValueError(
            "Full35 metrics must be finite values in [0,1]: " + ",".join(invalid)
        )
    return normalized


@dataclass(frozen=True)
class Full35MetricGateSpec:
    """One validated source of truth for every Full35 decision threshold."""

    metric_family: Literal["map50", "map50_95"] = "map50"
    total_max_drop: float = 0.015
    w8_incremental_max_drop: float = 0.01
    sham_max_absolute_drift: float = 0.01
    recovery_floor: float = 0.04

    def __post_init__(self) -> None:
        if self.metric_family not in _METRIC_FAMILY_KEYS:
            raise ValueError("metric_family must be map50 or map50_95")
        values = {
            "total_max_drop": self.total_max_drop,
            "w8_incremental_max_drop": self.w8_incremental_max_drop,
            "sham_max_absolute_drift": self.sham_max_absolute_drift,
            "recovery_floor": self.recovery_floor,
        }
        invalid = tuple(
            name
            for name, value in values.items()
            if not math.isfinite(value) or value < 0.0
        )
        if invalid:
            raise ValueError(
                "metric gate thresholds must be finite and non-negative: "
                + ", ".join(invalid)
            )
        if self.recovery_floor < self.total_max_drop:
            raise ValueError("recovery_floor must not be stricter than total_max_drop")

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_family": self.metric_family,
            "total_drop": self.total_max_drop,
            "w8_incremental_drop": self.w8_incremental_max_drop,
            "sham_absolute_drift": self.sham_max_absolute_drift,
            "recovery_floor": self.recovery_floor,
        }

    @property
    def metric_keys(self) -> tuple[str, ...]:
        return _METRIC_FAMILY_KEYS[self.metric_family]


@dataclass(frozen=True)
class Full35MetricSnapshot:
    """One complete eight-metric model validation snapshot."""

    run_id: str
    policy_id: str
    metric_contract_id: str
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.run_id or not self.policy_id or not self.metric_contract_id:
            raise ValueError(
                "metric snapshot run_id, policy_id and metric_contract_id "
                "must not be empty"
            )
        object.__setattr__(self, "metrics", _metrics(self.metrics))


@dataclass(frozen=True)
class Full35MetricCandidate:
    """Candidate metrics with enough identity to apply format-specific gates."""

    run_id: str
    format_id: str
    stage: Literal["ptq", "qat"]
    policy_id: str
    metric_contract_id: str
    metrics: Mapping[str, float]
    sham_metrics: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or not self.format_id
            or not self.policy_id
            or not self.metric_contract_id
        ):
            raise ValueError(
                "metric candidate run_id, format_id, policy_id and "
                "metric_contract_id must not be empty"
            )
        if self.stage not in {"ptq", "qat"}:
            raise ValueError("metric candidate stage must be ptq or qat")
        object.__setattr__(self, "metrics", _metrics(self.metrics))
        if self.sham_metrics is not None:
            object.__setattr__(self, "sham_metrics", _metrics(self.sham_metrics))
        if self.stage == "qat" and self.sham_metrics is None:
            raise ValueError("QAT metric candidate requires matched sham metrics")

    @property
    def is_w8(self) -> bool:
        return re.search(r"(?:^|-)w8(?:-|$)", self.format_id.lower()) is not None


@dataclass(frozen=True)
class Full35MetricGateResult:
    """All deltas and decisions; no metric is hidden behind a headline average."""

    candidate_run_id: str
    accepted_run_id: str
    matched_run_id: str
    spec: Full35MetricGateSpec
    total_deltas: dict[str, float]
    incremental_deltas: dict[str, float]
    sham_deltas: dict[str, float]
    total_gate_passed: bool
    w8_incremental_gate_applicable: bool
    w8_incremental_gate_passed: bool
    sham_gate_applicable: bool
    sham_gate_passed: bool
    worst_total_metric: str
    worst_total_delta: float
    worst_incremental_metric: str
    worst_incremental_delta: float
    maximum_sham_drift: float
    decision: Literal["green", "recover", "reject", "invalid"]
    passed: bool

    @property
    def metric_count(self) -> int:
        return len(self.total_deltas)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_run_id": self.candidate_run_id,
            "accepted_run_id": self.accepted_run_id,
            "matched_run_id": self.matched_run_id,
            "thresholds": self.spec.to_dict(),
            "total_deltas": self.total_deltas,
            "incremental_deltas": self.incremental_deltas,
            "sham_deltas": self.sham_deltas,
            "total_gate_passed": self.total_gate_passed,
            "w8_incremental_gate_applicable": self.w8_incremental_gate_applicable,
            "w8_incremental_gate_passed": self.w8_incremental_gate_passed,
            "sham_gate_applicable": self.sham_gate_applicable,
            "sham_gate_passed": self.sham_gate_passed,
            "worst_total_metric": self.worst_total_metric,
            "worst_total_delta": self.worst_total_delta,
            "worst_incremental_metric": self.worst_incremental_metric,
            "worst_incremental_delta": self.worst_incremental_delta,
            "maximum_sham_drift": self.maximum_sham_drift,
            "decision": self.decision,
            "passed": self.passed,
        }


class Full35MetricGate:
    """Apply accepted, incremental and matched-sham thresholds to all metrics."""

    def __init__(self, spec: Full35MetricGateSpec | None = None) -> None:
        self.spec = Full35MetricGateSpec() if spec is None else spec

    def evaluate(
        self,
        candidate: Full35MetricCandidate,
        accepted: Full35MetricSnapshot,
        matched: Full35MetricSnapshot,
    ) -> Full35MetricGateResult:
        contract_ids = {
            candidate.metric_contract_id,
            accepted.metric_contract_id,
            matched.metric_contract_id,
        }
        if len(contract_ids) != 1:
            raise ValueError(
                "metric contract mismatch; candidate, accepted and matched "
                "snapshots must use the same evaluator/data contract"
            )
        if candidate.policy_id != matched.policy_id:
            raise ValueError(
                "matched policy mismatch; candidate and matched snapshot "
                "must share activation placement and activation quantization"
            )
        expected_keys = self.spec.metric_keys
        for label, values in (
            ("candidate", candidate.metrics),
            ("accepted", accepted.metrics),
            ("matched", matched.metrics),
        ):
            if tuple(values) != expected_keys:
                raise ValueError(
                    f"metric family mismatch; {label} must use "
                    f"{self.spec.metric_family}"
                )
        total = {
            key: candidate.metrics[key] - accepted.metrics[key] for key in expected_keys
        }
        incremental = {
            key: candidate.metrics[key] - matched.metrics[key] for key in expected_keys
        }
        sham = (
            {
                key: candidate.sham_metrics[key] - matched.metrics[key]
                for key in expected_keys
            }
            if candidate.sham_metrics is not None
            else {}
        )
        worst_total_metric = min(total, key=total.__getitem__)
        worst_incremental_metric = min(incremental, key=incremental.__getitem__)
        worst_total_delta = total[worst_total_metric]
        worst_incremental_delta = incremental[worst_incremental_metric]
        maximum_sham_drift = max((abs(value) for value in sham.values()), default=0.0)
        total_passed = all(
            value >= -self.spec.total_max_drop for value in total.values()
        )
        incremental_applicable = candidate.is_w8
        incremental_passed = not incremental_applicable or all(
            value >= -self.spec.w8_incremental_max_drop
            for value in incremental.values()
        )
        sham_applicable = bool(sham)
        sham_passed = (
            not sham_applicable
            or maximum_sham_drift <= self.spec.sham_max_absolute_drift
        )

        if not sham_passed:
            decision: Literal["green", "recover", "reject", "invalid"] = "invalid"
        elif total_passed and incremental_passed:
            decision = "green"
        elif worst_total_delta >= -self.spec.recovery_floor:
            decision = "recover"
        else:
            decision = "reject"
        return Full35MetricGateResult(
            candidate_run_id=candidate.run_id,
            accepted_run_id=accepted.run_id,
            matched_run_id=matched.run_id,
            spec=self.spec,
            total_deltas=total,
            incremental_deltas=incremental,
            sham_deltas=sham,
            total_gate_passed=total_passed,
            w8_incremental_gate_applicable=incremental_applicable,
            w8_incremental_gate_passed=incremental_passed,
            sham_gate_applicable=sham_applicable,
            sham_gate_passed=sham_passed,
            worst_total_metric=worst_total_metric,
            worst_total_delta=worst_total_delta,
            worst_incremental_metric=worst_incremental_metric,
            worst_incremental_delta=worst_incremental_delta,
            maximum_sham_drift=maximum_sham_drift,
            decision=decision,
            passed=decision == "green",
        )
