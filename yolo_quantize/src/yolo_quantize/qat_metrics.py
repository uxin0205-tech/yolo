"""Dual-family accuracy gates and checkpoint selection for activation-aware QAT."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

MAP50_GATE_METRICS: tuple[str, ...] = (
    "coco/box/map50",
    "coco/person/box/map50",
    "bbat/box/map50",
    "bbat/pose/map50",
    "bbat/ball/box/map50",
    "bbat/bat/box/map50",
    "bbat/ball/pose/map50",
    "bbat/bat/pose/map50",
)
MAP50_95_GATE_METRICS: tuple[str, ...] = tuple(
    name.replace("/map50", "/map50_95") for name in MAP50_GATE_METRICS
)
QAT_GATE_METRICS: tuple[str, ...] = MAP50_GATE_METRICS + MAP50_95_GATE_METRICS


def _validated(metrics: Mapping[str, float]) -> dict[str, float]:
    missing = tuple(name for name in QAT_GATE_METRICS if name not in metrics)
    if missing:
        raise ValueError(f"metrics are missing required QAT gate values: {missing}")
    result = {str(name): float(value) for name, value in metrics.items()}
    invalid = {
        name: value
        for name, value in result.items()
        if not math.isfinite(value) or not 0.0 <= value <= 1.0
    }
    if invalid:
        raise ValueError(f"metrics must be finite values in [0, 1]: {invalid}")
    return result


@dataclass(frozen=True)
class Map50AccuracyGateReport:
    """Dual-family deployment and matched-sham decisions."""

    passed: bool
    deployment_passed: bool
    matched_sham_passed: bool | None
    maximum_drop: float
    maximum_map50_95_drop: float
    deltas: dict[str, float]
    failed_metrics: tuple[str, ...]
    primary_failed_metrics: tuple[str, ...]
    companion_failed_metrics: tuple[str, ...]
    worst_map50_delta: float
    worst_map50_95_delta: float
    matched_sham_max_absolute_drift: float | None
    matched_sham_deltas: dict[str, float]
    matched_sham_failed_metrics: tuple[str, ...]


class Map50AccuracyGate:
    """Reject checkpoints when any total mAP50 or mAP50-95 drop exceeds budget."""

    def __init__(
        self,
        baseline: Mapping[str, float],
        *,
        maximum_drop: float = 0.015,
        maximum_map50_95_drop: float = 0.04,
        matched_baseline: Mapping[str, float] | None = None,
        matched_max_absolute_drift: float | None = None,
    ) -> None:
        if not math.isfinite(maximum_drop) or not 0.0 <= maximum_drop <= 1.0:
            raise ValueError("maximum_drop must be finite and in [0, 1]")
        self.baseline = _validated(baseline)
        self.maximum_drop = float(maximum_drop)
        if (
            not math.isfinite(maximum_map50_95_drop)
            or not 0.0 <= maximum_map50_95_drop <= 1.0
        ):
            raise ValueError("maximum_map50_95_drop must be finite and in [0, 1]")
        self.maximum_map50_95_drop = float(maximum_map50_95_drop)
        if (matched_baseline is None) != (matched_max_absolute_drift is None):
            raise ValueError(
                "matched_baseline and matched_max_absolute_drift must be set together"
            )
        self.matched_baseline = (
            None if matched_baseline is None else _validated(matched_baseline)
        )
        if matched_max_absolute_drift is not None and (
            not math.isfinite(matched_max_absolute_drift)
            or not 0.0 <= matched_max_absolute_drift <= 1.0
        ):
            raise ValueError("matched_max_absolute_drift must be finite and in [0, 1]")
        self.matched_max_absolute_drift = (
            None
            if matched_max_absolute_drift is None
            else float(matched_max_absolute_drift)
        )

    def evaluate(self, candidate: Mapping[str, float]) -> Map50AccuracyGateReport:
        values = _validated(candidate)
        deltas = {name: values[name] - self.baseline[name] for name in QAT_GATE_METRICS}
        primary_failed = tuple(
            name
            for name in MAP50_GATE_METRICS
            if deltas[name] < -self.maximum_drop - 1e-12
        )
        companion_failed = tuple(
            name
            for name in MAP50_95_GATE_METRICS
            if deltas[name] < -self.maximum_map50_95_drop - 1e-12
        )
        matched_deltas = (
            {}
            if self.matched_baseline is None
            else {
                name: values[name] - self.matched_baseline[name]
                for name in QAT_GATE_METRICS
            }
        )
        matched_failed = (
            ()
            if self.matched_max_absolute_drift is None
            else tuple(
                name
                for name in QAT_GATE_METRICS
                if abs(matched_deltas[name]) > self.matched_max_absolute_drift + 1e-12
            )
        )
        deployment_passed = not primary_failed and not companion_failed
        matched_sham_passed = (
            None if self.matched_max_absolute_drift is None else not matched_failed
        )
        return Map50AccuracyGateReport(
            passed=deployment_passed and matched_sham_passed is not False,
            deployment_passed=deployment_passed,
            matched_sham_passed=matched_sham_passed,
            maximum_drop=self.maximum_drop,
            maximum_map50_95_drop=self.maximum_map50_95_drop,
            deltas=deltas,
            failed_metrics=(
                primary_failed
                + companion_failed
                + tuple(f"matched-sham/{name}" for name in matched_failed)
            ),
            primary_failed_metrics=primary_failed,
            companion_failed_metrics=companion_failed,
            worst_map50_delta=min(deltas[name] for name in MAP50_GATE_METRICS),
            worst_map50_95_delta=min(deltas[name] for name in MAP50_95_GATE_METRICS),
            matched_sham_max_absolute_drift=self.matched_max_absolute_drift,
            matched_sham_deltas=matched_deltas,
            matched_sham_failed_metrics=matched_failed,
        )


def _scores(metrics: Mapping[str, float]) -> dict[str, float]:
    values = _validated(metrics)
    detect_map50 = 0.5 * (values["coco/box/map50"] + values["coco/person/box/map50"])
    detect_map50_95 = 0.5 * (
        values["coco/box/map50_95"] + values["coco/person/box/map50_95"]
    )
    pose_map50 = 0.5 * (values["bbat/box/map50"] + values["bbat/pose/map50"])
    pose_map50_95 = 0.5 * (values["bbat/box/map50_95"] + values["bbat/pose/map50_95"])
    joint_map50 = (
        0.2 * values["coco/box/map50"]
        + 0.2 * values["coco/person/box/map50"]
        + 0.2 * values["bbat/box/map50"]
        + 0.4 * values["bbat/pose/map50"]
    )
    joint_map50_95 = (
        0.2 * values["coco/box/map50_95"]
        + 0.2 * values["coco/person/box/map50_95"]
        + 0.2 * values["bbat/box/map50_95"]
        + 0.4 * values["bbat/pose/map50_95"]
    )
    return {
        "best_detect": 0.5 * (detect_map50 + detect_map50_95),
        "best_pose": 0.5 * (pose_map50 + pose_map50_95),
        "best_joint": 0.5 * (joint_map50 + joint_map50_95),
    }


@dataclass(frozen=True)
class Map50SelectionResult:
    epoch: int
    selected: tuple[str, ...]
    scores: dict[str, float]


class Map50CheckpointSelectors:
    """Rank with both metric families and admit only gate-feasible joint checkpoints."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, object]] = {}

    def observe(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        gate: Map50AccuracyGateReport,
    ) -> Map50SelectionResult:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        values = _validated(metrics)
        scores = _scores(values)
        selected: list[str] = []
        for label in ("best_detect", "best_pose"):
            previous = self._state.get(label)
            if previous is None or scores[label] > float(previous["score"]):
                self._state[label] = {
                    "epoch": epoch,
                    "score": scores[label],
                    "metrics": dict(values),
                }
                selected.append(label)
        previous_matched = self._state.get("best_matched_sham")
        if gate.matched_sham_passed is True and (
            previous_matched is None
            or scores["best_joint"] > float(previous_matched["score"])
        ):
            self._state["best_matched_sham"] = {
                "epoch": epoch,
                "score": scores["best_joint"],
                "metrics": dict(values),
                "gate": {
                    "deployment_passed": gate.deployment_passed,
                    "matched_sham_passed": True,
                    "deltas": dict(gate.deltas),
                    "matched_sham_deltas": dict(gate.matched_sham_deltas),
                },
            }
            selected.append("best_matched_sham")
        previous_joint = self._state.get("best_joint")
        if gate.passed and (
            previous_joint is None
            or scores["best_joint"] > float(previous_joint["score"])
        ):
            self._state["best_joint"] = {
                "epoch": epoch,
                "score": scores["best_joint"],
                "metrics": dict(values),
                "gate": {"passed": True, "deltas": dict(gate.deltas)},
            }
            selected.append("best_joint")
        self._state["last"] = {
            "epoch": epoch,
            "score": scores["best_joint"],
            "metrics": dict(values),
            "gate": {
                "passed": gate.passed,
                "deltas": dict(gate.deltas),
                "failed_metrics": list(gate.failed_metrics),
            },
        }
        selected.append("last")
        return Map50SelectionResult(
            epoch=epoch,
            selected=tuple(selected),
            scores=scores,
        )

    def state_dict(self) -> dict[str, dict[str, object]]:
        return {
            label: {
                key: (
                    dict(value)
                    if isinstance(value, dict)
                    else list(value)
                    if isinstance(value, tuple)
                    else value
                )
                for key, value in record.items()
            }
            for label, record in self._state.items()
        }

    def load_state_dict(
        self,
        state: Mapping[str, Mapping[str, object]],
    ) -> None:
        allowed = {
            "best_detect",
            "best_pose",
            "best_joint",
            "best_matched_sham",
            "last",
        }
        unexpected = set(state) - allowed
        if unexpected:
            raise ValueError(f"unknown checkpoint selectors: {unexpected}")
        self._state = {str(label): dict(record) for label, record in state.items()}


__all__ = (
    "MAP50_95_GATE_METRICS",
    "MAP50_GATE_METRICS",
    "QAT_GATE_METRICS",
    "Map50AccuracyGate",
    "Map50AccuracyGateReport",
    "Map50CheckpointSelectors",
    "Map50SelectionResult",
)
