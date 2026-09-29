"""Accuracy hard gates, joint score, and independent checkpoint selectors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

GATE_METRICS: tuple[str, ...] = (
    "coco/box/map50_95",
    "coco/person/box/map50_95",
    "bbat/box/map50_95",
    "bbat/pose/map50_95",
    "bbat/ball/box/map50_95",
    "bbat/bat/box/map50_95",
    "bbat/ball/pose/map50_95",
    "bbat/bat/pose/map50_95",
)


def _validated_metrics(
    metrics: Mapping[str, float],
    required: tuple[str, ...] = GATE_METRICS,
) -> dict[str, float]:
    missing = tuple(name for name in required if name not in metrics)
    if missing:
        raise ValueError(f"metrics are missing required values: {missing}")
    resolved = {name: float(value) for name, value in metrics.items()}
    invalid = {
        name: value
        for name, value in resolved.items()
        if not math.isfinite(value) or not 0.0 <= value <= 1.0
    }
    if invalid:
        raise ValueError(f"metrics must be finite values in [0, 1]: {invalid}")
    return resolved


@dataclass(frozen=True)
class AccuracyGateReport:
    passed: bool
    maximum_drop: float
    deltas: dict[str, float]
    failed_metrics: tuple[str, ...]


class AccuracyGate:
    """Veto a candidate when any accepted mAP50-95 task drops too far."""

    def __init__(
        self,
        baseline: Mapping[str, float],
        *,
        maximum_drop: float = 0.08,
    ) -> None:
        if not 0 <= maximum_drop <= 1:
            raise ValueError("maximum_drop must be in [0, 1]")
        self.baseline = _validated_metrics(baseline)
        self.maximum_drop = float(maximum_drop)

    def evaluate(
        self,
        candidate: Mapping[str, float],
    ) -> AccuracyGateReport:
        values = _validated_metrics(candidate)
        deltas = {
            name: values[name] - self.baseline[name]
            for name in GATE_METRICS
        }
        failed = tuple(
            name
            for name in GATE_METRICS
            if deltas[name] < -self.maximum_drop - 1e-12
        )
        return AccuracyGateReport(
            passed=not failed,
            maximum_drop=self.maximum_drop,
            deltas=deltas,
            failed_metrics=failed,
        )


def joint_score(metrics: Mapping[str, float]) -> float:
    """Accepted selection score; it never replaces the per-metric hard gate."""

    values = _validated_metrics(metrics)
    return (
        0.2 * values["coco/box/map50_95"]
        + 0.2 * values["coco/person/box/map50_95"]
        + 0.2 * values["bbat/box/map50_95"]
        + 0.4 * values["bbat/pose/map50_95"]
    )


def detect_score(metrics: Mapping[str, float]) -> float:
    values = _validated_metrics(metrics)
    return 0.5 * (
        values["coco/box/map50_95"]
        + values["coco/person/box/map50_95"]
    )


def pose_score(metrics: Mapping[str, float]) -> float:
    values = _validated_metrics(metrics)
    return 0.5 * (
        values["bbat/box/map50_95"]
        + values["bbat/pose/map50_95"]
    )


@dataclass(frozen=True)
class SelectionResult:
    epoch: int
    selected: tuple[str, ...]
    scores: dict[str, float]


class CheckpointSelectors:
    """Track best Detect, Pose, gate-feasible joint, and unconditional last."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, object]] = {}

    def observe(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        gate: AccuracyGateReport,
    ) -> SelectionResult:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        values = _validated_metrics(metrics)
        scores = {
            "best_detect": detect_score(values),
            "best_pose": pose_score(values),
            "best_joint": joint_score(values),
        }
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
        previous_joint = self._state.get("best_joint")
        if gate.passed and (
            previous_joint is None
            or scores["best_joint"] > float(previous_joint["score"])
        ):
            self._state["best_joint"] = {
                "epoch": epoch,
                "score": scores["best_joint"],
                "metrics": dict(values),
                "gate": {
                    "passed": True,
                    "deltas": dict(gate.deltas),
                },
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
        return SelectionResult(
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
        allowed = {"best_detect", "best_pose", "best_joint", "last"}
        if not set(state) <= allowed:
            raise ValueError(f"unknown checkpoint selectors: {set(state) - allowed}")
        self._state = {
            str(label): dict(record)
            for label, record in state.items()
        }
