from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_combine.experiment_log import ExperimentLogger
from yolo_combine.metrics import (
    GATE_METRICS,
    AccuracyGate,
    CheckpointSelectors,
    joint_score,
)


def _baseline(value: float = 0.8) -> dict[str, float]:
    return {name: value for name in GATE_METRICS}


def test_accuracy_gate_is_per_metric_and_joint_score_cannot_hide_failure() -> None:
    baseline = _baseline()
    accepted = dict(baseline)
    accepted["bbat/ball/pose/map50_95"] = 0.72
    rejected = dict(accepted)
    rejected["bbat/ball/pose/map50_95"] = 0.719

    gate = AccuracyGate(baseline, maximum_drop=0.08)
    accepted_report = gate.evaluate(accepted)
    rejected_report = gate.evaluate(rejected)

    assert accepted_report.passed
    assert accepted_report.deltas["bbat/ball/pose/map50_95"] == pytest.approx(-0.08)
    assert not rejected_report.passed
    assert rejected_report.failed_metrics == ("bbat/ball/pose/map50_95",)
    assert joint_score(rejected) > 0


def test_checkpoint_selectors_track_detect_pose_joint_and_last_independently() -> None:
    gate = AccuracyGate(_baseline(0.5), maximum_drop=0.08)
    selectors = CheckpointSelectors()
    metrics = _baseline(0.6)
    first = selectors.observe(epoch=1, metrics=metrics, gate=gate.evaluate(metrics))
    worse_detect = dict(metrics)
    worse_detect["coco/box/map50_95"] = 0.55
    worse_detect["coco/person/box/map50_95"] = 0.55
    second = selectors.observe(
        epoch=2,
        metrics=worse_detect,
        gate=gate.evaluate(worse_detect),
    )

    assert set(first.selected) == {"best_detect", "best_pose", "best_joint", "last"}
    assert second.selected == ("last",)
    state = selectors.state_dict()
    assert state["best_detect"]["epoch"] == 1
    assert state["best_pose"]["epoch"] == 1
    assert state["best_joint"]["epoch"] == 1
    assert state["last"]["epoch"] == 2


def test_logger_writes_jsonl_csv_png_and_capability_report(tmp_path: Path) -> None:
    logger = ExperimentLogger(tmp_path, tensorboard="off")
    logger.log(
        "macro",
        step=1,
        values={
            "detect_mean_loss": 2.0,
            "pose_mean_loss": 4.0,
            "gradient/cosine": -0.25,
        },
    )
    logger.log(
        "epoch",
        step=1,
        values={
            "coco/box/map50_95": 0.51,
            "bbat/pose/map50_95": 0.83,
        },
    )
    plot = logger.plot("epoch")
    logger.close()

    assert (tmp_path / "events.jsonl").is_file()
    assert (tmp_path / "macro.csv").is_file()
    assert (tmp_path / "epoch.csv").is_file()
    assert plot.is_file() and plot.stat().st_size > 0
    capability = json.loads(
        (tmp_path / "logging-capabilities.json").read_text(encoding="utf-8")
    )
    assert capability["jsonl"] is True
    assert capability["csv"] is True
    assert capability["png"] is True
    assert capability["tensorboard"]["enabled"] is False
