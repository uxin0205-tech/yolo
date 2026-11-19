from __future__ import annotations

import pytest

from yolo_quantize.qat_metrics import (
    MAP50_95_GATE_METRICS,
    MAP50_GATE_METRICS,
    QAT_GATE_METRICS,
    Map50AccuracyGate,
    Map50CheckpointSelectors,
)


def _metrics(value: float) -> dict[str, float]:
    return {name: value for name in (*MAP50_GATE_METRICS, *MAP50_95_GATE_METRICS)}


def test_map50_gate_includes_activation_and_weight_total_drop() -> None:
    gate = Map50AccuracyGate(_metrics(0.8), maximum_drop=0.015)
    candidate = _metrics(0.79)
    candidate["coco/person/box/map50"] = 0.7849

    report = gate.evaluate(candidate)

    assert report.passed is False
    assert report.failed_metrics == ("coco/person/box/map50",)
    assert report.deltas["coco/person/box/map50"] == pytest.approx(-0.0151)


def test_map50_95_companion_gate_rejects_hidden_localization_loss() -> None:
    gate = Map50AccuracyGate(
        _metrics(0.8),
        maximum_drop=0.015,
        maximum_map50_95_drop=0.04,
    )
    candidate = _metrics(0.795)
    candidate["bbat/bat/pose/map50_95"] = 0.7599

    report = gate.evaluate(candidate)

    assert report.passed is False
    assert report.primary_failed_metrics == ()
    assert report.companion_failed_metrics == ("bbat/bat/pose/map50_95",)
    assert report.worst_map50_delta == pytest.approx(-0.005)
    assert report.worst_map50_95_delta == pytest.approx(-0.0401)


def test_map50_checkpoint_selector_only_promotes_joint_when_gate_passes() -> None:
    selector = Map50CheckpointSelectors()
    failed = Map50AccuracyGate(_metrics(0.8), maximum_drop=0.015).evaluate(
        _metrics(0.7)
    )

    first = selector.observe(epoch=0, metrics=_metrics(0.7), gate=failed)
    passed = Map50AccuracyGate(_metrics(0.8), maximum_drop=0.015).evaluate(
        _metrics(0.795)
    )
    second = selector.observe(epoch=1, metrics=_metrics(0.795), gate=passed)

    assert "best_joint" not in first.selected
    assert "best_joint" in second.selected
    assert selector.state_dict()["best_joint"]["epoch"] == 1


def test_map50_checkpoint_selector_round_trips_state() -> None:
    selector = Map50CheckpointSelectors()
    gate = Map50AccuracyGate(_metrics(0.8), maximum_drop=0.015).evaluate(
        _metrics(0.795)
    )
    selector.observe(epoch=3, metrics=_metrics(0.795), gate=gate)
    restored = Map50CheckpointSelectors()

    restored.load_state_dict(selector.state_dict())

    assert restored.state_dict() == selector.state_dict()


def test_checkpoint_selector_uses_map50_95_as_well_as_map50_for_ranking() -> None:
    baseline = _metrics(0.8)
    for name in MAP50_95_GATE_METRICS:
        baseline[name] = 0.7
    first_metrics = dict(baseline)
    second_metrics = dict(baseline)
    for name in MAP50_GATE_METRICS:
        second_metrics[name] = 0.79
    for name in MAP50_95_GATE_METRICS:
        second_metrics[name] = 0.72
    gate = Map50AccuracyGate(
        baseline,
        maximum_drop=0.015,
        maximum_map50_95_drop=0.04,
    )
    selector = Map50CheckpointSelectors()

    selector.observe(epoch=0, metrics=first_metrics, gate=gate.evaluate(first_metrics))
    selected = selector.observe(
        epoch=1,
        metrics=second_metrics,
        gate=gate.evaluate(second_metrics),
    )

    assert "best_joint" in selected.selected


def test_matched_sham_gate_rejects_absolute_recipe_drift() -> None:
    accepted = _metrics(0.8)
    matched = _metrics(0.79)
    candidate = _metrics(0.805)
    gate = Map50AccuracyGate(
        accepted,
        maximum_drop=0.015,
        maximum_map50_95_drop=0.04,
        matched_baseline=matched,
        matched_max_absolute_drift=0.01,
    )

    report = gate.evaluate(candidate)

    assert report.primary_failed_metrics == ()
    assert report.companion_failed_metrics == ()
    assert report.passed is False
    assert report.matched_sham_max_absolute_drift == 0.01
    assert set(report.matched_sham_failed_metrics) == set(QAT_GATE_METRICS)
    assert all(
        value == pytest.approx(0.015) for value in report.matched_sham_deltas.values()
    )


def test_matched_sham_selector_is_independent_from_deployment_gate() -> None:
    accepted = _metrics(0.8)
    matched = _metrics(0.78)
    candidate = _metrics(0.781)
    gate = Map50AccuracyGate(
        accepted,
        maximum_drop=0.015,
        maximum_map50_95_drop=0.04,
        matched_baseline=matched,
        matched_max_absolute_drift=0.01,
    )
    selector = Map50CheckpointSelectors()

    report = gate.evaluate(candidate)
    selected = selector.observe(epoch=4, metrics=candidate, gate=report)

    assert report.deployment_passed is False
    assert report.matched_sham_passed is True
    assert report.passed is False
    assert "best_joint" not in selected.selected
    assert "best_matched_sham" in selected.selected
    assert selector.state_dict()["best_matched_sham"]["epoch"] == 4
