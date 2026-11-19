from __future__ import annotations

import pytest

from yolo_quantize import (
    FULL35_MAP50_95_KEYS,
    FULL35_METRIC_KEYS,
    Full35MetricCandidate,
    Full35MetricGate,
    Full35MetricGateSpec,
    Full35MetricSnapshot,
)


def _metrics(value: float) -> dict[str, float]:
    return {key: value for key in FULL35_METRIC_KEYS}


def test_metric_gate_checks_all_eight_total_w8_and_sham_thresholds() -> None:
    accepted = Full35MetricSnapshot(
        run_id="accepted-j3",
        policy_id="accepted-silu",
        metric_contract_id="full35-formal-v1",
        metrics=_metrics(1.0),
    )
    matched = Full35MetricSnapshot(
        run_id="matched-parent",
        policy_id="qsilu-a8",
        metric_contract_id="full35-formal-v1",
        metrics=_metrics(0.98),
    )
    candidate = Full35MetricCandidate(
        run_id="candidate-w8",
        format_id="uniform-w8",
        stage="qat",
        policy_id="qsilu-a8",
        metric_contract_id="full35-formal-v1",
        metrics=_metrics(0.975),
        sham_metrics=_metrics(0.984),
    )

    result = Full35MetricGate().evaluate(candidate, accepted, matched)

    assert {
        "metric_keys": tuple(result.total_deltas),
        "metric_count": result.metric_count,
        "total_passed": result.total_gate_passed,
        "w8_incremental_applicable": result.w8_incremental_gate_applicable,
        "w8_incremental_passed": result.w8_incremental_gate_passed,
        "sham_applicable": result.sham_gate_applicable,
        "sham_passed": result.sham_gate_passed,
        "worst_total_delta": round(result.worst_total_delta, 6),
        "worst_incremental_delta": round(result.worst_incremental_delta, 6),
        "maximum_sham_drift": round(result.maximum_sham_drift, 6),
        "decision": result.decision,
        "passed": result.passed,
    } == {
        "metric_keys": FULL35_METRIC_KEYS,
        "metric_count": 8,
        "total_passed": False,
        "w8_incremental_applicable": True,
        "w8_incremental_passed": True,
        "sham_applicable": True,
        "sham_passed": True,
        "worst_total_delta": -0.025,
        "worst_incremental_delta": -0.005,
        "maximum_sham_drift": 0.004,
        "decision": "recover",
        "passed": False,
    }


def test_default_metric_gate_uses_the_v5_one_and_a_half_point_accuracy_budget() -> None:
    spec = Full35MetricGateSpec()

    assert spec.metric_family == "map50"
    assert spec.total_max_drop == 0.015
    assert spec.recovery_floor == 0.04
    assert all(key.endswith("/map50") for key in FULL35_METRIC_KEYS)


def test_metric_gate_fails_closed_on_metric_or_matched_policy_mismatch() -> None:
    accepted = Full35MetricSnapshot(
        run_id="accepted",
        policy_id="accepted-silu",
        metric_contract_id="formal-v1",
        metrics=_metrics(0.8),
    )
    matched = Full35MetricSnapshot(
        run_id="matched",
        policy_id="hardswish-a8",
        metric_contract_id="search-v1",
        metrics=_metrics(0.8),
    )
    candidate = Full35MetricCandidate(
        run_id="candidate",
        format_id="uniform-w8",
        stage="ptq",
        policy_id="qsilu-a8",
        metric_contract_id="formal-v1",
        metrics=_metrics(0.8),
    )

    with pytest.raises(ValueError, match="metric contract"):
        Full35MetricGate().evaluate(candidate, accepted, matched)

    matched_same_contract = Full35MetricSnapshot(
        run_id="matched",
        policy_id="hardswish-a8",
        metric_contract_id="formal-v1",
        metrics=_metrics(0.8),
    )
    with pytest.raises(ValueError, match="matched policy"):
        Full35MetricGate().evaluate(candidate, accepted, matched_same_contract)


def test_metric_gate_serializes_the_validated_custom_threshold_spec() -> None:
    spec = Full35MetricGateSpec(
        total_max_drop=0.03,
        w8_incremental_max_drop=0.005,
        sham_max_absolute_drift=0.002,
        recovery_floor=0.05,
    )
    accepted = Full35MetricSnapshot(
        run_id="accepted",
        policy_id="accepted-silu",
        metric_contract_id="formal-v1",
        metrics=_metrics(0.8),
    )
    matched = Full35MetricSnapshot(
        run_id="matched",
        policy_id="qsilu-a8",
        metric_contract_id="formal-v1",
        metrics=_metrics(0.8),
    )
    candidate = Full35MetricCandidate(
        run_id="candidate",
        format_id="uniform-w8",
        stage="ptq",
        policy_id="qsilu-a8",
        metric_contract_id="formal-v1",
        metrics=_metrics(0.8),
    )

    payload = Full35MetricGate(spec).evaluate(candidate, accepted, matched).to_dict()

    assert payload["thresholds"] == {
        "metric_family": "map50",
        "total_drop": 0.03,
        "w8_incremental_drop": 0.005,
        "sham_absolute_drift": 0.002,
        "recovery_floor": 0.05,
    }


def test_w8_incremental_failure_enters_recovery_even_when_total_gate_passes() -> None:
    accepted = Full35MetricSnapshot(
        run_id="accepted",
        policy_id="accepted-silu",
        metric_contract_id="search-v1",
        metrics=_metrics(0.8),
    )
    matched = Full35MetricSnapshot(
        run_id="matched",
        policy_id="qsilu-a8",
        metric_contract_id="search-v1",
        metrics=_metrics(0.799),
    )
    candidate = Full35MetricCandidate(
        run_id="candidate",
        format_id="uniform-w8",
        stage="ptq",
        policy_id="qsilu-a8",
        metric_contract_id="search-v1",
        metrics=_metrics(0.786),
    )

    result = Full35MetricGate().evaluate(candidate, accepted, matched)

    assert result.total_gate_passed is True
    assert result.w8_incremental_gate_passed is False
    assert result.decision == "recover"
    assert result.passed is False


def test_legacy_map50_95_contract_remains_explicitly_reproducible() -> None:
    legacy = {key: 0.8 for key in FULL35_MAP50_95_KEYS}
    spec = Full35MetricGateSpec(metric_family="map50_95", total_max_drop=0.04)
    accepted = Full35MetricSnapshot(
        run_id="accepted",
        policy_id="silu",
        metric_contract_id="legacy-v4",
        metrics=legacy,
    )
    matched = Full35MetricSnapshot(
        run_id="matched",
        policy_id="qsilu-a8",
        metric_contract_id="legacy-v4",
        metrics=legacy,
    )
    candidate = Full35MetricCandidate(
        run_id="candidate",
        format_id="uniform-w6",
        stage="ptq",
        policy_id="qsilu-a8",
        metric_contract_id="legacy-v4",
        metrics={key: 0.77 for key in FULL35_MAP50_95_KEYS},
    )

    result = Full35MetricGate(spec).evaluate(candidate, accepted, matched)

    assert result.metric_count == 8
    assert result.total_gate_passed is True
