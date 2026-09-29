import importlib
from pathlib import Path

import pytest


@pytest.fixture
def compare(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("summarize_continuous_qat").compare


def metrics():
    return {
        f"task{i}/{suffix}": 0.8 for i in range(8) for suffix in ("map50", "map50_95")
    }


def test_recovery_is_per_metric_and_both_gates_required(compare):
    accepted = metrics()
    ptq = metrics()
    ptq["task0/map50_95"] = 0.7
    qat = metrics()
    qat["task0/map50_95"] = 0.77
    result = compare(qat, accepted, ptq)
    assert result["passes_total_gate"]
    assert result["qat_minus_ptq"]["task0/map50_95"] == pytest.approx(0.07)
    qat["task0/map50_95"] = 0.75
    assert not compare(qat, accepted, ptq)["passes_total_gate"]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_metrics_rejected(compare, invalid):
    qat = metrics()
    qat["task0/map50"] = invalid
    with pytest.raises(ValueError):
        compare(qat, metrics(), metrics())
