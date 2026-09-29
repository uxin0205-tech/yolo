import importlib
from pathlib import Path

import pytest


@pytest.fixture
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_cumulative_ptq_queue")


def metrics():
    return {
        f"task{i}/{suffix}": 0.8 for i in range(8) for suffix in ("map50", "map50_95")
    }


def test_both_gates_required(api):
    values = metrics()
    values["task0/map50_95"] = 0.75
    assert not api.dual(values, metrics())["passes_total_gate"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_metrics_fail_closed(api, bad):
    values = metrics()
    values["task0/map50"] = bad
    with pytest.raises(ValueError):
        api.dual(values, metrics())


def test_missing_metric_rejected(api):
    values = metrics()
    values.pop("task0/map50")
    with pytest.raises(ValueError):
        api.dual(values, metrics())


def test_accepted_prefix_never_overwritten(api):
    with pytest.raises(ValueError):
        api.extend({"layer": "fixed-sd4"}, {"layer": "w6"})
    assert api.extend({"a": "w4"}, {"b": "w6"}) == {"a": "w4", "b": "w6"}


def test_gate_failure_does_not_advance_prefix(api):
    assert api.choose([{"passes_total_gate": False}], 100) is None
    record = {
        "candidate_id": "a",
        "passes_total_gate": True,
        "packed_weight_bytes": 110,
    }
    assert api.choose([record], 100) is None


def test_nomination_excludes_noop_and_postprocessed_output(api):
    def row(path, error):
        task = {
            "all_finite": True,
            "same_structure": True,
            "tensors": [
                {"path": "output[0]", "normalized_rmse": 999},
                {"path": "output[1].one2one.scores", "normalized_rmse": error},
            ],
        }
        return {
            "path": path,
            "region": "neck",
            "format": "fixed-sd4",
            "tasks": {"detect": task, "pose": task},
        }

    sites = {
        "a": {"elements": 100, "encoded_bits": 4},
        "b": {"elements": 10, "encoded_bits": 8},
    }
    result = api.nominate("neck", [row("a", 0), row("b", 0.1)], sites)
    assert result["path"] == "b" and result["raw_risk"] == 0.1


def test_immutable_plan_not_overwritten(api, tmp_path):
    path = tmp_path / "plan.json"
    api.write(path, {"a": 1}, immutable=True)
    with pytest.raises(FileExistsError):
        api.write(path, {"a": 2}, immutable=True)
    assert api.read(path) == {"a": 1}
