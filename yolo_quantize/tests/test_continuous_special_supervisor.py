import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_module():
    spec = importlib.util.spec_from_file_location(
        "continuous_supervisor",
        Path(__file__).resolve().parents[1] / "scripts/run_continuous_special_queue.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_queue(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "OUT", tmp_path)
    monkeypatch.setattr(module, "PLAN", tmp_path / "plan.json")
    monkeypatch.setattr(module, "REPORT", tmp_path / "report.json")
    parent = SimpleNamespace(
        inference_sha256="export",
        config_sha256="parent",
        plan_path=tmp_path / "parent-plan.json",
        metrics_path=tmp_path / "metrics.json",
    )
    (tmp_path / "parent-preflight.json").write_text(
        json.dumps(
            {"forward_parity": {"status": "passed"}, "inference": {"sha256": "export"}}
        )
    )
    (tmp_path / "operator-precision.json").write_text(
        json.dumps(
            {
                "status": "passed_diagnostic_precision_inventory",
                "parent_manifest_sha256": "parent",
            }
        )
    )
    metrics = {
        f"task{i}/{kind}": 0.5 for i in range(8) for kind in ("map50", "map50_95")
    }
    parent.metrics_path.write_text(json.dumps({"metrics": metrics}))
    (tmp_path / "parent-revalidation.json").write_text(
        json.dumps(
            {
                "status": "passed_search_parent_revalidation",
                "parent_manifest_sha256": "parent",
            }
        )
    )
    parent.plan_path.write_text(
        json.dumps(
            {"sources": {"accepted_metrics": {"path": str(parent.metrics_path)}}}
        )
    )
    candidates = tuple(SimpleNamespace(candidate_id=f"c{i}") for i in range(40))
    module.REPORT.write_text(
        json.dumps(
            {
                "status": "completed",
                "results": {
                    c.candidate_id: {
                        "status": "completed",
                        "metrics": metrics,
                        "all_search_metrics": metrics,
                    }
                    for c in candidates
                },
            }
        )
    )
    monkeypatch.setattr(module.LockedQATParentSpec, "from_yaml", lambda _: parent)
    monkeypatch.setattr(
        module.Full35MixedPolicySearchPlan,
        "from_yaml",
        lambda _: SimpleNamespace(
            candidates=candidates, locked_qat_parent_sha256="parent"
        ),
    )
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *a, **k: SimpleNamespace(pid=123, wait=lambda timeout: 0),
    )
    return module


def test_supervisor_waits_for_other_gpu_then_requires_decision(tmp_path, monkeypatch):
    module = fixture_queue(tmp_path, monkeypatch)
    answers = iter(("999", ""))
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **k: next(answers)
    )
    sleeps = []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    module.main()
    assert sleeps == [600]
    status = json.loads((tmp_path / "execution-status.json").read_text())
    assert status["status"] == "decision_required" and status["completed_jobs"] == 40
    result = json.loads((tmp_path / "special-dual-summary.json").read_text())
    assert all(c["passes_total_gate"] for c in result["candidates"])


def test_supervisor_invalid_trace_prevents_gpu_launch(tmp_path, monkeypatch):
    module = fixture_queue(tmp_path, monkeypatch)
    (tmp_path / "operator-precision.json").write_text(json.dumps({"status": "failed"}))
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *a, **k: pytest.fail("GPU must not be queried"),
    )
    with pytest.raises(ValueError, match="operator trace"):
        module.main()
    assert (
        json.loads((tmp_path / "execution-status.json").read_text())["status"]
        == "error"
    )
