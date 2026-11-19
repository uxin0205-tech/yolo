"""確保程序恢復不繞過使用者停止、來源鎖或完整續訓檔。"""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "operations/stage8_queue_recovery.py"
spec = importlib.util.spec_from_file_location("stage8_queue_recovery", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / "run"
    folder = root / "train-rep17_20_silu"
    put(root / "source.json", {"frozen_source": "original"})
    parent = tmp_path / "initial.pt"
    parent.write_bytes(b"parent fixture")
    spec = {"action": "train", "parent": {"checkpoint": str(parent), "sha256": recovery.digest(parent)},
            "recipe": {"epochs": 5}, "calibration": {"status": "passed"}}
    put(folder / "job.json", spec)
    put(folder / "config.json", {k: spec[k] for k in ["parent", "recipe", "calibration"]} |
        {"source_sha256": recovery.digest(root / "source.json")})
    (folder / "resume.pt").write_bytes(b"complete epoch fixture")
    put(folder / "resume.json", {"sha256": recovery.digest(folder / "resume.pt"), "next_epoch": 5})
    put(folder / "progress.json", {"epoch": 5, "step": 340})
    (folder / "worker.log").write_text("recorded failure\n")
    put(root / "supervisor.json", {"status": "error", "job": folder.name,
                                  "error": f"RuntimeError('{folder.name} worker exit=-11；log')"})
    return root


def test_recovers_only_verified_complete_epoch(run_root):
    result = recovery.validate_recovery(run_root, source_check=lambda root: None)
    assert result["next_epoch"] == 5


@pytest.mark.parametrize("stop", ["STOP", "train-rep17_20_silu/STOP"])
def test_stop_is_never_overridden(run_root, stop):
    (run_root / stop).touch()
    with pytest.raises(RuntimeError, match="STOP"):
        recovery.validate_recovery(run_root, source_check=lambda root: None)


@pytest.mark.parametrize("error", ["worker exit=1", "worker exit=-9", "non-finite loss"])
def test_other_errors_are_not_retried(run_root, error):
    state = json.loads((run_root / "supervisor.json").read_text())
    state["error"] = error
    put(run_root / "supervisor.json", state)
    with pytest.raises(RuntimeError, match="SIGSEGV"):
        recovery.validate_recovery(run_root, source_check=lambda root: None)


def test_corrupt_checkpoint_is_rejected(run_root):
    (run_root / "train-rep17_20_silu/resume.pt").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="SHA"):
        recovery.validate_recovery(run_root, source_check=lambda root: None)


def test_recipe_change_is_rejected(run_root):
    path = run_root / "train-rep17_20_silu/job.json"
    value = json.loads(path.read_text())
    value["recipe"]["epochs"] = 10
    put(path, value)
    with pytest.raises(RuntimeError, match="config"):
        recovery.validate_recovery(run_root, source_check=lambda root: None)


def test_source_guard_failure_is_propagated(run_root):
    def changed(root):
        raise ValueError("source changed")
    with pytest.raises(ValueError, match="source changed"):
        recovery.validate_recovery(run_root, source_check=changed)


@pytest.mark.parametrize("second_succeeds", [True, False])
def test_queue_retries_at_most_once_and_preserves_evidence(run_root, monkeypatch, second_succeeds):
    validate = recovery.validate_recovery
    monkeypatch.setattr(recovery, "validate_recovery", lambda root: validate(root, source_check=lambda root: None))
    monkeypatch.setattr(recovery.signal, "signal", lambda *args: None)
    launched = []

    class Queue:
        def __init__(self, *args, **kwargs):
            launched.append(args)
            self.pid = 10000 + len(launched)
            self.returncode = 0 if len(launched) == 2 and second_succeeds else 1
            state = json.loads((run_root / "supervisor.json").read_text())
            state.update(pid=self.pid, status="complete" if self.returncode == 0 else "error")
            put(run_root / "supervisor.json", state)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(recovery.subprocess, "Popen", Queue)
    output = run_root / "recovery"
    if second_succeeds:
        assert recovery.run(run_root, output) == 0
    else:
        with pytest.raises(RuntimeError, match="用完"):
            recovery.run(run_root, output)
    assert len(launched) == 2
    assert (output / "attempt-0.json").is_file() and (output / "attempt-1.json").is_file()
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == ("complete" if second_succeeds else "error")
