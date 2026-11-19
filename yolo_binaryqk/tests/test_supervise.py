import io
import json
from types import SimpleNamespace

import pytest
from binaryqk import supervise
from binaryqk.protocol import recipe_for
from binaryqk.run import GPUInUseError, write_json


def test_busy_gpu_waits_then_rechecks_without_launching(monkeypatch, tmp_path):
    now = [0.]
    calls = []

    def require_idle():
        calls.append(now[0])
        if len(calls) == 1:
            raise GPUInUseError("occupied")

    monkeypatch.setattr(supervise, "require_idle_gpu", require_idle)
    monkeypatch.setattr(supervise.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(supervise.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monitor = io.StringIO()
    assert supervise.wait_for_gpu(tmp_path, "B2", monitor)
    assert calls == [0., 600.]
    assert json.loads(monitor.getvalue())["status"] == "waiting-for-gpu"


def test_stop_interrupts_busy_wait(monkeypatch, tmp_path):
    def occupied():
        raise GPUInUseError("occupied")

    monkeypatch.setattr(supervise, "require_idle_gpu", occupied)
    monkeypatch.setattr(supervise.time, "sleep", lambda _: (tmp_path / "STOP").touch())
    assert supervise.wait_for_gpu(tmp_path, "B2", io.StringIO()) is False


def test_candidate_requires_full_result_and_checkpoint(monkeypatch, tmp_path):
    run = tmp_path / "B1"
    assert supervise.candidate_action(run, "B1") == "fresh"
    run.mkdir()
    with pytest.raises(ValueError, match="without verified epoch resume"):
        supervise.candidate_action(run, "B1")
    (run / "resume.pt").touch()
    (run / "resume.json").touch()
    assert supervise.candidate_action(run, "B1") == "resume"
    result = {"variant": "B1", "smoke": False, "phase": "screening", "recipe": recipe_for("screening"),
              "stop_reason": "budget",
              "history": [{"epoch": epoch, "evaluation": {"images": 5000, "formal": True}}
                          for epoch in range(11)], "best": {"checkpoint": "epoch-10.pt"}}
    write_json(run / "status.json", {"kind": "completed"})
    write_json(run / "result.json", result)
    loaded = []

    def load(path):
        loaded.append(path)
        return None, {"variant": "B1", "lineage": {"phase": "screening"}}

    monkeypatch.setattr(supervise.checkpoint, "load", load)
    assert supervise.candidate_action(run, "B1") == "skip-completed"
    assert loaded == [run / "epoch-10.pt"]
    result["history"][0]["evaluation"]["images"] = 40
    write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="full formal evaluation"):
        supervise.candidate_action(run, "B1")
    result["history"][0]["evaluation"]["images"] = 5000
    result["best"]["checkpoint"] = "../outside.pt"
    write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="checkpoint path"):
        supervise.candidate_action(run, "B1")


def test_continuation_skips_completed_and_retries_resource_race(monkeypatch, tmp_path):
    monkeypatch.setattr(supervise, "candidate_action",
                        lambda path, variant, smoke, phase, parent: "skip-completed" if variant == "B1" else "fresh")
    checks, launches = [], []
    monkeypatch.setattr(supervise, "wait_for_gpu",
                        lambda output, variant, monitor: checks.append(variant) or True)

    def popen(command, **kwargs):
        launches.append(command)
        code = 75 if len(launches) == 1 else 0
        return SimpleNamespace(pid=12345, poll=lambda: code, wait=lambda: code)

    monkeypatch.setattr(supervise.subprocess, "Popen", popen)
    assert supervise.run_child(tmp_path, "B1", "screening", False, io.StringIO()) == 0
    assert supervise.run_child(tmp_path, "B2", "screening", False, io.StringIO()) == 0
    assert checks == ["B2", "B2"]
    assert len(launches) == 2 and all(command[3] == "B2" for command in launches)


def test_queue_runs_all_screenings_then_only_selected_recovery(monkeypatch, tmp_path):
    from binaryqk import report
    calls = []
    monkeypatch.setattr(supervise, "run_child",
                        lambda output, variant, phase, smoke, monitor, parent=None:
                        calls.append((variant, phase, parent)) or 0)
    parent = tmp_path / "B2/epoch-10.pt"
    monkeypatch.setattr(supervise, "screening_selection", lambda output, smoke: ("B2", parent))
    monkeypatch.setattr(report, "summarize", lambda output: None)
    assert supervise._supervise_locked(tmp_path, False, False) == 0
    assert calls == [("B1", "screening", None), ("B2", "screening", None),
                     ("B3", "screening", None), ("B2", "recovery", parent)]
    assert json.loads((tmp_path / "supervisor.json").read_text())["next_stage_started"] is False


def test_existing_queue_cannot_have_two_supervisors(monkeypatch, tmp_path):
    monkeypatch.setattr(supervise, "ROOT", tmp_path)
    output = tmp_path / "yolo_binaryqk/artifacts/stage1/queue"
    output.mkdir(parents=True)
    with (output / ".supervisor.lock").open("a") as lock:
        supervise.fcntl.flock(lock, supervise.fcntl.LOCK_EX | supervise.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            supervise.supervise(output, continue_queue=True)


def test_source_change_cannot_silently_continue(tmp_path):
    supervise.verify_queue_source(tmp_path)
    path = tmp_path / 'source.json'
    sources = json.loads(path.read_text())
    sources['train.py'] = '0' * 64
    path.write_text(json.dumps(sources))
    with pytest.raises(ValueError, match='source changed'):
        supervise.verify_queue_source(tmp_path)
