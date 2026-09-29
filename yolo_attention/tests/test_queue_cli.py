from __future__ import annotations

import json
from pathlib import Path

from yolo_attention.cli import main
from yolo_attention.queue_model import JobKind, JobStatus, QueueJob, QueueResult, QueueState
from yolo_attention.queue_store import QueueStore

ROOT = Path(__file__).resolve().parents[1]


def _run(capsys, *args: str) -> tuple[int, dict[str, object]]:
    code = main([*args, "--json"])
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return code, payload


def test_queue_init_status_and_next_are_cpu_safe(tmp_path: Path, capsys) -> None:
    queue_root = tmp_path / "queue"
    baseline_run = ROOT / "artifacts" / "runs" / "b26-fp"
    before = (
        {
            (path.relative_to(baseline_run), path.stat().st_size, path.stat().st_mtime_ns)
            for path in baseline_run.rglob("*")
            if path.is_file()
        }
        if baseline_run.exists()
        else set()
    )

    init_code, initialized = _run(
        capsys,
        "queue",
        "init",
        "--queue-root",
        str(queue_root),
        "--project-root",
        str(ROOT),
    )
    status_code, status = _run(capsys, "queue", "status", "--queue-root", str(queue_root))
    next_code, next_job = _run(capsys, "queue", "next", "--queue-root", str(queue_root))
    dry_code, dry = _run(capsys, "queue", "run-next", "--queue-root", str(queue_root))

    assert (init_code, status_code, next_code, dry_code) == (0, 0, 0, 0)
    assert initialized["jobs"] == 6
    assert status["counts"] == {"blocked": 5, "ready": 1}
    assert next_job["job_id"] == "b26-fp"
    assert next_job["will_execute"] is False
    assert dry == next_job
    after = (
        {
            (path.relative_to(baseline_run), path.stat().st_size, path.stat().st_mtime_ns)
            for path in baseline_run.rglob("*")
            if path.is_file()
        }
        if baseline_run.exists()
        else set()
    )
    assert after == before


def test_queue_validate_checks_real_paths_without_running_jobs(tmp_path: Path, capsys) -> None:
    queue_root = tmp_path / "queue"
    _run(
        capsys,
        "queue",
        "init",
        "--queue-root",
        str(queue_root),
        "--project-root",
        str(ROOT),
    )

    code, report = _run(capsys, "queue", "validate", "--queue-root", str(queue_root))

    assert code == 0
    assert report["valid"] is True
    assert report["errors"] == []
    assert "optional quantization is locked" in report["warnings"]


def test_queue_init_refuses_to_overwrite_existing_state(tmp_path: Path, capsys) -> None:
    queue_root = tmp_path / "queue"
    args = (
        "queue",
        "init",
        "--queue-root",
        str(queue_root),
        "--project-root",
        str(ROOT),
    )
    first, _ = _run(capsys, *args)
    second = main([*args, "--json"])

    assert first == 0
    assert second != 0


def test_queue_run_next_execute_constructs_live_backend_lazily(tmp_path: Path, capsys, monkeypatch) -> None:
    queue_root = tmp_path / "queue"
    job = QueueJob(
        id="p0",
        run_name="P0",
        stage="validation",
        kind=JobKind.VALIDATE,
        order=0,
        status=JobStatus.READY,
    )
    QueueStore(queue_root).initialize(QueueState.initial((job,), project_root=str(tmp_path)))
    calls: list[Path] = []

    class FakeLiveBackend:
        def __init__(self, *, project_root):
            calls.append(Path(project_root))

        def execute(self, job, state):
            checkpoint = tmp_path / "parent.pt"
            metrics = tmp_path / "result.json"
            checkpoint.touch()
            metrics.write_text("{}\n", encoding="utf-8")
            return QueueResult(checkpoint_path=str(checkpoint), metrics_path=str(metrics))

    monkeypatch.setattr("yolo_attention.queue_backend.ResearchQueueBackend", FakeLiveBackend)

    code, payload = _run(
        capsys,
        "queue",
        "run-next",
        "--queue-root",
        str(queue_root),
        "--execute",
    )

    assert code == 0
    assert payload["will_execute"] is True
    assert calls == [tmp_path]
    assert QueueStore(queue_root).load().job("p0").status is JobStatus.SUCCEEDED
