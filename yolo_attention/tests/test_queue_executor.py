from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from yolo_attention.queue_executor import QueueExecutionError, QueueExecutor
from yolo_attention.queue_model import JobKind, JobStatus, QueueJob, QueueResult, QueueState
from yolo_attention.queue_store import QueueStore
from yolo_attention.queue_workflow import create_initial_state


class FakeBackend:
    def __init__(self, checkpoint: Path, *, failure: Exception | None = None) -> None:
        self.checkpoint = checkpoint
        self.failure = failure
        self.calls: list[str] = []

    def execute(self, job, state):
        self.calls.append(job.id)
        if self.failure is not None:
            raise self.failure
        metrics_path = self.checkpoint.with_suffix(".json")
        metrics_path.write_text('{"map50_95": 0.4}\n', encoding="utf-8")
        return QueueResult(
            map50_95=0.4,
            checkpoint_path=str(self.checkpoint),
            metrics_path=str(metrics_path),
        )


def _store(tmp_path: Path) -> QueueStore:
    root = tmp_path / "project"
    (root / "weights").mkdir(parents=True)
    (root / "weights" / "yolo26m.pt").touch()
    store = QueueStore(tmp_path / "queue")
    store.initialize(create_initial_state(root))
    return store


def test_run_next_without_execute_never_calls_backend_or_changes_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    checkpoint = tmp_path / "result.pt"
    checkpoint.touch()
    backend = FakeBackend(checkpoint)
    before = store.load()

    preview = QueueExecutor(store, backend=backend).run_next(execute=False)

    assert preview.job_id == "b26-fp"
    assert preview.will_execute is False
    assert backend.calls == []
    assert store.load() == before


def test_success_is_persisted_and_unblocks_exactly_one_next_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    checkpoint = tmp_path / "result.pt"
    checkpoint.touch()
    backend = FakeBackend(checkpoint)

    preview = QueueExecutor(store, backend=backend).run_next(execute=True)
    state = store.load()

    assert preview.job_id == "b26-fp"
    assert backend.calls == ["b26-fp"]
    assert state.job("b26-fp").status is JobStatus.SUCCEEDED
    assert state.job("b26-fp").attempts == 1
    assert state.job("p0").status is JobStatus.READY
    assert state.job("i-scr").status is JobStatus.BLOCKED


def test_failure_is_persisted_and_requires_explicit_retry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backend = FakeBackend(tmp_path / "unused.pt", failure=RuntimeError("GPU remains busy"))
    executor = QueueExecutor(store, backend=backend)

    with pytest.raises(QueueExecutionError, match="GPU remains busy"):
        executor.run_next(execute=True)

    failed = store.load()
    assert failed.job("b26-fp").status is JobStatus.FAILED
    assert failed.job("b26-fp").attempts == 1
    assert failed.job("b26-fp").failure_reason == "GPU remains busy"

    retried = executor.retry("b26-fp")
    assert retried.status is JobStatus.QUEUED
    assert store.load().job("b26-fp").status is JobStatus.QUEUED
    with pytest.raises(ValueError, match="failed or interrupted"):
        executor.retry("p0")


def test_child_result_with_catastrophic_map_drop_fails_closed(tmp_path: Path) -> None:
    parent_checkpoint = tmp_path / "parent.pt"
    child_checkpoint = tmp_path / "child.pt"
    parent_checkpoint.touch()
    child_checkpoint.touch()
    parent = QueueJob(
        id="parent",
        run_name="PARENT",
        stage="test",
        kind=JobKind.TRAIN,
        order=0,
        status=JobStatus.SUCCEEDED,
        result=QueueResult(map50_95=0.5, checkpoint_path=str(parent_checkpoint)),
    )
    child = QueueJob(
        id="child",
        run_name="CHILD",
        stage="test",
        kind=JobKind.TRAIN,
        order=1,
        status=JobStatus.READY,
        parent_job_ids=("parent",),
        model_parent_job_id="parent",
    )
    store = QueueStore(tmp_path / "queue")
    store.initialize(QueueState.initial((parent, child), project_root=str(tmp_path)))

    class CollapsedBackend(FakeBackend):
        def execute(self, job, state):
            result = super().execute(job, state)
            return replace(result, map50_95=0.001)

    executor = QueueExecutor(store, backend=CollapsedBackend(child_checkpoint))

    with pytest.raises(QueueExecutionError, match="catastrophic mAP drop"):
        executor.run_next(execute=True)

    assert store.load().job("child").status is JobStatus.FAILED


def test_r2_diagnostic_records_catastrophic_drop_without_blocking_queue(tmp_path: Path) -> None:
    parent_checkpoint = tmp_path / "parent.pt"
    child_checkpoint = tmp_path / "child.pt"
    parent_checkpoint.touch()
    child_checkpoint.touch()
    metrics_path = child_checkpoint.with_suffix(".json")
    parent = QueueJob(
        id="d2-1p",
        run_name="D2-1P",
        stage="bdcn-projection",
        kind=JobKind.EVALUATE,
        order=0,
        status=JobStatus.SUCCEEDED,
        result=QueueResult(map50_95=0.5, checkpoint_path=str(parent_checkpoint)),
    )
    diagnostic = QueueJob(
        id="r2-pshift",
        run_name="R2-PSHIFT",
        stage="bdcn-denominator",
        kind=JobKind.EVALUATE,
        order=1,
        status=JobStatus.READY,
        parent_job_ids=("d2-1p",),
        model_parent_job_id="d2-1p",
    )
    store = QueueStore(tmp_path / "queue")
    store.initialize(QueueState.initial((parent, diagnostic), project_root=str(tmp_path)))

    class CollapsedDiagnosticBackend(FakeBackend):
        def execute(self, job, state):
            result = super().execute(job, state)
            return replace(result, map50_95=0.001, metrics_path=str(metrics_path))

    executor = QueueExecutor(store, backend=CollapsedDiagnosticBackend(child_checkpoint))
    executor.run_next(execute=True)

    recorded = store.load().job("r2-pshift")
    assert recorded.status is JobStatus.SUCCEEDED
    assert recorded.result is not None
    assert recorded.result.map50_95 == 0.001


def test_succeeded_job_is_never_dispatched_again(tmp_path: Path) -> None:
    store = _store(tmp_path)
    checkpoint = tmp_path / "result.pt"
    checkpoint.touch()
    backend = FakeBackend(checkpoint)
    executor = QueueExecutor(store, backend=backend)
    executor.run_next(execute=True)
    state = store.load()
    jobs = tuple(replace(job, status=JobStatus.BLOCKED) if job.id == "p0" else job for job in state.jobs)
    store.save(replace(state, jobs=jobs))

    preview = executor.run_next(execute=False)

    assert preview.job_id == "p0"
    assert backend.calls == ["b26-fp"]


def test_rewind_selection_archives_invalid_descendants_and_restores_readiness(tmp_path: Path) -> None:
    project = tmp_path / "project"
    run = project / "artifacts" / "runs" / "v1-dyn"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text("invalid\n", encoding="utf-8")
    select = QueueJob(
        id="recovery-select",
        run_name="RECOVERY-SELECT",
        stage="selection",
        kind=JobKind.SELECT,
        order=0,
        status=JobStatus.SUCCEEDED,
        decision={"winners": ["w-dir"], "skipped": [], "reason": "old", "expand": []},
    )
    descendant = QueueJob(
        id="v1-dyn",
        run_name="V1-DYN",
        stage="scale",
        kind=JobKind.EVALUATE,
        order=1,
        status=JobStatus.SUCCEEDED,
        parent_job_ids=("recovery-select",),
    )
    store = QueueStore(tmp_path / "queue")
    store.initialize(QueueState.initial((select, descendant), project_root=str(project)))
    executor = QueueExecutor(store, backend=FakeBackend(tmp_path / "unused.pt"))

    rewound = executor.rewind_selection("recovery-select")

    state = store.load()
    assert rewound.status is JobStatus.READY
    assert [job.id for job in state.jobs] == ["recovery-select"]
    assert state.job("recovery-select").decision is None
    assert not run.exists()
    archived = list((project / "artifacts" / "invalidated").glob("*/runs/v1-dyn/metrics.json"))
    assert len(archived) == 1
    assert '"event": "rewound"' in store.events_path.read_text(encoding="utf-8")


def test_rewind_selection_archives_later_legacy_job_with_stale_parent_link(tmp_path: Path) -> None:
    project = tmp_path / "project"
    run = project / "artifacts" / "runs" / "d2-fp"
    run.mkdir(parents=True)
    selection = QueueJob(
        id="d1-confirm-select",
        run_name="D1-CONFIRM-SELECT",
        stage="selection",
        kind=JobKind.SELECT,
        order=40,
        status=JobStatus.SUCCEEDED,
        decision={"winners": ["d1-shared"], "skipped": [], "reason": "old", "expand": []},
    )
    legacy = QueueJob(
        id="d2-fp",
        run_name="D2-FP",
        stage="bdcn-projection",
        kind=JobKind.EVALUATE,
        order=41,
        status=JobStatus.SUCCEEDED,
        parent_job_ids=(),
    )
    store = QueueStore(tmp_path / "queue")
    store.initialize(QueueState.initial((selection, legacy), project_root=str(project)))

    QueueExecutor(store, backend=FakeBackend(tmp_path / "unused.pt")).rewind_selection(
        "d1-confirm-select"
    )

    assert [job.id for job in store.load().jobs] == ["d1-confirm-select"]
    assert not run.exists()
