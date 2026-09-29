from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_attention.queue_model import (
    JobStatus,
    QueueJob,
    QueueResult,
    QueueState,
    QueueValidationError,
)
from yolo_attention.queue_store import QueueLockedError, QueueStore


def test_queue_state_rejects_more_than_one_active_job() -> None:
    jobs = (
        QueueJob.minimal("a", order=0, status=JobStatus.QUEUED),
        QueueJob.minimal("b", order=1, status=JobStatus.RUNNING),
    )

    with pytest.raises(QueueValidationError, match="one active job"):
        QueueState.initial(jobs).validate()


def test_queue_state_rejects_dependency_cycle() -> None:
    jobs = (
        QueueJob.minimal("a", order=0, parent_job_ids=("b",)),
        QueueJob.minimal("b", order=1, parent_job_ids=("a",)),
    )

    with pytest.raises(QueueValidationError, match="cycle"):
        QueueState.initial(jobs).validate()


def test_queue_state_json_round_trip_preserves_nested_result() -> None:
    original = QueueState.initial(
        (
            QueueJob.minimal(
                "b26-fp",
                order=0,
                status=JobStatus.SUCCEEDED,
                result=QueueResult(
                    map50_95=0.401,
                    checkpoint_path="weights/yolo26m.pt",
                    metrics_path="artifacts/runs/b26-fp/metrics/queue-result.json",
                ),
            ),
        ),
        project_root="/tmp/project",
    )

    restored = QueueState.from_dict(original.to_dict())

    assert restored == original


def test_store_round_trip_refuses_overwrite_and_appends_event(tmp_path: Path) -> None:
    store = QueueStore(tmp_path / "queue")
    state = QueueState.initial(
        (QueueJob.minimal("b26-fp", order=0, status=JobStatus.READY),),
        project_root=str(tmp_path),
    )

    store.initialize(state)
    store.append_event("initialized", job_id=None, details={"jobs": 1})

    assert store.load() == state
    assert not store.temporary_path.exists()
    event = json.loads(store.events_path.read_text(encoding="utf-8").strip())
    assert event["event"] == "initialized"
    assert event["details"] == {"jobs": 1}
    with pytest.raises(FileExistsError):
        store.initialize(state)


def test_worker_lock_is_non_blocking(tmp_path: Path) -> None:
    store = QueueStore(tmp_path / "queue")
    store.initialize(
        QueueState.initial(
            (QueueJob.minimal("b26-fp", order=0, status=JobStatus.READY),),
            project_root=str(tmp_path),
        )
    )

    with store.worker_lock(), pytest.raises(QueueLockedError), store.worker_lock():
        pass
