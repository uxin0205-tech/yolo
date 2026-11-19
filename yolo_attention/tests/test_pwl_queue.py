from __future__ import annotations

from pathlib import Path

import pytest

from yolo_attention.cli import main
from yolo_attention.queue_model import JobKind, JobStatus, QueueJob, QueueResult, QueueState
from yolo_attention.queue_store import QueueStore
from yolo_attention.queue_workflow import append_pwl_validation

ROOT = Path(__file__).resolve().parents[1]


def test_pwl_validation_appends_two_serial_gpu_gated_jobs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    completed = QueueJob(
        id="v1-br",
        run_name="V1-BR",
        stage="a0",
        kind=JobKind.TRAIN,
        order=0,
        status=JobStatus.SUCCEEDED,
        result=QueueResult(checkpoint_path=str(checkpoint)),
        checkpoint_path=str(checkpoint),
    )
    state = QueueState.initial((completed,), project_root=str(ROOT))

    appended = append_pwl_validation(state)

    analysis = appended.job("pwl-score-analysis")
    comparison = appended.job("pwl-compare")
    assert analysis.kind is comparison.kind is JobKind.VALIDATE
    assert analysis.parent_job_ids == ("v1-br",)
    assert comparison.parent_job_ids == ("pwl-score-analysis",)
    assert analysis.parent_checkpoint == comparison.parent_checkpoint == str(checkpoint)
    assert analysis.requires_gpu and comparison.requires_gpu


def test_pwl_validation_cannot_be_appended_twice(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    state = QueueState.initial(
        (
            QueueJob(
                id="v1-br",
                run_name="V1-BR",
                stage="a0",
                kind=JobKind.TRAIN,
                order=0,
                status=JobStatus.SUCCEEDED,
                result=QueueResult(checkpoint_path=str(checkpoint)),
            ),
        ),
        project_root=str(ROOT),
    )
    with pytest.raises(ValueError, match="already exists"):
        append_pwl_validation(append_pwl_validation(state))



def test_cli_appends_pwl_validation_without_executing_gpu(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    queue_root = tmp_path / "queue"
    QueueStore(queue_root).initialize(
        QueueState.initial(
            (
                QueueJob(
                    id="v1-br",
                    run_name="V1-BR",
                    stage="a0",
                    kind=JobKind.TRAIN,
                    order=0,
                    status=JobStatus.SUCCEEDED,
                    result=QueueResult(checkpoint_path=str(checkpoint)),
                ),
            ),
            project_root=str(ROOT),
        )
    )

    assert main(["queue", "append-pwl-validation", "--queue-root", str(queue_root)]) == 0
    assert QueueStore(queue_root).load().job("pwl-score-analysis").status is JobStatus.READY
