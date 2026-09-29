"""Single-worker state machine; execution is impossible without an explicit gate."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .queue_model import JobStatus, QueueJob, QueueResult, QueueState
from .queue_policy import SelectionDecision
from .queue_store import QueueStore
from .queue_workflow import materialize_after_selection, next_runnable_job, refresh_readiness


class QueueExecutionError(RuntimeError):
    pass


MINIMUM_PARENT_MAP_RETENTION = 0.5
MAP_RETENTION_DIAGNOSTIC_JOB_IDS = frozenset({"r2-pshift"})


class QueueBackend(Protocol):
    def execute(self, job: QueueJob, state: QueueState) -> QueueResult | SelectionDecision: ...


@dataclass(frozen=True)
class QueuePreview:
    job_id: str | None
    run_name: str | None
    kind: str | None
    requires_gpu: bool
    will_execute: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "run_name": self.run_name,
            "kind": self.kind,
            "requires_gpu": self.requires_gpu,
            "will_execute": self.will_execute,
        }


def _replace_job(state: QueueState, replacement: QueueJob) -> QueueState:
    jobs = tuple(replacement if job.id == replacement.id else job for job in state.jobs)
    updated = replace(state, jobs=jobs, revision=state.revision + 1)
    updated.validate()
    return updated


def _preview(job: QueueJob | None, *, execute: bool) -> QueuePreview:
    return QueuePreview(
        job_id=job.id if job else None,
        run_name=job.run_name if job else None,
        kind=job.kind.value if job else None,
        requires_gpu=job.requires_gpu if job else False,
        will_execute=bool(execute and job is not None),
    )


def _validate_result(job: QueueJob, result: QueueResult, state: QueueState) -> None:
    if job.kind.value in {"train", "evaluate"} and (
        result.map50_95 is None or not math.isfinite(result.map50_95)
    ):
        raise QueueExecutionError(f"job {job.id} did not produce finite map50_95")
    if result.checkpoint_path is None or not Path(result.checkpoint_path).is_file():
        raise QueueExecutionError(f"job {job.id} did not produce a readable checkpoint")
    if result.metrics_path is None or not Path(result.metrics_path).is_file():
        raise QueueExecutionError(f"job {job.id} did not produce a readable metrics artifact")
    if (
        job.model_parent_job_id is not None
        and result.map50_95 is not None
        and job.id not in MAP_RETENTION_DIAGNOSTIC_JOB_IDS
    ):
        parent = state.job(job.model_parent_job_id)
        parent_map = parent.result.map50_95 if parent.result is not None else None
        if (
            parent_map is not None
            and parent_map > 0.0
            and result.map50_95 < parent_map * MINIMUM_PARENT_MAP_RETENTION
        ):
            raise QueueExecutionError(
                f"job {job.id} has catastrophic mAP drop: {result.map50_95:.6f} versus "
                f"parent {job.model_parent_job_id} at {parent_map:.6f}; manual review required"
            )


class QueueExecutor:
    EXPANDING_SELECTIONS = frozenset(
        {
            "architecture-select",
            "recovery-select",
            "scale-select",
            "a0-select",
            "n0-select",
            "normalization-select",
            "d0-select",
            "d1-select",
            "d1-confirm-select",
            "d2-select",
            "denominator-select",
            "final-select",
        }
    )

    def __init__(self, store: QueueStore, *, backend: QueueBackend) -> None:
        self.store = store
        self.backend = backend

    def preview_next(self) -> QueuePreview:
        state = refresh_readiness(self.store.load())
        return _preview(next_runnable_job(state), execute=False)

    def run_next(self, *, execute: bool) -> QueuePreview:
        if not execute:
            return self.preview_next()
        with self.store.worker_lock():
            state = refresh_readiness(self.store.load())
            job = next_runnable_job(state)
            preview = _preview(job, execute=True)
            if job is None:
                return preview
            if job.status is JobStatus.READY:
                job = replace(job, status=JobStatus.QUEUED)
                state = _replace_job(state, job)
                self.store.save(state)
                self.store.append_event("queued", job_id=job.id, details={})
            job = replace(
                job,
                status=JobStatus.RUNNING,
                attempts=job.attempts + 1,
                failure_reason=None,
            )
            state = _replace_job(state, job)
            self.store.save(state)
            self.store.append_event("running", job_id=job.id, details={"attempt": job.attempts})
            try:
                outcome = self.backend.execute(job, state)
                if isinstance(outcome, QueueResult):
                    _validate_result(job, outcome, state)
                    succeeded = replace(
                        job,
                        status=JobStatus.SUCCEEDED,
                        result=outcome,
                        metrics_path=outcome.metrics_path,
                        checkpoint_path=outcome.checkpoint_path,
                    )
                    state = _replace_job(state, succeeded)
                elif isinstance(outcome, SelectionDecision):
                    succeeded = replace(
                        job,
                        status=JobStatus.SUCCEEDED,
                        decision={
                            "winners": list(outcome.winners),
                            "skipped": list(outcome.skipped),
                            "reason": outcome.reason,
                            "expand": list(outcome.expand),
                        },
                    )
                    state = _replace_job(state, succeeded)
                    if job.id in self.EXPANDING_SELECTIONS:
                        state = materialize_after_selection(
                            state,
                            job.id,
                            outcome,
                            generated_root=self.store.generated_root,
                        )
                else:
                    raise QueueExecutionError(
                        f"backend returned unsupported outcome {type(outcome).__name__}"
                    )
                state = refresh_readiness(state)
                self.store.save(state)
                self.store.append_event("succeeded", job_id=job.id, details={})
                return preview
            except KeyboardInterrupt:
                interrupted = replace(job, status=JobStatus.INTERRUPTED, failure_reason="interrupted")
                self.store.save(_replace_job(state, interrupted))
                self.store.append_event("interrupted", job_id=job.id, details={})
                raise
            except Exception as exc:
                failed = replace(job, status=JobStatus.FAILED, failure_reason=str(exc))
                self.store.save(_replace_job(state, failed))
                self.store.append_event("failed", job_id=job.id, details={"reason": str(exc)})
                raise QueueExecutionError(f"job {job.id} failed: {exc}") from exc

    def run_all(self, *, execute: bool) -> list[QueuePreview]:
        if not execute:
            return [self.run_next(execute=False)]
        previews: list[QueuePreview] = []
        while True:
            preview = self.run_next(execute=True)
            if preview.job_id is None:
                return previews
            previews.append(preview)

    def rewind_selection(self, job_id: str) -> QueueJob:
        """Re-run a completed selection and archive every materialized descendant."""

        with self.store.worker_lock():
            state = self.store.load()
            selection = state.job(job_id)
            if selection.kind.value != "select" or selection.status is not JobStatus.SUCCEEDED:
                raise ValueError("rewind only accepts a succeeded selection job")
            if any(job.status in {JobStatus.QUEUED, JobStatus.RUNNING} for job in state.jobs):
                raise ValueError("another queue job is already active")
            # Dynamic workflow jobs are appended in strictly increasing order.
            # The order boundary also repairs legacy queues whose generated D2
            # jobs incorrectly referenced d1-select instead of d1-confirm-select.
            removed_ids = {job.id for job in state.jobs if job.order > selection.order}
            project_root = Path(state.project_root)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            invalidated_root = project_root / "artifacts" / "invalidated"
            invalidated_root.mkdir(parents=True, exist_ok=True)
            readme = invalidated_root / "README.md"
            if not readme.exists():
                readme.write_text(
                    "# Invalidated runs\n\nQueue rewind moves superseded run and generated configuration artifacts here.\n",
                    encoding="utf-8",
                )
            archive = invalidated_root / f"{timestamp}-{job_id}"
            archived: list[str] = []
            for removed_id in sorted(removed_ids):
                for source, group in (
                    (project_root / "artifacts" / "runs" / removed_id, "runs"),
                    (self.store.generated_root / removed_id, "generated"),
                ):
                    if not source.exists():
                        continue
                    destination = archive / group / removed_id
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(destination))
                    archived.append(str(destination))
            reset = replace(
                selection,
                status=JobStatus.BLOCKED,
                metrics_path=None,
                checkpoint_path=None,
                failure_reason=None,
                decision=None,
                result=None,
            )
            jobs = tuple(
                reset if job.id == job_id else job for job in state.jobs if job.id not in removed_ids
            )
            selections = {
                key: value
                for key, value in state.selections.items()
                if key != job_id and key not in removed_ids
            }
            updated = refresh_readiness(
                replace(
                    state,
                    jobs=jobs,
                    selections=selections,
                    revision=state.revision + 1,
                )
            )
            self.store.save(updated)
            self.store.append_event(
                "rewound",
                job_id=job_id,
                details={
                    "removed_jobs": sorted(removed_ids),
                    "archived": archived,
                },
            )
            return updated.job(job_id)

    def retry(self, job_id: str) -> QueueJob:
        with self.store.worker_lock():
            state = self.store.load()
            job = state.job(job_id)
            if job.status not in {JobStatus.FAILED, JobStatus.INTERRUPTED}:
                raise ValueError("retry only accepts a failed or interrupted job")
            if any(other.status in {JobStatus.QUEUED, JobStatus.RUNNING} for other in state.jobs):
                raise ValueError("another queue job is already active")
            retried = replace(job, status=JobStatus.QUEUED, failure_reason=None)
            self.store.save(_replace_job(state, retried))
            self.store.append_event("retried", job_id=job_id, details={"attempts": job.attempts})
            return retried

    def append_bdcn_v2(self) -> QueueState:
        """Append the immutable post-mainline BDCN defect-fix branch."""

        from .queue_workflow import append_bdcn_v2_fix

        with self.store.worker_lock():
            state = append_bdcn_v2_fix(self.store.load())
            state = refresh_readiness(state)
            self.store.save(state)
            self.store.append_event("bdcn_v2_appended", job_id=None, details={})
            return state

    def append_bdcn_v3(self) -> QueueState:
        """Append the fixed-control and stabilized learned-codebook branch."""

        from .queue_workflow import append_bdcn_v3_stable

        with self.store.worker_lock():
            state = append_bdcn_v3_stable(self.store.load())
            state = refresh_readiness(state)
            self.store.save(state)
            self.store.append_event("bdcn_v3_appended", job_id=None, details={})
            return state

    def append_pwl_validation(self) -> QueueState:
        """Append the PWL-only score analysis and final comparison branch."""

        from .queue_workflow import append_pwl_validation

        with self.store.worker_lock():
            state = append_pwl_validation(self.store.load())
            state = refresh_readiness(state)
            self.store.save(state)
            self.store.append_event("pwl_validation_appended", job_id=None, details={})
            return state
