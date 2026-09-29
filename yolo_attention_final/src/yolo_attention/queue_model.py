"""Single-worker experiment queue 使用的 typed、serializable state。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import StringEnum


class QueueValidationError(ValueError):
    pass


class JobKind(StringEnum):
    VALIDATE = "validate"
    TRAIN = "train"
    EVALUATE = "evaluate"
    SELECT = "select"


class JobStatus(StringEnum):
    BLOCKED = "blocked"
    READY = "ready"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    SKIPPED = "skipped"


TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.INTERRUPTED, JobStatus.SKIPPED}
)


@dataclass(frozen=True)
class QueueResult:
    map50_95: float | None = None
    map50: float | None = None
    map75: float | None = None
    maps: tuple[float, ...] = ()
    row_sum_max_error: float | None = None
    checkpoint_path: str | None = None
    metrics_path: str | None = None
    profile_path: str | None = None
    bdcn_bucket_histogram: tuple[int, ...] = ()
    bdcn_bucket_overflow_rate: float | None = None
    bdcn_last_bucket_rate: float | None = None
    bdcn_distance_max: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueueResult:
        payload = dict(data)
        payload["maps"] = tuple(payload.get("maps", ()))
        payload["bdcn_bucket_histogram"] = tuple(payload.get("bdcn_bucket_histogram", ()))
        return cls(**payload)


@dataclass(frozen=True)
class QueueJob:
    id: str
    run_name: str
    stage: str
    kind: JobKind
    order: int
    status: JobStatus = JobStatus.BLOCKED
    variant_path: str | None = None
    training_path: str | None = None
    evaluation_path: str | None = None
    parent_job_ids: tuple[str, ...] = ()
    model_parent_job_id: str | None = None
    parent_checkpoint: str | None = None
    requires_gpu: bool = False
    attempts: int = 0
    metrics_path: str | None = None
    checkpoint_path: str | None = None
    failure_reason: str | None = None
    decision: dict[str, Any] | None = None
    result: QueueResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", JobKind(self.kind))
        object.__setattr__(self, "status", JobStatus(self.status))
        object.__setattr__(self, "parent_job_ids", tuple(self.parent_job_ids))

    @classmethod
    def minimal(
        cls,
        job_id: str,
        *,
        order: int,
        status: JobStatus = JobStatus.BLOCKED,
        parent_job_ids: tuple[str, ...] = (),
        result: QueueResult | None = None,
    ) -> QueueJob:
        return cls(
            id=job_id,
            run_name=job_id.upper(),
            stage="test",
            kind=JobKind.VALIDATE,
            order=order,
            status=status,
            parent_job_ids=parent_job_ids,
            result=result,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        data["parent_job_ids"] = list(self.parent_job_ids)
        if self.result is not None:
            data["result"]["maps"] = list(self.result.maps)
            data["result"]["bdcn_bucket_histogram"] = list(self.result.bdcn_bucket_histogram)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueueJob:
        payload = dict(data)
        payload["parent_job_ids"] = tuple(payload.get("parent_job_ids", ()))
        if payload.get("result") is not None:
            payload["result"] = QueueResult.from_dict(payload["result"])
        return cls(**payload)


@dataclass(frozen=True)
class QueueState:
    schema_version: int
    project_root: str
    jobs: tuple[QueueJob, ...]
    created_at: str
    revision: int = 0
    selections: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def initial(
        cls,
        jobs: tuple[QueueJob, ...],
        *,
        project_root: str = ".",
    ) -> QueueState:
        return cls(
            schema_version=1,
            project_root=project_root,
            jobs=tuple(jobs),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def job(self, job_id: str) -> QueueJob:
        for job in self.jobs:
            if job.id == job_id:
                return job
        raise KeyError(f"unknown queue job {job_id!r}")

    def validate(self) -> None:
        if self.schema_version != 1:
            raise QueueValidationError("unsupported queue schema version")
        ids = [job.id for job in self.jobs]
        if len(ids) != len(set(ids)):
            raise QueueValidationError("queue job IDs must be unique")
        orders = [job.order for job in self.jobs]
        if len(orders) != len(set(orders)):
            raise QueueValidationError("queue job order values must be unique")
        known = set(ids)
        for job in self.jobs:
            if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", job.id):
                raise QueueValidationError(f"invalid queue job ID {job.id!r}")
            if job.order < 0 or job.attempts < 0:
                raise QueueValidationError("job order and attempts cannot be negative")
            missing = set(job.parent_job_ids) - known
            if missing:
                raise QueueValidationError(f"job {job.id!r} has missing dependencies {sorted(missing)}")
            if job.model_parent_job_id is not None and job.model_parent_job_id not in known:
                raise QueueValidationError(f"job {job.id!r} has missing model parent")
        active = [job.id for job in self.jobs if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}]
        if len(active) > 1:
            raise QueueValidationError("queue permits only one active job")
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        graph = {job.id: job.parent_job_ids for job in self.jobs}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(job_id: str) -> None:
            if job_id in visiting:
                raise QueueValidationError("queue dependency cycle detected")
            if job_id in visited:
                return
            visiting.add(job_id)
            for parent in graph[job_id]:
                visit(parent)
            visiting.remove(job_id)
            visited.add(job_id)

        for job_id in graph:
            visit(job_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_root": self.project_root,
            "jobs": [job.to_dict() for job in self.jobs],
            "created_at": self.created_at,
            "revision": self.revision,
            "selections": self.selections,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueueState:
        state = cls(
            schema_version=data["schema_version"],
            project_root=data["project_root"],
            jobs=tuple(QueueJob.from_dict(job) for job in data["jobs"]),
            created_at=data["created_at"],
            revision=data.get("revision", 0),
            selections=dict(data.get("selections", {})),
        )
        state.validate()
        return state
