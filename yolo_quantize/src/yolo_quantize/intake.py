"""Read-only intake checks for the upstream activation project."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

PRIMARY_METRICS = (
    "coco/box/map50_95",
    "bbat/box/map50_95",
    "bbat/pose/map50_95",
)
QUANTIZATION_PRIMARY_MAX_DROP = 0.04

ACTIVATION_REQUIRED_FILES = (
    ("qsilu_float_source", "src/activation_lab/activations.py"),
    ("qsilu_bittrue_source", "src/activation_lab/hardware.py"),
    ("activation_manifest", "training/full35/contracts/activation-manifest.yaml"),
    (
        "activation_accepted_baseline",
        "training/full35/contracts/accepted-full35-baseline.yaml",
    ),
    ("activation_recipe", "training/full35/activation-recipe.yaml"),
)

FULL35_REQUIRED_FILES = (
    ("full35_release_manifest", "MANIFEST.json"),
    ("full35_release_status", "RELEASE_STATUS.json"),
    ("full35_joint_config", "configs/joint.yaml"),
    ("full35_accepted_checkpoint", "weights/combined/inference/best_joint.pt"),
)


@dataclass(frozen=True)
class FileDigest:
    """SHA-256 evidence for one immutable intake dependency."""

    role: str
    owner: str
    relative_path: str
    bytes: int
    sha256: str
    expected_sha256: str | None = None

    @property
    def matches_expected(self) -> bool | None:
        if self.expected_sha256 is None:
            return None
        return self.sha256 == self.expected_sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "owner": self.owner,
            "relative_path": self.relative_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
            "matches_expected": self.matches_expected,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checksum_index(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            continue
        checksum, relative = fields
        checksums[relative.lstrip("*")] = checksum
    return checksums


@dataclass(frozen=True)
class ActivationJobSummary:
    """Queue evidence for one activation experiment job."""

    queue_name: str
    job_id: str
    run_name: str
    activation: str
    kind: str
    phase: str | None
    status: str
    upstream_gate_status: str
    quantization_eligibility: str
    classification: str
    gate_passed: bool | None
    worst_delta: float | None
    failed_metrics: tuple[str, ...]
    deltas: dict[str, float]
    metrics: dict[str, float]
    primary_metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "queue_name": self.queue_name,
            "job_id": self.job_id,
            "run_name": self.run_name,
            "activation": self.activation,
            "kind": self.kind,
            "phase": self.phase,
            "status": self.status,
            "upstream_gate_status": self.upstream_gate_status,
            "quantization_eligibility": self.quantization_eligibility,
            "classification": self.classification,
            "gate_passed": self.gate_passed,
            "worst_delta": self.worst_delta,
            "failed_metrics": list(self.failed_metrics),
            "deltas": self.deltas,
            "metrics": self.metrics,
            "primary_metrics": self.primary_metrics,
        }


@dataclass(frozen=True)
class ActivationQueueSummary:
    """Serializable terminal-state summary for one upstream queue."""

    queue_name: str
    queue_id: str
    config_path: str
    state_path: str
    completed: int
    pending: int
    blocked: int
    next_job: str | None
    updated_at: str | None
    runner_active: bool

    @property
    def terminal(self) -> bool:
        return self.pending == 0 and self.next_job is None

    def to_dict(self) -> dict[str, object]:
        return {
            "queue_name": self.queue_name,
            "queue_id": self.queue_id,
            "config_path": self.config_path,
            "state_path": self.state_path,
            "completed": self.completed,
            "pending": self.pending,
            "blocked": self.blocked,
            "next_job": self.next_job,
            "updated_at": self.updated_at,
            "runner_active": self.runner_active,
            "terminal": self.terminal,
        }


def _runner_lock_is_held(state_path: Path) -> bool:
    """Check the queue runner's advisory lock without changing its file."""

    lock_path = state_path.with_suffix(".lock")
    if not lock_path.is_file():
        return False
    with lock_path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
            return False


@dataclass(frozen=True)
class ActivationIntakeReport:
    """Whether upstream activation evidence is ready for quantization intake."""

    ready: bool
    blockers: tuple[str, ...]
    queues: tuple[ActivationQueueSummary, ...]
    jobs: tuple[ActivationJobSummary, ...]
    files: tuple[FileDigest, ...]
    finalization_only_blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def finalization_ready(self) -> bool:
        return self.ready

    @property
    def screening_blockers(self) -> tuple[str, ...]:
        finalization_only = set(self.finalization_only_blockers)
        blockers = [
            blocker for blocker in self.blockers if blocker not in finalization_only
        ]
        has_candidate = any(
            job.status == "completed"
            and job.kind == "train"
            and job.phase == "short_recovery"
            and job.quantization_eligibility == "screen_candidate"
            for job in self.jobs
        )
        if not has_candidate:
            blockers.append("no completed short-recovery screen candidate exists")
        return tuple(blockers)

    @property
    def screening_ready(self) -> bool:
        return not self.screening_blockers

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "ready": self.ready,
            "screening_ready": self.screening_ready,
            "finalization_ready": self.finalization_ready,
            "blockers": list(self.blockers),
            "screening_blockers": list(self.screening_blockers),
            "finalization_only_blockers": list(self.finalization_only_blockers),
            "warnings": list(self.warnings),
            "queues": [queue.to_dict() for queue in self.queues],
            "jobs": [job.to_dict() for job in self.jobs],
            "files": [file.to_dict() for file in self.files],
        }


@dataclass(frozen=True)
class ActivationIntake:
    """Inspect activation queues without mutating their workspace."""

    activation_root: Path
    full35_root: Path

    @classmethod
    def from_workspace(
        cls,
        *,
        activation_root: Path,
        full35_root: Path,
    ) -> ActivationIntake:
        return cls(
            activation_root=Path(activation_root),
            full35_root=Path(full35_root),
        )

    def analyze(self) -> ActivationIntakeReport:
        blockers: list[str] = []
        finalization_only_blockers: list[str] = []
        warnings: list[str] = []
        queues: list[ActivationQueueSummary] = []
        jobs: list[ActivationJobSummary] = []
        files: list[FileDigest] = []

        checksum_path = self.full35_root / "CHECKSUMS.sha256"
        full35_checksums = _checksum_index(checksum_path)
        if not checksum_path.is_file():
            blockers.append("Full35 checksum index is missing: CHECKSUMS.sha256")

        for owner, root, requirements, expected in (
            (
                "yolo_activation",
                self.activation_root,
                ACTIVATION_REQUIRED_FILES,
                {},
            ),
            ("full35", self.full35_root, FULL35_REQUIRED_FILES, full35_checksums),
        ):
            for role, relative_path in requirements:
                path = root / relative_path
                if not path.is_file():
                    blockers.append(
                        f"{owner} required file is missing: {relative_path}"
                    )
                    continue
                expected_sha256 = expected.get(relative_path)
                digest = FileDigest(
                    role=role,
                    owner=owner,
                    relative_path=relative_path,
                    bytes=path.stat().st_size,
                    sha256=_sha256(path),
                    expected_sha256=expected_sha256,
                )
                files.append(digest)
                if (
                    expected_sha256 is None
                    and owner == "full35"
                    and role != "full35_release_manifest"
                ):
                    blockers.append(
                        f"Full35 checksum is missing for required file: {relative_path}"
                    )
                elif digest.matches_expected is False:
                    blockers.append(f"Full35 checksum mismatch: {relative_path}")
        queue_directory = self.activation_root / "training/full35"
        queue_configs = tuple(sorted(queue_directory.glob("*queue.yaml")))

        if not queue_configs:
            blockers.append("no activation queue definitions were found")

        for queue_config in queue_configs:
            queue_config_relative = str(queue_config.relative_to(self.activation_root))
            files.append(
                FileDigest(
                    role=f"queue_config:{queue_config.stem}",
                    owner="yolo_activation",
                    relative_path=queue_config_relative,
                    bytes=queue_config.stat().st_size,
                    sha256=_sha256(queue_config),
                )
            )
            try:
                config = yaml.safe_load(queue_config.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError):
                blockers.append(f"{queue_config.stem} config is malformed YAML")
                continue
            if not isinstance(config, dict) or not isinstance(
                config.get("execution"), dict
            ):
                blockers.append(f"{queue_config.stem} config schema is malformed")
                continue
            state_relative = config["execution"].get("state")
            if not isinstance(state_relative, str) or not state_relative:
                blockers.append(f"{queue_config.stem} config has no state path")
                continue
            state_path = self.activation_root / state_relative
            if not state_path.is_file():
                blockers.append(
                    f"{queue_config.stem} state is missing: {state_relative}"
                )
                continue
            files.append(
                FileDigest(
                    role=f"queue_state:{queue_config.stem}",
                    owner="yolo_activation",
                    relative_path=state_relative,
                    bytes=state_path.stat().st_size,
                    sha256=_sha256(state_path),
                )
            )
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                blockers.append(f"{queue_config.stem} state is malformed JSON")
                continue
            if not isinstance(state, dict) or not isinstance(state.get("counts"), dict):
                blockers.append(f"{queue_config.stem} state schema is malformed")
                continue
            counts = state.get("counts", {})
            pending = int(counts.get("pending", 0))
            runner_active = _runner_lock_is_held(state_path)
            if runner_active:
                blockers.append(f"{queue_config.stem} runner lock is still held")
            queue_name = str(state.get("queue", queue_config.stem))
            queues.append(
                ActivationQueueSummary(
                    queue_name=queue_name,
                    queue_id=str(config.get("queue_id", "")),
                    config_path=str(queue_config.relative_to(self.activation_root)),
                    state_path=str(state_relative),
                    completed=int(counts.get("completed", 0)),
                    pending=pending,
                    blocked=int(counts.get("blocked", 0)),
                    next_job=(
                        str(state["next_job"])
                        if state.get("next_job") is not None
                        else None
                    ),
                    updated_at=(
                        str(state["updated_at"])
                        if state.get("updated_at") is not None
                        else None
                    ),
                    runner_active=runner_active,
                )
            )

            raw_specs = config.get("jobs", [])
            if not isinstance(raw_specs, list):
                blockers.append(f"{queue_config.stem} jobs schema is malformed")
                raw_specs = []
            specs = {
                str(spec["id"]): spec
                for spec in raw_specs
                if isinstance(spec, dict) and "id" in spec
            }
            baseline_metrics: dict[str, float] = {}
            gate_config = config.get("accuracy_gate")
            if isinstance(gate_config, dict) and gate_config.get("baseline"):
                baseline_path = queue_config.parent / str(gate_config["baseline"])
                if baseline_path.is_file():
                    try:
                        baseline_payload = yaml.safe_load(
                            baseline_path.read_text(encoding="utf-8")
                        )
                        raw_metrics = baseline_payload.get("metrics", {})
                        baseline_metrics = {
                            str(name): float(value)
                            for name, value in raw_metrics.items()
                        }
                    except (
                        AttributeError,
                        OSError,
                        TypeError,
                        ValueError,
                        yaml.YAMLError,
                    ):
                        blockers.append(
                            f"{queue_config.stem} baseline metrics are malformed"
                        )

            raw_state_jobs = state.get("jobs", [])
            if not isinstance(raw_state_jobs, list):
                blockers.append(f"{queue_config.stem} state jobs are malformed")
                raw_state_jobs = []
            for raw_job in raw_state_jobs:
                if not isinstance(raw_job, dict):
                    blockers.append(f"{queue_config.stem} state job is malformed")
                    continue
                job_id = str(raw_job.get("id", ""))
                spec = specs.get(job_id, {})
                status = str(raw_job.get("status", "unknown"))
                activation = str(spec.get("activation", "unknown"))
                kind = str(spec.get("kind", "unknown"))
                run_name = str(raw_job.get("run", spec.get("run_name", "")))
                gate = raw_job.get("gate")
                raw_gate_passed = gate.get("passed") if isinstance(gate, dict) else None
                gate_passed = (
                    raw_gate_passed if isinstance(raw_gate_passed, bool) else None
                )
                raw_deltas = gate.get("deltas", {}) if isinstance(gate, dict) else {}
                deltas = {str(name): float(value) for name, value in raw_deltas.items()}
                metrics = {
                    name: baseline_metrics[name] + delta
                    for name, delta in deltas.items()
                    if name in baseline_metrics
                }
                primary_deltas = {
                    name: deltas[name] for name in PRIMARY_METRICS if name in deltas
                }
                if status == "pending":
                    upstream_gate_status = "pending"
                    quantization_eligibility = "provisional"
                elif status == "blocked":
                    upstream_gate_status = "blocked"
                    quantization_eligibility = "not_run"
                elif activation == "silu":
                    upstream_gate_status = "control"
                    quantization_eligibility = "control_only"
                else:
                    if gate_passed is True:
                        upstream_gate_status = "passed"
                    elif gate_passed is False:
                        upstream_gate_status = "failed"
                    else:
                        upstream_gate_status = "not_evaluated"
                    if status != "completed" or len(primary_deltas) != len(
                        PRIMARY_METRICS
                    ):
                        quantization_eligibility = "insufficient_evidence"
                    elif min(primary_deltas.values()) >= -QUANTIZATION_PRIMARY_MAX_DROP:
                        quantization_eligibility = "screen_candidate"
                    else:
                        quantization_eligibility = "outside_screen_gate"
                classification = quantization_eligibility
                jobs.append(
                    ActivationJobSummary(
                        queue_name=queue_name,
                        job_id=job_id,
                        run_name=run_name,
                        activation=activation,
                        kind=kind,
                        phase=(
                            str(spec["phase"])
                            if spec.get("phase") is not None
                            else None
                        ),
                        status=status,
                        upstream_gate_status=upstream_gate_status,
                        quantization_eligibility=quantization_eligibility,
                        classification=classification,
                        gate_passed=gate_passed,
                        worst_delta=(
                            float(gate["worst_delta"])
                            if isinstance(gate, dict)
                            and gate.get("worst_delta") is not None
                            else None
                        ),
                        failed_metrics=tuple(
                            str(value)
                            for value in (
                                gate.get("failed_metrics", ())
                                if isinstance(gate, dict)
                                else ()
                            )
                        ),
                        deltas=deltas,
                        metrics=metrics,
                        primary_metrics={
                            name: metrics[name]
                            for name in PRIMARY_METRICS
                            if name in metrics
                        },
                    )
                )
                if status == "completed":
                    if kind == "validate":
                        run_relative = (
                            Path("artifacts/runs/full35/evaluations") / run_name
                        )
                        required_artifacts = (
                            ("activation-summary.json", "activation_summary"),
                            (
                                "epoch-0000/bittrue/metrics.json",
                                "completed_bittrue_metrics",
                            ),
                        )
                    else:
                        run_relative = Path("artifacts/runs/full35") / run_name
                        required_artifacts = (
                            ("activation-experiment.json", "activation_experiment"),
                            ("inference/best_joint.pt", "completed_checkpoint"),
                        )
                    for artifact_suffix, artifact_role in required_artifacts:
                        artifact_relative = run_relative / artifact_suffix
                        artifact_path = self.activation_root / artifact_relative
                        if not artifact_path.is_file():
                            blockers.append(
                                f"{job_id} completed {kind} run is missing: "
                                f"{artifact_relative.as_posix()}"
                            )
                            continue
                        files.append(
                            FileDigest(
                                role=f"{artifact_role}:{job_id}",
                                owner="yolo_activation",
                                relative_path=artifact_relative.as_posix(),
                                bytes=artifact_path.stat().st_size,
                                sha256=_sha256(artifact_path),
                            )
                        )
                    if kind == "train":
                        metric_paths = sorted(
                            (self.activation_root / run_relative).glob(
                                "validation/epoch-*/bittrue/metrics.json"
                            )
                        )
                        if not metric_paths:
                            blockers.append(
                                f"{job_id} completed train run has no BitTrue metrics"
                            )
                        for metric_path in metric_paths:
                            metric_relative = metric_path.relative_to(
                                self.activation_root
                            )
                            files.append(
                                FileDigest(
                                    role=f"completed_bittrue_metrics:{job_id}",
                                    owner="yolo_activation",
                                    relative_path=metric_relative.as_posix(),
                                    bytes=metric_path.stat().st_size,
                                    sha256=_sha256(metric_path),
                                )
                            )
            if pending:
                pending_message = f"{queue_config.stem} has {pending} pending job(s)"
                blockers.append(pending_message)
                finalization_only_blockers.append(pending_message)
                if not runner_active:
                    interrupted_message = (
                        f"{queue_config.stem} is interrupted: pending jobs remain "
                        "but its runner lock is not held"
                    )
                    blockers.append(interrupted_message)
                    finalization_only_blockers.append(interrupted_message)
                    warnings.append(interrupted_message)

        return ActivationIntakeReport(
            ready=not blockers,
            blockers=tuple(blockers),
            queues=tuple(queues),
            jobs=tuple(jobs),
            files=tuple(files),
            finalization_only_blockers=tuple(finalization_only_blockers),
            warnings=tuple(warnings),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="唯讀檢查 yolo_activation 是否可交接給量化專案",
    )
    parser.add_argument(
        "--activation-root",
        type=Path,
        default=Path("/home/uxin/yolo/yolo_activation"),
    )
    parser.add_argument(
        "--full35-root",
        type=Path,
        default=Path("/home/uxin/yolo/yolo_combine/final/full35"),
    )
    parser.add_argument(
        "--gate",
        choices=("screening", "finalization"),
        default="finalization",
    )
    args = parser.parse_args(argv)
    report = ActivationIntake.from_workspace(
        activation_root=args.activation_root,
        full35_root=args.full35_root,
    ).analyze()
    print(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    selected_ready = (
        report.screening_ready
        if args.gate == "screening"
        else report.finalization_ready
    )
    return 0 if selected_ready else 2


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
