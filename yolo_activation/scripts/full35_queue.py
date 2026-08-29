#!/usr/bin/env python3
"""Run the Full35 activation experiment queue serially with compact output."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = PROJECT_ROOT / "training/full35/experiment-queue.yaml"
DEFAULT_RECIPE = PROJECT_ROOT / "training/full35/activation-recipe.yaml"
RUN_ROOT = PROJECT_ROOT / "artifacts/runs/full35"
CLI = PROJECT_ROOT / "scripts/full35_activation.py"


@dataclass(frozen=True)
class QueueJob:
    job_id: str
    kind: str
    activation: str
    run_name: str
    depends_on: tuple[str, ...]
    require_dependency_gate: bool
    backend: str = "both"
    phase: str | None = None
    metric_selection: str = "final_epoch"
    detect_microbatch: int = 32
    regions: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueueConfig:
    path: Path
    state_path: Path
    log_root: Path
    maximum_drop: float
    gate_metrics: tuple[str, ...]
    baseline: dict[str, float]
    jobs: tuple[QueueJob, ...]


def _path(value: str, *, base: Path = PROJECT_ROOT) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def load_queue(path: str | Path = DEFAULT_QUEUE) -> QueueConfig:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Full35 queue requires schema_version: 1")
    execution = payload.get("execution")
    gate = payload.get("accuracy_gate")
    raw_jobs = payload.get("jobs")
    if not isinstance(execution, dict) or not isinstance(gate, dict):
        raise TypeError("queue execution and accuracy_gate must be mappings")
    if not isinstance(raw_jobs, list):
        raise TypeError("queue jobs must be a list")
    if execution.get("mode") != "serial":
        raise ValueError("Full35 GPU queue must be serial")
    if int(execution.get("maximum_concurrent_gpu_jobs", 0)) != 1:
        raise ValueError("Full35 queue permits exactly one GPU job")

    baseline_path = _path(str(gate["baseline"]), base=source.parent)
    baseline_payload = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline_payload, dict) or not isinstance(
        baseline_payload.get("metrics"), dict
    ):
        raise TypeError("queue baseline contract is malformed")
    gate_metrics = tuple(str(value) for value in gate.get("metrics", ()))
    baseline = {name: float(baseline_payload["metrics"][name]) for name in gate_metrics}

    jobs: list[QueueJob] = []
    seen: set[str] = set()
    run_names: set[str] = set()
    for raw in raw_jobs:
        if not isinstance(raw, dict):
            raise TypeError("queue job must be a mapping")
        job = QueueJob(
            job_id=str(raw["id"]),
            kind=str(raw["kind"]),
            activation=str(raw["activation"]),
            run_name=str(raw["run_name"]),
            depends_on=tuple(str(value) for value in raw.get("depends_on", ())),
            require_dependency_gate=bool(raw.get("require_dependency_gate")),
            backend=str(raw.get("backend", "both")),
            phase=str(raw["phase"]) if raw.get("phase") else None,
            metric_selection=str(raw.get("metric_selection", "final_epoch")),
            detect_microbatch=int(raw.get("detect_physical_microbatch", 32)),
            regions=tuple(str(value) for value in raw.get("regions", ())),
        )
        if job.kind not in {"validate", "train"}:
            raise ValueError(f"unsupported queue kind: {job.kind}")
        if job.kind == "train" and job.phase is None:
            raise ValueError(f"train job has no phase: {job.job_id}")
        if job.metric_selection not in {"final_epoch", "best_joint"}:
            raise ValueError(
                f"unsupported metric selection: {job.job_id}: {job.metric_selection}"
            )
        if job.kind == "validate" and job.metric_selection != "final_epoch":
            raise ValueError(
                f"validation job requires final_epoch selection: {job.job_id}"
            )
        if job.detect_microbatch < 1 or 128 % job.detect_microbatch:
            raise ValueError(
                f"Detect physical microbatch must divide 128: {job.job_id}"
            )
        if job.job_id in seen or job.run_name in run_names:
            raise ValueError(f"duplicate queue identity: {job.job_id}")
        unknown = set(job.depends_on) - seen
        if unknown:
            raise ValueError(
                f"queue dependencies must appear earlier: {job.job_id}: {sorted(unknown)}"
            )
        seen.add(job.job_id)
        run_names.add(job.run_name)
        jobs.append(job)

    return QueueConfig(
        path=source,
        state_path=_path(str(execution["state"])),
        log_root=_path(str(execution["verbose_logs"])),
        maximum_drop=float(gate["maximum_map50_95_drop"]),
        gate_metrics=gate_metrics,
        baseline=baseline,
        jobs=tuple(jobs),
    )


def _phase_epochs() -> dict[str, int]:
    payload = yaml.safe_load(DEFAULT_RECIPE.read_text(encoding="utf-8"))
    return {
        str(name): int(values.get("epochs", 0))
        for name, values in payload["phases"].items()
    }


def _metrics_path(job: QueueJob) -> Path:
    if job.kind == "validate":
        return (
            RUN_ROOT / "evaluations" / job.run_name / "epoch-0000/bittrue/metrics.json"
        )
    epochs = _phase_epochs()[str(job.phase)]
    return (
        RUN_ROOT
        / job.run_name
        / "validation"
        / f"epoch-{epochs - 1:04d}"
        / "bittrue/metrics.json"
    )


def _job_metrics(job: QueueJob) -> dict[str, float] | None:
    if job.metric_selection == "best_joint":
        run_dir = RUN_ROOT / job.run_name
        if not (run_dir / "activation-experiment.json").is_file():
            return None
        events_path = run_dir / "logs/events.jsonl"
        if not events_path.is_file():
            return None
        best_step: int | None = None
        best_rank: tuple[int, float] | None = None
        for line in events_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("kind") != "gate":
                continue
            values = event.get("values", {})
            score = values.get("score/best_joint")
            if score is None:
                continue
            rank = (
                int(float(values.get("passed", 0.0)) >= 0.5),
                float(score),
            )
            if best_rank is None or rank > best_rank:
                best_step = int(event["step"])
                best_rank = rank
        if best_step is None:
            return None
        path = (
            run_dir / "validation" / f"epoch-{best_step:04d}" / "bittrue/metrics.json"
        )
    else:
        path = _metrics_path(job)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("metrics")
    if not isinstance(raw, dict):
        raise TypeError(f"job metrics are malformed: {path}")
    return {str(name): float(value) for name, value in raw.items()}


def _gate(config: QueueConfig, job: QueueJob) -> dict[str, Any] | None:
    metrics = _job_metrics(job)
    if metrics is None:
        return None
    deltas = {
        name: metrics[name] - config.baseline[name] for name in config.gate_metrics
    }
    failed = tuple(
        name for name, delta in deltas.items() if delta < -config.maximum_drop - 1e-12
    )
    return {
        "passed": not failed,
        "failed_metrics": failed,
        "worst_delta": min(deltas.values()),
        "deltas": deltas,
    }


def _command(job: QueueJob, *, device: str) -> list[str]:
    command = [
        sys.executable,
        str(CLI),
        job.kind,
        "--activation",
        job.activation,
        "--device",
        device,
    ]
    if job.kind == "validate":
        command.extend(("--backend", job.backend))
    else:
        command.extend(("--phase", str(job.phase)))
        command.extend(("--detect-microbatch", str(job.detect_microbatch)))
    for region in job.regions:
        command.extend(("--region", region))
    if job.regions:
        command.extend(("--policy-id", job.job_id))
    command.extend(("--name", job.run_name))
    return command


def _atomic_state(config: QueueConfig, payload: dict[str, Any]) -> None:
    config.state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.state_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(config.state_path)


def queue_status(config: QueueConfig) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    known: dict[str, dict[str, Any]] = {}
    next_job: str | None = None
    for job in config.jobs:
        gate = _gate(config, job)
        dependency_states = [known[name] for name in job.depends_on]
        blocked = any(
            state["status"] == "blocked" for state in dependency_states
        ) or bool(
            job.require_dependency_gate
            and any(
                state.get("gate") is not None and not state["gate"]["passed"]
                for state in dependency_states
            )
        )
        complete = gate is not None
        status = "completed" if complete else "blocked" if blocked else "pending"
        if (
            next_job is None
            and status == "pending"
            and all(state["status"] == "completed" for state in dependency_states)
        ):
            next_job = job.job_id
        item = {
            "id": job.job_id,
            "run": job.run_name,
            "status": status,
            "gate": gate,
        }
        known[job.job_id] = item
        jobs.append(item)
    return {
        "queue": config.path.stem,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "next_job": next_job,
        "counts": {
            name: sum(item["status"] == name for item in jobs)
            for name in ("completed", "pending", "blocked")
        },
        "jobs": jobs,
    }


def _compact(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, separators=(",", ":")), flush=True)


def run_queue(config: QueueConfig, *, device: str, max_jobs: int) -> int:
    config.log_root.mkdir(parents=True, exist_ok=True)
    lock_path = config.state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _compact("queue_locked", lock=str(lock_path))
            return 3
        completed_this_run = 0
        while True:
            status = queue_status(config)
            _atomic_state(config, status)
            job_id = status["next_job"]
            if job_id is None:
                _compact("queue_static_complete", counts=status["counts"])
                return 0
            if max_jobs and completed_this_run >= max_jobs:
                _compact("queue_yield", next_job=job_id, completed=completed_this_run)
                return 0
            job = next(value for value in config.jobs if value.job_id == job_id)
            log_path = config.log_root / f"{job.job_id}.log"
            _compact("job_start", job=job.job_id, run=job.run_name)
            with log_path.open("ab") as output:
                result = subprocess.run(
                    _command(job, device=device),
                    cwd=PROJECT_ROOT,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                _compact(
                    "job_failed",
                    job=job.job_id,
                    exit_code=result.returncode,
                    log=str(log_path),
                )
                return result.returncode
            gate = _gate(config, job)
            if gate is None:
                _compact("job_incomplete", job=job.job_id, log=str(log_path))
                return 4
            completed_this_run += 1
            _compact(
                "job_complete",
                job=job.job_id,
                gate=gate["passed"],
                worst_delta=round(float(gate["worst_delta"]), 6),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full35 serial activation queue")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    run = commands.add_parser("run")
    run.add_argument("--device", default="0")
    run.add_argument("--max-jobs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_queue(args.queue)
    if args.command == "status":
        status = queue_status(config)
        _atomic_state(config, status)
        _compact(
            "queue_status",
            next_job=status["next_job"],
            counts=status["counts"],
        )
        return 0
    if args.max_jobs < 0:
        raise ValueError("--max-jobs must be non-negative")
    return run_queue(config, device=args.device, max_jobs=args.max_jobs)


if __name__ == "__main__":
    raise SystemExit(main())
