#!/usr/bin/env python3
"""Fail-fast Full35 seed0 pipeline from diagnostic P1 through JOINT validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from yolo_combine.gpu_queue import GPUIdlePolicy, NvidiaSMIProbe


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]
    expected_files: tuple[Path, ...]
    output_root: Path | None
    needs_gpu: bool = True


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite_json(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    numbers = 0

    def visit(value: Any) -> None:
        nonlocal numbers
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, float)):
            numbers += 1
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite value in {path}: {value}")
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        raise TypeError(f"unexpected JSON value in {path}: {type(value).__name__}")

    visit(payload)
    if numbers == 0:
        raise ValueError(f"JSON artifact contains no numeric evidence: {path}")


def _finite_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV artifact contains no rows: {path}")
    numeric_values = 0
    for row in rows:
        for value in row.values():
            if value is None or not value.strip():
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            numeric_values += 1
            if not math.isfinite(numeric):
                raise ValueError(f"non-finite value in {path}: {value}")
    if numeric_values == 0:
        raise ValueError(f"CSV artifact contains no numeric evidence: {path}")


def _validate_artifact(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".pt":
        if path.stat().st_size < 1024:
            raise ValueError(f"checkpoint is unexpectedly small: {path}")
    elif path.suffix == ".json":
        _finite_json(path)
    elif path.suffix == ".csv":
        _finite_csv(path)


def _step_complete(step: PipelineStep) -> bool:
    if not step.expected_files:
        return False
    for path in step.expected_files:
        try:
            _validate_artifact(path)
        except (OSError, TypeError, ValueError):
            return False
    return True


def _wait_for_idle_gpu(
    *,
    gpu_index: int,
    policy: GPUIdlePolicy,
    step_name: str,
) -> None:
    probe = NvidiaSMIProbe()
    streak = 0
    while streak < policy.stable_polls:
        snapshot = probe.snapshot(gpu_index)
        streak = streak + 1 if policy.accepts(snapshot) else 0
        print(
            f"[{_now()}] step={step_name} GPU{gpu_index} "
            f"free={snapshot.free_memory_mib}MiB "
            f"util={snapshot.utilization_percent}% "
            f"processes={len(snapshot.compute_processes)} "
            f"idle_streak={streak}/{policy.stable_polls}",
            flush=True,
        )
        if streak < policy.stable_polls:
            time.sleep(policy.poll_seconds)


def build_steps(root: Path) -> tuple[PipelineStep, ...]:
    python = root / ".venv" / "bin" / "python"
    variant = root / "variants" / "full35"
    pose_root = variant / "artifacts" / "pose"
    diagnostic_name = "p0-full35-p1-diagnostic-f0p3-b128-e2-seed0"
    p1_name = "p0-full35-p1-b128-e17-seed0"
    p2_name = "p0-full35-p2-b64a2-e22-seed0"
    p3_name = "p0-full35-p3-b32a4-e100max-seed0"
    diagnostic = pose_root / "diagnostic" / diagnostic_name
    p1 = pose_root / p1_name
    p2 = pose_root / p2_name
    p3 = pose_root / p3_name
    baseline = variant / "artifacts" / "standalone-baseline" / "p3-seed0"
    gate = variant / "baselines" / "formal-gate-p3-seed0.json"
    joint = (
        variant
        / "artifacts"
        / "fusion"
        / "formal"
        / "full35-joint-adamw-seed0"
    )
    evaluation = (
        variant
        / "artifacts"
        / "fusion"
        / "formal"
        / "evaluations"
        / "queued-final-seed0"
        / "epoch-0000"
    )

    def pose_expected(run: Path, *, diagnostic_run: bool = False) -> tuple[Path, ...]:
        values = (
            run / "weights" / "best.pt",
            run / "weights" / "last.pt",
            run / "results.csv",
        )
        return values + (
            (run / "diagnostic-run.json",)
            if diagnostic_run
            else (run / "pose-stage-complete.json",)
        )

    return (
        PipelineStep(
            name="diagnostic_p1_f0p3_b128_e2",
            command=(
                str(python),
                "variants/full35/run.py",
                "pose",
                "--stage",
                "p1",
                "--device",
                "0",
                "--fraction",
                "0.3",
                "--epochs",
                "2",
                "--name",
                diagnostic_name,
            ),
            expected_files=pose_expected(diagnostic, diagnostic_run=True),
            output_root=diagnostic,
        ),
        PipelineStep(
            name="formal_pose_p1",
            command=(
                str(python),
                "variants/full35/run.py",
                "pose",
                "--stage",
                "p1",
                "--device",
                "0",
                "--name",
                p1_name,
            ),
            expected_files=pose_expected(p1),
            output_root=p1,
        ),
        PipelineStep(
            name="formal_pose_p2",
            command=(
                str(python),
                "variants/full35/run.py",
                "pose",
                "--stage",
                "p2",
                "--device",
                "0",
                "--name",
                p2_name,
                "--initial-checkpoint",
                str(p1 / "weights" / "best.pt"),
            ),
            expected_files=pose_expected(p2),
            output_root=p2,
        ),
        PipelineStep(
            name="formal_pose_p3",
            command=(
                str(python),
                "variants/full35/run.py",
                "pose",
                "--stage",
                "p3",
                "--device",
                "0",
                "--name",
                p3_name,
                "--initial-checkpoint",
                str(p2 / "weights" / "best.pt"),
            ),
            expected_files=pose_expected(p3),
            output_root=p3,
        ),
        PipelineStep(
            name="standalone_float_bittrue_baseline",
            command=(
                str(python),
                "variants/full35/baseline.py",
                "--device",
                "0",
                "--name",
                "p3-seed0",
                "--backend",
                "both",
                "--pose-checkpoint",
                str(p3 / "weights" / "best.pt"),
            ),
            expected_files=(
                gate,
                baseline / "validation" / "float" / "metrics.json",
                baseline / "validation" / "bittrue" / "metrics.json",
            ),
            output_root=baseline,
        ),
        PipelineStep(
            name="formal_preflight",
            command=(
                str(python),
                "variants/full35/joint.py",
                "preflight",
            ),
            expected_files=(),
            output_root=None,
            needs_gpu=False,
        ),
        PipelineStep(
            name="joint_j1_j2_adamw",
            command=(
                str(python),
                "variants/full35/joint.py",
                "train",
                "--device",
                "0",
                "--name",
                "full35-joint-adamw-seed0",
            ),
            expected_files=(
                joint / "checkpoints" / "best_detect.pt",
                joint / "checkpoints" / "best_pose.pt",
                joint / "checkpoints" / "best_joint.pt",
                joint / "checkpoints" / "last.pt",
                joint / "logs" / "validation.csv",
            ),
            output_root=joint,
        ),
        PipelineStep(
            name="joint_final_float_bittrue_validation",
            command=(
                str(python),
                "variants/full35/joint.py",
                "validate",
                "--device",
                "0",
                "--backend",
                "both",
                "--name",
                "queued-final-seed0",
                "--checkpoint",
                str(joint / "checkpoints" / "best_joint.pt"),
            ),
            expected_files=(
                evaluation / "float" / "metrics.json",
                evaluation / "bittrue" / "metrics.json",
            ),
            output_root=evaluation.parent,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(
            "variants/full35/artifacts/queue/"
            "full35-seed0-pipeline/pipeline-status.json"
        ),
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=30_000)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--stable-polls", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--plan", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    state_path = (
        args.state.expanduser().resolve()
        if args.state.is_absolute()
        else (root / args.state).resolve()
    )
    steps = build_steps(root)
    if args.plan:
        print(
            json.dumps(
                [
                    {
                        **asdict(step),
                        "expected_files": [str(path) for path in step.expected_files],
                        "output_root": (
                            str(step.output_root) if step.output_root else None
                        ),
                    }
                    for step in steps
                ],
                indent=2,
                default=str,
            )
        )
        return 0

    policy = GPUIdlePolicy(
        min_free_memory_mib=args.min_free_mib,
        max_utilization_percent=args.max_utilization,
        stable_polls=args.stable_polls,
        poll_seconds=args.poll_seconds,
    )
    history: list[dict[str, Any]] = []
    started_at = _now()
    for index, step in enumerate(steps):
        if _step_complete(step):
            history.append(
                {"step": step.name, "status": "skipped_complete", "at": _now()}
            )
            continue
        if step.output_root is not None and step.output_root.exists():
            raise RuntimeError(
                f"partial or invalid output blocks safe resume for {step.name}: "
                f"{step.output_root}"
            )
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "waiting_for_gpu" if step.needs_gpu else "running",
                "started_at_utc": started_at,
                "updated_at_utc": _now(),
                "current_step": step.name,
                "current_step_index": index,
                "steps_total": len(steps),
                "history": history,
                "policy": asdict(policy),
            },
        )
        if step.needs_gpu:
            _wait_for_idle_gpu(
                gpu_index=args.gpu_index,
                policy=policy,
                step_name=step.name,
            )
        step_started = _now()
        print(
            f"[{step_started}] START {step.name}: {' '.join(step.command)}",
            flush=True,
        )
        completed = subprocess.run(step.command, cwd=root, check=False)
        if completed.returncode != 0:
            history.append(
                {
                    "step": step.name,
                    "status": "failed",
                    "started_at_utc": step_started,
                    "finished_at_utc": _now(),
                    "return_code": completed.returncode,
                }
            )
            _atomic_json(
                state_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "started_at_utc": started_at,
                    "updated_at_utc": _now(),
                    "current_step": step.name,
                    "history": history,
                },
            )
            return completed.returncode
        for expected in step.expected_files:
            _validate_artifact(expected)
        history.append(
            {
                "step": step.name,
                "status": "completed",
                "started_at_utc": step_started,
                "finished_at_utc": _now(),
                "return_code": 0,
            }
        )
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "running",
                "started_at_utc": started_at,
                "updated_at_utc": _now(),
                "current_step": step.name,
                "history": history,
            },
        )

    _atomic_json(
        state_path,
        {
            "schema_version": 1,
            "status": "completed",
            "started_at_utc": started_at,
            "finished_at_utc": _now(),
            "history": history,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
