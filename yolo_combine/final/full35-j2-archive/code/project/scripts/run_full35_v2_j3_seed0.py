#!/usr/bin/env python3
"""Queue the isolated Full35 v2 J3 challenger and conditional final validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from yolo_combine.gpu_queue import GPUIdlePolicy, NvidiaSMIProbe


PARENT_RUN_NAME = "full35-joint-adamw-v2-j0e8-j1e20-j2e80-seed0"
RUN_NAME = "full35-joint-adamw-v2-j3-b32-challenger-seed0"
EVALUATION_NAME = "full35-v2-j3-best-joint-seed0"


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


def _wait_for_idle_gpu(
    *,
    gpu_index: int,
    policy: GPUIdlePolicy,
    state_path: Path,
) -> None:
    probe = NvidiaSMIProbe()
    streak = 0
    while streak < policy.stable_polls:
        snapshot = probe.snapshot(gpu_index)
        streak = streak + 1 if policy.accepts(snapshot) else 0
        payload = {
            "schema_version": 1,
            "status": "waiting_for_gpu",
            "updated_at_utc": _now(),
            "run_name": RUN_NAME,
            "parent_run_name": PARENT_RUN_NAME,
            "gpu": {
                "index": gpu_index,
                "free_memory_mib": snapshot.free_memory_mib,
                "utilization_percent": snapshot.utilization_percent,
                "compute_processes": [
                    asdict(process) for process in snapshot.compute_processes
                ],
                "idle_streak": streak,
                "required_streak": policy.stable_polls,
            },
            "policy": asdict(policy),
        }
        _atomic_json(state_path, payload)
        print(
            f"[{payload['updated_at_utc']}] GPU{gpu_index} "
            f"free={snapshot.free_memory_mib}MiB "
            f"util={snapshot.utilization_percent}% "
            f"processes={len(snapshot.compute_processes)} "
            f"idle={streak}/{policy.stable_polls}",
            flush=True,
        )
        if streak < policy.stable_polls:
            time.sleep(policy.poll_seconds)


def _run(
    command: Sequence[str],
    *,
    root: Path,
    state_path: Path,
    status: str,
) -> None:
    _atomic_json(
        state_path,
        {
            "schema_version": 1,
            "status": status,
            "updated_at_utc": _now(),
            "run_name": RUN_NAME,
            "parent_run_name": PARENT_RUN_NAME,
            "command": list(command),
        },
    )
    completed = subprocess.run(tuple(command), cwd=root, check=False)
    if completed.returncode:
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "failed",
                "failed_step": status,
                "return_code": completed.returncode,
                "updated_at_utc": _now(),
                "run_name": RUN_NAME,
                "parent_run_name": PARENT_RUN_NAME,
            },
        )
        raise subprocess.CalledProcessError(completed.returncode, command)


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
            "full35-v2-j3-b32-seed0/queue-status.json"
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
    python = root / ".venv/bin/python"
    formal_root = root / "variants/full35/artifacts/fusion/formal"
    parent_run_dir = formal_root / PARENT_RUN_NAME
    run_dir = formal_root / RUN_NAME
    evaluation_dir = formal_root / "evaluations" / EVALUATION_NAME
    parent_last = parent_run_dir / "checkpoints/last.pt"
    parent_best = parent_run_dir / "checkpoints/best_joint.pt"
    candidate_best = run_dir / "checkpoints/best_joint.pt"
    candidate_last = run_dir / "checkpoints/last.pt"

    commands = {
        "preflight": (
            str(python),
            "variants/full35/joint.py",
            "preflight",
        ),
        "train": (
            str(python),
            "variants/full35/joint.py",
            "train",
            "--device",
            str(args.gpu_index),
            "--name",
            RUN_NAME,
            "--resume",
            str(parent_last),
            "--enable-j3",
            "--j3-detect-microbatch",
            "32",
        ),
        "validate": (
            str(python),
            "variants/full35/joint.py",
            "validate",
            "--device",
            str(args.gpu_index),
            "--backend",
            "both",
            "--name",
            EVALUATION_NAME,
            "--checkpoint",
            str(candidate_best),
        ),
    }
    plan = {
        "run_name": RUN_NAME,
        "parent_run_name": PARENT_RUN_NAME,
        "run_dir": str(run_dir),
        "evaluation_dir": str(evaluation_dir),
        "state": str(state_path),
        "exact_resume": str(parent_last),
        "acceptance_reference": str(parent_best),
        "commands": commands,
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return 0

    for required in (parent_last, parent_best):
        if not required.is_file():
            raise FileNotFoundError(required)
    if run_dir.exists():
        raise FileExistsError(f"J3 output already exists; refuse overwrite: {run_dir}")
    if evaluation_dir.exists():
        raise FileExistsError(
            f"J3 evaluation output already exists; refuse overwrite: {evaluation_dir}"
        )

    _run(
        commands["preflight"],
        root=root,
        state_path=state_path,
        status="preflight",
    )
    policy = GPUIdlePolicy(
        min_free_memory_mib=args.min_free_mib,
        max_utilization_percent=args.max_utilization,
        stable_polls=args.stable_polls,
        poll_seconds=args.poll_seconds,
    )
    _wait_for_idle_gpu(
        gpu_index=args.gpu_index,
        policy=policy,
        state_path=state_path,
    )
    _run(
        commands["train"],
        root=root,
        state_path=state_path,
        status="training_j3",
    )
    if not candidate_last.is_file():
        raise FileNotFoundError(
            f"J3 completed without a last checkpoint: {candidate_last}"
        )

    candidate_improved = candidate_best.is_file()
    if candidate_improved:
        _run(
            commands["validate"],
            root=root,
            state_path=state_path,
            status="final_validation",
        )
    selected = candidate_best if candidate_improved else parent_best
    _atomic_json(
        state_path,
        {
            "schema_version": 1,
            "status": "completed",
            "updated_at_utc": _now(),
            "run_name": RUN_NAME,
            "parent_run_name": PARENT_RUN_NAME,
            "run_dir": str(run_dir),
            "candidate_improved_parent_best": candidate_improved,
            "candidate_best_joint": (
                str(candidate_best) if candidate_improved else None
            ),
            "selected_checkpoint": str(selected),
            "evaluation_dir": (
                str(evaluation_dir) if candidate_improved else None
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
