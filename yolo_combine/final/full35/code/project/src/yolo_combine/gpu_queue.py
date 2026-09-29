"""Persistent, fail-closed execution of one command after a GPU becomes idle."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    name: str
    used_memory_mib: int


@dataclass(frozen=True)
class GPUSnapshot:
    index: int
    uuid: str
    total_memory_mib: int
    free_memory_mib: int
    utilization_percent: int
    compute_processes: tuple[GPUProcess, ...]


@dataclass(frozen=True)
class GPUIdlePolicy:
    min_free_memory_mib: int = 30_000
    max_utilization_percent: int = 10
    stable_polls: int = 3
    poll_seconds: int = 60

    def __post_init__(self) -> None:
        if self.min_free_memory_mib < 1:
            raise ValueError("min_free_memory_mib must be positive")
        if not 0 <= self.max_utilization_percent <= 100:
            raise ValueError("max_utilization_percent must be in [0, 100]")
        if self.stable_polls < 1:
            raise ValueError("stable_polls must be positive")
        if self.poll_seconds < 1:
            raise ValueError("poll_seconds must be positive")

    def accepts(self, snapshot: GPUSnapshot) -> bool:
        return (
            not snapshot.compute_processes
            and snapshot.free_memory_mib >= self.min_free_memory_mib
            and snapshot.utilization_percent <= self.max_utilization_percent
        )


class GPUProbe(Protocol):
    def snapshot(self, index: int) -> GPUSnapshot: ...


class NvidiaSMIProbe:
    """Read target-GPU memory/utilization and UUID-filtered compute processes."""

    @staticmethod
    def _run(arguments: Sequence[str]) -> str:
        completed = subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def snapshot(self, index: int) -> GPUSnapshot:
        if index < 0:
            raise ValueError("GPU index cannot be negative")
        gpu_output = self._run(
            (
                f"--id={index}",
                "--query-gpu=uuid,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            )
        )
        rows = list(csv.reader(gpu_output.splitlines()))
        if len(rows) != 1 or len(rows[0]) != 4:
            raise RuntimeError(f"unexpected nvidia-smi GPU output: {gpu_output!r}")
        uuid, total, free, utilization = (value.strip() for value in rows[0])

        process_output = self._run(
            (
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            )
        )
        processes: list[GPUProcess] = []
        if process_output:
            for row in csv.reader(process_output.splitlines()):
                if len(row) != 4:
                    raise RuntimeError(
                        f"unexpected nvidia-smi process output row: {row!r}"
                    )
                process_uuid, pid, name, used_memory = (
                    value.strip() for value in row
                )
                if process_uuid == uuid:
                    processes.append(
                        GPUProcess(
                            pid=int(pid),
                            name=name,
                            used_memory_mib=int(used_memory),
                        )
                    )
        return GPUSnapshot(
            index=index,
            uuid=uuid,
            total_memory_mib=int(total),
            free_memory_mib=int(free),
            utilization_percent=int(utilization),
            compute_processes=tuple(sorted(processes, key=lambda item: item.pid)),
        )


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


class QueuedGPUJob:
    """Wait for a stable idle window, launch once, and preserve durable status."""

    def __init__(
        self,
        *,
        queue_id: str,
        gpu_index: int,
        policy: GPUIdlePolicy,
        command: Sequence[str],
        working_directory: str | Path,
        state_path: str | Path,
        log_path: str | Path,
        lock_path: str | Path,
        probe: GPUProbe | None = None,
    ) -> None:
        if not queue_id or any(character.isspace() for character in queue_id):
            raise ValueError("queue_id must be a non-empty token")
        if gpu_index < 0:
            raise ValueError("gpu_index cannot be negative")
        if not command:
            raise ValueError("queued command cannot be empty")
        self.queue_id = queue_id
        self.gpu_index = gpu_index
        self.policy = policy
        self.command = tuple(str(value) for value in command)
        self.working_directory = Path(working_directory).expanduser().resolve()
        self.state_path = Path(state_path).expanduser().resolve()
        self.log_path = Path(log_path).expanduser().resolve()
        self.lock_path = Path(lock_path).expanduser().resolve()
        self.probe = probe or NvidiaSMIProbe()
        if not self.working_directory.is_dir():
            raise FileNotFoundError(self.working_directory)

    def _state(self, status: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "queue_id": self.queue_id,
            "status": status,
            "updated_at_utc": _now(),
            "gpu_index": self.gpu_index,
            "policy": asdict(self.policy),
            "command": list(self.command),
            "working_directory": str(self.working_directory),
            "log_path": str(self.log_path),
            **extra,
        }
        _atomic_json(self.state_path, payload)

    def run(self) -> int:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"queue lock is already held: {self.lock_path}"
                ) from error
            return self._run_locked()

    def _run_locked(self) -> int:
        queued_at = _now()
        idle_streak = 0
        poll_count = 0
        self._state("queued", queued_at_utc=queued_at, idle_streak=idle_streak)
        while idle_streak < self.policy.stable_polls:
            poll_count += 1
            try:
                snapshot = self.probe.snapshot(self.gpu_index)
            except Exception as error:
                idle_streak = 0
                self._state(
                    "probe_error",
                    queued_at_utc=queued_at,
                    poll_count=poll_count,
                    idle_streak=idle_streak,
                    error=f"{type(error).__name__}: {error}",
                )
                print(f"[{_now()}] GPU probe failed: {error}", flush=True)
                time.sleep(self.policy.poll_seconds)
                continue

            idle = self.policy.accepts(snapshot)
            idle_streak = idle_streak + 1 if idle else 0
            self._state(
                "waiting" if idle_streak < self.policy.stable_polls else "ready",
                queued_at_utc=queued_at,
                poll_count=poll_count,
                idle_streak=idle_streak,
                snapshot=asdict(snapshot),
            )
            print(
                f"[{_now()}] GPU {self.gpu_index}: free={snapshot.free_memory_mib}MiB "
                f"util={snapshot.utilization_percent}% processes="
                f"{len(snapshot.compute_processes)} idle_streak="
                f"{idle_streak}/{self.policy.stable_polls}",
                flush=True,
            )
            if idle_streak < self.policy.stable_polls:
                time.sleep(self.policy.poll_seconds)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        launched_at = _now()
        with self.log_path.open("ab", buffering=0) as output:
            process = subprocess.Popen(
                self.command,
                cwd=self.working_directory,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            self._state(
                "running",
                queued_at_utc=queued_at,
                launched_at_utc=launched_at,
                child_pid=process.pid,
                poll_count=poll_count,
            )
            print(
                f"[{launched_at}] launched pid={process.pid}: "
                f"{' '.join(self.command)}",
                flush=True,
            )
            return_code = process.wait()
        finished_at = _now()
        self._state(
            "completed" if return_code == 0 else "failed",
            queued_at_utc=queued_at,
            launched_at_utc=launched_at,
            finished_at_utc=finished_at,
            return_code=return_code,
            poll_count=poll_count,
        )
        print(
            f"[{finished_at}] command exited with code {return_code}",
            flush=True,
        )
        return return_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wait for a stable idle GPU window, then run one command once."
    )
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=30_000)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    job = QueuedGPUJob(
        queue_id=args.queue_id,
        gpu_index=args.gpu_index,
        policy=GPUIdlePolicy(
            min_free_memory_mib=args.min_free_mib,
            max_utilization_percent=args.max_utilization,
            stable_polls=args.stable_polls,
            poll_seconds=args.poll_seconds,
        ),
        command=command,
        working_directory=args.working_directory,
        state_path=args.state,
        log_path=args.log,
        lock_path=args.lock,
    )
    return job.run()


__all__ = (
    "GPUIdlePolicy",
    "GPUProcess",
    "GPUSnapshot",
    "NvidiaSMIProbe",
    "QueuedGPUJob",
    "main",
)

