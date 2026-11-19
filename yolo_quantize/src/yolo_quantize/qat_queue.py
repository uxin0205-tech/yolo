"""Fail-closed queue for the matched-sham to QAT handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from .qat_runtime import Full35QATRuntime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class QueueRecorder:
    root: Path
    queue_id: str
    last_signature: str | None = None

    def transition(self, kind: str, **values: Any) -> None:
        payload = {
            "schema_version": 1,
            "queue_id": self.queue_id,
            "kind": kind,
            "time_unix": time.time(),
            "values": values,
        }
        signature = json.dumps(
            {"kind": kind, "values": values},
            ensure_ascii=False,
            sort_keys=True,
        )
        if signature == self.last_signature:
            return
        self.last_signature = signature
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        _atomic_json(self.root / "status.json", payload)


def _pid_alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    try:
        fields = stat.read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return False
    return len(fields) >= 3 and fields[2] != "Z"


def _process_tokens(pid: int) -> tuple[str, ...]:
    command = Path(f"/proc/{pid}/cmdline").read_bytes()
    return tuple(
        token.decode("utf-8", errors="strict")
        for token in command.split(b"\0")
        if token
    )


def _is_expected_sham_command(tokens: Sequence[str], plan_name: str) -> bool:
    pairs = tuple(pairwise(tokens))
    return (
        ("-m", "yolo_quantize.qat_runtime") in pairs
        and ("--arm", "sham") in pairs
        and any(
            left == "--plan" and Path(right).name == plan_name for left, right in pairs
        )
        and "--execute-reviewed-plan" in tokens
    )


def _compute_pids(device_index: int) -> tuple[int, ...]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={device_index}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    )


def _validate_qat_completion(runtime: Full35QATRuntime) -> dict[str, Any]:
    run_dir = runtime.plan.run_root / runtime.run_name("qat")
    manifest_path = run_dir / "qat-experiment.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"QAT completion manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "plan_sha256": runtime.plan.config_sha256,
        "candidate_id": runtime.plan.candidate_id,
        "arm": "qat",
        "run_name": runtime.run_name("qat"),
        "formal_validation": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"QAT completion field {field} drifted: "
                f"{payload.get(field)!r} != {value!r}"
            )
    if payload.get("completed_stages") != ["j3"]:
        raise ValueError("QAT completion does not contain exactly the j3 stage")
    epochs_completed = int(payload.get("epochs_completed", -1))
    epochs_planned = runtime.plan.training.epochs
    if not 0 < epochs_completed <= epochs_planned:
        raise ValueError("QAT completion epoch count is outside the reviewed budget")
    if epochs_completed != epochs_planned:
        early_stop = payload.get("early_stop")
        if not isinstance(early_stop, dict):
            raise ValueError("short QAT completion has no early-stop evidence")
        values = early_stop.get("values")
        if not isinstance(values, dict):
            raise ValueError("short QAT completion early-stop values are malformed")
        if (
            runtime.effective_patience < 1
            or int(values.get("patience", -1)) != runtime.effective_patience
            or int(values.get("stale_epochs", -1)) < runtime.effective_patience
            or int(values.get("should_stop", 0)) != 1
        ):
            raise ValueError("short QAT completion did not exhaust reviewed patience")
    paths = payload.get("checkpoint_paths")
    digests = payload.get("checkpoint_sha256")
    if not isinstance(paths, dict) or not isinstance(digests, dict):
        raise TypeError("QAT completion has no checkpoint path/hash mappings")
    checkpoint = Path(str(paths.get("best_joint", ""))).resolve()
    if run_dir.resolve() not in checkpoint.parents or not checkpoint.is_file():
        raise FileNotFoundError("QAT best_joint checkpoint is missing or outside run")
    actual_digest = _sha256(checkpoint)
    if digests.get("best_joint") != actual_digest:
        raise ValueError("QAT best_joint checkpoint SHA-256 drifted")
    return {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_digest,
        "epochs_completed": epochs_completed,
    }


def _resolve_qat_resume(run_dir: Path, *, allowed: bool) -> Path | None:
    if not run_dir.exists():
        return None
    if not allowed:
        raise FileExistsError(
            f"QAT run directory exists without completion evidence: {run_dir}"
        )
    checkpoint = (run_dir / "checkpoints" / "last.pt").resolve()
    if run_dir.resolve() not in checkpoint.parents or not checkpoint.is_file():
        raise FileNotFoundError("incomplete QAT run has no in-scope last checkpoint")
    return checkpoint


def run_queue(
    *,
    plan: Path,
    observed_sham_pid: int | None,
    device_index: int,
    poll_seconds: int,
    queue_root: Path,
    resume_incomplete_qat: bool = False,
    patience_continuation: Path | None = None,
) -> dict[str, Any]:
    if poll_seconds < 30:
        raise ValueError("queue poll interval must be at least 30 seconds")
    runtime = Full35QATRuntime.from_yaml(
        plan,
        patience_continuation=patience_continuation,
    )
    recorder = QueueRecorder(queue_root.resolve(), queue_id="v19-sham-to-qat-v1")
    sham_manifest = (
        runtime.plan.run_root / runtime.run_name("sham") / "qat-experiment.json"
    )
    sham_evidence: dict[str, Any] | None = None
    if sham_manifest.is_file():
        sham_evidence = runtime._assert_sham_ready()
    else:
        if observed_sham_pid is None:
            raise RuntimeError(
                "sham is incomplete and no observed sham PID was provided"
            )
        tokens = _process_tokens(observed_sham_pid)
        if not _is_expected_sham_command(tokens, runtime.plan.config_path.name):
            raise ValueError("observed PID is not the reviewed V19 sham command")
        recorder.transition(
            "observing_sham",
            pid=observed_sham_pid,
            manifest=str(sham_manifest),
            poll_seconds=poll_seconds,
        )
        while True:
            alive = _pid_alive(observed_sham_pid)
            if sham_manifest.is_file():
                sham_evidence = runtime._assert_sham_ready()
                if not alive:
                    break
                recorder.transition(
                    "sham_manifest_ready_waiting_exit",
                    pid=observed_sham_pid,
                    checkpoint_sha256=sham_evidence["checkpoint_sha256"],
                )
            elif not alive:
                raise RuntimeError(
                    "observed sham exited without a valid completion manifest"
                )
            time.sleep(poll_seconds)
    if sham_evidence is None:
        raise RuntimeError("sham evidence was not resolved")
    recorder.transition("sham_validated", **sham_evidence)

    qat_run_dir = runtime.plan.run_root / runtime.run_name("qat")
    if (qat_run_dir / "qat-experiment.json").is_file():
        completion = _validate_qat_completion(runtime)
        recorder.transition("queue_complete", reused_existing=True, **completion)
        return completion
    resume_checkpoint = _resolve_qat_resume(
        qat_run_dir,
        allowed=resume_incomplete_qat,
    )
    if runtime.patience_continuation is not None:
        if resume_checkpoint is None:
            raise FileNotFoundError(
                "patience continuation requires an incomplete QAT run"
            )
        resume_checkpoint = runtime.patience_continuation.resume_checkpoint
    while True:
        gpu_pids = _compute_pids(device_index)
        if not gpu_pids:
            break
        recorder.transition("waiting_for_gpu", pids=list(gpu_pids))
        time.sleep(poll_seconds)

    command_parts = [
        sys.executable,
        "-m",
        "yolo_quantize.qat_runtime",
        "--plan",
        str(runtime.plan.config_path),
        "--arm",
        "qat",
        "--device",
        str(device_index),
        "--execute-reviewed-plan",
    ]
    if resume_checkpoint is not None:
        command_parts.extend(("--resume", str(resume_checkpoint)))
    if runtime.patience_continuation is not None:
        command_parts.extend(
            (
                "--patience-continuation",
                str(runtime.patience_continuation.config_path),
            )
        )
    command = tuple(command_parts)
    log_path = recorder.root / "v19-qat.log"
    environment = os.environ.copy()
    source = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=runtime.plan.config_path.parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        recorder.transition(
            "qat_started",
            pid=process.pid,
            command=list(command),
            log=str(log_path),
            resumed=resume_checkpoint is not None,
            resume_checkpoint=(
                str(resume_checkpoint) if resume_checkpoint is not None else None
            ),
        )
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"V19 QAT exited with return code {return_code}")
    completion = _validate_qat_completion(runtime)
    recorder.transition("queue_complete", reused_existing=False, **completion)
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V19 matched-sham to QAT queue")
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"),
    )
    parser.add_argument("--observed-sham-pid", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument(
        "--queue-root",
        type=Path,
        default=Path("artifacts/queues/v19-sham-to-qat-v1"),
    )
    parser.add_argument("--execute-reviewed-plan", action="store_true")
    parser.add_argument("--resume-incomplete-qat", action="store_true")
    parser.add_argument("--patience-continuation", type=Path)
    args = parser.parse_args(argv)
    if not args.execute_reviewed_plan:
        parser.error("queue execution requires --execute-reviewed-plan")
    recorder = QueueRecorder(args.queue_root.resolve(), "v19-sham-to-qat-v1")
    try:
        completion = run_queue(
            plan=args.plan.resolve(),
            observed_sham_pid=args.observed_sham_pid,
            device_index=args.device,
            poll_seconds=args.poll_seconds,
            queue_root=args.queue_root,
            resume_incomplete_qat=args.resume_incomplete_qat,
            patience_continuation=args.patience_continuation,
        )
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as error:
        recorder.transition(
            "queue_failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        print(
            json.dumps(
                {"status": "failed", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {"status": "complete", **completion},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "QueueRecorder",
    "_is_expected_sham_command",
    "_resolve_qat_resume",
    "_validate_qat_completion",
    "main",
    "run_queue",
)
