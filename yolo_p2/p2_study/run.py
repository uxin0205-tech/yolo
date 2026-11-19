"""Crash-safe, single-instance controller for the complete P2 study."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from p2_study import ARTIFACTS, ROOT
from p2_study.worker import load_config, write_json


def now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def stages() -> list[dict]:
    """Return the fixed fair experiment schedule."""
    items = [
        {"id": "preflight", "args": ["preflight"]},
        {"id": "static_tests", "args": ["static"]},
        {"id": "prepare_weights", "args": ["prepare"]},
        {"id": "autobatch", "args": ["autobatch"]},
    ]
    for phase in ("gate1", "health"):
        for experiment in ("A1",):
            items.append(
                {
                    "id": f"{phase}_{experiment}",
                    "args": ["train", "--phase", phase, "--experiment", experiment],
                    "phase": phase,
                    "experiment": experiment,
                }
            )
    items.extend(
        [
            {
                "id": "formal_A1",
                "args": ["train", "--phase", "formal", "--experiment", "A1"],
                "phase": "formal",
                "experiment": "A1",
            },
            {
                "id": "staged_A2_head",
                "args": ["train_staged", "--stage", "head", "--experiment", "A2"],
                "phase": "staged",
                "experiment": "A2_head",
            },
            {
                "id": "formal_A2",
                "args": ["train_staged", "--stage", "full", "--experiment", "A2"],
                "phase": "formal",
                "experiment": "A2",
            },
        ]
    )
    for command in ("validate", "benchmark"):
        for experiment in ("A0", "A1", "A2"):
            items.append(
                {
                    "id": f"{command}_{experiment}",
                    "args": [command, "--experiment", experiment],
                    "experiment": experiment,
                }
            )
    items.append({"id": "analysis", "args": [], "analysis": True})
    return items


def load_state(config_path: Path) -> dict:
    """Load state or initialize a new immutable schedule."""
    path = ARTIFACTS / "state.json"
    if path.is_file():
        state = json.loads(path.read_text())
        active_stages = {item["id"] for item in stages()}
        state["stages"] = {stage_id: value for stage_id, value in state["stages"].items() if stage_id in active_stages}
        for item in stages():
            state["stages"].setdefault(item["id"], {"status": "pending", "attempts": []})
        return state
    return {
        "status": "pending",
        "created_at": now(),
        "updated_at": now(),
        "root": str(ROOT),
        "config": str(config_path),
        "pid": os.getpid(),
        "stages": {item["id"]: {"status": "pending", "attempts": []} for item in stages()},
    }


def save_state(state: dict) -> None:
    """Update state atomically."""
    state["updated_at"] = now()
    state["pid"] = os.getpid()
    write_json(ARTIFACTS / "state.json", state)


def checkpoint_for(item: dict) -> Path | None:
    """Locate the resumable last checkpoint for a training stage."""
    if "phase" not in item:
        return None
    path = ARTIFACTS / "runs" / item["phase"] / item["experiment"] / "weights/last.pt"
    return path if path.is_file() else None


def failure_report(state: dict, stage_id: str, log: Path) -> None:
    """Write a truthful terminal failure report."""
    attempts = state["stages"][stage_id]["attempts"]
    lines = [
        "# YOLO11m P2 實驗失敗報告",
        "",
        f"- 失敗階段：`{stage_id}`",
        f"- 嘗試次數：{len(attempts)}",
        f"- 最後日誌：`{log}`",
        f"- 時間：{now()}",
        "",
        "排程已停止，未產生勝者結論。請修復上述錯誤後使用 `./p2_study/ctl.sh resume`。",
        "",
    ]
    (ARTIFACTS / "FAILURE_REPORT.md").write_text("\n".join(lines))


def recover_oom(item: dict, state: dict) -> None:
    """Halve the global batch and preserve invalidated runs so comparisons remain fair."""
    manifest_path = ARTIFACTS / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    old_batch = int(manifest["batch"])
    new_batch = old_batch // 2
    if new_batch < 1:
        raise RuntimeError("OOM occurred at batch=1; no fair automatic reduction remains")
    manifest.update(
        {"batch": new_batch, "previous_batch": old_batch, "reduced_after_oom": item["id"], "updated_at": now()}
    )
    write_json(manifest_path, manifest)
    invalidated = ARTIFACTS / "invalidated" / f"batch_{old_batch}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    invalidated.mkdir(parents=True, exist_ok=True)
    if item.get("phase"):
        current = ARTIFACTS / "runs" / item["phase"] / item["experiment"]
        if current.exists():
            destination = invalidated / item["phase"] / item["experiment"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(current), str(destination))
    formal = ARTIFACTS / "runs/formal"
    completed_formal = any(
        state["stages"].get(f"formal_{name}", {}).get("status") == "completed" for name in ("A1", "A2")
    )
    if completed_formal and formal.exists():
        destination = invalidated / "formal"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(formal), str(destination))
        for stage_id, stage in state["stages"].items():
            if stage_id.startswith(("staged_", "formal_", "validate_", "benchmark_")) or stage_id == "analysis":
                stage.update({"status": "pending", "attempts": []})
        state["restart_schedule"] = True


def run_stage(item: dict, config_path: Path, state: dict) -> bool:
    """Run a stage with at most two real failures; interrupted controllers remain resumable."""
    stage = state["stages"][item["id"]]
    log_dir = ARTIFACTS / "logs/stages"
    log_dir.mkdir(parents=True, exist_ok=True)
    failed_attempts = sum(
        attempt.get("exit_code") not in (None, 0) and not attempt.get("resolved_by_code_fix")
        for attempt in stage["attempts"]
    )
    while failed_attempts < 2:
        attempt_number = len(stage["attempts"]) + 1
        args = [sys.executable, "-m", "p2_study.worker", *item["args"], "--config", str(config_path)]
        checkpoint = checkpoint_for(item)
        if attempt_number > 1 and checkpoint:
            args.extend(["--resume", str(checkpoint)])
        log = log_dir / f"{item['id']}.attempt{attempt_number}.log"
        attempt = {
            "attempt": attempt_number,
            "started_at": now(),
            "command": args,
            "log": str(log),
            "checkpoint": str(checkpoint) if checkpoint else None,
        }
        stage["status"] = "running"
        stage["attempts"].append(attempt)
        state["status"] = "running"
        state["current_stage"] = item["id"]
        save_state(state)
        started = time.monotonic()
        with log.open("a") as stream:
            stream.write(f"\n[{now()}] attempt {attempt_number}\n")
            stream.flush()
            result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        attempt.update(
            {"ended_at": now(), "duration_seconds": time.monotonic() - started, "exit_code": result.returncode}
        )
        if result.returncode == 0:
            stage["status"] = "completed"
            stage["completed_at"] = now()
            save_state(state)
            return True
        text = log.read_text(errors="replace")[-200_000:]
        lower = text.lower()
        attempt["error_type"] = (
            "oom"
            if "out of memory" in lower
            else "nan"
            if re.search(r"(?<![a-z])(nan|inf)(?![a-z])", lower)
            else "error"
        )
        failed_attempts += 1
        if attempt["error_type"] == "oom" and failed_attempts == 1:
            recover_oom(item, state)
        save_state(state)
    stage["status"] = "failed"
    state["status"] = "failed"
    state["failed_stage"] = item["id"]
    failure_report(state, item["id"], log)
    save_state(state)
    return False


def parse_args() -> argparse.Namespace:
    """Parse controller arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Acquire the controller lock and execute unfinished stages in order."""
    args = parse_args()
    config = load_config(args.config)
    config_path = Path(config["config_path"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lock_stream = (ARTIFACTS / "controller.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("Another P2 study controller is already running") from error
    state_path = ARTIFACTS / "state.json"
    if state_path.exists() and not args.resume:
        raise SystemExit("state.json already exists; use --resume to continue without duplicating work")
    state = load_state(config_path)
    if "formal_A0" in state["stages"] and state["stages"]["formal_A0"]["status"] != "completed":
        state["stages"]["formal_A0"].update(
            {
                "status": "skipped",
                "skipped_at": now(),
                "reason": "User selected the official YOLO11m checkpoint baseline",
            }
        )
    state["status"] = "running"
    save_state(state)
    schedule = stages()
    index = 0
    while index < len(schedule):
        item = schedule[index]
        if state["stages"][item["id"]]["status"] == "completed":
            index += 1
            continue
        if item.get("analysis"):
            command = [sys.executable, "-m", "p2_study.analyze", "--config", str(config_path)]
            item = {**item, "args": []}
            state["stages"][item["id"]]["status"] = "running"
            log = ARTIFACTS / "logs/stages/analysis.attempt1.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
            state["stages"][item["id"]]["attempts"].append(
                {"attempt": 1, "command": command, "log": str(log), "exit_code": result.returncode, "ended_at": now()}
            )
            if result.returncode:
                state["stages"][item["id"]]["status"] = "failed"
                state["status"] = "failed"
                failure_report(state, item["id"], log)
                save_state(state)
                return
            state["stages"][item["id"]]["status"] = "completed"
            save_state(state)
        elif not run_stage(item, config_path, state):
            return
        if state.pop("restart_schedule", False):
            save_state(state)
            index = 0
        else:
            index += 1
    state["status"] = "completed"
    state["completed_at"] = now()
    state.pop("current_stage", None)
    save_state(state)


if __name__ == "__main__":
    main()
