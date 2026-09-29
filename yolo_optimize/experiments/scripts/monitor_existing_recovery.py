#!/usr/bin/env python3
"""接手既有 recovery PID；每 400 秒完整監測，退出偵測不讀訓練指標。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from supervise_recovery import (
    WORKSPACE, RUNNER, _query_gpu, _disk_snapshot, _last_progress_event,
    _summary_status, _timestamp, _write_jsonl,
)


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if root == WORKSPACE or not root.is_relative_to(WORKSPACE) or not root.is_dir():
        raise ValueError("run 必須是 workspace 內既有子目錄")
    token = identity(args.pid)
    if token is None:
        raise ValueError("指定訓練 PID 已結束")
    command = Path(f"/proc/{args.pid}/cmdline").read_bytes().split(b"\0")
    command = [part.decode() for part in command if part]
    if str(RUNNER) not in command or "train" not in command or "--output" not in command:
        raise ValueError("PID 不是本專案 recovery train")
    if Path(command[command.index("--output") + 1]).resolve() != root:
        raise ValueError("PID output 與指定 run 不符")
    log = root.parent / "logs" / f"{root.name}.monitor-400.jsonl"
    if args.detach:
        error_log = log.with_suffix(".stderr.log")
        with error_log.open("x", encoding="utf-8") as stderr:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                "--pid", str(args.pid), "--run-dir", str(root)],
                cwd=WORKSPACE, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=stderr, start_new_session=True)
        print(json.dumps({"monitor_pid": child.pid, "training_pid": args.pid,
                          "interval_seconds": 400, "log": str(log)}), flush=True)
        return
    with log.open("x", encoding="utf-8"):
        pass
    next_probe = time.monotonic()
    while True:
        alive = identity(args.pid) == token
        now = time.monotonic()
        if not alive or now >= next_probe:
            status, error = _summary_status(root)
            event = {"event": "monitor" if alive else "process_exit_observed",
                "timestamp": _timestamp(), "pid": args.pid, "interval_seconds": 400,
                "process_alive": alive, "returncode": None,
                "exit_code_observable": False, "summary_status": status,
                "summary_error": error, "gpu": _query_gpu(), "disk": _disk_snapshot(),
                "progress": _last_progress_event(root / "progress.jsonl")}
            _write_jsonl(log, event)
            if not alive:
                return
            next_probe = now + 400
        time.sleep(10)  # 只觀察 PID 是否退出，不讀 loss／AP／GPU。


if __name__ == "__main__":
    main()
