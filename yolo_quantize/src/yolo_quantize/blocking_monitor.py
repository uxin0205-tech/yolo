"""單一shell子程序內等待狀態事件；平時不讀log或查GPU。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

TERMINAL = {
    "complete",
    "completed",
    "error",
    "failed",
    "cancelled",
    "paused",
    "decision_required",
    "cpu_preflight_complete_pending_integration",
}
FIELDS = (
    "status",
    "current_index",
    "current_candidate",
    "current_arm",
    "completed_jobs",
    "error",
)


def snapshot(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not set(FIELDS).issubset(value):
            raise ValueError("status JSON missing required monitor fields")
        result = {key: value[key] for key in FIELDS}
        jobs = result["completed_jobs"]
        if isinstance(jobs, list):
            result["completed_jobs"] = len(jobs)
        elif isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 0:
            raise ValueError("completed_jobs must be a list or nonnegative integer")
        return result
    except (OSError, ValueError, TypeError) as error:
        return {
            "status": "error",
            "current_index": None,
            "current_candidate": None,
            "current_arm": None,
            "completed_jobs": 0,
            "error": {"kind": "monitor_read_error", "message": str(error)},
        }


def wait_for_event(path: Path, *, interval: float = 600, sleep=time.sleep) -> dict:
    if interval <= 0:
        raise ValueError("monitor interval must be positive")
    baseline = snapshot(path)
    if baseline["status"] in TERMINAL or baseline["error"] is not None:
        return baseline
    while True:
        sleep(interval)
        current = snapshot(path)
        if current != baseline:
            return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            wait_for_event(args.status), ensure_ascii=False, separators=(",", ":")
        )
    )


if __name__ == "__main__":
    main()
