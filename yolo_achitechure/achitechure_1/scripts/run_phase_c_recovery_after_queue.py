#!/usr/bin/env python3
"""Relay a running queue into deferred Phase C without operator prompts."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWN_WORKER_MANIFEST_ERROR = "run does not use batch=16, nbs=16, workers=4"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _write_relay_state(path: Path, status: str, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updated_at": _utc_now(), **values}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _wait_for_source(
    *,
    source_state: Path,
    source_run_tag: str,
    source_full35_state: Path,
    workers: int,
    relay_state: Path,
) -> None:
    relayed_worker_manifest_error = False
    while True:
        if not source_state.is_file():
            _write_relay_state(relay_state, "waiting_for_source_state", source_state=str(source_state))
            time.sleep(30)
            continue
        source = _read_json(source_state)
        status = source.get("status")
        if status in {"waiting_for_phase_c_gpu", "completed"}:
            _write_relay_state(relay_state, "source_ready", source_status=status)
            return
        if status == "failed":
            message = str(source.get("error", {}).get("message", ""))
            complete = (
                PROJECT_ROOT
                / "artifacts/runs"
                / f"a2-partial75-phase-a1-{source_run_tag}"
                / "training-complete.json"
            )
            if (
                not relayed_worker_manifest_error
                and KNOWN_WORKER_MANIFEST_ERROR in message
                and complete.is_file()
            ):
                relayed_worker_manifest_error = True
                _write_relay_state(
                    relay_state,
                    "resuming_after_known_manifest_mismatch",
                    source_error=message,
                    completed_a1=str(complete),
                )
                print(
                    f"[{_utc_now()}] A1 completed; resuming the same queue with workers={workers}",
                    flush=True,
                )
                from achitechure_1.queue import run_partial75_continuation_queue

                run_partial75_continuation_queue(
                    PROJECT_ROOT,
                    source_run_tag,
                    source_full35_state=source_full35_state,
                    workers=workers,
                )
                continue
            raise RuntimeError(f"source queue failed and cannot be relayed safely: {message}")
        _write_relay_state(
            relay_state,
            "waiting_for_source_queue",
            source_status=status,
            current_job=source.get("current_job"),
        )
        time.sleep(30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--source-run-tag", required=True)
    parser.add_argument("--source-full35-state", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    args = parser.parse_args()

    source_state = args.source_state.resolve()
    source_full35_state = args.source_full35_state.resolve()
    relay_state = (
        PROJECT_ROOT / "artifacts/queues" / f"phase-c-{args.run_tag}" / "relay-state.json"
    ).resolve()
    try:
        _wait_for_source(
            source_state=source_state,
            source_run_tag=args.source_run_tag,
            source_full35_state=source_full35_state,
            workers=args.workers,
            relay_state=relay_state,
        )
        _write_relay_state(relay_state, "starting_phase_c_recovery", source_state=str(source_state))
        from achitechure_1.queue import run_phase_c_recovery_queue

        state = run_phase_c_recovery_queue(
            PROJECT_ROOT,
            args.run_tag,
            source_state=source_state,
            workers=args.workers,
        )
        _write_relay_state(relay_state, "completed", queue_state=str(state))
        print(f"Phase-C recovery queue state: {state}", flush=True)
        return 0
    except BaseException as exc:
        _write_relay_state(
            relay_state,
            "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
