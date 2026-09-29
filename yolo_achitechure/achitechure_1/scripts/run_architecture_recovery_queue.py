#!/usr/bin/env python3
"""Run Full35 and then Partial75 without operator prompts between jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from achitechure_1.queue import run_architecture_queue, run_partial75_continuation_queue

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--continue-after-full35-state", type=Path)
    args = parser.parse_args()
    if args.continue_after_full35_state:
        state = run_partial75_continuation_queue(
            PROJECT_ROOT,
            args.run_tag,
            source_full35_state=args.continue_after_full35_state,
            workers=args.workers,
        )
    else:
        state = run_architecture_queue(PROJECT_ROOT, args.run_tag, workers=args.workers)
    print(f"queue state: {state}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
