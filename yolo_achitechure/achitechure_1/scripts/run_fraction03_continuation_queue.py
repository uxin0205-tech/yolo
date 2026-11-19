#!/usr/bin/env python3
"""執行 B batch16、C batch8×accumulate2、workers=6 的 30% 對稱續跑佇列。"""

from __future__ import annotations

import argparse
from pathlib import Path

from achitechure_1.queue import run_fraction03_continuation_queue

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    parser.add_argument("--full35-source", type=Path)
    parser.add_argument("--partial75-source", type=Path)
    args = parser.parse_args()
    state = run_fraction03_continuation_queue(
        PROJECT_ROOT,
        args.run_tag,
        full35_source=args.full35_source,
        partial75_source=args.partial75_source,
        workers=args.workers,
    )
    print(f"queue state: {state}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
