#!/usr/bin/env python3
"""執行 fraction=1.0、Phase C batch8×2、patience 可覆寫的對稱佇列。"""

from __future__ import annotations

import argparse
from pathlib import Path

from achitechure_1.queue import run_fraction10_phase_c_queue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "artifacts/queues/fraction10-phase-b-rtx5060ti-fraction10-b16-workers6-r1/state.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--source-state", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--minimum-available-ram-gib", type=float, default=0.5)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=9.0)
    args = parser.parse_args()
    state = run_fraction10_phase_c_queue(
        PROJECT_ROOT,
        args.run_tag,
        source_state=args.source_state,
        workers=args.workers,
        patience=args.patience,
        minimum_available_ram_bytes=int(args.minimum_available_ram_gib * (1 << 30)),
        minimum_free_vram_bytes=int(args.minimum_free_vram_gib * (1 << 30)),
    )
    print(f"queue state: {state}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
