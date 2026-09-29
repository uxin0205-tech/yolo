#!/usr/bin/env python3
"""以 batch16、workers6、fraction=1.0 執行兩架構 Phase-B 對照佇列。"""

from __future__ import annotations

import argparse
from pathlib import Path

from achitechure_1.queue import run_fraction10_phase_b_control_queue

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    parser.add_argument("--full35-source", type=Path)
    parser.add_argument("--partial75-source", type=Path)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=12.0)
    args = parser.parse_args()
    state = run_fraction10_phase_b_control_queue(
        PROJECT_ROOT,
        args.run_tag,
        full35_source=args.full35_source,
        partial75_source=args.partial75_source,
        workers=args.workers,
        minimum_free_vram_bytes=int(args.minimum_free_vram_gib * (1 << 30)),
    )
    print(f"queue state: {state}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
