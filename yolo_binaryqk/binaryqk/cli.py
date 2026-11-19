"""本階段 CPU 準備入口，不會隱式啟動 GPU 或下載權重。"""

import argparse
import json
from pathlib import Path

import torch

from .calibration import calibrate, read_calibration
from .diagnostics import smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("smoke")
    check.add_argument("--calibration", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 4:
        parser.error("CPU preparation supports 1–4 threads")
    if args.output.exists():
        parser.error("output already exists; use a fresh diagnostic path")
    torch.set_num_threads(args.threads)
    if args.command == "calibrate":
        calibrate(args.output)
    else:
        report = smoke(args.calibration, read_calibration(args.calibration))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
