#!/usr/bin/env python3
"""準備 ball／bat detection 驗證集，或驗證單一 Bit-True checkpoint。"""

from __future__ import annotations

import argparse
from pathlib import Path

from achitechure_1.ball_bat_evaluation import (
    prepare_ball_bat_detect_dataset,
    validate_ball_bat_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/uxin0/yolo/original/pose/dataset")
DEFAULT_DERIVED = Path("/home/uxin0/yolo/original/pose/detect_dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--output", type=Path, default=DEFAULT_DERIVED)

    validate = commands.add_parser("validate")
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--data", type=Path, default=DEFAULT_DERIVED / "coco80/data.yaml")
    validate.add_argument("--batch", type=int, default=8)
    validate.add_argument("--workers", type=int, choices=range(0, 9), default=6)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        two_class, coco80 = prepare_ball_bat_detect_dataset(
            source_root=args.source,
            output_root=args.output,
            coco_data=PROJECT_ROOT / "configs/coco2017.yaml",
        )
        print(f"ball／bat 2-class detection 設定：{two_class}", flush=True)
        print(f"ball／bat COCO80 驗證設定：{coco80}", flush=True)
        return 0
    if args.command == "validate":
        path = validate_ball_bat_checkpoint(
            checkpoint=args.checkpoint,
            data=args.data,
            run_dir=args.run_dir,
            batch=args.batch,
            workers=args.workers,
        )
        print(f"ball／bat 驗證結果：{path}", flush=True)
        return 0
    raise AssertionError(f"未處理的命令：{args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
