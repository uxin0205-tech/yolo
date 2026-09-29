#!/usr/bin/env python3
"""Create a two-class detect checkpoint from a YOLO pose checkpoint.

Only parameters with matching names and shapes are transferred. The pose-only
keypoint branch is intentionally dropped; the detection head is rebuilt with
the two BBT5 classes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose-weights",
        type=Path,
        default=SCRIPT_DIR.parents[1] / "pose_dataset" / "weight" / "yolo11m_bat.pt",
    )
    parser.add_argument(
        "--model-yaml",
        type=Path,
        default=SCRIPT_DIR / "yolo11m_detect_2cls.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "weights" / "yolo11m_bat_detect_init.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pose_weights.is_file():
        raise FileNotFoundError(args.pose_weights)
    if not args.model_yaml.is_file():
        raise FileNotFoundError(args.model_yaml)

    model = YOLO(str(args.model_yaml))
    if model.task != "detect":
        raise RuntimeError(f"expected a detect model, got task={model.task!r}")

    model.load(str(args.pose_weights))
    model.model.names = {0: "ball", 1: "bat"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"saved detect checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
