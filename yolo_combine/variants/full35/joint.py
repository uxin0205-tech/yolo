#!/usr/bin/env python3
"""Full35-only formal fusion entrypoint."""

from pathlib import Path

from yolo_combine.joint_cli import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "configs" / "joint.yaml")

