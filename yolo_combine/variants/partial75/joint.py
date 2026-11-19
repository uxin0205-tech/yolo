#!/usr/bin/env python3
"""Partial75-only fallback fusion entrypoint."""

from pathlib import Path

from yolo_combine.joint_cli import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent / "configs" / "joint.yaml")

