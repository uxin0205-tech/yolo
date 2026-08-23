#!/usr/bin/env python3
"""Architecture-locked Full35 experiment entrypoint."""

from pathlib import Path

from yolo_combine.variant_cli import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
