#!/usr/bin/env python3
"""Final Full35 CLI: preflight, train/resume, validate and infer."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "code/project/src"))

from yolo_combine.joint_cli import main

if __name__ == "__main__":
    main(ROOT / "configs/joint.yaml")
