#!/usr/bin/env python3
"""Run the real-data graph-shared Full35 smoke without CUDA."""

import argparse
import json
from pathlib import Path

from yolo_combine.cpu_joint_smoke import run_real_cpu_smoke
from yolo_combine.joint_config import JointExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pose-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = JointExperimentConfig.load(root / "configs" / "joint.yaml")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "artifacts" / "formal-joint-cpu-smoke.json"
    )
    report = run_real_cpu_smoke(
        config,
        output_root=output.parent / "formal-joint-cpu-smoke",
        pose_checkpoint=args.pose_checkpoint,
        steps=args.steps,
        imgsz=args.imgsz,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

