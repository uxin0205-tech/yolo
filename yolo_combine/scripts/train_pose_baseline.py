#!/usr/bin/env python3
"""Train the independent P0 Full35 Pose26 ball/bat baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from yolo_combine.baselines import train_pose_baseline
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.pose_stages import POSE_STAGES, pose_stage
from yolo_combine.source import SourceBundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path("/home/uxin/yolo/yolo_achitechure/achitechure_1/final")
DEFAULT_BBAT5_REGISTRY = Path("/home/uxin/yolo/configs/datasets/bbat5-v1.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", choices=("full35", "partial75"), default="full35"
    )
    parser.add_argument("--stage", choices=tuple(POSE_STAGES), default="smoke")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--bbat5-registry", type=Path, default=DEFAULT_BBAT5_REGISTRY)
    parser.add_argument(
        "--pose-view",
        type=Path,
        default=PROJECT_ROOT / "artifacts/datasets/bbat5-v1-runtime",
    )
    parser.add_argument(
        "--project", type=Path, default=PROJECT_ROOT / "artifacts/runs/p0"
    )
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    stage = pose_stage(args.stage)
    initial_checkpoint = stage.validate_transition(args.initial_checkpoint)
    source = SourceBundle(args.bundle, architecture=args.architecture)
    source.verify_manifest()
    prepared = prepare_bbt5_view(args.bbat5_registry, args.pose_view)
    pose_data = prepared.yaml
    name = args.name or f"p0-{args.architecture}-{stage.name}-seed{args.seed}"
    epochs = args.epochs if args.epochs is not None else stage.epochs
    imgsz = args.imgsz if args.imgsz is not None else stage.imgsz
    batch = args.batch if args.batch is not None else stage.batch
    workers = args.workers if args.workers is not None else stage.workers
    fraction = args.fraction if args.fraction is not None else stage.fraction
    val = args.val if args.val is not None else stage.val
    plots = args.plots if args.plots is not None else stage.plots
    report = train_pose_baseline(
        source,
        data_yaml=pose_data,
        project=args.project,
        name=name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=args.device,
        seed=args.seed,
        fraction=fraction,
        val=val,
        plots=plots,
        exist_ok=args.exist_ok,
        overrides=stage.trainer_overrides(),
        initial_checkpoint=initial_checkpoint,
    )
    payload = asdict(report)
    payload.update(
        {
            "architecture": args.architecture,
            "dataset": str(pose_data),
            "dataset_registry": str(args.bbat5_registry.resolve()),
            "dataset_view": asdict(prepared),
            "seed": args.seed,
            "stage": asdict(stage),
            "resolved_training": {
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "workers": workers,
                "fraction": fraction,
                "val": val,
                "plots": plots,
            },
            "selection_status": (
                "smoke-only"
                if fraction < 1.0 or not val or imgsz < 640
                else "formal-bbat5-v1"
            ),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
