#!/usr/bin/env python3
"""Run one or more real COCO/BBT5 joint optimizer steps."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from yolo_combine import (
    SourceBundle,
    TaskLossRouter,
    build_joint_train_loaders,
    prepare_bbt5_view,
    run_joint_steps,
)
from yolo_combine.freezing import InheritedFreezeGuard
from yolo_combine.source import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path("/home/uxin/yolo/yolo_achitechure/achitechure_1/final")
DEFAULT_BBAT5_REGISTRY = Path("/home/uxin/yolo/configs/datasets/bbat5-v1.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", choices=("full35", "partial75"), default="full35"
    )
    parser.add_argument("--ratio", choices=("1:1", "2:1"), default="1:1")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--pose-head-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--detect-data",
        type=Path,
        default=PROJECT_ROOT / "configs/data/coco2017.yaml",
    )
    parser.add_argument(
        "--bbat5-registry",
        type=Path,
        default=DEFAULT_BBAT5_REGISTRY,
    )
    parser.add_argument(
        "--pose-view",
        type=Path,
        default=PROJECT_ROOT / "artifacts/cache-views/bbat5-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source = SourceBundle(args.bundle, architecture=args.architecture)
    source.verify_manifest()
    model = (
        source.build_shared(pose_head_checkpoint=args.pose_head_checkpoint)
        .to(device)
        .train()
    )
    inherited_guard = InheritedFreezeGuard.capture(model)
    prepared = prepare_bbt5_view(args.bbat5_registry, args.pose_view)
    pose_data = prepared.yaml
    loaders = build_joint_train_loaders(
        model,
        detect_yaml=args.detect_data,
        pose_yaml=pose_data,
        device=device,
        batch_size=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        fraction=args.fraction,
        seed=args.seed,
    )
    router = TaskLossRouter(model, epochs=1, imgsz=args.imgsz)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.SGD(trainable, lr=args.lr)
    report = run_joint_steps(
        router,
        optimizer,
        loaders,
        steps=args.steps,
        detect_per_step=2 if args.ratio == "2:1" else 1,
    )
    inherited_guard.assert_unchanged(model)
    payload = asdict(report)
    payload.update(
        {
            "architecture": args.architecture,
            "pose_head_checkpoint": (
                str(args.pose_head_checkpoint.resolve())
                if args.pose_head_checkpoint
                else None
            ),
            "pose_head_sha256": (
                file_sha256(args.pose_head_checkpoint)
                if args.pose_head_checkpoint
                else None
            ),
            "ratio": args.ratio,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "frozen_paths": inherited_guard.paths,
            "detect_images": len(loaders.detect.dataset),
            "pose_images": len(loaders.pose.dataset),
            "dataset_registry": str(args.bbat5_registry.resolve()),
            "pose_view": asdict(prepared),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
