"""Folder-locked command interface for one architecture experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .baselines import train_pose_baseline
from .cpu_validation import validate_workspace_on_cpu
from .data import prepare_bbt5_view, prepare_coco_detect_subset
from .freezing import InheritedFreezeGuard
from .joint import run_joint_steps
from .loaders import build_joint_train_loaders
from .pose_stages import POSE_STAGES, pose_stage
from .source import SourceBundle, file_sha256
from .training import TaskLossRouter
from .variants import VariantWorkspace


def _parser(workspace: VariantWorkspace) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"variants/{workspace.architecture}/run.py",
        description=f"Architecture-locked commands for {workspace.architecture}.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="Read-only source/config/hash audit.")
    audit.add_argument("--verify-hashes", action="store_true")

    cpu = commands.add_parser(
        "cpu-check",
        help="Run model/data/forward/materialization/backward checks without CUDA.",
    )
    cpu.add_argument("--imgsz", type=int, default=64)
    cpu.add_argument("--seed", type=int, default=0)
    cpu.add_argument("--skip-joint-step", action="store_true")
    cpu.add_argument("--no-write-report", action="store_true")
    joint = commands.add_parser(
        "joint-smoke",
        help="Run one real COCO/BBT5 optimizer step on CPU.",
    )
    joint.add_argument("--ratio", choices=("1:1", "2:1"), default="2:1")
    joint.add_argument("--steps", type=int, default=1)
    joint.add_argument("--batch", type=int, default=1)
    joint.add_argument("--imgsz", type=int, default=64)
    joint.add_argument("--workers", type=int, default=0)
    joint.add_argument("--coco-images", type=int, default=128)
    joint.add_argument("--seed", type=int, default=0)
    joint.add_argument("--lr", type=float, default=1e-5)
    joint.add_argument("--pose-head-checkpoint", type=Path, default=None)

    pose = commands.add_parser("pose", help="Run a folder-isolated Pose stage.")
    pose.add_argument("--stage", choices=tuple(POSE_STAGES), default="smoke")
    pose.add_argument("--initial-checkpoint", type=Path, default=None)
    pose.add_argument("--name", default=None)
    pose.add_argument("--epochs", type=int, default=None)
    pose.add_argument("--imgsz", type=int, default=None)
    pose.add_argument("--batch", type=int, default=None)
    pose.add_argument("--workers", type=int, default=None)
    pose.add_argument("--device", default="0")
    pose.add_argument("--seed", type=int, default=0)
    pose.add_argument("--fraction", type=float, default=None)
    pose.add_argument("--val", action=argparse.BooleanOptionalAction, default=None)
    pose.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    pose.add_argument("--exist-ok", action="store_true")
    return parser


def _audit(workspace: VariantWorkspace, verify_hashes: bool) -> dict[str, object]:
    report = workspace.audit(verify_hashes=verify_hashes)
    return {
        "architecture": workspace.architecture,
        "role": workspace.role,
        "root": str(workspace.root),
        "run_root": str(workspace.run_root),
        "source_bundle": str(workspace.source_bundle),
        "dataset_contract": {
            "registry": str(workspace.bbat5_registry),
            "version": workspace.bbat5_version,
        },
        "datasets": {name: str(value) for name, value in workspace.datasets.items()},
        "audit": {
            "ok": report.ok,
            "missing_paths": list(report.missing_paths),
            "hash_mismatches": list(report.hash_mismatches),
            "hashes_verified": verify_hashes,
        },
    }


def _pose(workspace: VariantWorkspace, args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    stage = pose_stage(args.stage)
    initial_checkpoint = stage.validate_transition(args.initial_checkpoint)
    source = SourceBundle(workspace.source_bundle, architecture=workspace.architecture)
    source.verify_manifest()
    prepared = prepare_bbt5_view(
        workspace.bbat5_registry,
        workspace.pose_view_root,
    )
    name = args.name or f"p0-{workspace.architecture}-{stage.name}-seed{args.seed}"
    resolved = {
        "epochs": args.epochs if args.epochs is not None else stage.epochs,
        "imgsz": args.imgsz if args.imgsz is not None else stage.imgsz,
        "batch": args.batch if args.batch is not None else stage.batch,
        "workers": args.workers if args.workers is not None else stage.workers,
        "fraction": args.fraction if args.fraction is not None else stage.fraction,
        "val": args.val if args.val is not None else stage.val,
        "plots": args.plots if args.plots is not None else stage.plots,
    }
    report = train_pose_baseline(
        source,
        data_yaml=prepared.yaml,
        project=workspace.pose_run_root,
        name=name,
        device=args.device,
        seed=args.seed,
        exist_ok=args.exist_ok,
        overrides=stage.trainer_overrides(),
        initial_checkpoint=initial_checkpoint,
        **resolved,
    )
    return {
        **asdict(report),
        "architecture": workspace.architecture,
        "workspace": str(workspace.root),
        "dataset_view": asdict(prepared),
        "seed": args.seed,
        "stage": asdict(stage),
        "resolved_training": resolved,
    }


def _joint_smoke(
    workspace: VariantWorkspace, args: argparse.Namespace
) -> dict[str, object]:
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    source = SourceBundle(workspace.source_bundle, architecture=workspace.architecture)
    source.verify_manifest()
    model = (
        source.build_shared(pose_head_checkpoint=args.pose_head_checkpoint)
        .to(device)
        .train()
    )
    guard = InheritedFreezeGuard.capture(model)
    prepared = prepare_bbt5_view(
        workspace.bbat5_registry,
        workspace.pose_view_root,
    )
    detect_view = prepare_coco_detect_subset(
        workspace.datasets["coco_detect"],
        workspace.run_root / "cache-views" / "coco-detect-smoke",
        limit=args.coco_images,
    )
    loaders = build_joint_train_loaders(
        model,
        detect_yaml=detect_view.yaml,
        pose_yaml=prepared.yaml,
        device=device,
        batch_size=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        fraction=1.0,
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
    guard.assert_unchanged(model)
    return {
        **asdict(report),
        "architecture": workspace.architecture,
        "device": "cpu",
        "ratio": args.ratio,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "frozen_paths": guard.paths,
        "detect_images": len(loaders.detect.dataset),
        "pose_images": len(loaders.pose.dataset),
        "pose_view": asdict(prepared),
        "detect_view": asdict(detect_view),
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
    }


def main(variant_root: str | Path, argv: Sequence[str] | None = None) -> None:
    workspace = VariantWorkspace.load(variant_root)
    args = _parser(workspace).parse_args(argv)
    if args.command == "audit":
        payload = _audit(workspace, args.verify_hashes)
        print(json.dumps(payload, indent=2, sort_keys=True))
        audit = payload["audit"]
        if not isinstance(audit, dict) or not audit["ok"]:
            raise SystemExit(1)
        return
    if args.command == "cpu-check":
        if torch.cuda.is_initialized():
            raise RuntimeError("cpu-check refuses to run after CUDA was initialized")
        payload = validate_workspace_on_cpu(
            workspace,
            imgsz=args.imgsz,
            seed=args.seed,
            joint_step=not args.skip_joint_step,
        )
        rendered = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        if not args.no_write_report:
            workspace.cpu_report_path.parent.mkdir(parents=True, exist_ok=True)
            workspace.cpu_report_path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return
    if args.command == "joint-smoke":
        if torch.cuda.is_initialized():
            raise RuntimeError("joint-smoke refuses to run after CUDA was initialized")
        payload = _joint_smoke(workspace, args)
        rendered = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        workspace.joint_smoke_report_path.parent.mkdir(parents=True, exist_ok=True)
        workspace.joint_smoke_report_path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return
    if args.command == "pose":
        print(json.dumps(_pose(workspace, args), indent=2, sort_keys=True, default=str))
        return
    raise AssertionError(f"unhandled command: {args.command}")
