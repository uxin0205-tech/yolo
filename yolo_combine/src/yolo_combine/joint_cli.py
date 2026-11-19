"""Architecture-locked CLI for formal joint preflight, training, validation, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import torch

from .data import prepare_bbt5_view
from .factory import FusionModelFactory
from .formal_training import FormalJointTrainingSession
from .inference import SharedDualPredictor, load_combined_weights
from .joint_config import JointExperimentConfig
from .source import SourceBundle
from .validation import JointValidator, ValidationSettings
from .xnor import XNORExecutionConfig


def _parser(config: JointExperimentConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"variants/{config.architecture}/joint.py",
        description=f"{config.architecture} shared-trunk Detect/Pose commands",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Read-only formal readiness report.")

    train = commands.add_parser("train", help="Run formal single-GPU joint training.")
    train.add_argument("--device", default="0")
    train.add_argument("--name", default=None)
    train.add_argument("--resume", type=Path, default=None)
    train.add_argument("--enable-j3", action="store_true")
    train.add_argument(
        "--j3-detect-microbatch",
        type=int,
        default=None,
        help="J3-only physical Detect microbatch; logical batch remains unchanged.",
    )

    validate = commands.add_parser("validate", help="Run independent Detect and Pose validators.")
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--device", default="0")
    validate.add_argument("--name", default="manual")
    validate.add_argument(
        "--backend",
        choices=("float", "bittrue", "both"),
        default="both",
    )
    validate.add_argument("--prefer-live", action="store_true")

    infer = commands.add_parser("infer", help="Single-trunk image inference.")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--source", type=Path, nargs="+", required=True)
    infer.add_argument("--task", choices=("detect", "pose", "both"), default="both")
    infer.add_argument("--device", default="0")
    infer.add_argument("--conf", type=float, default=0.25)
    infer.add_argument("--iou", type=float, default=0.7)
    infer.add_argument("--max-det", type=int, default=300)
    infer.add_argument("--output-json", type=Path, default=None)
    infer.add_argument("--prefer-live", action="store_true")
    return parser


def _torch_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not value.isdigit():
        raise ValueError("device must be one CUDA index or cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(f"cuda:{int(value)}")


def _load_model(
    config: JointExperimentConfig,
    checkpoint: Path,
    *,
    device: torch.device,
    prefer_ema: bool,
):
    source = SourceBundle(config.source_bundle, architecture=config.architecture)
    factory = FusionModelFactory(
        source,
        detect_data_yaml=config.detect_data,
        pose_data_yaml=config.pose_data,
        xnor=XNORExecutionConfig(token_tile=config.xnor_token_tile),
    )
    built = factory.build(
        checkpoint_kind="float",
        allow_untrained_pose_head=True,
    )
    model = built.model.to(device)
    loaded = load_combined_weights(model, checkpoint, prefer_ema=prefer_ema)
    return source, model, built.report, loaded


def _serialize_results(results: dict[str, list[Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for task, task_results in results.items():
        records = []
        for result in task_results:
            boxes = result.boxes
            item: dict[str, Any] = {
                "path": str(result.path),
                "names": {str(key): value for key, value in result.names.items()},
                "detections": [],
            }
            keypoints = result.keypoints.data.cpu() if result.keypoints is not None else None
            for index in range(len(boxes)):
                detection = {
                    "xyxy": [float(value) for value in boxes.xyxy[index].cpu()],
                    "confidence": float(boxes.conf[index].cpu()),
                    "class_id": int(boxes.cls[index].cpu()),
                }
                if keypoints is not None:
                    detection["keypoints"] = keypoints[index].tolist()
                item["detections"].append(detection)
            records.append(item)
        payload[task] = records
    return payload


def main(config_path: str | Path, argv: Sequence[str] | None = None) -> None:
    config = JointExperimentConfig.load(config_path)
    args = _parser(config).parse_args(argv)
    if args.command == "preflight":
        report = config.preflight()
        payload = {
            "ready": report.ready,
            "architecture": config.architecture,
            "resolved": config.as_dict(),
            "blockers": list(report.blockers),
            "warnings": list(report.warnings),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not report.ready:
            raise SystemExit(1)
        return
    if args.command == "train":
        if args.j3_detect_microbatch is not None and not (
            args.enable_j3 and args.resume is not None
        ):
            raise ValueError(
                "--j3-detect-microbatch requires --enable-j3 and --resume"
            )
        report = FormalJointTrainingSession(
            config,
            device=args.device,
            run_name=args.name,
            detect_microbatch_size=args.j3_detect_microbatch,
        ).run(
            resume=args.resume,
            enable_j3=args.enable_j3 or None,
        )
        print(json.dumps(report, default=str, indent=2, sort_keys=True))
        return
    if args.command == "validate":
        device = _torch_device(args.device)
        source, model, factory_report, loaded = _load_model(
            config,
            args.checkpoint,
            device=device,
            prefer_ema=not args.prefer_live,
        )
        output = config.run_root / "evaluations" / args.name
        pose_view = prepare_bbt5_view(
            config.registry,
            output / "datasets" / "bbat5-v1-runtime",
        )
        validator = JointValidator(
            source,
            detect_data_yaml=config.detect_data,
            pose_data_yaml=pose_view.yaml,
            output_root=output,
            settings=ValidationSettings(
                imgsz=config.imgsz,
                detect_batch_size=config.detect_val_batch_size,
                pose_batch_size=config.pose_val_batch_size,
                detect_workers=config.detect_workers,
                pose_workers=config.pose_workers,
                device=str(device),
                plots=config.validation_plots,
                save_coco_json=config.save_coco_json,
            ),
        )
        kinds = (
            ("float", "bittrue")
            if args.backend == "both"
            else (args.backend,)
        )
        reports = validator.validate_backends(model, epoch=0, kinds=kinds)
        print(
            json.dumps(
                {
                    "loaded": loaded,
                    "factory": factory_report.as_dict(),
                    "metrics": {name: value.metrics for name, value in reports.items()},
                },
                default=str,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "infer":
        device = _torch_device(args.device)
        _, model, factory_report, loaded = _load_model(
            config,
            args.checkpoint,
            device=device,
            prefer_ema=not args.prefer_live,
        )
        images = []
        paths = []
        for path in args.source:
            resolved = path.expanduser().resolve()
            image = cv2.imread(str(resolved))
            if image is None:
                raise ValueError(f"cannot decode input image: {resolved}")
            images.append(image)
            paths.append(str(resolved))
        predictor = SharedDualPredictor(
            model,
            imgsz=config.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            amp=config.amp,
        )
        payload = {
            "loaded": loaded,
            "factory_complete": factory_report.complete,
            "task": args.task,
            "results": _serialize_results(
                predictor.predict(images, task=args.task, paths=paths)
            ),
        }
        rendered = json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n"
        if args.output_json is not None:
            target = args.output_json.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return
    raise AssertionError(f"unhandled command: {args.command}")
