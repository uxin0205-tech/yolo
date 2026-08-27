#!/usr/bin/env python3
"""Generate same-evaluator Partial75 standalone metrics when explicitly requested."""

import argparse
import json
from pathlib import Path

from yolo_combine.data import prepare_bbt5_view
from yolo_combine.joint_config import JointExperimentConfig
from yolo_combine.source import SourceBundle
from yolo_combine.standalone_baseline import StandaloneBaselineValidator
from yolo_combine.validation import ValidationSettings
from yolo_combine.xnor import XNORExecutionConfig, install_xnor_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="p3-seed0")
    parser.add_argument("--backend", choices=("float", "bittrue", "both"), default="both")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = JointExperimentConfig.load(root / "configs" / "joint.yaml")
    source = SourceBundle(config.source_bundle, architecture="partial75")
    source.verify_manifest()
    source.verify_environment()
    source.activate_code()
    install_xnor_backend(XNORExecutionConfig(token_tile=config.xnor_token_tile))
    output = root / "artifacts" / "standalone-baseline" / args.name
    pose_view = prepare_bbt5_view(
        config.registry,
        output / "datasets" / "bbat5-v1-runtime",
    )
    validator = StandaloneBaselineValidator(
        source,
        pose_checkpoint=args.pose_checkpoint,
        detect_data_yaml=config.detect_data,
        pose_data_yaml=pose_view.yaml,
        output_root=output / "validation",
        settings=ValidationSettings(
            imgsz=config.imgsz,
            detect_batch_size=config.detect_val_batch_size,
            pose_batch_size=config.pose_val_batch_size,
            detect_workers=config.detect_workers,
            pose_workers=config.pose_workers,
            device=args.device,
            plots=True,
        ),
    )
    kinds = ("float", "bittrue") if args.backend == "both" else (args.backend,)
    results = validator.validate_backends(kinds)
    gate_file = None
    if "bittrue" in results:
        gate_file = validator.write_gate_file(
            results["bittrue"],
            root / "baselines" / f"formal-gate-{args.name}.json",
        )
    print(
        json.dumps(
            {
                "metrics": {name: result.metrics for name, result in results.items()},
                "gate_file": str(gate_file) if gate_file else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

