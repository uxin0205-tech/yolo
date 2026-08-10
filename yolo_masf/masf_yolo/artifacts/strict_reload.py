"""Fresh-process strict checkpoint reconstruction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from masf_yolo.models.builder import build_model
from masf_yolo.variants import get_variant

from .checkpoints import load_canonical_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data-hash", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--environment-hash", required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    variant = get_variant(args.variant)
    model = build_model(variant)
    load_canonical_checkpoint(
        model,
        args.checkpoint,
        variant,
        expected_data_hash=args.data_hash,
        expected_config_hash=args.config_hash,
        expected_environment_hash=args.environment_hash,
    )
    detect = model.model[-1]
    report = {
        "strict_load": True,
        "variant": variant.variant_id,
        "strides": model.stride.tolist(),
        "detect_scales": detect.nl,
        "validation_ran": False,
    }
    if args.data is not None:
        from ultralytics import YOLO

        from masf_yolo.models.builder import TEMPLATE_PATH

        wrapper = YOLO(str(TEMPLATE_PATH), task="detect", verbose=False)
        wrapper.model = model
        metrics = wrapper.val(
            data=str(args.data.resolve()),
            split=args.split,
            device=args.device,
            imgsz=args.imgsz,
            project=str(args.checkpoint.resolve().parent / "strict_validation"),
            name=variant.variant_id.lower(),
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        report["validation_ran"] = True
        report["validation"] = {
            key: float(value) for key, value in metrics.results_dict.items()
        }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
