"""CPU-only feasibility and transfer inspection for the clean study."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import ultralytics
from ultralytics import YOLO

from ..contracts import sha256_file
from .builder import build_clean_model
from .contracts import CLEAN_EXPERIMENTS, load_clean_config
from .plan import build_clean_plan
from .profiles import clean_profile


def inspect_clean_study(config_path: Path) -> dict:
    config = load_clean_config(config_path)
    config.verify_initializer()
    if ultralytics.__version__ != config.values["environment"]["ultralytics"]:
        raise RuntimeError("installed Ultralytics version differs from the clean contract")
    source = YOLO(str(config.initializer_path), task="detect").model
    if source.model[-1].nc != 80 or len(source.names) != 80:
        raise RuntimeError("pinned clean initializer is not an 80-class detector")
    source_nc = source.model[-1].nc
    source_names_count = len(source.names)
    del source
    gc.collect()
    architectures: dict[tuple[str, str | None], dict] = {}
    for name, spec in CLEAN_EXPERIMENTS.items():
        key = (spec.family, spec.variant)
        if key in architectures:
            continue
        model = build_clean_model(name, config.initializer_path)
        report = model.masf_transfer_report
        architectures[key] = {
            "representative": name,
            "family": spec.family,
            "variant": spec.variant,
            "strides": [float(value) for value in model.stride.tolist()],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "matched_tensors": len(report["matched"]),
            "new_tensors": len(report["missing"]),
            "shape_mismatches": len(report["shape_mismatch"]),
            "unexpected_source_tensors": len(report["unexpected"]),
        }
        del model
        gc.collect()
    return {
        "ok": True,
        "training_started": False,
        "ultralytics": ultralytics.__version__,
        "config_hash": config.config_hash,
        "initializer": {
            "path": config.values["initializer"]["path"],
            "sha256": sha256_file(config.initializer_path),
            "nc": source_nc,
            "names_count": source_names_count,
        },
        "dataset_visibility": config.values["dataset"]["visibility"],
        "fairness": {
            "strict_tier": [
                name for name, spec in CLEAN_EXPERIMENTS.items()
                if spec.comparison_tier == "strict_fair"
            ],
            "shared_schedule": {
                key: value for key, value in clean_profile(
                    "B0-Clean", seed=42, model="<model>", data="<train-val-only>", project="<project>"
                ).items() if key not in {"model", "name", "project"}
            },
            "optimization_control_is_ranked_separately": True,
        },
        "architectures": list(architectures.values()),
        "jobs": build_clean_plan(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/clean/clean_ablation.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_clean_study(args.config)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
