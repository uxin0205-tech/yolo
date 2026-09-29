"""Explicit single-job Clean worker; importing or inspecting never starts training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from ..artifacts.io import atomic_write_json
from ..contracts import sha256_file
from ..training.runner import run_training
from .builder import build_clean_model
from .contracts import CLEAN_EXPERIMENTS, load_clean_config
from .data_view import write_train_val_view
from .profiles import clean_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/clean/clean_ablation.yaml"))
    parser.add_argument("--experiment", choices=tuple(CLEAN_EXPERIMENTS), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_clean_config(args.config)
    config.verify_initializer()
    spec = config.experiment(args.experiment)
    if args.stage == "smoke" and spec.comparison_tier != "strict_fair":
        raise ValueError("smoke is defined only for strict-fair experiments")
    if spec.parent_required != (args.parent_checkpoint is not None):
        raise ValueError("parent checkpoint is required only for P2-Control-Clean-Full")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise ValueError("resume checkpoint does not exist")
    config.assert_split_use(split="train", purpose="fit")
    config.assert_split_use(split="val", purpose="selection")

    artifact_root = config.root / "artifacts" / "clean-bbt5-ablation"
    training_data = artifact_root / "data" / "train_val_only.yaml"
    write_train_val_view(config.locked_data_yaml, training_data)
    model = None if args.resume_checkpoint is not None else build_clean_model(
        args.experiment, config.initializer_path
    )
    if args.parent_checkpoint is not None and args.resume_checkpoint is None:
        parent = YOLO(str(args.parent_checkpoint), task="detect").model
        assert model is not None
        model.load_state_dict(parent.state_dict(), strict=True)
    project = artifact_root / "training"
    profile = clean_profile(
        args.experiment,
        seed=args.seed,
        model="clean-in-memory",
        data=str(training_data),
        project=str(project),
        stage=args.stage,
        patience=config.values["training"]["early_stopping_patience"],
    )
    result = run_training(model, profile, resume_path=args.resume_checkpoint)
    payload = {
        "stage": args.stage,
        "experiment": args.experiment,
        "seed": args.seed,
        "comparison_tier": spec.comparison_tier,
        "initializer_sha256": config.values["initializer"]["sha256"],
        "config_hash": config.config_hash,
        "data_visibility": {"train": "fit", "val": "selection", "test": "not_read"},
        "parent_checkpoint": str(args.parent_checkpoint) if args.parent_checkpoint else None,
        "resumed_from": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "best": str(result.best),
        "best_sha256": sha256_file(result.best),
        "last": str(result.last),
        "last_sha256": sha256_file(result.last),
        "save_dir": str(result.save_dir),
    }
    atomic_write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
