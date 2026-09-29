"""Command line entry point for the 10-epoch attention-only experiment plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import convert_coco_annotations, ensure_yolo_layout, make_manifest, write_dataset_yaml
from .formal_audit import audit_formal_plan
from .report import build_summary
from .training_profiles import make_training_overrides
from .variants.definitions import (
    KD_INHERITING_VARIANTS,
    NON_KD_BIAS_VARIANTS,
    T6_CANDIDATES,
    get_variant,
    materialize_non_kd_bias_variant,
    materialize_t6_candidate,
    materialize_t7_variant,
)
from .workflow import create_run_artifact, execute_zero_training_validation, finalize_training_artifact, run_local_checks


def _update_json(path: Path, updates: dict) -> None:
    value = json.loads(path.read_text()) if path.exists() else {}
    value.update(updates)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="binary_attention")
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="run numerical self-checks; no G0-G5 engineering gate")
    verify.add_argument("--variant", default="T2")

    run = sub.add_parser("run", help="schedule or execute one formal QAT fine-tuning run")
    run.add_argument("--variant", required=True)
    run.add_argument("--stage", choices=["full", "validation"], required=True)
    run.add_argument("--data-manifest", required=True)
    run.add_argument("--data", default=None, help="Ultralytics dataset YAML; required with --execute")
    run.add_argument("--source-weights", default="../original/weight/yolo11m.pt")
    run.add_argument("--init-checkpoint", default=None, help="optional strict checkpoint; formal paper run defaults to the FP source")
    run.add_argument("--batch", type=int, default=None)
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--device", default="0")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--kd-components", default=None, help="selected T6 KD components, e.g. positional+feature")
    run.add_argument("--base-variant", default=None, help="selected upstream base, e.g. T3 for T6/T7 or T7-D for N4")
    run.add_argument("--bias-type", choices=["none", "dense_2d", "decomposed_2d"], default=None)
    run.add_argument("--magnitude-mode", choices=["fp", "int8", "int4"], default=None)
    run.add_argument("--execute", action="store_true")

    manifest = sub.add_parser("make-manifest", help="make deterministic full COCO train manifest")
    manifest.add_argument("--coco", default="../coco2017")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--seed", type=int, default=0)

    prep = sub.add_parser("prepare-data", help="convert official COCO annotations into YOLO labels")
    prep.add_argument("--coco", default="../coco2017")

    datacfg = sub.add_parser("make-data-config", help="write an Ultralytics dataset YAML")
    datacfg.add_argument("--train", required=True)
    datacfg.add_argument("--val", default="../coco2017/images/val2017")
    datacfg.add_argument("--output", required=True)

    sub.add_parser("report", help="rebuild JSON/CSV summaries and figures")
    sub.add_parser("audit", help="fail-closed audit of all required research artifacts")
    args = parser.parse_args()

    if args.command == "verify":
        print(run_local_checks(Path.cwd(), args.variant))
        return
    if args.command == "make-manifest":
        print(make_manifest(Path(args.coco), Path(args.output), None, args.seed))
        return
    if args.command == "prepare-data":
        coco = Path(args.coco)
        ensure_yolo_layout(coco)
        print(convert_coco_annotations(coco))
        return
    if args.command == "make-data-config":
        print(write_dataset_yaml(Path(args.output), Path(args.train), Path(args.val)))
        return
    if args.command == "report":
        print(build_summary(Path.cwd()))
        return
    if args.command == "audit":
        result = audit_formal_plan(Path.cwd())
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not result["ok"]:
            raise SystemExit(1)
        return

    variant = get_variant(args.variant)
    if variant.id in T6_CANDIDATES or variant.id == "T6":
        if not args.base_variant or (variant.id == "T6" and not args.kd_components):
            raise SystemExit(f"{variant.id} requires --base-variant and selected --kd-components")
        try:
            variant = materialize_t6_candidate(
                variant.id,
                base_variant=args.base_variant,
                components=tuple(args.kd_components.split("+")) if args.kd_components else None,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    elif variant.id in KD_INHERITING_VARIANTS:
        if not args.base_variant or not args.kd_components:
            raise SystemExit(f"{variant.id} requires selected T6 --base-variant and --kd-components")
        try:
            variant = materialize_t7_variant(
                variant.id,
                base_variant=args.base_variant,
                kd_components=tuple(args.kd_components.split("+")),
                bias_type=args.bias_type,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    elif variant.id in NON_KD_BIAS_VARIANTS:
        try:
            variant = materialize_non_kd_bias_variant(
                variant.id,
                bias_type=args.bias_type,
                magnitude_mode=args.magnitude_mode,
                parent_variant=args.base_variant,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.bias_type or args.magnitude_mode:
        raise SystemExit(f"{variant.id} does not accept inherited bias/magnitude options")
    manifest_path = Path(args.data_manifest)
    if not manifest_path.exists() or not manifest_path.with_suffix(".json").exists():
        raise SystemExit(f"full COCO manifest and sidecar are required: {manifest_path}")
    if args.stage == "validation" and not args.variant.startswith("E"):
        raise SystemExit("only E0/E1-S/E1 use the zero-training validation stage")
    if args.stage == "full" and args.variant.startswith("E"):
        raise SystemExit("E0/E1-S/E1 are zero-training validation variants; use --stage validation")

    run = create_run_artifact(
        Path.cwd(), variant, args.stage, sys.argv[:], data_manifest=manifest_path
    )
    _update_json(
        run / "status.json",
        {
            "data_manifest": str(manifest_path.resolve()),
            "source_weights": str(Path(args.source_weights).resolve()),
            "init_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
            "finetune_checkpoint": str(Path(args.init_checkpoint or args.source_weights).resolve()),
            "teacher_checkpoint": str(Path(args.source_weights).resolve()) if variant.use_distillation else None,
            "qat_finetuning": bool(variant.use_qat),
            "paper_profile": (
                "10-epoch attention-only QAT fine-tuning"
                if args.stage == "full" and variant.use_qat
                else "10-epoch attention-only FP fine-tuning"
                if args.stage == "full"
                else None
            ),
            "trainable_scope": "attention_only" if args.stage == "full" else "none",
            "device": args.device,
            "seed": 0 if args.seed is None else args.seed,
        },
    )
    if not args.execute:
        print(run)
        return
    if not args.data:
        raise SystemExit("--data is required with --execute")

    import torch

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("requested GPU but CUDA is unavailable; run formal training on the configured GPU host")

    if args.stage == "validation":
        execute_zero_training_validation(
            run, variant, Path(args.data), Path(args.source_weights), args.device
        )
        print(run)
        return

    from .trainer import BinaryDetectionTrainer, _validation_counter_model, freeze_to_attention_only

    overrides = make_training_overrides(
        stage=args.stage,
        model=str(run / "model.yaml"),
        data=args.data,
        device=args.device,
        project=str(run / "ultralytics"),
        name="train",
        seed=args.seed,
        batch=args.batch,
        workers=args.workers,
    )
    _update_json(run / "training_args.json", overrides)
    _update_json(run / "status.json", {"batch": overrides["batch"], "workers": overrides["workers"], "epochs": overrides["epochs"]})
    trainer = BinaryDetectionTrainer(
        overrides=overrides,
        variant=variant,
        model_yaml=run / "model.yaml",
        source_weights=Path(args.source_weights),
        init_weights=Path(args.init_checkpoint) if args.init_checkpoint else None,
    )
    execution = trainer.execution_profile()
    _update_json(run / "training_args.json", execution)
    _update_json(run / "status.json", execution)
    trainer.train()
    trainable = trainer.trainable_profile()
    weight_provenance = {
        "checkpoint_weight_source": "epoch_ema",
        "metrics_weight_source": "epoch_ema",
        "ema_checkpoint_epoch": overrides["epochs"],
    }
    _update_json(run / "training_args.json", {**trainable, **weight_provenance})
    _update_json(run / "status.json", {**trainable, **weight_provenance})
    artifact_model = _validation_counter_model(trainer)
    freeze_to_attention_only(artifact_model)
    finalize_training_artifact(run, artifact_model, variant, source_weights=Path(args.source_weights))
    print(run)


if __name__ == "__main__":
    main()
