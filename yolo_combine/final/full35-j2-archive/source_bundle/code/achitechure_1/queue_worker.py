"""One-process queue jobs so CUDA and DataLoader memory die between stages."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fixed_common(
    root: Path,
    workers: int,
    *,
    batch: int = 16,
    nbs: int = 16,
    fraction: float | None = None,
):
    from .config import CommonTrainingConfig

    common = CommonTrainingConfig.from_yaml(root / "configs/training/common.yaml")
    return replace(
        common,
        batch=batch,
        nbs=nbs,
        workers=workers,
        fraction=common.fraction if fraction is None else fraction,
        gradient_accumulation=nbs != batch,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m achitechure_1.queue_worker")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    commands = parser.add_subparsers(dest="command", required=True)

    probe = commands.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)

    inspect = commands.add_parser("inspect-candidate")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.add_argument("--kind", choices=("float", "bittrue"), required=True)
    inspect.add_argument("--output", type=Path, required=True)

    train = commands.add_parser("train")
    train.add_argument("--variant", choices=("full35", "partial75"), required=True)
    train.add_argument("--phase", choices=("a1", "a2", "b", "c"), required=True)
    train.add_argument("--weights", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--nbs", type=int, default=16)
    train.add_argument("--validation-batch", type=int, default=16)
    train.add_argument("--fraction", type=float)
    train.add_argument("--patience", type=int)
    train.add_argument("--resume-incomplete", action="store_true")

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--checkpoint", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--batch", type=int, default=16)

    profile = commands.add_parser("profile-phase-c")
    profile.add_argument("--checkpoint", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--batch", type=int, default=8)
    profile.add_argument("--accumulate", type=int, default=2)
    profile.add_argument("--steps", type=int, default=2)
    profile.add_argument("--amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "probe":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("the autonomous training queue requires a CUDA GPU")
        free, total = torch.cuda.mem_get_info(0)
        _write_json(
            args.output,
            {
                "cuda_available": True,
                "device": 0,
                "gpu_name": torch.cuda.get_device_name(0),
                "free_vram_bytes": int(free),
                "total_vram_bytes": int(total),
            },
        )
        return 0
    if args.command == "inspect-candidate":
        from ultralytics import YOLO

        from .masf import P3MASFFull35, P3MASFPartial75
        from .model import inspect_yolo26_graph

        model = YOLO(str(args.checkpoint.resolve())).model
        graph = inspect_yolo26_graph(model)
        masf = getattr(model.model[graph.p3_index], "p3_masf", None)
        if isinstance(masf, P3MASFFull35):
            variant = "full35"
        elif isinstance(masf, P3MASFPartial75):
            variant = "partial75"
        else:
            raise TypeError("checkpoint 不含可辨識的 Full35／Partial75 P3 MASF")
        normalizations = [
            module.config.normalization.value
            for module in model.modules()
            if module.__class__.__name__ == "HardwareFriendlyAttention"
        ]
        expected_normalization = "piecewise_linear" if args.kind == "float" else "bit_true_pwl"
        if normalizations != [expected_normalization, expected_normalization]:
            raise ValueError(
                f"{args.kind} checkpoint attention backend 不符：{normalizations}"
            )
        _write_json(
            args.output,
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "kind": args.kind,
                "variant": variant,
                "graph": asdict(graph),
                "attention_normalizations": normalizations,
                "alpha": float(masf.alpha.detach().cpu()),
            },
        )
        return 0
    if args.command == "train":
        from .config import load_phase_spec
        from .training import launch_phase, resume_phase

        phase = load_phase_spec(root / f"configs/training/phase-{args.phase}.yaml")
        if args.patience is not None:
            if args.patience < 1:
                raise ValueError("patience 必須為正整數")
            phase = replace(phase, patience=args.patience)
        common = _fixed_common(
            root,
            args.workers,
            batch=args.batch,
            nbs=args.nbs,
            fraction=args.fraction,
        )
        shared = {
            "project_root": root,
            "masf_variant": args.variant,
            "attention_config": root / "configs/attention/float-pwl-final.yaml",
            "bittrue_config": root / "configs/attention/bittrue-pwl-final.yaml",
            "phase": phase,
            "common": common,
            "run_id": args.run_id,
            "validation_batch": args.validation_batch,
        }
        if args.resume_incomplete:
            resume_phase(**shared)
        else:
            launch_phase(weights=args.weights.resolve(), **shared)
        return 0
    if args.command == "materialize":
        from .checkpoint import materialize_bittrue_checkpoint

        materialize_bittrue_checkpoint(
            args.checkpoint.resolve(),
            root / "configs/attention/bittrue-pwl-final.yaml",
            args.output.resolve(),
        )
        return 0
    if args.command == "validate":
        from .evaluation import validate_bittrue

        common = _fixed_common(root, args.workers, batch=args.batch, nbs=args.batch)
        validate_bittrue(
            checkpoint=args.checkpoint.resolve(),
            data=(root / common.data).resolve(),
            run_dir=args.run_dir.resolve(),
            imgsz=common.imgsz,
            batch=common.batch,
            device=common.device,
            workers=common.workers,
        )
        return 0
    if args.command == "profile-phase-c":
        import torch

        from .profiling import profile_training_step

        try:
            profile_training_step(
                checkpoint=args.checkpoint.resolve(),
                float_attention_config=root / "configs/attention/float-pwl-final.yaml",
                output=args.output.resolve(),
                batch=args.batch,
                accumulate=args.accumulate,
                steps=args.steps,
                amp=args.amp,
            )
        except torch.cuda.OutOfMemoryError as exc:
            _write_json(
                args.output,
                {
                    "status": "oom",
                    "error": str(exc),
                    "checkpoint": str(args.checkpoint.resolve()),
                    "batch": args.batch,
                    "accumulate": args.accumulate,
                    "effective_batch": args.batch * args.accumulate,
                    "amp": args.amp,
                    "gpu": torch.cuda.get_device_name(0),
                },
            )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
