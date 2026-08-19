"""One-process queue jobs so CUDA and DataLoader memory die between stages."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fixed_common(root: Path, workers: int, *, batch: int = 16, nbs: int = 16):
    from .config import CommonTrainingConfig

    common = CommonTrainingConfig.from_yaml(root / "configs/training/common.yaml")
    return replace(
        common,
        batch=batch,
        nbs=nbs,
        workers=workers,
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

    train = commands.add_parser("train")
    train.add_argument("--variant", choices=("full35", "partial75"), required=True)
    train.add_argument("--phase", choices=("a1", "a2", "b", "c"), required=True)
    train.add_argument("--weights", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--nbs", type=int, default=16)

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
    if args.command == "train":
        from .config import load_phase_spec
        from .training import launch_phase

        launch_phase(
            project_root=root,
            weights=args.weights.resolve(),
            masf_variant=args.variant,
            attention_config=root / "configs/attention/float-pwl-final.yaml",
            bittrue_config=root / "configs/attention/bittrue-pwl-final.yaml",
            phase=load_phase_spec(root / f"configs/training/phase-{args.phase}.yaml"),
            common=_fixed_common(root, args.workers, batch=args.batch, nbs=args.nbs),
            run_id=args.run_id,
        )
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
            )
        except torch.cuda.OutOfMemoryError as exc:
            _write_json(
                args.output,
                {
                    "status": "oom",
                    "error": str(exc),
                    "batch": args.batch,
                    "accumulate": args.accumulate,
                    "effective_batch": args.batch * args.accumulate,
                    "gpu": torch.cuda.get_device_name(0),
                },
            )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
