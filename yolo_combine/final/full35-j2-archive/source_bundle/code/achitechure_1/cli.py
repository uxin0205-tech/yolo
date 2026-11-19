"""Single fail-closed command surface for the achitechure_1 experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m achitechure_1.cli")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--variant", choices=("full35", "partial75"), required=True)
    prepare.add_argument("--parent", type=Path, default=Path("inputs/parent/best.pt"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--execute", action="store_true")

    train = commands.add_parser("train")
    train.add_argument("--variant", choices=("full35", "partial75"), required=True)
    train.add_argument("--phase", choices=("a1", "a2", "b", "c", "tune"), required=True)
    train.add_argument("--weights", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--masf-lr", type=float)
    train.add_argument("--execute", action="store_true")

    materialize = commands.add_parser("materialize-bittrue")
    materialize.add_argument("--checkpoint", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--execute", action="store_true")

    validate = commands.add_parser("validate")
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--execute", action="store_true")

    profile = commands.add_parser("profile")
    profile.add_argument("--checkpoint", type=Path, required=True)
    profile.add_argument("--kind", choices=("inference", "training"), default="inference")
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--warmup", type=int, default=20)
    profile.add_argument("--iterations", type=int, default=100)
    profile.add_argument("--steps", type=int, default=3)
    profile.add_argument("--amp", action="store_true")
    profile.add_argument("--execute", action="store_true")

    gate = commands.add_parser("gate")
    gate.add_argument("--parent-name", required=True)
    gate.add_argument("--parent-map", type=float, required=True)
    gate.add_argument("--child-name", required=True)
    gate.add_argument("--child-map", type=float, required=True)
    gate.add_argument("--policy", choices=("rollback-0.001", "phase-c-improve"), default="rollback-0.001")

    select = commands.add_parser("select")
    select.add_argument("--candidate", type=Path, action="append", required=True)

    export = commands.add_parser("export-masf")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--execute", action="store_true")

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--variant", choices=("full35", "partial75"), required=True)
    smoke.add_argument("--device", default="cpu")
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _phase(root: Path, name: str, masf_lr: float | None):
    from .config import load_phase_spec
    from .phases import tuning_phase

    if name == "tune":
        if masf_lr is None:
            raise ValueError("--masf-lr is required for tune")
        return tuning_phase(masf_lr)
    if masf_lr is not None:
        raise ValueError("--masf-lr is only valid for tune")
    return load_phase_spec(root / f"configs/training/phase-{name}.yaml")


def _smoke(root: Path, variant: str, device_name: str) -> dict[str, object]:
    import torch
    from ultralytics import YOLO

    from .model import graft_p3_masf, inspect_yolo26_graph

    device = torch.device(device_name)
    model = YOLO(str(root / "inputs/parent/best.pt")).model.to(device)
    graft = graft_p3_masf(model, variant)
    model.eval()
    sample = torch.randn(1, 3, 64, 64, device=device)
    with torch.no_grad():
        inference = model(sample)
    model.train()
    prediction = model(sample)["one2many"]
    loss = prediction["boxes"].float().mean() + prediction["scores"].float().mean()
    loss = loss + sum(value.float().mean() for value in prediction["feats"])
    loss.backward()
    gradients = [parameter.grad for parameter in model.model[graft.p3_index].p3_masf.parameters()]
    return {
        "variant": variant,
        "device": str(device),
        "graph": asdict(inspect_yolo26_graph(model)),
        "inference_output": type(inference).__name__,
        "finite_loss": bool(torch.isfinite(loss)),
        "all_masf_gradients": all(value is not None and torch.isfinite(value).all() for value in gradients),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "preflight":
        from .preflight import run_preflight, write_preflight

        report = run_preflight(root)
        write_preflight(report, root / "artifacts/preflight.json")
        _print(report)
        return 0 if report["valid"] else 1
    if args.command == "prepare":
        payload = {
            "variant": args.variant,
            "parent": str(_resolve(root, args.parent).resolve()),
            "output": str(_resolve(root, args.output).resolve()),
            "will_execute": args.execute,
        }
        if args.execute:
            from .checkpoint import prepare_variant_checkpoint

            path, report = prepare_variant_checkpoint(
                _resolve(root, args.parent), args.variant, _resolve(root, args.output)
            )
            payload.update(path=str(path), transfer=asdict(report))
        _print(payload)
        return 0
    if args.command == "train":
        from .config import (
            FORMAL_VALIDATION_BATCH,
            CommonTrainingConfig,
            training_config_for_phase,
        )

        phase = _phase(root, args.phase, args.masf_lr)
        common = training_config_for_phase(
            CommonTrainingConfig.from_yaml(root / "configs/training/common.yaml"),
            phase.name,
        )
        payload = {
            "variant": args.variant,
            "phase": asdict(phase),
            "weights": str(_resolve(root, args.weights).resolve()),
            "run_id": args.run_id,
            "batch": common.batch,
            "nbs": common.nbs,
            "workers": common.workers,
            "fraction": common.fraction,
            "amp": common.amp,
            "will_execute": args.execute,
        }
        if args.execute:
            from .training import launch_phase

            launch_phase(
                project_root=root,
                weights=_resolve(root, args.weights),
                masf_variant=args.variant,
                attention_config=root / "configs/attention/float-pwl-final.yaml",
                bittrue_config=root / "configs/attention/bittrue-pwl-final.yaml",
                phase=phase,
                common=common,
                run_id=args.run_id,
                validation_batch=FORMAL_VALIDATION_BATCH,
            )
            payload["status"] = "completed"
        _print(payload)
        return 0
    if args.command == "materialize-bittrue":
        payload = {"checkpoint": str(args.checkpoint), "output": str(args.output), "will_execute": args.execute}
        if args.execute:
            from .checkpoint import materialize_bittrue_checkpoint

            payload["output"] = str(
                materialize_bittrue_checkpoint(
                    _resolve(root, args.checkpoint),
                    root / "configs/attention/bittrue-pwl-final.yaml",
                    _resolve(root, args.output),
                )
            )
        _print(payload)
        return 0
    if args.command == "validate":
        from .config import CommonTrainingConfig

        common = CommonTrainingConfig.from_yaml(root / "configs/training/common.yaml")
        run = root / "artifacts/validation" / args.run_id
        payload = {"checkpoint": str(args.checkpoint), "run": str(run), "will_execute": args.execute}
        if args.execute:
            from .evaluation import validate_bittrue

            payload["metrics"] = str(
                validate_bittrue(
                    checkpoint=_resolve(root, args.checkpoint),
                    data=_resolve(root, Path(common.data)),
                    run_dir=run,
                    imgsz=common.imgsz,
                    batch=common.batch,
                    device=common.device,
                    workers=common.workers,
                )
            )
        _print(payload)
        return 0
    if args.command == "profile":
        if args.amp and args.kind != "training":
            raise ValueError("--amp 僅適用於 training profile")
        payload = {
            "kind": args.kind,
            "amp": args.amp,
            "output": str(args.output),
            "will_execute": args.execute,
        }
        if args.execute:
            from .profiling import profile_checkpoint, profile_training_step

            if args.kind == "inference":
                path = profile_checkpoint(
                    checkpoint=_resolve(root, args.checkpoint),
                    output=_resolve(root, args.output),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            else:
                path = profile_training_step(
                    checkpoint=_resolve(root, args.checkpoint),
                    float_attention_config=root / "configs/attention/float-pwl-final.yaml",
                    output=_resolve(root, args.output),
                    steps=args.steps,
                    amp=args.amp,
                )
            payload["output"] = str(path)
        _print(payload)
        return 0
    if args.command == "gate":
        from .selection import phase_c_candidate, phase_gate

        if args.policy == "phase-c-improve":
            selected = phase_c_candidate(args.parent_name, args.parent_map, args.child_name, args.child_map)
        else:
            selected = phase_gate(args.parent_name, args.parent_map, args.child_name, args.child_map)
        _print({"selected": selected, "rolled_back": selected == args.parent_name})
        return 0
    if args.command == "select":
        from .selection import Candidate, choose_architecture

        candidates = tuple(Candidate(**json.loads(path.read_text(encoding="utf-8"))) for path in args.candidate)
        _print({"winner": asdict(choose_architecture(candidates))})
        return 0
    if args.command == "export-masf":
        payload = {"output": str(args.output), "will_execute": args.execute}
        if args.execute:
            from .checkpoint import export_masf_state

            payload["output"] = str(export_masf_state(_resolve(root, args.checkpoint), _resolve(root, args.output)))
        _print(payload)
        return 0
    if args.command == "smoke":
        report = _smoke(root, args.variant, args.device)
        _print(report)
        return 0 if report["finite_loss"] and report["all_masf_gradients"] else 1
    raise AssertionError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
