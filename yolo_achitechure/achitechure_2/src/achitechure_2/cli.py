"""Dry-run-first command surface for the Stage-Aware Lite-C3k2 workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import CANDIDATES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m achitechure_2")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("preflight", help="check runtime, pycocotools, CUDA, and handoff gate")

    intake = commands.add_parser("intake", help="validate the formal achitechure_1 handoff")
    intake.add_argument("--manifest", type=Path, required=True)
    intake.add_argument("--execute", action="store_true")

    graph = commands.add_parser("inspect", help="inspect a checkpoint graph")
    graph.add_argument("--checkpoint", type=Path, required=True)
    graph.add_argument("--allow-missing-masf", action="store_true")

    build = commands.add_parser("build", help="materialize an independent C0/C1/C2/C3 candidate")
    build.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    build.add_argument("--output", type=Path)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--execute", action="store_true")

    train = commands.add_parser("train", help="run one gated training stage")
    train.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--stage", choices=("smoke", "formal", "extension", "qat"), required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--smoke-epochs", type=int, default=3)
    train.add_argument("--execute", action="store_true")

    extend = commands.add_parser("extension-gate")
    extend.add_argument("--metrics", type=Path, required=True, help="JSON array of 100 mAP50-95 values")
    extend.add_argument("--best-epoch", type=int, required=True)
    extend.add_argument("--early-stopped", action="store_true")

    assess = commands.add_parser("assess")
    assess.add_argument("--c0", type=Path, required=True)
    assess.add_argument("--candidate", type=Path, action="append", required=True)
    assess.add_argument("--execute", action="store_true", help="write artifacts/selection.json")

    fused = commands.add_parser("fuse-reference", help="materialize Q0 fused FP32 reference")
    fused.add_argument("--candidate", required=True)
    fused.add_argument("--checkpoint", type=Path, required=True)
    fused.add_argument("--output", type=Path, required=True)
    fused.add_argument("--execute", action="store_true")

    quant = commands.add_parser("quant-prepare", help="prepare Conv fake-quant simulation")
    quant.add_argument("--candidate", required=True)
    quant.add_argument("--checkpoint", type=Path, required=True)
    quant.add_argument("--output", type=Path, required=True)
    quant.add_argument("--execute", action="store_true")

    calibrate = commands.add_parser("quant-calibrate", help="calibrate a prepared Q1 simulation")
    calibrate.add_argument("--checkpoint", type=Path, required=True)
    calibrate.add_argument("--calibration-tensors", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--batch-size", type=int, default=16)
    calibrate.add_argument("--device", default="0")
    calibrate.add_argument("--max-batches", type=int)
    calibrate.add_argument("--execute", action="store_true")

    materialize = commands.add_parser("materialize-bittrue")
    materialize.add_argument("--checkpoint", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--execute", action="store_true")

    validate = commands.add_parser("validate-bittrue")
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--execute", action="store_true")

    profile = commands.add_parser("profile")
    profile.add_argument("--checkpoint", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--warmup", type=int, default=20)
    profile.add_argument("--iterations", type=int, default=100)
    profile.add_argument("--execute", action="store_true")

    gaps = commands.add_parser("quant-report")
    gaps.add_argument("--q0", type=float, required=True)
    gaps.add_argument("--q1", type=float, required=True)
    gaps.add_argument("--q2", type=float, required=True)
    return parser


def _load_metrics(path: Path):
    from .decisions import CandidateMetrics

    return CandidateMetrics(**json.loads(path.read_text(encoding="utf-8")))


def _assert_quant_candidate(root: Path, candidate: str) -> None:
    if candidate == "C0":
        return
    selection = root / "artifacts/selection.json"
    if not selection.is_file():
        raise RuntimeError("C_best has not been selected; quantization is limited to C0 until then")
    selected = json.loads(selection.read_text(encoding="utf-8")).get("c_best")
    if not isinstance(selected, dict) or selected.get("metrics", {}).get("candidate_id") != candidate:
        raise RuntimeError(f"quantization is limited to C0 and recorded C_best, not {candidate}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()

    if args.command == "preflight":
        import importlib.util

        import torch
        import ultralytics

        accepted = root / "artifacts/intake/accepted.json"
        payload = {
            "valid": bool(
                torch.cuda.is_available()
                and importlib.util.find_spec("pycocotools") is not None
                and accepted.is_file()
                and ultralytics.__version__ == "8.4.90"
            ),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pycocotools_installed": importlib.util.find_spec("pycocotools") is not None,
            "accepted_handoff": accepted.is_file(),
        }
        _print(payload)
        return 0 if payload["valid"] else 1

    if args.command == "intake":
        from .intake import HandoffManifest, validate_handoff, write_intake

        manifest = HandoffManifest.load(_resolve(root, args.manifest))
        payload: dict[str, Any] = {
            "manifest": str(manifest.source_manifest),
            "variant": manifest.variant,
            "float_checkpoint": str(manifest.float_checkpoint.path),
            "bittrue_checkpoint": str(manifest.bittrue_checkpoint.path),
            "will_execute": args.execute,
        }
        if args.execute:
            report = validate_handoff(manifest.source_manifest, project_root=root)
            destination = write_intake(report, root / "artifacts/intake/accepted.json")
            payload.update(status="accepted", report=str(destination), validation=asdict(report))
        _print(payload)
        return 0

    if args.command == "inspect":
        from ultralytics import YOLO

        from .graph import inspect_graph

        checkpoint = _resolve(root, args.checkpoint).resolve()
        report = inspect_graph(YOLO(str(checkpoint)).model, require_masf=not args.allow_missing_masf)
        _print(report.to_dict())
        return 0

    if args.command == "build":
        from .intake import require_accepted_intake

        intake = require_accepted_intake(root)
        output = _resolve(
            root,
            args.output or Path(f"artifacts/candidates/{args.candidate.lower()}/float-parent.pt"),
        ).resolve()
        payload = {
            "candidate": args.candidate,
            "same_float_parent": intake["float_checkpoint"],
            "output": str(output),
            "seed": args.seed,
            "will_execute": args.execute,
        }
        if args.execute:
            from ultralytics import YOLO

            from .candidate import build_candidate, write_build_report
            from .intake import file_sha256

            source = Path(intake["float_checkpoint"]["path"])
            yolo = YOLO(str(source))
            model, report = build_candidate(yolo.model, args.candidate, seed=args.seed)
            yolo.model = model
            output.parent.mkdir(parents=True, exist_ok=False)
            yolo.save(str(output))
            report_path = write_build_report(report, output.parent / "transfer-report.json")
            lineage = {
                "candidate_id": args.candidate,
                "parent": {"path": str(source), "sha256": file_sha256(source)},
                "checkpoint": {"path": str(output), "sha256": file_sha256(output)},
                "report": str(report_path),
            }
            (output.parent / "lineage.json").write_text(
                json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            payload.update(status="built", report=report.to_dict(), checkpoint_sha256=file_sha256(output))
        _print(payload)
        return 0

    if args.command == "train":
        from .training import STAGE_RULES

        checkpoint = _resolve(root, args.checkpoint).resolve()
        payload = {
            "candidate": args.candidate,
            "checkpoint": str(checkpoint),
            "stage": args.stage,
            "stage_rules": STAGE_RULES[args.stage],
            "run_id": args.run_id,
            "will_execute": args.execute,
        }
        if args.stage == "qat":
            _assert_quant_candidate(root, args.candidate)
        if args.execute:
            from .training import launch_training

            payload["completion"] = str(
                launch_training(
                    project_root=root,
                    checkpoint=checkpoint,
                    candidate_id=args.candidate,
                    stage=args.stage,
                    run_id=args.run_id,
                    smoke_epochs=args.smoke_epochs,
                )
            )
        _print(payload)
        return 0

    if args.command == "extension-gate":
        from .decisions import should_extend

        metrics = json.loads(_resolve(root, args.metrics).read_text(encoding="utf-8"))
        _print(asdict(should_extend(metrics, best_epoch=args.best_epoch, early_stopped=args.early_stopped)))
        return 0

    if args.command == "assess":
        from .decisions import choose_c_best, classify_candidate, trigger_c3_p5_fallback, trigger_r1

        c0 = _load_metrics(_resolve(root, args.c0))
        decisions = [classify_candidate(_load_metrics(_resolve(root, path)), c0) for path in args.candidate]
        winner = choose_c_best(decisions)
        payload = {
            "c0": asdict(c0),
            "decisions": [asdict(item) for item in decisions],
            "triggers": {
                "c3_p5": any(trigger_c3_p5_fallback(item) for item in decisions),
                "r1": any(trigger_r1(item) for item in decisions),
            },
            "c_best": asdict(winner) if winner else None,
            "quantization_mainline_gated": winner is None,
        }
        if args.execute:
            destination = root / "artifacts/selection.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            payload["selection_artifact"] = str(destination)
        _print(payload)
        return 0

    if args.command == "fuse-reference":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        _assert_quant_candidate(root, args.candidate)
        output = _resolve(root, args.output).resolve()
        payload = {
            "candidate": args.candidate,
            "output": str(output),
            "will_execute": args.execute,
        }
        if args.execute:
            from ultralytics import YOLO

            from .graph import inspect_graph
            from .quantization import make_fused_reference

            yolo = YOLO(str(_resolve(root, args.checkpoint).resolve()))
            yolo.model = make_fused_reference(yolo.model)
            inspect_graph(yolo.model)
            output.parent.mkdir(parents=True, exist_ok=False)
            yolo.save(str(output))
            inspect_graph(YOLO(str(output)).model)
            payload["status"] = "fused-and-reloaded"
        _print(payload)
        return 0

    if args.command == "quant-prepare":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        _assert_quant_candidate(root, args.candidate)
        output = _resolve(root, args.output).resolve()
        payload = {
            "candidate": args.candidate,
            "checkpoint": str(_resolve(root, args.checkpoint).resolve()),
            "output": str(output),
            "simulation_only": True,
            "will_execute": args.execute,
        }
        if args.execute:
            import torch
            from ultralytics import YOLO

            from .quantization import prepare_w8a8_simulation, quant_scope_dict

            yolo = YOLO(str(_resolve(root, args.checkpoint).resolve()))
            prepared, scope = prepare_w8a8_simulation(yolo.model)
            output.parent.mkdir(parents=True, exist_ok=False)
            torch.save({"model": prepared, "simulation_only": True}, output)
            scope_path = output.parent / "quant-scope.json"
            scope_path.write_text(
                json.dumps(quant_scope_dict(scope), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            payload.update(scope=quant_scope_dict(scope), scope_artifact=str(scope_path))
        _print(payload)
        return 0

    if args.command == "quant-calibrate":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        output = _resolve(root, args.output).resolve()
        payload = {"output": str(output), "simulation_only": True, "will_execute": args.execute}
        if args.execute:
            import torch

            from .quantization import calibrate_w8a8

            checkpoint = torch.load(_resolve(root, args.checkpoint), map_location="cpu", weights_only=False)
            model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
            if not isinstance(model, torch.nn.Module):
                raise TypeError("quant calibration checkpoint must contain model")
            images = torch.load(
                _resolve(root, args.calibration_tensors), map_location="cpu", weights_only=True
            )
            if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1:] != (3, 640, 640):
                raise ValueError("calibration tensors must have shape [N, 3, 640, 640]")
            count = calibrate_w8a8(model, images.split(args.batch_size), max_batches=args.max_batches)
            output.parent.mkdir(parents=True, exist_ok=False)
            torch.save({**checkpoint, "model": model, "calibration_batches": count}, output)
            payload["calibration_batches"] = count
        _print(payload)
        return 0

    if args.command == "materialize-bittrue":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        output = _resolve(root, args.output).resolve()
        payload = {"output": str(output), "will_execute": args.execute}
        if args.execute:
            from achitechure_1.checkpoint import materialize_bittrue_checkpoint

            payload["output"] = str(
                materialize_bittrue_checkpoint(
                    _resolve(root, args.checkpoint),
                    root.parent / "achitechure_1/configs/attention/bittrue-pwl-final.yaml",
                    output,
                )
            )
        _print(payload)
        return 0

    if args.command == "validate-bittrue":
        from .config import load_yaml
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        common = load_yaml(root / "configs/training/common.yaml")
        run = root / "artifacts/validation" / args.run_id
        payload = {"run": str(run), "will_execute": args.execute}
        if args.execute:
            from achitechure_1.evaluation import validate_bittrue

            payload["metrics"] = str(
                validate_bittrue(
                    checkpoint=_resolve(root, args.checkpoint),
                    data=_resolve(root, Path(common["data"])),
                    run_dir=run,
                    imgsz=int(common["imgsz"]),
                    batch=int(common["batch"]),
                    device=str(common["device"]),
                    workers=int(common["workers"]),
                )
            )
        _print(payload)
        return 0

    if args.command == "profile":
        from .intake import require_accepted_intake

        require_accepted_intake(root)
        payload = {"output": str(_resolve(root, args.output)), "will_execute": args.execute}
        if args.execute:
            from achitechure_1.profiling import profile_checkpoint

            payload["output"] = str(
                profile_checkpoint(
                    checkpoint=_resolve(root, args.checkpoint),
                    output=_resolve(root, args.output),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )
        _print(payload)
        return 0

    if args.command == "quant-report":
        from .quantization import robustness_report

        _print(asdict(robustness_report(args.q0, args.q1, args.q2)))
        return 0
    raise AssertionError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
