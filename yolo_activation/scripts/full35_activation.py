#!/usr/bin/env python3
"""Full35 production activation preflight, manifest, validation, and recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activation_lab.activations import ActivationName, available_activations
from activation_lab.training.domain import StaticPolicy
from activation_lab.training.full35 import (
    Full35ActivationExperiment,
    load_full35_manifest,
    uniform_full35_policy,
)

DEFAULT_RECIPE = PROJECT_ROOT / "training/full35/activation-recipe.yaml"


def _policy(args: argparse.Namespace) -> StaticPolicy:
    activation = args.activation
    assignments: list[tuple[str, ActivationName]] = []
    for raw in args.region:
        name, separator, value = raw.partition("=")
        if not separator or not name or value not in available_activations():
            raise ValueError(f"invalid --region assignment: {raw}")
        assignments.append((name, value))  # type: ignore[arg-type]
    if assignments:
        policy_id = args.policy_id or "static--" + "__".join(
            f"{name}={value}" for name, value in sorted(assignments)
        )
        return StaticPolicy(
            policy_id=policy_id,
            default_activation="silu",
            region_assignments=tuple(sorted(assignments)),
        )
    return uniform_full35_policy(activation)


def _finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    return True


def _write(payload: Any, output: Path | None = None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    if output is not None:
        target = output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--activation",
        choices=available_activations(),
        default="silu",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="REGION=ACTIVATION",
    )
    parser.add_argument("--policy-id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full35 Activation 正式實驗工具")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="檢查 release、資料、hash 與環境")
    preflight.add_argument("--skip-hashes", action="store_true")
    preflight.add_argument("--output", type=Path)

    manifest = commands.add_parser("manifest", help="從 accepted checkpoint 盤點 SiLU")
    manifest.add_argument("--approve", action="store_true")
    manifest.add_argument("--output", type=Path)

    smoke = commands.add_parser("smoke", help="CPU/GPU 小張量 policy wiring smoke")
    _add_policy_arguments(smoke)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--output", type=Path)

    validate = commands.add_parser("validate", help="完整 COCO/BBAT5 formal validation")
    _add_policy_arguments(validate)
    validate.add_argument("--device", default="0")
    validate.add_argument(
        "--backend", choices=("float", "bittrue", "both"), default="both"
    )
    validate.add_argument("--name", required=True)
    validate.add_argument("--output", type=Path)

    train = commands.add_parser(
        "train", help="從 accepted best_joint 執行獨立 recovery"
    )
    _add_policy_arguments(train)
    train.add_argument("--device", default="0")
    train.add_argument(
        "--phase",
        choices=(
            "short_recovery",
            "region_sensitivity",
            "policy_search",
            "finalist_seed1",
            "finalist_seed2",
            "claim_ablation",
            "qat_if_needed",
        ),
        required=True,
    )
    train.add_argument("--detect-microbatch", type=int, default=32)
    train.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = Full35ActivationExperiment.from_yaml(args.recipe)
    if args.command == "preflight":
        report = experiment.preflight(verify_hashes=not args.skip_hashes)
        _write(
            {
                "ready": report.ready,
                "blockers": report.blockers,
                "warnings": report.warnings,
                "resolved": report.resolved,
            },
            args.output,
        )
        return 0 if report.ready else 2
    if args.command == "manifest":
        manifest = experiment.build_manifest(approve=args.approve)
        target = experiment.save_manifest(manifest, args.output)
        _write(
            {
                "path": str(target),
                "model_id": manifest.model_id,
                "reviewed": manifest.reviewed,
                "model_source_sha256": manifest.model_source_sha256,
                "site_count": len(manifest.sites),
                "regions": manifest.regions,
                "region_counts": {
                    region: sum(site.region == region for site in manifest.sites)
                    for region in manifest.regions
                },
            }
        )
        return 0
    manifest = load_full35_manifest(experiment.config)
    policy = _policy(args)
    if args.command == "smoke":
        loaded = experiment.load_policy_model(manifest, policy)
        device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
        model = loaded.model.to(device).eval()
        sample = torch.randn(1, 3, 64, 64, device=device)
        with torch.inference_mode():
            output = model(sample, task="both")
        _write(
            {
                "policy_id": policy.policy_id,
                "changed_paths": len(loaded.applied.changed_paths),
                "unchanged_paths": len(loaded.applied.unchanged_paths),
                "output_keys": sorted(output),
                "output_finite": _finite(output),
            },
            args.output,
        )
        return 0
    if args.command == "validate":
        backends = ("float", "bittrue") if args.backend == "both" else (args.backend,)
        payload = experiment.validate(
            manifest,
            policy,
            run_name=args.name,
            device=args.device,
            backends=backends,
        )
        _write(payload, args.output)
        return 0
    if args.command == "train":
        report = experiment.run_recovery(
            manifest,
            policy,
            phase_name=args.phase,
            run_name=args.name,
            device=args.device,
            detect_microbatch_size=args.detect_microbatch,
        )
        _write(report)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
