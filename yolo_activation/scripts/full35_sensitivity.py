#!/usr/bin/env python3
"""Run a resumable, full-validation Full35 region sensitivity matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activation_lab.activations import available_activations
from activation_lab.training import StaticPolicy
from activation_lab.training.full35 import (
    Full35ActivationExperiment,
    load_full35_manifest,
)

DEFAULT_RECIPE = PROJECT_ROOT / "training/full35/activation-recipe.yaml"
PRIMARY_METRICS = (
    "coco/box/map50_95",
    "bbat/box/map50_95",
    "bbat/pose/map50_95",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full35 全量 one-region-at-a-time sensitivity"
    )
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument(
        "--activation",
        choices=available_activations(),
        default="poly_shift",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--backend", choices=("float", "bittrue", "both"), default="both"
    )
    parser.add_argument("--prefix", default="sensitivity-zero-shot")
    parser.add_argument("--region", action="append", default=[])
    return parser.parse_args()


def _baseline(path: Path) -> dict[str, float]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if not isinstance(metrics, dict):
        raise TypeError("accepted Full35 baseline metrics are missing")
    return {name: float(metrics[name]) for name in PRIMARY_METRICS}


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    experiment = Full35ActivationExperiment.from_yaml(args.recipe)
    manifest = load_full35_manifest(experiment.config)
    available_regions = tuple(sorted(manifest.regions))
    regions = tuple(args.region) if args.region else available_regions
    unknown = sorted(set(regions) - set(available_regions))
    if unknown:
        raise ValueError(f"unknown Full35 regions: {unknown}")
    backends = ("float", "bittrue") if args.backend == "both" else (args.backend,)
    if "bittrue" not in backends:
        raise ValueError("formal sensitivity selection requires the bittrue backend")

    baseline_path = (
        PROJECT_ROOT / "training/full35/contracts/accepted-full35-baseline.yaml"
    )
    baseline = _baseline(baseline_path)
    summary_path = (
        experiment.config.run_root
        / "sensitivity"
        / f"{args.prefix}-{args.activation}"
        / "sensitivity-summary.json"
    )
    rows: list[dict[str, Any]] = []
    for region in regions:
        run_name = f"{args.prefix}-{args.activation}-{region}-seed1"
        result_path = (
            experiment.config.run_root
            / "evaluations"
            / run_name
            / "activation-summary.json"
        )
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            status = "reused"
        else:
            policy = StaticPolicy(
                policy_id=f"region--{region}--{args.activation}",
                default_activation="silu",
                region_assignments=((region, args.activation),),
            )
            result = experiment.validate(
                manifest,
                policy,
                run_name=run_name,
                device=args.device,
                backends=backends,
            )
            status = "completed"
        metrics = result["metrics"]["bittrue"]
        row = {
            "region": region,
            "activation": args.activation,
            "status": status,
            "changed_site_count": len(result["changed_paths"]),
            "metrics": {name: float(metrics[name]) for name in PRIMARY_METRICS},
            "deltas": {
                name: float(metrics[name]) - baseline[name] for name in PRIMARY_METRICS
            },
            "run_name": run_name,
        }
        rows.append(row)
        _save(
            summary_path,
            {
                "schema_version": 1,
                "checkpoint": str(experiment.config.checkpoint),
                "checkpoint_sha256": experiment.config.checkpoint_sha256,
                "joint_config": str(experiment.config.joint_config),
                "data_fraction": experiment.config.fraction,
                "resampling": experiment.config.resampling,
                "activation": args.activation,
                "selection_backend": "bittrue",
                "baseline": baseline,
                "regions_requested": list(regions),
                "regions_completed": len(rows),
                "rows": rows,
            },
        )
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
