#!/usr/bin/env python3
"""Convert Full35 activation histograms into region-level decision support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activation_lab.activations import build_activation

DEFAULT_PROFILE_ROOT = (
    PROJECT_ROOT / "artifacts/runs/full35/profiling/profile-full-train-seed1"
)
DEFAULT_SENSITIVITY = (
    PROJECT_ROOT / "artifacts/runs/full35/sensitivity/"
    "sensitivity-zero-shot-poly_shift/sensitivity-summary.json"
)
CANDIDATES = (
    "hardswish",
    "relu",
    "qsilu_pq",
    "poly_shift",
    "poly_quality",
)
TRAINING_ONLY_REGIONS = frozenset({"detect_one2many", "pose_one2many", "pose_flow"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full35 profile 區域決策分析")
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--sensitivity", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _weighted_errors(
    counts: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
) -> dict[str, dict[str, float]]:
    bins = counts.numel()
    width = (maximum - minimum) / bins
    centers = torch.linspace(
        minimum + width / 2,
        maximum - width / 2,
        bins,
        dtype=torch.float64,
        requires_grad=True,
    )
    total = counts.sum()
    if total <= 0:
        raise ValueError("histogram contains no observations")
    weights = counts.to(dtype=torch.float64) / total
    silu = build_activation("silu").to(dtype=torch.float64)(centers)
    silu_derivative = torch.autograd.grad(silu.sum(), centers)[0].detach()
    occupied = counts > 0
    reports: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATES:
        values = build_activation(candidate).to(dtype=torch.float64)(centers)
        derivative = torch.autograd.grad(values.sum(), centers, retain_graph=True)[0]
        error = values.detach() - silu.detach()
        derivative_error = derivative.detach() - silu_derivative
        reports[candidate] = {
            "mae_vs_silu": float((weights * error.abs()).sum()),
            "rmse_vs_silu": float((weights * error.square()).sum().sqrt()),
            "derivative_mae_vs_silu": float((weights * derivative_error.abs()).sum()),
            "occupied_bin_max_abs_error_vs_silu": float(error[occupied].abs().max()),
        }
    return reports


def _aggregate_regions(report: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    total_elements = 0
    for site in report["sites"].values():
        region = site["region"]
        histogram = site["histogram"]
        counts = torch.tensor(histogram["counts"], dtype=torch.int64)
        current = grouped.setdefault(
            region,
            {
                "site_count": 0,
                "activation_elements": 0,
                "counts": torch.zeros_like(counts),
                "minimum": float(histogram["minimum"]),
                "maximum": float(histogram["maximum"]),
            },
        )
        if (
            current["minimum"] != float(histogram["minimum"])
            or current["maximum"] != float(histogram["maximum"])
            or current["counts"].numel() != counts.numel()
        ):
            raise ValueError(f"incompatible histograms in region {region}")
        current["site_count"] += 1
        current["activation_elements"] += int(site["count"])
        current["counts"] += counts
        total_elements += int(site["count"])

    deployment_elements = sum(
        current["activation_elements"]
        for region, current in grouped.items()
        if region not in TRAINING_ONLY_REGIONS
    )
    result: dict[str, Any] = {}
    for region, current in sorted(grouped.items()):
        result[region] = {
            "site_count": current["site_count"],
            "activation_elements": current["activation_elements"],
            "activation_element_fraction": (
                current["activation_elements"] / total_elements
            ),
            "deployment_graph": region not in TRAINING_ONLY_REGIONS,
            "deployment_activation_element_fraction": (
                current["activation_elements"] / deployment_elements
                if region not in TRAINING_ONLY_REGIONS
                else 0.0
            ),
            "candidate_errors": _weighted_errors(
                current["counts"],
                minimum=current["minimum"],
                maximum=current["maximum"],
            ),
        }
    return {
        "dataset_images": report["dataset_images"],
        "images_seen": report["images_seen"],
        "observed_site_count": report["observed_site_count"],
        "missing_sites": report["missing_sites"],
        "activation_elements": total_elements,
        "deployment_activation_elements": deployment_elements,
        "regions": result,
    }


def _sensitivity_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in payload["rows"]:
        deltas = raw["deltas"]
        worst_drop = max(0.0, *(-float(value) for value in deltas.values()))
        if worst_drop > 0.005:
            tier = "high"
        elif worst_drop > 0.001:
            tier = "medium"
        else:
            tier = "low"
        rows.append(
            {
                **raw,
                "deployment_graph": raw["region"] not in TRAINING_ONLY_REGIONS,
                "worst_zero_shot_drop": worst_drop,
                "risk_tier": tier,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    profile_root = args.profile_root.expanduser().resolve()
    tasks = {
        task: json.loads(
            (profile_root / f"{task}-activation-stats.json").read_text(encoding="utf-8")
        )
        for task in ("detect", "pose")
    }
    sensitivity_path = args.sensitivity.expanduser().resolve()
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "checkpoint_sha256": tasks["detect"]["checkpoint_sha256"],
        "profile_root": str(profile_root),
        "sensitivity_summary": str(sensitivity_path),
        "training_only_regions": sorted(TRAINING_ONLY_REGIONS),
        "tasks": {task: _aggregate_regions(report) for task, report in tasks.items()},
        "zero_shot_sensitivity": _sensitivity_rows(sensitivity),
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else profile_root / "decision-support.json"
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"decision_support={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
