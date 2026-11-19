#!/usr/bin/env python3
"""从本机 Full35 artifacts 产生可提交的最终 JSON／CSV 摘要。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
SELECTOR_METRICS = (
    "coco/box/map50_95",
    "coco/person/box/map50_95",
    "bbat/box/map50_95",
    "bbat/pose/map50_95",
    "bbat/ball/box/map50_95",
    "bbat/bat/box/map50_95",
    "bbat/ball/pose/map50_95",
    "bbat/bat/pose/map50_95",
)


def _load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def _baseline() -> dict[str, float]:
    path = PROJECT_ROOT / "training/full35/contracts/accepted-full35-baseline.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {name: float(value) for name, value in document["metrics"].items()}


def _gate_row(job: dict[str, Any], baseline: dict[str, float]) -> dict[str, Any]:
    gate = job["gate"]
    deltas = {name: float(gate["deltas"][name]) for name in SELECTOR_METRICS}
    return {
        "id": job["id"],
        "run": job["run"],
        "status": job["status"],
        "metrics": {name: baseline[name] + deltas[name] for name in SELECTOR_METRICS},
        "deltas": deltas,
        "worst_delta": float(gate["worst_delta"]),
        "passed": bool(gate["passed"]),
        "failed_metrics": gate["failed_metrics"],
    }


def _read_gate_step(relative_path: str, step: int) -> dict[str, float]:
    values: dict[str, float] = {}
    with (PROJECT_ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["step"]) == step:
                values[row["metric"]] = float(row["value"])
    return values


def build_results() -> dict[str, Any]:
    baseline = _baseline()
    zero_shot: list[dict[str, Any]] = []
    pattern = "artifacts/runs/full35/evaluations/zero-shot-uniform-*-seed1/activation-summary.json"
    for path in sorted(PROJECT_ROOT.glob(pattern)):
        document = json.loads(path.read_text(encoding="utf-8"))
        activation = document["policy"]["default_activation"]
        selector = document["metrics"]["bittrue"]
        deltas = {
            name: float(selector[name]) - baseline[name] for name in SELECTOR_METRICS
        }
        zero_shot.append(
            {
                "activation": activation,
                "run": document["run_name"],
                "changed_sites": len(document["changed_paths"]),
                "selector_backend": "bittrue",
                "selector_metrics": {
                    name: float(selector[name]) for name in SELECTOR_METRICS
                },
                "selector_deltas": deltas,
                "worst_delta": min(deltas.values()),
                "passed": all(value >= -0.015 for value in deltas.values()),
                "full_metrics": document["metrics"],
            }
        )

    static_queue = _load_json("artifacts/runs/full35/queue/queue-state.json")
    completed_static = [
        _gate_row(job, baseline)
        for job in static_queue["jobs"]
        if job["gate"] is not None
    ]
    finalist_queue = _load_json(
        "artifacts/runs/full35/queue/finalist-seed1/queue-state.json"
    )
    completed_finalist = [
        _gate_row(job, baseline)
        for job in finalist_queue["jobs"]
        if job["gate"] is not None
    ]

    provisional_gate = _read_gate_step(
        "artifacts/runs/full35/finalist-seed1-uniform-qsilu-pq-seed1/logs/gate.csv", 0
    )
    provisional_deltas = {
        name: provisional_gate[f"delta/{name}"] for name in SELECTOR_METRICS
    }
    provisional = {
        "id": "finalist-seed1-qsilu-pq",
        "run": "finalist-seed1-uniform-qsilu-pq-seed1",
        "status": "stopped_by_user_after_epoch_1_and_epoch_2_macro_106",
        "completion_marker": False,
        "publish_as_final": False,
        "metrics": {
            name: baseline[name] + provisional_deltas[name] for name in SELECTOR_METRICS
        },
        "deltas": provisional_deltas,
        "worst_delta": min(provisional_deltas.values()),
        "passed": False,
        "failed_metrics": [
            name for name, value in provisional_deltas.items() if value < -0.015
        ],
        "score_best_joint": provisional_gate["score/best_joint"],
    }

    sensitivity = _load_json(
        "artifacts/runs/full35/sensitivity/sensitivity-zero-shot-poly_shift/"
        "sensitivity-summary.json"
    )
    profile = _load_json(
        "artifacts/runs/full35/profiling/profile-full-train-seed1/profile-summary.json"
    )
    return {
        "schema_version": 1,
        "release_date": "2026-08-29",
        "experiment_id": "full35-sipa-bcsp-v1",
        "selection": {
            "backend": "bittrue",
            "maximum_map50_95_drop": 0.015,
            "metrics": list(SELECTOR_METRICS),
            "rule": "每一项相对 accepted SiLU baseline delta 均不得低于 -0.015",
        },
        "data": {
            "fraction": 1.0,
            "resampling": False,
            "coco_train_images": 118287,
            "coco_val_images": 5000,
            "bbat5_formal_train_images": 5964,
            "bbat5_formal_val_images": 683,
            "bbat5_version": "canonical bbat5-v1",
        },
        "baseline": {
            "id": "full35-j3-best-joint-bittrue-epoch58",
            "checkpoint_sha256": (
                "d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c"
            ),
            "metrics": baseline,
        },
        "profile": profile,
        "uniform_zero_shot": zero_shot,
        "static_queue": {
            "counts": static_queue["counts"],
            "completed_results": completed_static,
            "blocked_job_ids": [
                job["id"] for job in static_queue["jobs"] if job["status"] == "blocked"
            ],
        },
        "poly_shift_region_zero_shot": sensitivity,
        "finalist_queue": {
            "counts": finalist_queue["counts"],
            "completed_results": completed_finalist,
            "provisional_stopped_results": [provisional],
        },
        "physical_batch_probes": [
            {
                "batch": 128,
                "result": "oom_first_forward",
                "process_gib": 30.66,
                "free_mib": 326.31,
                "next_allocation_mib": 400,
            },
            {
                "batch": 64,
                "result": "oom_first_forward",
                "process_gib": 30.59,
                "free_mib": 400.31,
                "next_allocation_mib": 400,
            },
            {
                "batch": 32,
                "result": "oom_first_forward",
                "process_gib": 30.92,
                "free_mib": 62.31,
                "next_allocation_mib": 100,
            },
            {"batch": 16, "result": "stable_full_training"},
        ],
        "published_weights": [
            {
                "role": "accepted_silu_quantization_parent",
                "bytes": 106825541,
                "sha256": (
                    "d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c"
                ),
            },
            {
                "role": "qsilu_pq_10epoch_recovery_quantization_candidate",
                "bytes": 106825541,
                "sha256": (
                    "7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e"
                ),
            },
        ],
    }


def write_reports(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    results = build_results()
    json_path = report_dir / "full35-activation-results.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = report_dir / "full35-activation-results.csv"
    fieldnames = ["phase", "id", "status", "passed", "worst_delta"]
    for metric in SELECTOR_METRICS:
        fieldnames.extend((metric, f"delta/{metric}"))
    rows: list[dict[str, Any]] = []
    baseline = results["baseline"]["metrics"]
    baseline_row: dict[str, Any] = {
        "phase": "baseline",
        "id": results["baseline"]["id"],
        "status": "accepted",
        "passed": True,
        "worst_delta": 0.0,
    }
    for metric in SELECTOR_METRICS:
        baseline_row[metric] = baseline[metric]
        baseline_row[f"delta/{metric}"] = 0.0
    rows.append(baseline_row)

    for item in results["uniform_zero_shot"]:
        row = {
            "phase": "uniform_zero_shot",
            "id": item["activation"],
            "status": "completed",
            "passed": item["passed"],
            "worst_delta": item["worst_delta"],
        }
        for metric in SELECTOR_METRICS:
            row[metric] = item["selector_metrics"][metric]
            row[f"delta/{metric}"] = item["selector_deltas"][metric]
        rows.append(row)

    for item in results["static_queue"]["completed_results"]:
        if item["id"].startswith("zero-shot"):
            continue
        row = {
            "phase": "short_recovery",
            "id": item["id"],
            "status": item["status"],
            "passed": item["passed"],
            "worst_delta": item["worst_delta"],
        }
        for metric in SELECTOR_METRICS:
            row[metric] = item["metrics"][metric]
            row[f"delta/{metric}"] = item["deltas"][metric]
        rows.append(row)

    for item in (
        results["finalist_queue"]["completed_results"]
        + results["finalist_queue"]["provisional_stopped_results"]
    ):
        row = {
            "phase": "finalist_seed1",
            "id": item["id"],
            "status": item["status"],
            "passed": item["passed"],
            "worst_delta": item["worst_delta"],
        }
        for metric in SELECTOR_METRICS:
            row[metric] = item["metrics"][metric]
            row[f"delta/{metric}"] = item["deltas"][metric]
        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    write_reports(args.output)


if __name__ == "__main__":
    main()
