#!/usr/bin/env python3
"""Run resumable all-non-SiLU Full35 Q3 sensitivity on CPU without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

# This runner is intentionally CPU-only. Hide every CUDA device before importing
# PyTorch (or any package that may import it indirectly), so even framework startup
# cannot create a CUDA context.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activation_lab.activations import available_activations, build_activation
from activation_lab.hardware import hardware_proxy
from activation_lab.training import (
    Full35ActivationExperiment,
    StaticPolicy,
    load_full35_manifest,
    load_manifest,
)

ACCEPTED_RECIPE = PROJECT_ROOT / "training/full35/activation-recipe.yaml"
QSILU_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/runs/full35/short-recovery-v2-lr01-uniform-qsilu-pq-seed1"
    / "inference/best_joint.pt"
)
DEFAULT_RUN_ROOT = PROJECT_ROOT / "artifacts/runs/full35/q3-nonsilu-cpu"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/q3-nonsilu-cpu"
PROFILE_ROOT = PROJECT_ROOT / "artifacts/runs/full35/profiling/profile-full-train-seed1"
PROFILE_FILES = {
    "detect": PROFILE_ROOT / "detect-activation-stats.json",
    "pose": PROFILE_ROOT / "pose-activation-stats.json",
}
DECISION_SUPPORT_PATH = PROFILE_ROOT / "decision-support.json"
DEFAULT_CANDIDATES = ("hardswish", "leaky_relu_1_8")
REGIONS = (
    "backbone_attention",
    "backbone_deep",
    "backbone_early",
    "detect_one2many",
    "detect_one2one",
    "masf",
    "neck",
    "neck_attention",
    "pose_flow",
    "pose_one2many",
    "pose_one2one",
)
TRAINING_ONLY_REGIONS = frozenset({"detect_one2many", "pose_flow", "pose_one2many"})
GATE_METRICS = (
    "coco/box/map50_95",
    "coco/person/box/map50_95",
    "bbat/box/map50_95",
    "bbat/pose/map50_95",
    "bbat/ball/box/map50_95",
    "bbat/bat/box/map50_95",
    "bbat/ball/pose/map50_95",
    "bbat/bat/pose/map50_95",
)
SUMMARY_METRICS = GATE_METRICS[:4]
METRIC_LABELS = {
    "coco/box/map50_95": "COCO Δ",
    "coco/person/box/map50_95": "Person Δ",
    "bbat/box/map50_95": "BBAT box Δ",
    "bbat/pose/map50_95": "BBAT pose Δ",
}
METRIC_COLORS = {
    "coco/box/map50_95": "#2563EB",
    "coco/person/box/map50_95": "#7C3AED",
    "bbat/box/map50_95": "#D97706",
    "bbat/pose/map50_95": "#059669",
}
DISPLAY_NAMES = {
    "hardswish": "Hardswish",
    "leaky_relu_1_8": "Dyadic LeakyReLU 1/8",
}
REGION_LABELS = {
    "backbone_attention": "backbone\nattention",
    "backbone_deep": "backbone\ndeep",
    "backbone_early": "backbone\nearly",
    "detect_one2many": "detect\none-to-many",
    "detect_one2one": "detect\none-to-one",
    "masf": "MASF",
    "neck": "neck",
    "neck_attention": "neck\nattention",
    "pose_flow": "pose\nflow",
    "pose_one2many": "pose\none-to-many",
    "pose_one2one": "pose\none-to-one",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "以已完成 qSiLU recovery 權重執行全非 SiLU 的 CPU 11-region Q3 "
            "zero-shot sensitivity；不訓練、不更新權重"
        )
    )
    parser.add_argument("--candidate", action="append", choices=available_activations())
    parser.add_argument("--region", action="append", choices=REGIONS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只建立並稽核 qSiLU manifest，不執行資料集驗證",
    )
    parser.add_argument(
        "--eager-qsilu",
        action="store_true",
        help="停用逐位等價的 CPU Inductor qSiLU kernel，僅供除錯",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _configure_cpu(threads: int) -> None:
    if threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except RuntimeError:
        pass


def _assert_cpu_only() -> dict[str, Any]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    if visible_devices != "" or cuda_available or cuda_device_count != 0:
        raise RuntimeError(
            "Q3 CPU safety guard failed: "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}, "
            f"torch.cuda.is_available()={cuda_available}, "
            f"torch.cuda.device_count()={cuda_device_count}"
        )
    return {
        "cuda_visible_devices": visible_devices,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": cuda_device_count,
    }


def _install_cpu_qsilu_factory(*, compile_kernel: bool) -> str:
    """Share one stateless qSiLU module across paths and optionally compile it.

    The compiled module executes the exact registry implementation. A 4-D
    boundary probe must be bitwise identical before the factory is installed;
    otherwise the Q3 run fails closed and cannot silently change its baseline.
    """

    from activation_lab.training import model as policy_model

    eager_factory = policy_model.build_activation
    eager_qsilu = eager_factory("qsilu_pq").eval()
    if compile_kernel:
        shared_qsilu = torch.compile(
            eager_qsilu,
            backend="inductor",
            fullgraph=True,
            dynamic=True,
        )
        execution = "torch.compile-inductor-dynamic-bitwise-verified"
    else:
        shared_qsilu = eager_qsilu
        execution = "eager-shared-module"

    probe = torch.tensor(
        [
            -12.0,
            -8.0,
            -4.0,
            -2.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            2.0,
            4.0,
            8.0,
            12.0,
        ],
        dtype=torch.float32,
    ).reshape(1, 1, 3, 4)
    with torch.inference_mode():
        expected = eager_qsilu(probe)
        actual = shared_qsilu(probe)
    if not torch.equal(actual, expected):
        maximum_error = float((actual - expected).abs().max().item())
        raise RuntimeError(
            "compiled CPU qSiLU failed bitwise verification: "
            f"max_abs_error={maximum_error}"
        )

    def shared_factory(name: str) -> torch.nn.Module:
        if name == "qsilu_pq":
            return shared_qsilu
        return eager_factory(name)

    policy_model.build_activation = shared_factory
    return execution


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK TC", "Droid Sans Fallback"],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "svg.hashsalt": "full35-nonsilu-q3-cpu-v1",
        }
    )


def _validate_candidates(candidates: tuple[str, ...]) -> None:
    invalid = sorted(set(candidates) & {"silu", "relu", "qsilu_pq"})
    if invalid:
        raise ValueError(
            "Q3 candidate must be simpler than the qSiLU fallback and cannot contain "
            f"SiLU/ReLU/qSiLU: {invalid}"
        )
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Q3 candidates must be non-empty and unique")


def _policy_counts(manifest: Any, policy: StaticPolicy) -> dict[str, int]:
    return dict(Counter(policy.resolve(site) for site in manifest.sites))


def _audit_all_nonsilu(manifest: Any, policy: StaticPolicy) -> dict[str, int]:
    counts = _policy_counts(manifest, policy)
    if counts.get("silu", 0):
        raise ValueError(
            f"all-non-SiLU policy {policy.policy_id!r} still contains SiLU: {counts}"
        )
    if sum(counts.values()) != len(manifest.sites):
        raise ValueError("policy activation counts do not cover the complete manifest")
    return counts


def _build_experiments(
    run_root: Path, report_root: Path
) -> tuple[Full35ActivationExperiment, Any, Full35ActivationExperiment, Any]:
    accepted = Full35ActivationExperiment.from_yaml(ACCEPTED_RECIPE)
    accepted_config = replace(accepted.config, run_root=run_root)
    accepted = Full35ActivationExperiment(accepted_config)
    accepted_manifest = load_full35_manifest(accepted.config)

    checkpoint = QSILU_CHECKPOINT.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"completed qSiLU checkpoint is missing: {checkpoint}")
    checkpoint_sha256 = _sha256(checkpoint)
    manifest_path = report_root / "q3-qsilu-recovery-activation-manifest.yaml"
    qsilu_config = replace(
        accepted.config,
        experiment_id="full35-nonsilu-q3-cpu-v1",
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        manifest=manifest_path,
        run_root=run_root,
    )
    qsilu = Full35ActivationExperiment(qsilu_config)
    if manifest_path.is_file():
        qsilu_manifest = load_manifest(manifest_path)
    else:
        qsilu_manifest = qsilu.build_manifest(approve=True)
        qsilu.save_manifest(qsilu_manifest, manifest_path)
    if not qsilu_manifest.reviewed:
        raise ValueError("Q3 qSiLU manifest must be reviewed")
    if qsilu_manifest.model_source_sha256 != checkpoint_sha256:
        raise ValueError("Q3 qSiLU manifest checkpoint SHA-256 drifted")
    if len(qsilu_manifest.sites) != 190:
        raise ValueError(
            f"Q3 qSiLU manifest must cover 190 sites, got {len(qsilu_manifest.sites)}"
        )
    regions = {site.region for site in qsilu_manifest.sites}
    if regions != set(REGIONS):
        raise ValueError(
            f"Q3 manifest region contract drifted: missing={sorted(set(REGIONS) - regions)}, "
            f"unexpected={sorted(regions - set(REGIONS))}"
        )
    return accepted, accepted_manifest, qsilu, qsilu_manifest


def _validate_cached_payload(
    payload: dict[str, Any],
    experiment: Full35ActivationExperiment,
    policy: StaticPolicy,
) -> None:
    if payload.get("checkpoint_sha256") != experiment.config.checkpoint_sha256:
        raise ValueError(
            "cached Q3 result checkpoint SHA-256 does not match experiment"
        )
    if payload.get("data_fraction") != 1.0 or payload.get("resampling"):
        raise ValueError("cached Q3 result changed the formal dataset contract")
    if payload.get("policy") != {
        "policy_id": policy.policy_id,
        "default_activation": policy.default_activation,
        "region_assignments": dict(policy.region_assignments),
        "site_assignments": dict(policy.site_assignments),
    }:
        raise ValueError("cached Q3 result policy does not match requested policy")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"bittrue"}:
        raise ValueError("cached Q3 result must contain the CPU bittrue backend only")


def _load_or_validate(
    experiment: Full35ActivationExperiment,
    manifest: Any,
    policy: StaticPolicy,
    *,
    run_name: str,
) -> tuple[dict[str, Any], str, float | None]:
    result_path = (
        experiment.config.run_root
        / "evaluations"
        / run_name
        / "activation-summary.json"
    )
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_cached_payload(payload, experiment, policy)
        return payload, "reused", None
    started = time.monotonic()
    payload = experiment.validate(
        manifest,
        policy,
        run_name=run_name,
        device="cpu",
        backends=("bittrue",),
        plots=False,
    )
    elapsed = time.monotonic() - started
    _validate_cached_payload(payload, experiment, policy)
    return payload, "completed", elapsed


def _metrics(payload: dict[str, Any]) -> dict[str, float]:
    values = payload["metrics"]["bittrue"]
    missing = [metric for metric in GATE_METRICS if metric not in values]
    if missing:
        raise ValueError(f"Q3 result is missing gate metrics: {missing}")
    return {metric: float(values[metric]) for metric in GATE_METRICS}


def _deltas(values: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {metric: values[metric] - baseline[metric] for metric in GATE_METRICS}


def _profile_weighted_errors(
    manifest: Any, candidates: tuple[str, ...]
) -> dict[tuple[str, str], dict[str, float]]:
    site_regions = {site.module_path: site.region for site in manifest.sites}
    sums: dict[tuple[str, str], dict[str, float]] = {
        (candidate, region): {"count": 0.0, "abs": 0.0, "square": 0.0}
        for candidate in candidates
        for region in REGIONS
    }
    qsilu = build_activation("qsilu_pq").to(dtype=torch.float64)
    modules = {
        candidate: build_activation(candidate).to(dtype=torch.float64)
        for candidate in candidates
    }
    with torch.no_grad():
        for profile_path in PROFILE_FILES.values():
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            for path, stats in payload["sites"].items():
                region = site_regions.get(path)
                if region is None:
                    continue
                histogram = stats["histogram"]
                counts = torch.tensor(histogram["counts"], dtype=torch.float64)
                total = float(counts.sum())
                if total <= 0:
                    continue
                minimum = float(histogram["minimum"])
                maximum = float(histogram["maximum"])
                width = (maximum - minimum) / int(histogram["bins"])
                centers = torch.linspace(
                    minimum + 0.5 * width,
                    maximum - 0.5 * width,
                    int(histogram["bins"]),
                    dtype=torch.float64,
                )
                reference = qsilu(centers)
                for candidate, module in modules.items():
                    error = module(centers) - reference
                    aggregate = sums[(candidate, region)]
                    aggregate["count"] += total
                    aggregate["abs"] += float((error.abs() * counts).sum())
                    aggregate["square"] += float((error.square() * counts).sum())
    result: dict[tuple[str, str], dict[str, float]] = {}
    for key, aggregate in sums.items():
        count = aggregate["count"]
        result[key] = {
            "occupied_histogram_count": count,
            "weighted_mae_vs_qsilu": aggregate["abs"] / count
            if count
            else float("nan"),
            "weighted_rmse_vs_qsilu": (
                (aggregate["square"] / count) ** 0.5 if count else float("nan")
            ),
        }
    return result


def _region_metadata() -> dict[str, dict[str, Any]]:
    payload = json.loads(DECISION_SUPPORT_PATH.read_text(encoding="utf-8"))
    metadata: dict[str, dict[str, Any]] = {}
    for region in REGIONS:
        detect = payload["tasks"]["detect"]["regions"].get(region)
        pose = payload["tasks"]["pose"]["regions"].get(region)
        metadata[region] = {
            "training_only": region in TRAINING_ONLY_REGIONS,
            "detect_applicable": detect is not None,
            "pose_applicable": pose is not None,
            "detect_deployment_fraction": float(
                detect["deployment_activation_element_fraction"] if detect else 0.0
            ),
            "pose_deployment_fraction": float(
                pose["deployment_activation_element_fraction"] if pose else 0.0
            ),
            "detect_site_count": int(detect["site_count"]) if detect else 0,
            "pose_site_count": int(pose["site_count"]) if pose else 0,
        }
    return metadata


def _row(
    *,
    candidate: str,
    region: str,
    payload: dict[str, Any],
    status: str,
    elapsed_seconds: float | None,
    activation_counts: dict[str, int],
    local_baseline: dict[str, float],
    global_baseline: dict[str, float],
    profile_error: dict[str, float],
    region_metadata: dict[str, Any],
) -> dict[str, Any]:
    values = _metrics(payload)
    local_deltas = _deltas(values, local_baseline)
    global_deltas = _deltas(values, global_baseline)
    candidate_proxy = hardware_proxy(candidate)
    qsilu_proxy = hardware_proxy("qsilu_pq")
    return {
        "candidate": candidate,
        "region": region,
        "status": status,
        "run_name": payload["run_name"],
        "elapsed_seconds": elapsed_seconds,
        "activation_counts": activation_counts,
        "changed_path_count": len(payload["changed_paths"]),
        "training_only": region_metadata["training_only"],
        "detect_deployment_fraction": region_metadata["detect_deployment_fraction"],
        "pose_deployment_fraction": region_metadata["pose_deployment_fraction"],
        "metrics": values,
        "local_deltas_vs_uniform_qsilu": local_deltas,
        "global_deltas_vs_accepted_silu": global_deltas,
        "local_worst_delta": min(local_deltas.values()),
        "global_worst_delta": min(global_deltas.values()),
        "global_gate_passed": min(global_deltas.values()) >= -0.015,
        "profile_proxy": profile_error,
        "hardware_proxy": asdict(candidate_proxy),
        "per_element_savings_vs_qsilu": {
            "variable_multiplications": (
                qsilu_proxy.variable_multiplications
                - candidate_proxy.variable_multiplications
            ),
            "constant_multiplications": (
                qsilu_proxy.constant_multiplications
                - candidate_proxy.constant_multiplications
            ),
            "add_subtract": qsilu_proxy.add_subtract - candidate_proxy.add_subtract,
            "range_operations": (
                qsilu_proxy.range_operations - candidate_proxy.range_operations
            ),
        },
    }


def _flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        base = {
            "candidate": row["candidate"],
            "region": row["region"],
            "status": row["status"],
            "training_only": row["training_only"],
            "changed_path_count": row["changed_path_count"],
            "elapsed_seconds": row["elapsed_seconds"],
            "detect_deployment_fraction": row["detect_deployment_fraction"],
            "pose_deployment_fraction": row["pose_deployment_fraction"],
            "local_worst_delta": row["local_worst_delta"],
            "global_worst_delta": row["global_worst_delta"],
            "global_gate_passed": row["global_gate_passed"],
            "profile_weighted_mae_vs_qsilu": row["profile_proxy"][
                "weighted_mae_vs_qsilu"
            ],
            "profile_weighted_rmse_vs_qsilu": row["profile_proxy"][
                "weighted_rmse_vs_qsilu"
            ],
        }
        for metric in GATE_METRICS:
            base[f"metric::{metric}"] = row["metrics"][metric]
            base[f"local_delta::{metric}"] = row["local_deltas_vs_uniform_qsilu"][
                metric
            ]
            base[f"global_delta::{metric}"] = row["global_deltas_vs_accepted_silu"][
                metric
            ]
        flattened.append(base)
    return flattened


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = _flatten_rows(rows)
    if not flattened:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def _save_figure(figure: plt.Figure, root: Path, stem: str) -> None:
    figure.savefig(root / f"{stem}.png", bbox_inches="tight")
    figure.savefig(
        root / f"{stem}.svg",
        bbox_inches="tight",
        metadata={"Date": "2026-08-31"},
    )
    plt.close(figure)


def _plot_delta_matrix(
    summary: dict[str, Any], root: Path, *, global_reference: bool
) -> None:
    candidates = tuple(summary["candidates"])
    rows_by_key = {(row["candidate"], row["region"]): row for row in summary["rows"]}
    figure, axes = plt.subplots(
        len(candidates),
        1,
        figsize=(22, 6.2 * len(candidates)),
        squeeze=False,
        constrained_layout=True,
    )
    for index, candidate in enumerate(candidates):
        axis = axes[index, 0]
        positions = np.arange(len(REGIONS))
        width = 0.19
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            values: list[float] = []
            for region in REGIONS:
                row = rows_by_key.get((candidate, region))
                if row is None:
                    values.append(float("nan"))
                    continue
                key = (
                    "global_deltas_vs_accepted_silu"
                    if global_reference
                    else "local_deltas_vs_uniform_qsilu"
                )
                values.append(100.0 * row[key][metric])
            axis.bar(
                positions + (metric_index - 1.5) * width,
                values,
                width=width,
                color=METRIC_COLORS[metric],
                label=METRIC_LABELS[metric],
            )
        for region_index, region in enumerate(REGIONS):
            if region in TRAINING_ONLY_REGIONS:
                axis.axvspan(
                    region_index - 0.48,
                    region_index + 0.48,
                    color="#E5E7EB",
                    alpha=0.55,
                    zorder=0,
                )
        axis.axhline(0.0, color="#111827", linewidth=1)
        if global_reference:
            axis.axhline(
                -1.5,
                color="#DC2626",
                linestyle="--",
                linewidth=1.5,
                label="正式 gate：−1.50 pp",
            )
        axis.set_xticks(
            positions,
            [REGION_LABELS[region] for region in REGIONS],
            fontsize=10,
        )
        axis.set_ylabel("ΔmAP50-95（百分點）")
        reference = "accepted SiLU" if global_reference else "uniform recovered qSiLU"
        completed = sum((candidate, region) in rows_by_key for region in REGIONS)
        axis.set_title(
            f"{DISPLAY_NAMES.get(candidate, candidate)}｜相對 {reference}｜"
            f"{completed}/{len(REGIONS)} regions"
        )
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=5 if global_reference else 4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle(
        "Full35 全非 SiLU：CPU 11-region zero-shot sensitivity\n"
        "灰底為 training-only region；bit-true weight backend；沒有重新訓練",
        fontsize=19,
        fontweight="bold",
    )
    stem = (
        "nonsilu-q3-11-region-global"
        if global_reference
        else "nonsilu-q3-11-region-local"
    )
    _save_figure(figure, root, stem)


def _markdown_table(rows: list[dict[str, Any]], candidate: str) -> list[str]:
    lines = [
        f"### {DISPLAY_NAMES.get(candidate, candidate)}",
        "",
        (
            "| Region | 類型 | COCO Δ | Person Δ | BBAT box Δ | BBAT pose Δ | "
            "八項最差 global Δ | Gate |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    by_region = {row["region"]: row for row in rows if row["candidate"] == candidate}
    for region in REGIONS:
        row = by_region.get(region)
        if row is None:
            lines.append(f"| `{region}` | 待執行 | — | — | — | — | — | — |")
            continue
        deltas = row["local_deltas_vs_uniform_qsilu"]
        region_type = "training-only" if row["training_only"] else "deployment"
        values = [f"{deltas[metric]:+.6f}" for metric in SUMMARY_METRICS]
        lines.append(
            f"| `{region}` | {region_type} | {' | '.join(values)} | "
            f"{row['global_worst_delta']:+.6f} | "
            f"{'通過' if row['global_gate_passed'] else '未通過'} |"
        )
    return lines


def _write_report(summary: dict[str, Any], root: Path) -> None:
    expected = len(summary["candidates"]) * len(summary["regions_requested"])
    completed = len(summary["rows"])
    accepted = summary.get("accepted_silu_cpu_baseline", {}).get("metrics", {})
    qsilu = summary.get("uniform_qsilu_cpu_baseline", {}).get("metrics", {})
    lines = [
        "# Full35 全非 SiLU CPU Q3：11-region Zero-shot Sensitivity",
        "",
        (
            f"> 狀態：{completed}/{expected} candidate-region cases。這是 zero-shot validation，"
            "沒有 optimizer、backward、epoch 訓練或權重更新。"
        ),
        "",
        "## 實驗契約",
        "",
        "- 最終 policy 的 190 個 activation sites 均不得為 SiLU。",
        "- 基底權重：已完成的 qSiLU 10-epoch short-recovery `best_joint.pt`。",
        "- 裝置：CPU；backend：bit-true weights；資料比例 100%，不抽樣、不重新切分。",
        "- 資料：COCO2017 與 Canonical BBAT5 v1。",
        "- 主表四項 Δ 相對同裝置 uniform recovered qSiLU；正式 gate 仍相對同裝置 accepted SiLU 稽核八項。",
        "",
        "## CPU Baseline",
        "",
        "| Metric | accepted SiLU | uniform recovered qSiLU | qSiLU Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in SUMMARY_METRICS:
        if metric in accepted and metric in qsilu:
            lines.append(
                f"| {METRIC_LABELS[metric].removesuffix(' Δ')} | "
                f"{accepted[metric]:.6f} | {qsilu[metric]:.6f} | "
                f"{qsilu[metric] - accepted[metric]:+.6f} |"
            )
        else:
            lines.append(f"| {METRIC_LABELS[metric].removesuffix(' Δ')} | — | — | — |")
    lines.extend(
        [
            "",
            "## 圖表",
            "",
            "![相對 uniform qSiLU 的 11-region sensitivity](nonsilu-q3-11-region-local.png)",
            "",
            "![相對 accepted SiLU 的正式 gate 視圖](nonsilu-q3-11-region-global.png)",
            "",
            "灰底 region 是 training-only graph；其替換結果不代表部署推論節省。",
            "",
            "## Region 結果",
            "",
        ]
    )
    for candidate in summary["candidates"]:
        lines.extend(_markdown_table(summary["rows"], candidate))
        lines.append("")
    lines.extend(
        [
            "## 證據邊界",
            "",
            "- 這是逐 region 一次只替換一區的一階 zero-shot sensitivity，不能取代多區 mixed policy 完整驗證。",
            "- CPU 與 GPU 不保證 bitwise 相同；本報告所有 delta 都使用同一 CPU backend baseline，避免跨裝置相減。",
            "- `leaky_relu_1_8` 在本階段以 float graph 的精確 1/8 係數執行；真正 INT8／RTL bit-exact 屬於後續量化階段。",
            "- 運算量欄位是解析 proxy，不是 latency、功耗或面積實測。",
            "",
            "機器可讀結果：[`summary.json`](summary.json)／[`rows.csv`](rows.csv)。",
        ]
    )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(summary: dict[str, Any], report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_root / "summary.json", summary)
    _write_csv(report_root / "rows.csv", summary["rows"])
    _configure_plot_style()
    _plot_delta_matrix(summary, report_root, global_reference=False)
    _plot_delta_matrix(summary, report_root, global_reference=True)
    _write_report(summary, report_root)


def main() -> int:
    args = _parse_args()
    candidates = tuple(args.candidate or DEFAULT_CANDIDATES)
    regions = tuple(args.region or REGIONS)
    _validate_candidates(candidates)
    cpu_safety = _assert_cpu_only()
    _configure_cpu(args.torch_threads)
    activation_execution = _install_cpu_qsilu_factory(
        compile_kernel=not args.eager_qsilu
    )
    run_root = args.run_root.expanduser().resolve()
    report_root = args.report_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    accepted, accepted_manifest, qsilu, qsilu_manifest = _build_experiments(
        run_root, report_root
    )
    q_policy = StaticPolicy(
        policy_id="q3-cpu-reference-uniform-qsilu",
        default_activation="qsilu_pq",
    )
    q_counts = _audit_all_nonsilu(qsilu_manifest, q_policy)
    if q_counts != {"qsilu_pq": 190}:
        raise ValueError(f"uniform qSiLU policy audit failed: {q_counts}")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "full35-nonsilu-q3-cpu-v1",
        "stage": "Q3_region_zero_shot_sensitivity",
        "training_performed": False,
        "device": "cpu",
        "cpu_safety": cpu_safety,
        "backend": "bittrue",
        "validation_plots": False,
        "activation_execution": activation_execution,
        "torch_threads": args.torch_threads,
        "data_fraction": 1.0,
        "resampling": False,
        "accepted_checkpoint": str(accepted.config.checkpoint),
        "accepted_checkpoint_sha256": accepted.config.checkpoint_sha256,
        "qsilu_checkpoint": str(qsilu.config.checkpoint),
        "qsilu_checkpoint_sha256": qsilu.config.checkpoint_sha256,
        "manifest": str(qsilu.config.manifest),
        "manifest_site_count": len(qsilu_manifest.sites),
        "candidates": list(candidates),
        "regions_requested": list(regions),
        "training_only_regions": sorted(TRAINING_ONLY_REGIONS),
        "accepted_silu_cpu_baseline": {},
        "uniform_qsilu_cpu_baseline": {},
        "rows": [],
    }
    if args.prepare_only:
        _write_outputs(summary, report_root)
        print(
            json.dumps(
                {
                    "prepared": True,
                    "manifest": str(qsilu.config.manifest),
                    "activation_counts": q_counts,
                    "training_performed": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    accepted_policy = StaticPolicy(
        policy_id="q3-cpu-reference-accepted-silu",
        default_activation="silu",
    )
    accepted_payload, accepted_status, accepted_elapsed = _load_or_validate(
        accepted,
        accepted_manifest,
        accepted_policy,
        run_name="q3-cpu-reference-accepted-silu",
    )
    accepted_metrics = _metrics(accepted_payload)
    summary["accepted_silu_cpu_baseline"] = {
        "status": accepted_status,
        "elapsed_seconds": accepted_elapsed,
        "metrics": accepted_metrics,
    }
    _write_outputs(summary, report_root)
    print(
        json.dumps(
            {
                "baseline": "accepted_silu",
                "status": accepted_status,
                "elapsed_seconds": accepted_elapsed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    qsilu_payload, qsilu_status, qsilu_elapsed = _load_or_validate(
        qsilu,
        qsilu_manifest,
        q_policy,
        run_name="q3-cpu-reference-uniform-qsilu",
    )
    qsilu_metrics = _metrics(qsilu_payload)
    summary["uniform_qsilu_cpu_baseline"] = {
        "status": qsilu_status,
        "elapsed_seconds": qsilu_elapsed,
        "metrics": qsilu_metrics,
        "global_deltas_vs_accepted_silu": _deltas(qsilu_metrics, accepted_metrics),
        "activation_counts": q_counts,
    }
    _write_outputs(summary, report_root)
    print(
        json.dumps(
            {
                "baseline": "uniform_qsilu",
                "status": qsilu_status,
                "elapsed_seconds": qsilu_elapsed,
                "global_worst_delta": min(
                    _deltas(qsilu_metrics, accepted_metrics).values()
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    profile_errors = _profile_weighted_errors(qsilu_manifest, candidates)
    metadata = _region_metadata()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for region in regions:
            policy = StaticPolicy(
                policy_id=f"q3--base-qsilu--{region}-{candidate}",
                default_activation="qsilu_pq",
                region_assignments=((region, candidate),),
            )
            counts = _audit_all_nonsilu(qsilu_manifest, policy)
            payload, status, elapsed = _load_or_validate(
                qsilu,
                qsilu_manifest,
                policy,
                run_name=f"q3-cpu-qsilu-{candidate}-{region}",
            )
            row = _row(
                candidate=candidate,
                region=region,
                payload=payload,
                status=status,
                elapsed_seconds=elapsed,
                activation_counts=counts,
                local_baseline=qsilu_metrics,
                global_baseline=accepted_metrics,
                profile_error=profile_errors[(candidate, region)],
                region_metadata=metadata[region],
            )
            rows.append(row)
            summary["rows"] = rows
            _write_outputs(summary, report_root)
            print(
                json.dumps(
                    {
                        "candidate": candidate,
                        "region": region,
                        "status": status,
                        "elapsed_seconds": elapsed,
                        "local_worst_delta": row["local_worst_delta"],
                        "global_worst_delta": row["global_worst_delta"],
                        "global_gate_passed": row["global_gate_passed"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = len(candidates) * len(regions)
    if len(rows) != expected:
        raise RuntimeError(f"Q3 matrix incomplete: {len(rows)} != {expected}")
    print(f"summary={report_root / 'summary.json'}", flush=True)
    print(f"report={report_root / 'README.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
