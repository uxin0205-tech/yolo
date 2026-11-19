#!/usr/bin/env python3
"""產生 Activation 最終報告的精度、函數誤差與運算量圖表。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activation_lab.activations import (
    ActivationName,
    build_activation,
)
from activation_lab.hardware import hardware_proxy

RESULTS_PATH = PROJECT_ROOT / "reports/full35-activation-results.json"
PROFILE_PATH = (
    PROJECT_ROOT
    / "artifacts/runs/full35/profiling/profile-full-train-seed1/decision-support.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/figures"

ACTIVATIONS: tuple[ActivationName, ...] = (
    "hardswish",
    "qsilu_pq",
    "poly_quality",
    "poly_shift",
)
CURVE_ACTIVATIONS: tuple[ActivationName, ...] = (
    "silu",
    *ACTIVATIONS,
)
DISPLAY_NAMES: dict[str, str] = {
    "silu": "SiLU",
    "hardswish": "Hardswish",
    "qsilu_pq": "qSiLU-PQ",
    "poly_quality": "SIPA Quality",
    "poly_shift": "SIPA Shift",
}
COLORS: dict[str, str] = {
    "silu": "#111827",
    "hardswish": "#DC2626",
    "qsilu_pq": "#2563EB",
    "poly_quality": "#D97706",
    "poly_shift": "#059669",
}
SELECTOR_METRICS: tuple[str, ...] = (
    "coco/box/map50_95",
    "coco/person/box/map50_95",
    "bbat/box/map50_95",
    "bbat/pose/map50_95",
    "bbat/ball/box/map50_95",
    "bbat/bat/box/map50_95",
    "bbat/ball/pose/map50_95",
    "bbat/bat/pose/map50_95",
)
METRIC_LABELS: tuple[str, ...] = (
    "COCO Δ",
    "Person Δ",
    "BBAT5 box Δ",
    "BBAT5 pose Δ",
)
SUMMARY_METRICS = SELECTOR_METRICS[:4]

# 由完整 train profiling 的 deployment_activation_elements / dataset_images 得到。
# 若本機仍有原始 profiling artifact，生成時會再核對一次。
DEPLOYMENT_ELEMENTS_PER_IMAGE = {
    "detect": 71_577_600,
    "pose": 72_652_800,
}
FUSED_CONSTANT_PRODUCTS = {
    "silu": 0,
    "hardswish": 1,
    "qsilu_pq": 2,
    "poly_quality": 6,
    "poly_shift": 6,
}
FUSED_ADD_SUBTRACT_UPPER_BOUND: dict[str, int] = {
    "silu": 0,
    "hardswish": 1,
    # qSiLU 最壞區段含兩個 dyadic coefficient network 與結果累加。
    "qsilu_pq": 8,
    "poly_quality": 5,
    "poly_shift": 5,
}
CONSTANT_CLASS = {
    "silu": "無",
    "hardswish": "一般常數（未融合時的 1/6）",
    "qsilu_pq": "dyadic／APoT",
    "poly_quality": "一般常數",
    "poly_shift": "dyadic／APoT",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK TC", "Droid Sans Fallback"],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.titlesize": 15,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "svg.hashsalt": "activation-report-2026-08-31",
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="產生不含 ReLU 的 Activation 精度、中央誤差與運算量圖表"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _load_results() -> dict[str, Any]:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def _accuracy_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = results["baseline"]["metrics"]
    rows: list[dict[str, Any]] = []

    zero_shot = {item["activation"]: item for item in results["uniform_zero_shot"]}
    recovery = {
        item["id"].removeprefix("short-recovery-uniform-").replace("-", "_"): item
        for item in results["static_queue"]["completed_results"]
        if item["id"].startswith("short-recovery-uniform-")
    }
    for stage, items in (("zero_shot", zero_shot), ("short_recovery", recovery)):
        for activation in ACTIVATIONS:
            item = items[activation]
            values = item.get("selector_metrics", item.get("metrics"))
            deltas = item.get("selector_deltas", item.get("deltas"))
            for metric in SELECTOR_METRICS:
                rows.append(
                    {
                        "stage": stage,
                        "activation": activation,
                        "metric": metric,
                        "baseline": float(baseline[metric]),
                        "value": float(values[metric]),
                        "delta": float(deltas[metric]),
                        "passed": bool(item["passed"]),
                        "worst_delta": float(item["worst_delta"]),
                    }
                )
    return rows


def _write_accuracy_csv(rows: list[dict[str, Any]], output_dir: Path) -> None:
    path = output_dir / "activation-accuracy-summary.csv"
    summary_rows = [row for row in rows if row["metric"] in SUMMARY_METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def _plot_accuracy(rows: list[dict[str, Any]], output_dir: Path) -> None:
    by_key = {(row["stage"], row["activation"], row["metric"]): row for row in rows}
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    full_activations: tuple[ActivationName, ...] = ("silu", *ACTIVATIONS)
    zoom_activations: tuple[ActivationName, ...] = (
        "silu",
        "qsilu_pq",
        "poly_quality",
        "poly_shift",
    )
    metric_colors = ("#2563EB", "#7C3AED", "#D97706", "#059669")
    stages = (
        ("zero_shot", "Zero-shot", (-19.0, 1.2), (-2.5, 0.6)),
        ("short_recovery", "10-epoch short recovery", (-3.5, 1.0), (-2.2, 0.8)),
    )

    for row_index, (stage, title, full_limits, zoom_limits) in enumerate(stages):
        for column_index, (activation_names, limits, suffix) in enumerate(
            (
                (full_activations, full_limits, "全候選"),
                (zoom_activations, zoom_limits, "近基準候選放大"),
            )
        ):
            axis = axes[row_index, column_index]
            positions = np.arange(len(activation_names))
            width = 0.18
            for metric_index, (metric, label, color) in enumerate(
                zip(SUMMARY_METRICS, METRIC_LABELS, metric_colors, strict=True)
            ):
                values = [
                    0.0
                    if activation == "silu"
                    else 100.0 * by_key[(stage, activation, metric)]["delta"]
                    for activation in activation_names
                ]
                offset = (metric_index - 1.5) * width
                bars = axis.bar(
                    positions + offset,
                    values,
                    width=width,
                    label=label,
                    color=color,
                )
                if column_index == 1:
                    axis.bar_label(
                        bars,
                        labels=[f"{value:+.2f}" for value in values],
                        padding=2,
                        fontsize=8,
                        rotation=90,
                    )
            axis.axhline(0.0, color="#111827", linewidth=1)
            axis.axhline(
                -1.5,
                color="#DC2626",
                linewidth=1.3,
                linestyle="--",
                label="-1.50 pp gate",
            )
            axis.set_ylim(*limits)
            axis.set_xticks(
                positions,
                [DISPLAY_NAMES[activation] for activation in activation_names],
            )
            axis.set_ylabel("相對 accepted SiLU 的 ΔmAP50-95（百分點）")
            axis.set_title(f"{title}｜{suffix}")
            axis.grid(axis="y", alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.93)
    )
    figure.suptitle(
        "Full35 Activation 四項 ΔmAP50-95 分組長條圖\n"
        "Accepted SiLU 明列為 Δ=0；ReLU 已排除；正式 gate 仍稽核八項",
        fontsize=19,
        fontweight="bold",
    )
    _save_figure(figure, output_dir, "activation-accuracy-overview")


def _evaluate_curves(x: np.ndarray) -> dict[str, np.ndarray]:
    tensor = torch.from_numpy(x)
    values: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name in CURVE_ACTIVATIONS:
            output = build_activation(name).to(dtype=torch.float64)(tensor)
            values[name] = output.cpu().numpy()
    return values


def _curve_metrics() -> list[dict[str, Any]]:
    x = np.linspace(-8.0, 8.0, 400_001, dtype=np.float64)
    values = _evaluate_curves(x)
    target = values["silu"]
    rows: list[dict[str, Any]] = []
    for name in CURVE_ACTIVATIONS:
        error = values[name] - target
        rows.append(
            {
                "activation": name,
                "grid_min": -8.0,
                "grid_max": 8.0,
                "samples": len(x),
                "mse_vs_silu": float(np.mean(np.square(error))),
                "mae_vs_silu": float(np.mean(np.abs(error))),
                "rmse_vs_silu": float(np.sqrt(np.mean(np.square(error)))),
                "max_abs_error_vs_silu": float(np.max(np.abs(error))),
            }
        )
    return rows


def _write_curve_metrics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    path = output_dir / "activation-function-metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_central_error(output_dir: Path) -> None:
    x = np.linspace(-4.0, 4.0, 40_001, dtype=np.float64)
    values = _evaluate_curves(x)
    target = values["silu"]

    figure = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
    curve_axis = figure.add_subplot(grid[0, :])
    all_error_axis = figure.add_subplot(grid[1, 0])
    zoom_axis = figure.add_subplot(grid[1, 1])

    for name in CURVE_ACTIVATIONS:
        width = 3.0 if name == "silu" else 2.0
        style = "--" if name == "hardswish" else "-"
        curve_axis.plot(
            x,
            values[name],
            label=DISPLAY_NAMES[name],
            color=COLORS[name],
            linewidth=width,
            linestyle=style,
        )
    curve_axis.set_title("中央區間函數曲線：負谷、原點與 1／2／4 節點")
    curve_axis.set_xlabel("輸入 x")
    curve_axis.set_ylabel("Activation 輸出")
    curve_axis.grid(alpha=0.22)
    curve_axis.legend(ncol=5, loc="upper left", fontsize=11)

    for name in ACTIVATIONS:
        error = values[name] - target
        style = "--" if name == "hardswish" else "-"
        all_error_axis.plot(
            x,
            error,
            label=DISPLAY_NAMES[name],
            color=COLORS[name],
            linewidth=2.1,
            linestyle=style,
        )
    all_error_axis.axhline(0.0, color="#6B7280", linewidth=1)
    all_error_axis.set_title("所有保留候選：signed error")
    all_error_axis.set_xlabel("輸入 x")
    all_error_axis.set_ylabel("候選輸出 − SiLU 輸出")
    all_error_axis.grid(alpha=0.22)
    all_error_axis.legend(fontsize=10)

    proposed = ("qsilu_pq", "poly_quality", "poly_shift")
    for name in proposed:
        error = values[name] - target
        zoom_axis.plot(
            x,
            error,
            label=(f"{DISPLAY_NAMES[name]}  max |e|={np.max(np.abs(error)):.5f}"),
            color=COLORS[name],
            linewidth=2.5,
        )
    for knot in (-4, -2, -1, 1, 2, 4):
        zoom_axis.axvline(knot, color="#9CA3AF", linewidth=0.7, alpha=0.35)
    zoom_axis.axhline(0.0, color="#6B7280", linewidth=1)
    zoom_axis.set_ylim(-0.03, 0.03)
    zoom_axis.set_title("近似候選獨立尺度放大：移除 Hardswish 尺度壓縮")
    zoom_axis.set_xlabel("輸入 x")
    zoom_axis.set_ylabel("候選輸出 − SiLU 輸出")
    zoom_axis.grid(alpha=0.22)
    zoom_axis.legend(fontsize=10, loc="lower left")

    figure.suptitle(
        "Activation 中央函數與誤差放大圖（x ∈ [-4, 4]；不含 ReLU）",
        fontsize=19,
        fontweight="bold",
    )
    _save_figure(figure, output_dir, "activation-central-error-enlarged")


def _verify_deployment_elements() -> None:
    if not PROFILE_PATH.exists():
        return
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    for task, expected in DEPLOYMENT_ELEMENTS_PER_IMAGE.items():
        document = profile["tasks"][task]
        actual = document["deployment_activation_elements"] / document["dataset_images"]
        if actual != expected:
            raise ValueError(
                f"{task} deployment activation elements changed: {actual} != {expected}"
            )


def _operation_rows() -> list[dict[str, Any]]:
    _verify_deployment_elements()
    rows: list[dict[str, Any]] = []
    for name in CURVE_ACTIVATIONS:
        proxy = hardware_proxy(cast(ActivationName, name))
        row: dict[str, Any] = {
            "activation": name,
            **asdict(proxy),
            "constant_class": CONSTANT_CLASS[name],
            "fused_selected_constant_products": FUSED_CONSTANT_PRODUCTS[name],
            "fused_add_subtract_upper_bound": FUSED_ADD_SUBTRACT_UPPER_BOUND[name],
        }
        for task, elements in DEPLOYMENT_ELEMENTS_PER_IMAGE.items():
            row[f"{task}_deployment_elements_per_image"] = elements
            row[f"{task}_variable_multiplications_per_image"] = (
                elements * proxy.variable_multiplications
            )
            row[f"{task}_transcendentals_per_image"] = (
                elements * proxy.transcendental_operations
            )
            row[f"{task}_fused_constant_products_per_image"] = (
                elements * FUSED_CONSTANT_PRODUCTS[name]
            )
        rows.append(row)
    return rows


def _write_operation_csv(rows: list[dict[str, Any]], output_dir: Path) -> None:
    path = output_dir / "activation-operation-estimate.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_operations(rows: list[dict[str, Any]], output_dir: Path) -> None:
    by_name = {row["activation"]: row for row in rows}
    names = [DISPLAY_NAMES[name] for name in CURVE_ACTIVATIONS]
    positions = np.arange(len(names))

    figure, axes = plt.subplots(2, 1, figsize=(16, 11), constrained_layout=True)
    operation_names = (
        ("variable_multiplications", "資料乘法／平方", "#2563EB"),
        ("constant_multiplications", "reference 常數乘法", "#D97706"),
        ("add_subtract", "加／減", "#059669"),
        ("range_operations", "abs／比較／clip", "#7C3AED"),
        ("transcendental_operations", "sigmoid 類", "#DC2626"),
    )
    bar_width = 0.15
    for index, (field, label, color) in enumerate(operation_names):
        offset = (index - 2) * bar_width
        values = [by_name[name][field] for name in CURVE_ACTIVATIONS]
        axes[0].bar(
            positions + offset, values, width=bar_width, label=label, color=color
        )
    axes[0].set_xticks(positions, names)
    axes[0].set_ylabel("每個 activation element 的靜態 proxy 次數")
    axes[0].set_title("Reference 實作的操作向量（不同操作不可直接相加成 latency）")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(ncol=3, fontsize=10)
    axes[0].annotate(
        "qSiLU 目前 eager reference 先算四個分支：8 個常數乘積。\n融合 coefficient-select 後估計只選 2 個 dyadic 乘積。",
        xy=(2, by_name["qsilu_pq"]["constant_multiplications"]),
        xytext=(3.05, 8.8),
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#F9FAFB", "ec": "#9CA3AF"},
    )

    width = 0.34
    detect = [
        by_name[name]["detect_variable_multiplications_per_image"] / 1e6
        for name in CURVE_ACTIVATIONS
    ]
    pose = [
        by_name[name]["pose_variable_multiplications_per_image"] / 1e6
        for name in CURVE_ACTIVATIONS
    ]
    axes[1].bar(
        positions - width / 2, detect, width, label="COCO Detect graph", color="#2563EB"
    )
    axes[1].bar(
        positions + width / 2, pose, width, label="BBAT5 Pose graph", color="#D97706"
    )
    axes[1].set_xticks(positions, names)
    axes[1].set_ylabel("資料乘法／平方估計（百萬次／640×640 image）")
    axes[1].set_title(
        "模型級投影：Detect 71.5776M、Pose 72.6528M deployment activation elements"
    )
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend()
    for index, name in enumerate(CURVE_ACTIVATIONS):
        top = max(detect[index], pose[index])
        note = ""
        if name == "silu":
            note = "另有同數量 sigmoid"
        elif name == "qsilu_pq":
            note = "另有 abs+4 compares"
        elif name == "hardswish":
            note = "另有 clip+scale"
        axes[1].text(index, top + 7, note, ha="center", fontsize=9, color="#374151")

    figure.suptitle(
        "Activation 運算量解析估計（不含 ReLU；不是 latency／功耗實測）",
        fontsize=19,
        fontweight="bold",
    )
    _save_figure(figure, output_dir, "activation-operation-estimate")


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    figure.savefig(
        output_dir / f"{stem}.svg",
        bbox_inches="tight",
        metadata={"Date": "2026-08-31"},
    )
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    results = _load_results()
    accuracy_rows = _accuracy_rows(results)
    curve_rows = _curve_metrics()
    operation_rows = _operation_rows()

    _write_accuracy_csv(accuracy_rows, args.output_dir)
    _write_curve_metrics(curve_rows, args.output_dir)
    _write_operation_csv(operation_rows, args.output_dir)
    _plot_accuracy(accuracy_rows, args.output_dir)
    _plot_central_error(args.output_dir)
    _plot_operations(operation_rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
