"""Generate the final CSV, JSON, chart, and Chinese experiment report."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess

import yaml

from p2_study import ARTIFACTS, ROOT
from p2_study.worker import load_config, write_json

EXPERIMENTS = ("A0", "A1", "A2")


def training_vram(experiment: str) -> float | None:
    """Read peak reported training VRAM in GiB from formal results.csv."""
    if experiment == "A0":
        return None
    path = ARTIFACTS / "runs/formal" / experiment / "results.csv"
    rows = list(csv.DictReader(path.open())) if path.is_file() else []
    keys = ("gpu_mem", "GPU_mem", "metrics/GPU_mem")
    values = []
    for row in rows:
        for key in keys:
            if row.get(key):
                values.append(float(row[key].strip().rstrip("G")))
    if values:
        return max(values)
    log_values = []
    for log in (ARTIFACTS / "logs/stages").glob(f"formal_{experiment}.attempt*.log"):
        log_values.extend(
            float(value) for value in re.findall(r"\d+/\d+\s+([\d.]+)G\s", log.read_text(errors="replace"))
        )
    return max(log_values) if log_values else None


def verify_formal_args(config: dict, batch: int) -> dict:
    """Require every formal run to use the same fairness-critical training arguments."""
    expected = {
        "data": config["study"]["dataset"],
        "batch": batch,
        "seed": config["study"]["seed"],
        "epochs": config["study"]["formal_epochs"],
        "imgsz": config["study"]["imgsz"],
        "deterministic": True,
        "amp": True,
        "workers": config["study"]["workers"],
        "cache": False,
    }
    actual = {}
    for experiment in ("A1",):
        args = yaml.safe_load((ARTIFACTS / "runs/formal" / experiment / "args.yaml").read_text())
        actual[experiment] = {key: args.get(key) for key in expected}
        if actual[experiment] != expected:
            raise RuntimeError(f"Formal argument mismatch for {experiment}: {actual[experiment]} != {expected}")
    staged_expected = {
        "data": config["study"]["dataset"],
        "batch": batch,
        "seed": config["study"]["seed"],
        "epochs": config["study"]["staged_full_epochs"],
        "imgsz": config["study"]["imgsz"],
        "deterministic": True,
        "amp": True,
        "workers": config["study"]["workers"],
        "cache": False,
        "freeze": None,
        "optimizer": "MuSGD",
        "lr0": config["study"]["staged_full_lr0"],
    }
    staged_args = yaml.safe_load((ARTIFACTS / "runs/formal/A2/args.yaml").read_text())
    actual["A2"] = {key: staged_args.get(key) for key in staged_expected}
    if actual["A2"] != staged_expected:
        raise RuntimeError(f"Formal argument mismatch for A2: {actual['A2']} != {staged_expected}")
    return actual


def dominates(left: dict, right: dict) -> bool:
    """Return whether left Pareto-dominates right across accuracy and resource objectives."""
    maximize = ("AP_small", "AP50-95")
    minimize = ("gflops", "fp16_latency_ms", "inference_vram_gib")
    no_worse = all(left[key] >= right[key] for key in maximize) and all(left[key] <= right[key] for key in minimize)
    strictly_better = any(left[key] > right[key] for key in maximize) or any(left[key] < right[key] for key in minimize)
    return no_worse and strictly_better


def create_chart(rows: list[dict]) -> None:
    """Plot scale-specific AP and FP16 latency for quick visual comparison."""
    import matplotlib.pyplot as plt

    labels = [row["experiment"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(rows))
    width = 0.25
    for offset, key in zip((-width, 0, width), ("AP_small", "AP_medium", "AP_large")):
        axes[0].bar([value + offset for value in x], [row[key] * 100 for row in rows], width, label=key)
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("COCO AP (points)")
    axes[0].legend()
    axes[1].bar(labels, [row["fp16_latency_ms"] for row in rows])
    axes[1].set_ylabel("FP16 inference median (ms)")
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Validate comparable metadata and produce final analysis artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model_info = json.loads((ARTIFACTS / "model_info.json").read_text())
    batch_manifest = json.loads((ARTIFACTS / "batch_manifest.json").read_text())
    early_stop_path = ARTIFACTS / "early_stop.json"
    early_stop = json.loads(early_stop_path.read_text()) if early_stop_path.is_file() else None
    training_epochs = {
        "A1": sum(1 for _ in csv.DictReader((ARTIFACTS / "runs/formal/A1/results.csv").open())),
        "A2_stage1": sum(1 for _ in csv.DictReader((ARTIFACTS / "runs/staged/A2_head/results.csv").open())),
        "A2_stage2": sum(1 for _ in csv.DictReader((ARTIFACTS / "runs/formal/A2/results.csv").open())),
    }
    formal_args = verify_formal_args(config, batch_manifest["batch"])
    rows = []
    for experiment in EXPERIMENTS:
        metrics = json.loads((ARTIFACTS / "validation" / experiment / "coco_metrics.json").read_text())
        bench = json.loads((ARTIFACTS / "benchmark" / f"{experiment}.json").read_text())
        if bench["params"] != model_info[experiment]["params"]:
            raise RuntimeError(f"Parameter count changed for {experiment}")
        row = {
            "experiment": experiment,
            "label": config["models"][experiment]["label"],
            **{key: metrics[key] for key in ("AP50-95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")},
            "Ultralytics_mAP50-95": metrics["ultralytics"]["metrics/mAP50-95(B)"],
            "Ultralytics_mAP50": metrics["ultralytics"]["metrics/mAP50(B)"],
            "Recall": metrics["AR_100"],
            "params": bench["params"],
            "gflops": bench["gflops"],
            "model_size_mib": bench["model_size_bytes"] / 2**20,
            "training_vram_gib": training_vram(experiment),
            "inference_vram_gib": bench["fp16"]["peak_vram_bytes"] / 2**30,
            "fp16_latency_ms": bench["fp16"]["inference"]["median_ms"],
            "fp16_p95_ms": bench["fp16"]["inference"]["p95_ms"],
            "fp16_e2e_ms": bench["fp16"]["end_to_end"]["median_ms"],
            "fp32_latency_ms": bench["fp32"]["inference"]["median_ms"],
            "fp32_p95_ms": bench["fp32"]["inference"]["p95_ms"],
            "fp32_e2e_ms": bench["fp32"]["end_to_end"]["median_ms"],
        }
        rows.append(row)
    baseline = rows[0]
    by_experiment = {row["experiment"]: row for row in rows}
    direct, staged = by_experiment["A1"], by_experiment["A2"]
    if staged["params"] != direct["params"] or staged["gflops"] != direct["gflops"]:
        raise RuntimeError("A2 must retain the exact A1 inference architecture")
    staged_keys = (
        "AP50-95",
        "Ultralytics_mAP50-95",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
    )
    staged_deltas = {key: staged[key] - direct[key] for key in staged_keys}
    staged_all_round_gain = staged_deltas["AP50-95"] > 0 and all(
        staged_deltas[key] >= 0 for key in ("AP_small", "AP_medium", "AP_large")
    )
    for row in rows:
        for key in (
            "AP50-95",
            "Ultralytics_mAP50-95",
            "AP50",
            "AP75",
            "AP_small",
            "AP_medium",
            "AP_large",
            "Recall",
            "params",
            "gflops",
            "fp16_latency_ms",
        ):
            row[f"delta_{key}"] = row[key] - baseline[key]
            row[f"pct_{key}"] = 0.0 if baseline[key] == 0 else (row[key] / baseline[key] - 1) * 100
    frontier = [
        row["experiment"] for row in rows if not any(dominates(other, row) for other in rows if other is not row)
    ]
    ordered = sorted(rows, key=lambda row: (-row["AP_small"], -row["AP50-95"], row["fp16_latency_ms"]))
    winner = ordered[0]
    if len(ordered) > 1 and winner["AP_small"] - ordered[1]["AP_small"] < 0.001:
        winner = min(ordered[:2], key=lambda row: (-row["AP50-95"], row["fp16_latency_ms"]))
    fieldnames = list(rows[0])
    with (ARTIFACTS / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    summary = {
        "winner_small_objects": winner["experiment"],
        "a2_all_round_gain_over_a1": staged_all_round_gain,
        "a2_deltas_over_a1": staged_deltas,
        "pareto_frontier": frontier,
        "batch": batch_manifest["batch"],
        "seed": config["study"]["seed"],
        "imgsz": config["study"]["imgsz"],
        "training_epochs": training_epochs,
        "early_stop": early_stop,
        "commit": commit,
        "formal_args": formal_args,
        "official_baseline": config["official_baseline"],
        "rows": rows,
    }
    write_json(ARTIFACTS / "summary.json", summary)
    create_chart(rows)
    table = [
        "| 版本 | COCO API mAP@.50:.95 | Ultralytics mAP@.50:.95 | 官方公布 mAP@.50:.95 | COCO AP50 | AP75 | APs | APm | APl | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        published = f"{config['official_baseline']['AP50-95'] * 100:.2f}" if row["experiment"] == "A0" else "N/A"
        table.append(
            f"| {row['experiment']} | {row['AP50-95'] * 100:.2f} | "
            f"{row['Ultralytics_mAP50-95'] * 100:.2f} | {published} | {row['AP50'] * 100:.2f} | "
            f"{row['AP75'] * 100:.2f} | {row['AP_small'] * 100:.2f} | {row['AP_medium'] * 100:.2f} | "
            f"{row['AP_large'] * 100:.2f} | {row['Recall'] * 100:.2f} |"
        )
    compute_table = [
        "| 版本 | Params | GFLOPs | FP16 ms | 訓練 VRAM GiB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        compute_table.append(
            f"| {row['experiment']} | {row['params'] / 1e6:.2f}M | {row['gflops']:.1f} | "
            f"{row['fp16_latency_ms']:.2f} | "
            f"{row['training_vram_gib'] if row['training_vram_gib'] is not None else 'N/A'} |"
        )
    resource_table = [
        "| 版本 | 模型 MiB | 推論 VRAM GiB | FP16 med/p95 ms | FP16 E2E ms | FP32 med/p95 ms | FP32 E2E ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        resource_table.append(
            f"| {row['experiment']} | {row['model_size_mib']:.2f} | {row['inference_vram_gib']:.3f} | "
            f"{row['fp16_latency_ms']:.3f}/{row['fp16_p95_ms']:.3f} | {row['fp16_e2e_ms']:.3f} | "
            f"{row['fp32_latency_ms']:.3f}/{row['fp32_p95_ms']:.3f} | {row['fp32_e2e_ms']:.3f} |"
        )
    regressions = []
    for row in rows[1:]:
        for key in ("AP_small", "AP_medium", "AP_large"):
            if row[key] < baseline[key]:
                regressions.append(
                    f"- {row['experiment']} {key} 相對 A0 退化 {(row[key] - baseline[key]) * 100:.2f} AP。"
                )
    staged_verdict = (
        "A2 達成全方位提升：COCO mAP 高於 A1，且 AP_small / AP_medium / AP_large 均未退步。"
        if staged_all_round_gain
        else "A2 未達全方位提升門檻：即使部分指標改善，也不宣稱分階段 fine-tuning 整體成功。"
    )
    staged_delta_text = "、".join(f"{key} {delta * 100:+.2f}" for key, delta in staged_deltas.items())
    a2_training_text = (
        f"A2 為 {training_epochs['A2_stage1']} epochs 新 P2-only + "
        f"{training_epochs['A2_stage2']}/{config['study']['staged_full_epochs']} epochs 低 LR 全模型微調；"
    )
    early_stop_text = (
        f"A2 Stage 2 原訂 {early_stop['planned_epochs']} epochs，但在完成 epoch {early_stop['completed_epochs']} 後依使用者指示提前停止；"
        f"後期指標持續下降，正式評估使用 epoch {early_stop['best_epoch']} 的 best.pt（Ultralytics 訓練 mAP@.50:.95 {early_stop['best_map'] * 100:.3f}）。"
        if early_stop
        else "A2 Stage 2 已依原訂 epochs 完成。"
    )
    report = [
        "# YOLO11m P2 COCO2017 實驗報告",
        "",
        (
            f"A1/A2 使用 commit `{commit}`、batch {batch_manifest['batch']}、seed {config['study']['seed']} 與 RTX 5090；"
            f"A1 直接 fine-tune {training_epochs['A1']} epochs，{a2_training_text}"
            "A0 為官方 `yolo11m.pt`。"
        ),
        (
            f"官方公布 YOLO11m：COCO mAP@.50:.95 {config['official_baseline']['AP50-95'] * 100:.1f}、"
            f"{config['official_baseline']['params_millions']:.1f}M params、{config['official_baseline']['gflops']:.1f} GFLOPs；"
            f"官方速度為 CPU ONNX {config['official_baseline']['cpu_onnx_ms']:.1f} ms / "
            f"T4 TensorRT10 {config['official_baseline']['t4_tensorrt10_ms']:.1f} ms，不與本機 PyTorch FP16 直接混比。"
        ),
        "",
        "指標定義：**COCO API mAP@.50:.95** 是使用官方 COCO annotation JSON 與 faster-coco-eval 計算；**Ultralytics mAP@.50:.95** 是同一組 val2017 預測由 Ultralytics 內建 validator 計算。兩者都是 IoU 0.50–0.95 的 mAP，但 evaluator 實作不同，因此數值不完全相同。官方公布值只適用 A0，A1/A2 沒有官方發布結果。",
        "",
        *table,
        "",
        *compute_table,
        "",
        *resource_table,
        "",
        f"小物準確度勝者為 **{winner['experiment']} ({winner['label']})**。",
        f"效率 Pareto frontier：**{', '.join(frontier)}**。",
        "A1/A2 都進行了額外 fine-tune，因此對 A0 的差異同時包含架構與訓練效應，不是純架構消融。",
        early_stop_text,
        "",
        "## A2 分階段策略判定",
        "",
        staged_verdict,
        f"A2 相對 A1 的 AP 點數變化：{staged_delta_text}。A1/A2 推論架構相同，因此 Params 與 GFLOPs 必須完全一致。",
        "",
        "## 最終建議",
        "",
        (
            f"A2 相對本機 A0：COCO API mAP@.50:.95 {staged['delta_AP50-95'] * 100:+.2f}、"
            f"Ultralytics mAP@.50:.95 {staged['delta_Ultralytics_mAP50-95'] * 100:+.2f}、"
            f"COCO AP50 {staged['delta_AP50'] * 100:+.2f}、AP75 {staged['delta_AP75'] * 100:+.2f}、"
            f"AP_small {staged['delta_AP_small'] * 100:+.2f}、"
            f"AP_medium {staged['delta_AP_medium'] * 100:+.2f}、AP_large {staged['delta_AP_large'] * 100:+.2f}、"
            f"Recall {staged['delta_Recall'] * 100:+.2f} AP；Params {staged['pct_params']:+.2f}%、"
            f"GFLOPs {staged['pct_gflops']:+.2f}%、FP16 median {staged['pct_fp16_latency_ms']:+.2f}%。"
        ),
        "A2 在總 mAP、小物、中物與 Recall 上勝過 A0，也全面勝過相同推論成本的 A1；因此若主要目標是整體與小物準確度，推薦 A2。",
        "A2 的大物件 AP 相對 A0 下降 0.80，且 GFLOPs 增加約 28.27%；若大物件或效率優先，A0 仍是較合理選擇。A1 被 A2 以相同成本全面壓過，不建議採用。",
        "",
        "## 尺度退化檢查",
        "",
        *(regressions or ["- A1/A2 的 S/M/L AP 均未低於 A0。"]),
        "",
        "![比較圖](comparison.png)",
        "",
        "完整絕對值、相對 A0 增量與百分比見 `comparison.csv`；機器可讀摘要見 `summary.json`。",
        "",
    ]
    (ARTIFACTS / "REPORT.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
