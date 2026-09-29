#!/usr/bin/env python3
"""由交付包內保存的訓練與驗證證據建立最終研究報告。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _bundle import BUNDLE_ROOT, atomic_json, file_sha256, load_models

REPORT_ROOT = BUNDLE_ROOT / "reports"
TRAINING_ROOT = REPORT_ROOT / "training"
FIGURE_ROOT = REPORT_ROOT / "figures"

TRAINING_RUNS = (
    ("full35-b-f03", "full35", "b", 0.3, 16),
    ("full35-c-f03", "full35", "c", 0.3, 8),
    ("full35-b-f10", "full35", "b", 1.0, 16),
    ("full35-c-f10", "full35", "c", 1.0, 8),
    ("partial75-b-f03", "partial75", "b", 0.3, 16),
    ("partial75-c-f03", "partial75", "c", 0.3, 8),
    ("partial75-b-f10", "partial75", "b", 1.0, 16),
    ("partial75-c-f10", "partial75", "c", 1.0, 8),
)

PROFILE_FILES = {
    "a0": "a0-bittrue-inference-fp16-rtx5060ti-2026-08-24.json",
    "full35-a2": "full35-a2-bittrue-inference-fp16-rtx5060ti-2026-08-24.json",
    "partial75-a2": "partial75-a2-bittrue-inference-fp16-rtx5060ti-2026-08-24.json",
}


def read_json(path: Path) -> dict[str, Any]:
    """讀取 JSON object。"""

    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """讀取 telemetry JSONL，並容忍歷史檔案的 NUL padding。"""

    records = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.replace("\x00", "").strip()
        if line:
            records.append(json.loads(line))
    return records


def read_csv(path: Path) -> list[dict[str, float]]:
    """把 Ultralytics results.csv 轉為數值列。"""

    with path.open(encoding="utf-8", newline="") as handle:
        return [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def loss_snapshot(row: dict[str, float]) -> dict[str, float]:
    """擷取單一 epoch 的 train／val losses。"""

    return {
        key: row[key]
        for key in (
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
        )
    }


def summarize_training(
    model_id: str,
    architecture: str,
    stage: str,
    fraction: float,
    batch: int,
    ap_lookup: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """整理單次訓練、資源與 final Bit-True metric。"""

    root = TRAINING_ROOT / model_id
    manifest = read_json(root / "manifest.json")
    complete = read_json(root / "training-complete.json")
    epochs = read_csv(root / "results.csv")
    telemetry = read_jsonl(root / "resource-telemetry.jsonl")
    if not epochs or not telemetry:
        raise RuntimeError(f"{model_id} 缺少訓練或 telemetry 資料")
    if not all(math.isfinite(value) for row in epochs for value in row.values()):
        raise RuntimeError(f"{model_id} results.csv 含 NaN/Inf")
    best_row = max(epochs, key=lambda row: row["metrics/mAP50-95(B)"])
    final_row = epochs[-1]
    ap = ap_lookup[model_id]
    min_ram = min(record["ram_available_bytes"] for record in telemetry)
    min_free_vram = min(record["vram_free_bytes"] for record in telemetry)
    peak_allocated = max(record["vram_peak_allocated_bytes"] for record in telemetry)
    max_pss = max(record["process_pss_bytes"] for record in telemetry)
    settings = manifest["common"]
    resources = manifest["resource_safety"]
    record = {
        "model_id": model_id,
        "architecture": architecture,
        "stage": stage,
        "train_fraction": fraction,
        "status": complete["status"],
        "stop_reason": complete["stop_reason"],
        "requested_epochs": complete["requested_epochs"],
        "completed_epochs": len(epochs),
        "best_epoch": int(best_row["epoch"]),
        "best_validation_map50_95_during_training": best_row["metrics/mAP50-95(B)"],
        "best_fitness_recorded": complete["best_fitness"],
        "final_epoch_map50_95": final_row["metrics/mAP50-95(B)"],
        "final_bittrue_coco_map50_95": ap["coco_internal"]["ap50_95"],
        "gate": ap["gate"],
        "elapsed_seconds": complete["elapsed_seconds"],
        "mean_epoch_seconds": complete["elapsed_seconds"] / len(epochs),
        "physical_batch": batch,
        "nbs": settings["nbs"],
        "effective_batch": settings["nbs"],
        "workers": settings["workers"],
        "in_training_validation_workers": 0,
        "amp": settings["amp"],
        "patience": complete["patience"],
        "seed": settings["seed"],
        "optimizer": settings["optimizer"],
        "imgsz": settings["imgsz"],
        "alpha": read_json(REPORT_ROOT / "raw" / model_id / "coco-internal.json").get("alpha"),
        "contains_nan_or_inf": False,
        "loss_at_best_epoch": loss_snapshot(best_row),
        "loss_at_final_epoch": loss_snapshot(final_row),
        "resource": {
            "minimum_available_ram_bytes": min_ram,
            "minimum_free_vram_bytes": min_free_vram,
            "maximum_process_pss_bytes": max_pss,
            "peak_vram_allocated_bytes": peak_allocated,
            "configured_ram_fail_floor_bytes": resources["minimum_available_ram_bytes"],
            "configured_vram_fail_floor_bytes": resources["minimum_free_vram_bytes"],
            "telemetry_samples": len(telemetry),
            "oom": False,
        },
        "partial75_channels": (
            {"context": 64, "bit_exact_bypass": 192} if architecture == "partial75" else None
        ),
        "evidence": {
            "training_plot": f"training/{model_id}/results.png",
            "results_csv": f"training/{model_id}/results.csv",
            "telemetry": f"training/{model_id}/resource-telemetry.jsonl",
            "manifest": f"training/{model_id}/manifest.json",
        },
    }
    return record, epochs


def write_training_csv(records: list[dict[str, Any]]) -> None:
    """輸出一列一訓練的摘要 CSV。"""

    rows = []
    for record in records:
        resource = record["resource"]
        rows.append(
            {
                key: record[key]
                for key in (
                    "model_id",
                    "architecture",
                    "stage",
                    "train_fraction",
                    "physical_batch",
                    "nbs",
                    "effective_batch",
                    "workers",
                    "patience",
                    "completed_epochs",
                    "best_epoch",
                    "best_validation_map50_95_during_training",
                    "final_epoch_map50_95",
                    "final_bittrue_coco_map50_95",
                    "gate",
                    "elapsed_seconds",
                    "mean_epoch_seconds",
                    "alpha",
                    "contains_nan_or_inf",
                )
            }
        )
        rows[-1].update(resource)
    path = REPORT_ROOT / "training-summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure: Any, stem: str) -> None:
    """輸出 PNG／SVG，並正規化 Matplotlib SVG 的行尾空白。"""

    for suffix in ("png", "svg"):
        path = FIGURE_ROOT / f"{stem}.{suffix}"
        figure.savefig(path, dpi=180)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")


def plot_training_curves(histories: dict[str, list[dict[str, float]]]) -> None:
    """比較每個架構、phase 與 fraction 的 mAP50-95 軌跡。"""

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for axis, architecture, stage in zip(
        axes.ravel(),
        ("full35", "full35", "partial75", "partial75"),
        ("b", "c", "b", "c"),
        strict=True,
    ):
        for fraction, color in ((0.3, "#e76f51"), (1.0, "#277da1")):
            model_id = f"{architecture}-{stage}-f{'03' if fraction == 0.3 else '10'}"
            rows = histories[model_id]
            axis.plot(
                [row["epoch"] for row in rows],
                [row["metrics/mAP50-95(B)"] for row in rows],
                marker="o",
                linewidth=1.8,
                label=f"fraction={fraction}",
                color=color,
            )
        axis.set_title(f"{architecture} Phase {stage.upper()}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation mAP50-95")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Full35 / Partial75 Training Curves")
    figure.tight_layout()
    save_figure(figure, "training-map-comparison")
    plt.close(figure)


def plot_bbt5(ap_models: list[dict[str, Any]]) -> None:
    """把使用者 BBT5 overall／ball／bat AP50-95 放在同一圖。"""

    labels = [item["id"] for item in ap_models]
    scopes = (
        ("Overall", lambda item: item["bbt5_internal"]["map50_95"]),
        ("Sports ball", lambda item: item["bbt5_internal"]["sports_ball"]["ap50_95"]),
        ("Baseball bat", lambda item: item["bbt5_internal"]["baseball_bat"]["ap50_95"]),
    )
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = ["#577590" if item["gate"] in ("immutable", "retained") else "#f9844a" for item in ap_models]
    for axis, (title, getter) in zip(axes, scopes, strict=True):
        values = [getter(item) for item in ap_models]
        axis.bar(labels, values, color=colors)
        axis.set_ylabel("AP50-95")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(max(0.0, min(values) - 0.025), max(values) + 0.025)
    axes[-1].tick_params(axis="x", rotation=40)
    figure.suptitle("BBT5 detect_dataset Bit-True AP50-95")
    figure.tight_layout()
    save_figure(figure, "bbt5-ap-comparison")
    plt.close(figure)


def plot_resources(records: list[dict[str, Any]]) -> None:
    """顯示每次訓練的 RAM 與 VRAM 實測界線。"""

    labels = [record["model_id"] for record in records]
    gib = 1 << 30
    min_ram = [record["resource"]["minimum_available_ram_bytes"] / gib for record in records]
    ram_floors = [record["resource"]["configured_ram_fail_floor_bytes"] / gib for record in records]
    peak_vram = [record["resource"]["peak_vram_allocated_bytes"] / gib for record in records]
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].bar(labels, min_ram, color="#43aa8b")
    axes[0].scatter(labels, ram_floors, color="#d62828", marker="D", label="Configured RAM fail floor")
    axes[0].set_ylabel("Minimum MemAvailable (GiB)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, peak_vram, color="#577590")
    axes[1].set_ylabel("Peak allocated VRAM (GiB)")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Training Resource Telemetry")
    figure.tight_layout()
    save_figure(figure, "training-resources")
    plt.close(figure)


def build_weight_index(
    models: list[dict[str, Any]], ap_lookup: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """建立單一權重入口與可過濾 registry。"""

    records = []
    for model in models:
        metrics = ap_lookup[model["id"]]
        if model["stage"] in ("b", "c"):
            manifest = read_json(TRAINING_ROOT / model["id"] / "manifest.json")
            common_settings = manifest["common"]
            phase = manifest["phase"]
            training_contract: dict[str, Any] = {
                "data": common_settings["data"],
                "imgsz": common_settings["imgsz"],
                "physical_batch": common_settings["batch"],
                "nbs": common_settings["nbs"],
                "effective_batch": common_settings["nbs"],
                "workers": common_settings["workers"],
                "amp": common_settings["amp"],
                "optimizer": common_settings["optimizer"],
                "requested_epochs": phase["epochs"],
                "patience": phase["patience"],
                "learning_rates": phase["learning_rates"],
                "weight_decay": phase["weight_decay"],
                "momentum": phase["momentum"],
                "manifest": f"reports/training/{model['id']}/manifest.json",
                "training_hardware": manifest["environment"]["gpu"],
            }
            float_source_uri = read_json(TRAINING_ROOT / model["id"] / "training-complete.json")["best_checkpoint"]
        elif model["stage"] == "a2":
            training_contract = {
                "scope": "從 commit 7f0bd61 匯入的 accepted A2；本工作區未重跑 A2",
                "data": "COCO2017",
                "imgsz": 640,
                "physical_batch": 16,
                "seed": 0,
                "training_hardware": "RTX 4080 SUPER（來源檔名契約）",
            }
            float_source_uri = (
                f"inputs/continuation/{model['architecture']}-accepted-a2/float-best.pt"
            )
        else:
            training_contract = {
                "scope": "immutable A0；本工作項目不訓練",
                "data": None,
                "imgsz": 640,
                "physical_batch": None,
                "seed": 0,
                "training_hardware": None,
            }
            float_source_uri = None
        raw_metric = read_json(REPORT_ROOT / "raw" / model["id"] / "coco-internal.json")
        common = {
            "model_id": model["id"],
            "architecture": model["architecture"],
            "stage": model["stage"],
            "train_fraction": model["train_fraction"],
            "seed": 0,
            "gate_status": model["gate"],
            "parent_id": None if model["stage"] in ("a0", "a2") else f"{model['architecture']}-a2",
            "training_contract": training_contract,
            "evaluation_contract": {
                "hardware": "NVIDIA GeForce RTX 5060 Ti",
                "software": "Python 3.12.13 / PyTorch 2.11.0+cu128 / Ultralytics 8.4.90",
                "imgsz": 640,
                "validation_batch": 8,
                "workers": 6,
                "selection_backend": "bit_true_pwl",
            },
            "metric_summary": {
                "coco_internal_map50_95": metrics["coco_internal"]["ap50_95"],
                "bbt5_overall_map50_95": metrics["bbt5_internal"]["map50_95"],
                "bbt5_ball_ap50_95": metrics["bbt5_internal"]["sports_ball"]["ap50_95"],
                "bbt5_bat_ap50_95": metrics["bbt5_internal"]["baseball_bat"]["ap50_95"],
            },
        }
        records.append(
            {
                **common,
                "id": f"{model['id']}:bittrue",
                "kind": "bittrue-evaluation",
                "status": "evaluation-candidate",
                "uri": model["bittrue"],
                "sha256": model["bittrue_sha256"],
                "state_dict_sha256": model["state_dict_sha256"],
                "source_uri": raw_metric["checkpoint"]["path"],
                "metric_source": f"reports/raw/{model['id']}/coco-internal.json",
            }
        )
        if model["float"] is not None:
            records.append(
                {
                    **common,
                    "id": f"{model['id']}:float-best",
                    "kind": "float-best",
                    "status": "resume-or-rematerialize",
                    "uri": model["float"],
                    "sha256": model["float_sha256"],
                    "state_dict_sha256": None,
                    "source_uri": float_source_uri,
                    "metric_source": None,
                }
            )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canonical_root": "weights/",
        "records": records,
        "runtime_last_policy": "last.pt 僅為中斷恢復產物，不是最終候選；未重複封裝，來源路徑記在 training-complete.json。",
    }
    atomic_json(BUNDLE_ROOT / "weights/index.json", payload)
    with (BUNDLE_ROOT / "weights/index.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "id",
            "model_id",
            "architecture",
            "stage",
            "train_fraction",
            "kind",
            "status",
            "gate_status",
            "parent_id",
            "uri",
            "sha256",
            "state_dict_sha256",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    readme = """# 權重索引

此資料夾是交付包的唯一正式權重入口。`bittrue/` 是所有 AP 排名實際使用的 Bit-True PWL checkpoint；`float/` 是可接續訓練或重新 materialize 的 best checkpoint。

請先查 `index.json`（完整 machine-readable metadata）或 `index.csv`（快速篩選）。每筆都含架構、phase、fraction、seed、gate 狀態、parent、用途、SHA256 與主要 COCO／BBT5 指標。

Ultralytics `last.pt` 只用於同一 run 的斷電恢復，不是可發布候選，因此未重複複製到本資料夾；其原始路徑與 resume 紀錄保存在 `../reports/training/*/training-complete.json`。
"""
    (BUNDLE_ROOT / "weights/README.md").write_text(readme, encoding="utf-8")
    return payload


def build_markdown(
    training: list[dict[str, Any]],
    ap_models: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> str:
    """建立最終訓練、資源與 selection 報告。"""

    ap_lookup = {item["id"]: item for item in ap_models}
    deltas = []
    for architecture in ("full35", "partial75"):
        parent = ap_lookup[f"{architecture}-a2"]["bbt5_internal"]
        child = ap_lookup[f"{architecture}-c-f10"]["bbt5_internal"]
        deltas.append(
            (
                architecture,
                child["map50_95"] - parent["map50_95"],
                child["sports_ball"]["ap50_95"] - parent["sports_ball"]["ap50_95"],
                child["baseball_bat"]["ap50_95"] - parent["baseball_bat"]["ap50_95"],
            )
        )
    lines = [
        "# RTX 5060 Ti Full35／Partial75 最終報告",
        "",
        "報告日期：2026-08-24（Asia/Taipei）",
        "",
        "## 最終結論",
        "",
        "- 完整資料 Phase C 已完成：Full35 10 epochs、Partial75 11 epochs，兩者皆由 patience 7 early stopping，沒有 OOM、NaN 或 Inf。",
        "- Full35 C-100% 的正式 Bit-True COCO mAP50-95 為 0.501395；Partial75 C-100% 為 0.500428，均低於各自 A2，因此 gate 都 rollback，accepted checkpoint 沒有改變。",
        "- Full35 A2 與 Partial75 A2 的差距小於 0.001；依既定規則用同機 FP16 p50 latency、GFLOPs、Params 打破平手，Partial75 是兩個 P3-MASF 架構中的正式工程 winner。",
        "- Partial75 A2 的 COCO mAP50-95 為 0.506754，與 A0 幾乎相同；A0 latency 仍較快。因此不能宣稱 P3-MASF 已帶來實質整體 accuracy 或端到端效率提升。",
        "- 使用者指定的 BBT5 `detect_dataset` 是最重要的局部指標：overall 與 bat 最高都是 Full35 C-30%（0.412511／0.515345），ball 最高仍是 A0（0.318082）。這些局部結果不能繞過 COCO overall gate。",
        "",
        "![BBT5 AP 比較](figures/bbt5-ap-comparison.png)",
        "",
        "## 正式訓練摘要",
        "",
        "| Run | Fraction | Batch / nbs | Epochs | Best epoch | Train best mAP | Final Bit-True COCO mAP | Mean epoch | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in training:
        lines.append(
            f"| {record['model_id']} | {record['train_fraction']:.1f} | {record['physical_batch']} / {record['nbs']} | "
            f"{record['completed_epochs']} | {record['best_epoch']} | "
            f"{record['best_validation_map50_95_during_training']:.6f} | "
            f"{record['final_bittrue_coco_map50_95']:.6f} | {record['mean_epoch_seconds'] / 60:.1f} min | "
            f"{record['gate']} |"
        )
    lines += [
        "",
        "![訓練 mAP 軌跡](figures/training-map-comparison.png)",
        "",
        "每個 run 原始 `results.csv`、Ultralytics `results.png`、F1／PR／P／R curves、confusion matrix、args、manifest 與 telemetry 均保存在 `training/<model-id>/`。Phase B 固定 physical batch 16；只有 Phase C 使用 physical batch 8、`nbs=16`，等效 batch 16。",
        "",
        "## 使用者 detect_dataset 關鍵指標",
        "",
        "資料契約：`/home/uxin0/yolo/original/pose/detect_dataset/coco80/data.yaml`；567 images、sports ball 301 instances、baseball bat 484 instances。",
        "",
        "### 完整資料 Phase C 相對各自 A2 的變化",
        "",
        "| 架構 | Overall AP Δ | Ball AP Δ | Bat AP Δ | 判讀 |",
        "|---|---:|---:|---:|---|",
    ]
    for architecture, overall_delta, ball_delta, bat_delta in deltas:
        lines.append(
            f"| {architecture} | {overall_delta:+.6f} | {ball_delta:+.6f} | {bat_delta:+.6f} | "
            "BBT5 bat 提升，但 ball 下降；COCO overall gate 仍 rollback。 |"
        )
    lines += [
        "",
        "| Model | Overall AP | Ball P | Ball R | Ball AP | Bat P | Bat R | Bat AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in ap_models:
        bbt5 = item["bbt5_internal"]
        ball = bbt5["sports_ball"]
        bat = bbt5["baseball_bat"]
        lines.append(
            f"| {item['id']} | {bbt5['map50_95']:.6f} | {ball['precision']:.6f} | "
            f"{ball['recall']:.6f} | {ball['ap50_95']:.6f} | {bat['precision']:.6f} | "
            f"{bat['recall']:.6f} | {bat['ap50_95']:.6f} |"
        )
    lines += [
        "",
        "AP50、AP75、F1、canonical COCO API 與每類完整數值見 `FULL35_PARTIAL75_AP.md`、`full35-partial75-ap.json` 與長格式 CSV。",
        "",
        "## 同機 FP16 效率",
        "",
        "固定 RTX 5060 Ti、imgsz 640、batch 1、FP16、20 warmup、100 iterations。",
        "",
        "| Model | Params | GFLOPs | P3-MASF MACs | p50 latency | Peak VRAM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id in ("a0", "full35-a2", "partial75-a2"):
        profile = profiles[model_id]
        lines.append(
            f"| {model_id} | {profile['parameters']:,} | {profile['gflops']:.3f} | "
            f"{profile['p3_masf_macs']:,} | {profile['latency_ms']['p50']:.3f} ms | "
            f"{profile['peak_vram_bytes'] / (1 << 20):.1f} MiB |"
        )
    lines += [
        "",
        "## RAM／VRAM 安全性",
        "",
        "八個正式 B/C run 都保存 batch-interval telemetry，且皆未發生 OOM。本次完整資料 Phase C 的 fail-closed RAM floor 是使用者指定的 0.5 GiB；較早的 runs 保留各自 1.5 GiB 契約。訓練期間每個 epoch 會執行 GC、CUDA cache 清理與 `malloc_trim`。",
        "",
        "![資源監控](figures/training-resources.png)",
        "",
        "## 限制與判讀邊界",
        "",
        "- 所有候選只有 seed 0，微小差距仍可能是單 seed 變異。",
        "- BBT5 valid 有 93/567 images（16.4%）的 COCO ID 出現在 COCO train2017；只適合候選間相對比較，不代表完全獨立泛化。",
        "- Ultralytics internal 與 canonical COCO API 是不同 evaluator，兩套數字分表保存，不混用排名。",
        "- fraction=0.3 與 fraction=1.0 是不同訓練資料契約；30% 結果不能當成完整資料訓練的絕對替代。",
        "- `EXPERIMENT_SPEC.md` 規劃的 winner LR tuning T1／T2／T3 尚未執行；本報告宣告的是 Full35／Partial75 架構 winner，不是完成 tuning 後的最終超參數 winner。",
        "- 本次沒有刪除 runtime artifacts，也沒有 commit 或 push。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap_report = read_json(REPORT_ROOT / "full35-partial75-ap.json")
    ap_models = ap_report["models"]
    ap_lookup = {item["id"]: item for item in ap_models}
    training = []
    histories = {}
    for values in TRAINING_RUNS:
        record, epochs = summarize_training(*values, ap_lookup)
        training.append(record)
        histories[record["model_id"]] = epochs
    profiles = {
        model_id: read_json(REPORT_ROOT / "raw/profiles" / filename)
        for model_id, filename in PROFILE_FILES.items()
    }
    if any(profile["precision"] != "fp16" or profile["gpu"] != "NVIDIA GeForce RTX 5060 Ti" for profile in profiles.values()):
        raise RuntimeError("latency profile 不是正式 RTX 5060 Ti FP16 契約")
    plot_training_curves(histories)
    plot_bbt5(ap_models)
    plot_resources(training)
    write_training_csv(training)
    weight_index = build_weight_index(load_models(), ap_lookup)
    partial = ap_lookup["partial75-a2"]
    full = ap_lookup["full35-a2"]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_date_taipei": "2026-08-24",
        "environment": {
            "python": "3.12.13",
            "torch": "2.11.0+cu128",
            "ultralytics": "8.4.90",
            "gpu": "NVIDIA GeForce RTX 5060 Ti",
            "imgsz": 640,
            "seed": 0,
        },
        "phase_c_queue": {
            "status": "completed",
            "physical_batch": 8,
            "nbs": 16,
            "workers": 6,
            "patience": 7,
            "fraction": 1.0,
            "ram_fail_floor_bytes": 536870912,
        },
        "selection": {
            "architecture_winner": "partial75",
            "scope": "Full35 與 Partial75 的 P3-MASF 架構比較",
            "reason": "COCO mAP50-95 差距小於 0.001，Partial75 的 FP16 p50 latency、GFLOPs 與 Params 均較低。",
            "overall_accuracy_improvement_vs_a0_claimed": False,
            "full35_a2_coco_map50_95": full["coco_internal"]["ap50_95"],
            "partial75_a2_coco_map50_95": partial["coco_internal"]["ap50_95"],
            "winner_lr_tuning_t1_t2_t3_completed": False,
        },
        "bbt5_phase_c_delta_vs_a2": {
            architecture: {
                "overall_map50_95": overall_delta,
                "sports_ball_ap50_95": ball_delta,
                "baseball_bat_ap50_95": bat_delta,
            }
            for architecture, overall_delta, ball_delta, bat_delta in (
                (
                    architecture,
                    ap_lookup[f"{architecture}-c-f10"]["bbt5_internal"]["map50_95"]
                    - ap_lookup[f"{architecture}-a2"]["bbt5_internal"]["map50_95"],
                    ap_lookup[f"{architecture}-c-f10"]["bbt5_internal"]["sports_ball"]["ap50_95"]
                    - ap_lookup[f"{architecture}-a2"]["bbt5_internal"]["sports_ball"]["ap50_95"],
                    ap_lookup[f"{architecture}-c-f10"]["bbt5_internal"]["baseball_bat"]["ap50_95"]
                    - ap_lookup[f"{architecture}-a2"]["bbt5_internal"]["baseball_bat"]["ap50_95"],
                )
                for architecture in ("full35", "partial75")
            )
        },
        "latency_profiles": profiles,
        "training_runs": training,
        "accuracy_report": "full35-partial75-ap.json",
        "weight_index": "../weights/index.json",
        "weight_records": len(weight_index["records"]),
        "plots": [
            "figures/training-map-comparison.png",
            "figures/bbt5-ap-comparison.png",
            "figures/training-resources.png",
        ],
    }
    atomic_json(REPORT_ROOT / "rtx5060ti-final-0824.json", payload)
    markdown = build_markdown(training, ap_models, profiles)
    (REPORT_ROOT / "RTX5060TI_FINAL_0824.md").write_text(markdown, encoding="utf-8")
    print(f"已產生 {len(training)} 個 training summaries、{len(weight_index['records'])} 個權重索引項目")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
