#!/usr/bin/env python3
"""由 final/reports/raw 產生統一的 Markdown、JSON 與長格式 CSV。"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bundle import BUNDLE_ROOT, atomic_json, load_models

REPORT_ROOT = BUNDLE_ROOT / "reports"


def read_json(path: Path) -> dict[str, Any]:
    """讀取 JSON object。"""

    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    """以固定六位小數顯示 AP；缺值保留破折號。"""

    return "—" if value is None else f"{value:.6f}"


def internal_coco(raw: dict[str, Any]) -> dict[str, Any]:
    """擷取 Ultralytics internal COCO metrics。"""

    return {
        "precision": None,
        "recall": raw["recall"],
        "ap50": raw["map50"],
        "ap75": raw["map75"],
        "ap50_95": raw["map50_95"],
        "ap_s": raw.get("ap_s"),
        "ap_m": raw.get("ap_m"),
        "ap_l": raw.get("ap_l"),
        "sports_ball": {"ap50_95": raw["sports_ball_class_32_ap"]},
        "baseball_bat": {"ap50_95": raw["baseball_bat_class_34_ap"]},
        "cuda_peak_vram_bytes": raw.get("cuda_peak_vram_bytes"),
        "speed_ms": raw.get("speed_ms"),
    }


def internal_bbt5(raw: dict[str, Any]) -> dict[str, Any]:
    """擷取 detect_dataset BBT5 metrics。"""

    overall = dict(raw["overall"])
    precision = overall["precision"]
    recall = overall["recall"]
    overall["f1"] = 2 * precision * recall / (precision + recall)
    return {
        **overall,
        "sports_ball": raw["per_class"]["sports_ball"],
        "baseball_bat": raw["per_class"]["baseball_bat"],
        "cuda_peak_vram_bytes": raw.get("cuda_peak_vram_bytes"),
        "speed_ms": raw.get("speed_ms"),
    }


def long_rows(model: dict[str, Any], coco: dict[str, Any], canonical: dict[str, Any], bbt5: dict[str, Any]) -> list[dict[str, Any]]:
    """建立 dataset／evaluator／scope 長格式列。"""

    base = {
        "model_id": model["id"],
        "architecture": model["architecture"],
        "stage": model["stage"],
        "train_fraction": model["train_fraction"],
        "gate": model["gate"],
        "bittrue_sha256": model["bittrue_sha256"],
    }
    rows: list[dict[str, Any]] = []

    def append(dataset: str, evaluator: str, scope: str, values: dict[str, Any]) -> None:
        rows.append(
            {
                **base,
                "dataset": dataset,
                "evaluator": evaluator,
                "scope": scope,
                "images": values.get("images"),
                "instances": values.get("instances"),
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f1": values.get("f1"),
                "ap50": values.get("ap50"),
                "ap75": values.get("ap75"),
                "ap50_95": values.get("ap50_95"),
                "ap_s": values.get("ap_s"),
                "ap_m": values.get("ap_m"),
                "ap_l": values.get("ap_l"),
                "ar100": values.get("ar100"),
            }
        )

    append("coco2017-val", "ultralytics-internal", "overall", coco)
    append("coco2017-val", "ultralytics-internal", "sports_ball", coco["sports_ball"])
    append("coco2017-val", "ultralytics-internal", "baseball_bat", coco["baseball_bat"])
    for scope, values in canonical["metrics"].items():
        append("coco2017-val", "canonical-coco-api", scope, values)
    append("bbt5-detect-valid", "ultralytics-internal", "overall", bbt5)
    append("bbt5-detect-valid", "ultralytics-internal", "sports_ball", bbt5["sports_ball"])
    append("bbt5-detect-valid", "ultralytics-internal", "baseball_bat", bbt5["baseball_bat"])
    return rows


def build_markdown(models: list[dict[str, Any]]) -> str:
    """建立人類可讀總結。"""

    lines = [
        "# Full35／Partial75 兩資料集 AP 總報告",
        "",
        f"本報告涵蓋 {len(models)} 個已保存 Bit-True 候選，測試矩陣為 {len(models)} × 2 datasets，沒有缺值。Full35／Partial75 的 A2 重複轉檔已用 state-dict SHA256 合併；原始 checkpoint 的映射見 `../checkpoint-inventory.json`。",
        "",
        "固定環境為 Python 3.12.13、PyTorch 2.11.0+cu128、Ultralytics 8.4.90、RTX 5060 Ti 16 GiB；imgsz 640、validation batch 8、workers 6、Bit-True PWL。",
        "",
        "## COCO2017 val：Ultralytics internal（正式 selection 口徑）",
        "",
        "| Model | Fraction | AP50-95 | AP50 | AP75 | Ball AP50-95 | Bat AP50-95 | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in models:
        c = item["coco_internal"]
        fraction = "—" if item["train_fraction"] is None else str(item["train_fraction"])
        lines.append(
            f"| {item['id']} | {fraction} | {fmt(c['ap50_95'])} | {fmt(c['ap50'])} | "
            f"{fmt(c['ap75'])} | {fmt(c['sports_ball']['ap50_95'])} | "
            f"{fmt(c['baseball_bat']['ap50_95'])} | {item['gate']} |"
        )
    lines += [
        "",
        "## COCO2017 val：canonical COCO API",
        "",
        "此表由每個候選既有的 `predictions.json` 重新計算。它適合查 canonical AP50／AP75／size AP，但不能和上一表的 internal AP 混成同一排名欄位。",
        "",
        "| Model | Overall AP | Ball AP | Ball AP50 | Ball AP75 | Bat AP | Bat AP50 | Bat AP75 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        metrics = item["coco_canonical"]["metrics"]
        ball = metrics["sports_ball"]
        bat = metrics["baseball_bat"]
        lines.append(
            f"| {item['id']} | {fmt(metrics['overall']['ap50_95'])} | {fmt(ball['ap50_95'])} | "
            f"{fmt(ball['ap50'])} | {fmt(ball['ap75'])} | {fmt(bat['ap50_95'])} | "
            f"{fmt(bat['ap50'])} | {fmt(bat['ap75'])} |"
        )
    lines += [
        "",
        "## BBT5 detect_dataset valid：Ultralytics internal",
        "",
        "資料契約是 `/home/uxin0/yolo/original/pose/detect_dataset/coco80/data.yaml`；不是直接使用 pose dataset。Valid 有 567 images、301 sports-ball instances、484 baseball-bat instances。",
        "",
        "### Overall",
        "",
        "| Model | Precision | Recall | F1 | AP50-95 | AP50 | AP75 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in models:
        b = item["bbt5_internal"]
        lines.append(
            f"| {item['id']} | {fmt(b['precision'])} | {fmt(b['recall'])} | {fmt(b['f1'])} | "
            f"{fmt(b['map50_95'])} | {fmt(b['map50'])} | {fmt(b['map75'])} |"
        )
    for title, key in (("Sports ball（COCO class 32）", "sports_ball"), ("Baseball bat（COCO class 34）", "baseball_bat")):
        lines += [
            "",
            f"### {title}",
            "",
            "| Model | Images | Instances | Precision | Recall | F1 | AP50-95 | AP50 | AP75 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in models:
            values = item["bbt5_internal"][key]
            lines.append(
                f"| {item['id']} | {values['images']} | {values['instances']} | "
                f"{fmt(values['precision'])} | {fmt(values['recall'])} | {fmt(values['f1'])} | "
                f"{fmt(values['ap50_95'])} | {fmt(values['ap50'])} | {fmt(values['ap75'])} |"
            )
    coco_winner = max(models, key=lambda item: item["coco_internal"]["ap50_95"])
    bbt5_winner = max(models, key=lambda item: item["bbt5_internal"]["map50_95"])
    ball_winner = max(models, key=lambda item: item["bbt5_internal"]["sports_ball"]["ap50_95"])
    bat_winner = max(models, key=lambda item: item["bbt5_internal"]["baseball_bat"]["ap50_95"])
    lines += [
        "",
        "## 結論",
        "",
        f"- 正式 COCO internal 最高是 {coco_winner['id']}（{coco_winner['coco_internal']['ap50_95']:.6f}），但相對 A0 沒有可主張的實質 accuracy 提升。",
        f"- 使用者提供的 BBT5 `detect_dataset`：overall 最高是 {bbt5_winner['id']}（{bbt5_winner['bbt5_internal']['map50_95']:.6f}）；sports-ball AP50-95 最高是 {ball_winner['id']}（{ball_winner['bbt5_internal']['sports_ball']['ap50_95']:.6f}）；baseball-bat AP50-95 最高是 {bat_winner['id']}（{bat_winner['bbt5_internal']['baseball_bat']['ap50_95']:.6f}）。",
        "- Full35／Partial75 C-30% 的 BBT5 局部收益沒有轉成 COCO 整體收益，兩者 COCO internal AP50-95 都約 0.497，因此 gate rollback 合理。",
        "- 完整資料 C-100% 的 Full35／Partial75 也都低於各自 A2，沒有改變 rollback 或 accepted checkpoint。",
        "- BBT5 valid 有 93/567 images（16.4%）的 COCO ID 出現在 COCO train2017；結果適合模型間相對比較，不代表獨立資料泛化。",
        "- 報告所有 AP 都來自 `weights/bittrue/`；`weights/float/` 只供接續訓練／重新 materialize，沒有拿 Float 與 Bit-True 混成同一張排名表。",
        "",
        "完整精度與 raw metrics 請用 `full35-partial75-ap.json`；逐 dataset／evaluator／scope 比較請用 `full35-partial75-ap.csv`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    records = load_models()
    output_models: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for record in records:
        raw_root = REPORT_ROOT / "raw" / record["id"]
        coco_raw = read_json(raw_root / "coco-internal.json")
        bbt5_raw = read_json(raw_root / "bbt5-internal.json")
        canonical = read_json(raw_root / "coco-canonical.json")
        for raw in (coco_raw, bbt5_raw):
            if raw["checkpoint"]["sha256"] != record["bittrue_sha256"]:
                raise RuntimeError(f"{record['id']} raw metrics 與 registry checkpoint SHA 不符")
        coco = internal_coco(coco_raw)
        bbt5 = internal_bbt5(bbt5_raw)
        item = {**record, "coco_internal": coco, "coco_canonical": canonical, "bbt5_internal": bbt5}
        output_models.append(item)
        rows.extend(long_rows(record, coco, canonical, bbt5))

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison_backend": "bit_true_pwl",
        "validation_contract": {"imgsz": 640, "batch": 8, "workers": 6},
        "datasets": {
            "coco2017": "configs/datasets/coco2017.yaml",
            "bbt5": "configs/datasets/bbt5-coco80.yaml",
        },
        "models": output_models,
    }
    atomic_json(REPORT_ROOT / "full35-partial75-ap.json", payload)
    csv_path = REPORT_ROOT / "full35-partial75-ap.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    markdown = build_markdown(output_models)
    temporary = REPORT_ROOT / "FULL35_PARTIAL75_AP.md.tmp"
    temporary.write_text(markdown, encoding="utf-8")
    temporary.replace(REPORT_ROOT / "FULL35_PARTIAL75_AP.md")
    print(f"已產生 {len(output_models)} models、{len(rows)} metric rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
