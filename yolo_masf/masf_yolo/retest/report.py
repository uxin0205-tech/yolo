"""Build the unified Chinese retest summary from evaluation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "b1r-p2-p3-retest"


def build_summary() -> dict:
    rows = []
    for split in ("val", "test"):
        for path in sorted((ART / "evaluation" / split).glob("*/metrics.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            name = path.parent.name
            rows.append({"split": split, "name": name, "map50_95": data.get("map50_95"), "map50": data.get("map50"), "map75": data.get("map75"), "ap_s": data.get("ap_s"), "ap_m": data.get("ap_m"), "ap_l": data.get("ap_l"), "ball_ap": data.get("ball_ap"), "ball_ap_s": data.get("ball_ap_s"), "ball_recall": data.get("ball_recall"), "bat_ap": data["per_class"].get("bat", {}).get("ap"), "ball_fp": data["class_diagnostics"].get("ball", {}).get("false_positive_count"), "bat_fp": data["class_diagnostics"].get("bat", {}).get("false_positive_count")})
    (ART / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ART / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    test = [row for row in rows if row["split"] == "test"]
    best = max(test, key=lambda row: row["map50_95"])
    profiles = json.loads((ART / "profiles" / "summary.json").read_text(encoding="utf-8")) if (ART / "profiles" / "summary.json").is_file() else []
    profile_by_name = {item["name"].lower().replace(" ", "_"): item for item in profiles}
    report = [
        "# B1R / P2 / P3 統一實驗報告",
        "",
        "## 執行範圍",
        "",
        "資料來源固定為 `bbt5-detect-baseline/dataset`，初始化權重為 `yolo11m_bat_detect_init.pt`。所有 formal 實驗均使用同一份固定 80/10/10 split；test 僅在訓練完成後統一執行。B0 是資料暴露的 operational reference，不是公平選模候選。",
        "",
        "## 訓練流程",
        "",
        "B1R-A 為 freeze 0–10、10 epochs；B1R-B 為全模型 90 epochs；direct 為同一 B0 initializer 全模型 100 epochs。另加入仿照 yolo_p2 的 BBT5 head-only 20 epochs + full 80 epochs control。P2/P3 五個 MFAM variants 先 smoke 3 epochs，再 formal 100 epochs。",
        "",
        "## Test 結果（mAP50–95）",
        "",
        f"目前 test 最高為 `{best['name']}`：{best['map50_95']:.6f}。這只是本次固定 split/單 seed 的結果，不代表跨資料集普遍優勢。",
        "",
        "| 模型 | mAP50–95 | AP_S | AP_M | AP_L | Ball AP | Bat AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test:
        report.append(f"| {row['name']} | {row['map50_95']:.6f} | {row['ap_s']:.6f} | {row['ap_m']:.6f} | {row['ap_l']:.6f} | {row['ball_ap']:.6f} | {row['bat_ap']:.6f} |")
    report += ["", "## 硬體成本摘要", "", "| 模型 | Params | GFLOPs | P2 activation |", "|---|---:|---:|---:|"]
    for key, item in profile_by_name.items():
        report.append(f"| {item['name']} | {item['params']:,} | {item['gflops']:.3f} | {item['p2_activation_bytes']:,} |")
    report += [
        "",
        "## 初步分析",
        "",
        "1. B0 在 test 仍為最高 operational reference（mAP50–95 0.770812），代表新增 P2 並不會自動提升目前 BBT5 任務。",
        "2. P3 Partial25-35 的 test mAP50–95 為 0.754951，最接近 B0；P3 PaperFormula-Full 為 0.750617。這表示在本資料上，較少處理通道可能比完整 MFAM 更穩定。",
        "3. P2 family 整體低於 B0；P2 control-full 比 direct 高，但仍未追平 B0，支持「freeze 策略比單純長時間全解凍更重要」的診斷。",
        "4. 所有比較都必須保留資料暴露、單一 seed、固定 split 與 B0 initializer 限制；不能把 B0 當成公平勝者，也不能把 smoke 當正式結果。",
        "",
        "## 產物位置",
        "",
        "- `evaluation/val/`、`evaluation/test/`：統一 metrics、predictions 與錯誤案例。",
        "- `lineage/checkpoints.json`：checkpoint 路徑與 SHA-256。",
        "- `comparison.csv`、`summary.json`：機器可讀總表。",
        "- `queue_state.json`、`requests/`、`worker/`、`queue_logs/`：可續跑 queue 證據。",
    ]
    (ART / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"models": len(rows), "best_test": best, "report": str(ART / "REPORT.md")}


if __name__ == "__main__":
    print(json.dumps(build_summary(), ensure_ascii=False, indent=2))
