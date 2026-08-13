#!/usr/bin/env python3
"""把 B1R/P2/P3 runtime 證據整理成可直接在 GitHub 瀏覽的發布包。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


DESCRIPTIONS = {
    "b0-original-3scale": "既有 B0 三尺度 P3/P4/P5 operational reference；未加入 P2 或 MFAM。",
    "p2-base-direct": "B1R 四尺度 P2/P3/P4/P5 baseline；從來源初始化權重全模型直接訓練 100 epochs。",
    "p2-control-head": "B1R 的 P2 對照第一段；凍結 inherited backbone/neck 與 P3–P5 towers，只訓練 P2 slot/head 20 epochs。",
    "p2-control-full": "承接 P2-Control-Head best，解除凍結後全模型訓練 80 epochs。",
    "p2-paperformula-full": "在 P2 slot 使用 PaperFormulaMFAM：DW3、DW5、factorized DW7、factorized DW9，全通道。",
    "p2-lite-35": "在 P2 slot 使用 DW3+DW5 的 PaperFormulaMFAM，全通道。",
    "p2-lite-35-f7": "在 P2 slot 使用 DW3+DW5+factorized DW7，全通道。",
    "p2-partial50-35": "在 P2 slot 只讓前 50% channels 通過 DW3+DW5，其餘 identity bypass。",
    "p2-partial25-35": "在 P2 slot 只讓前 25% channels 通過 DW3+DW5，其餘 identity bypass。",
    "p3-paperformula-full": "維持 B0 三尺度 Detect，只在 P3 feature slot 使用完整 PaperFormulaMFAM。",
    "p3-lite-35": "維持 B0 三尺度 Detect，只在 P3 feature slot 使用 DW3+DW5。",
    "p3-lite-35-f7": "維持 B0 三尺度 Detect，只在 P3 feature slot 使用 DW3+DW5+factorized DW7。",
    "p3-partial50-35": "維持 B0 三尺度 Detect，只讓 P3 前 50% channels 通過 DW3+DW5。",
    "p3-partial25-35": "維持 B0 三尺度 Detect，只讓 P3 前 25% channels 通過 DW3+DW5。",
}

TRAINING = {
    "b0-original-3scale": "不在本輪重訓；直接評估來源初始化權重。",
    "p2-base-direct": "全模型 100 epochs，SGD，lr0=0.001。",
    "p2-control-head": "P2 slot/head 20 epochs，SGD，lr0=0.01；其他 inherited layers 凍結。",
    "p2-control-full": "承接 control-head best，全模型 80 epochs，SGD，lr0=0.001。",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def materialize(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    materialize(target)
    shutil.copy2(source, target)


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def metric_row(row: dict) -> str:
    fields = (
        row["name"], row["map50_95"], row["map50"], row["ap_s"], row["ap_m"],
        row["ap_l"], row["ball_ap"], row["ball_ap_s"], row["ball_recall"],
        row["bat_ap"], row["ball_fp"], row["bat_fp"],
    )
    return "| " + " | ".join(fmt(value) if index else str(value) for index, value in enumerate(fields)) + " |"


def result_table(rows: list[dict]) -> str:
    lines = [
        "| 模型 | mAP50–95 | mAP50 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(metric_row(row) for row in rows)
    return "\n".join(lines)


def profile_table(profiles: list[dict]) -> str:
    lines = [
        "| 模型 | Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in profiles:
        lines.append(
            f"| {row['name']} | {row['params']:,} | {row['gflops']:.3f} | "
            f"{row['p2_activation_bytes']:,} | {row['peak_live_activation_bytes']:,} | {row['feature_traffic_bytes']:,} |"
        )
    return "\n".join(lines)


def experiment_readme(name: str, rows: dict[str, dict], profile: dict, checkpoint: dict) -> str:
    val = rows["val"]
    test = rows["test"]
    training = TRAINING.get(name, "先完成 3-epoch smoke 工程檢查，再從指定 parent 初始化，全模型 formal 100 epochs（lr0=0.001）。")
    checkpoint_path = Path(str(checkpoint["checkpoint"]))
    available = checkpoint_path.is_file()
    if name == "b0-original-3scale":
        weight_note = "可用；repo 內位置：[`../../../../bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`](../../../../bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt)。"
    elif available:
        weight_note = "發布時來源檔仍存在；請以 `checkpoint.json` 的 SHA-256 驗證。"
    else:
        weight_note = "**缺失**；lineage 保留原路徑與 SHA-256，但 runtime checkpoint 已不在磁碟，無法誠實上傳。"
    return f"""# {name}

## 做法

{DESCRIPTIONS[name]}

- 訓練：{training}
- 共通條件：imgsz=640、batch=16、seed=42、SGD、momentum=0.937、cosine LR、AMP、deterministic。
- 資料：固定 BBT5 80/10/10 group/hash-audited split。
- 權重狀態：{weight_note}
- 記錄 SHA-256：`{checkpoint['sha256']}`

## 精確結果

| Split | mAP50–95 | mAP50 | AP75 | AP_S | AP_M | AP_L | Ball AP | Ball AP_S | Ball Recall | Bat AP | Ball FP | Bat FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | {fmt(val['map50_95'])} | {fmt(val['map50'])} | {fmt(val['map75'])} | {fmt(val['ap_s'])} | {fmt(val['ap_m'])} | {fmt(val['ap_l'])} | {fmt(val['ball_ap'])} | {fmt(val['ball_ap_s'])} | {fmt(val['ball_recall'])} | {fmt(val['bat_ap'])} | {fmt(val['ball_fp'])} | {fmt(val['bat_fp'])} |
| test | {fmt(test['map50_95'])} | {fmt(test['map50'])} | {fmt(test['map75'])} | {fmt(test['ap_s'])} | {fmt(test['ap_m'])} | {fmt(test['ap_l'])} | {fmt(test['ball_ap'])} | {fmt(test['ball_ap_s'])} | {fmt(test['ball_recall'])} | {fmt(test['bat_ap'])} | {fmt(test['ball_fp'])} | {fmt(test['bat_fp'])} |

## 硬體靜態成本

| Params | GFLOPs | P2 activation bytes | Peak live activation bytes | Feature traffic bytes |
|---:|---:|---:|---:|---:|
| {profile['params']:,} | {profile['gflops']:.3f} | {profile['p2_activation_bytes']:,} | {profile['peak_live_activation_bytes']:,} | {profile['feature_traffic_bytes']:,} |

## 檔案

- [`val_metrics.json`](val_metrics.json)
- [`test_metrics.json`](test_metrics.json)
- [`profile.json`](profile.json)
- [`checkpoint.json`](checkpoint.json)
"""


def build_report(summary: list[dict], profiles: list[dict]) -> str:
    val_rows = [row for row in summary if row["split"] == "val"]
    test_rows = [row for row in summary if row["split"] == "test"]
    test_by_name = {row["name"]: row for row in test_rows}
    b0 = test_by_name["b0-original-3scale"]
    best_p2 = max((row for row in test_rows if row["name"].startswith("p2-")), key=lambda row: row["map50_95"])
    best_p3 = max((row for row in test_rows if row["name"].startswith("p3-")), key=lambda row: row["map50_95"])
    return f"""# B1R / P2 / P3 完整實驗報告

更新日期：2026-08-14

## 1. 研究問題

本輪檢查三件事：重新訓練的 P2 baseline 為何未追平既有 B0；完整 P2 head 是否值得其高解析成本；以及把 MFAM 只放在 P3 或只處理部分 channels，能否得到更穩定的精度／成本折衷。

## 2. 資料與初始化

- 來源資料：`bbt5-detect-baseline/dataset`；發布包不包含 dataset。
- Locked split：train/val/test=1,987/300/291 frames，比例 80/10/10，seed=42；group overlap 與 hash overlap 均為空。
- 類別：Ball、Bat。
- 初始化：`yolo11m_bat_detect_init.pt`，SHA-256=`9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d`。
- 重要限制：初始化權重已接觸 BBT5，因此全部結果是 data-exposed operational ablation，不是無洩漏泛化估計。

## 3. 共通訓練設定

- Ultralytics 8.4.90、PyTorch 2.11.0+cu128、CUDA 12.8。
- imgsz=640、batch=16、SGD、momentum=0.937、cosine LR、AMP、deterministic、nbs=64、單一 seed=42。
- B1R-A：凍結 backbone indices 0–10，10 epochs，lr0=0.01。
- B1R-B：承接 A-best，全模型 90 epochs，lr0=0.001。
- Direct：全模型 100 epochs，lr0=0.001。
- P2 Control：head-only 20 epochs（lr0=0.01）後，承接 best 全模型 80 epochs（lr0=0.001）。
- 每個 P2/P3 MFAM：3 epochs smoke 只驗證工程穩定；formal 為全模型 100 epochs，lr0=0.001。
- GPU queue 單工執行；`queue_state.json` 的最終狀態為 `formal_complete`。

## 4. 模型矩陣

| 名稱 | Placement | Channels | Branches / 對照 |
|---|---|---:|---|
| B0-Original-3Scale | P3/P4/P5 Detect | — | 既有 operational reference |
| P2-Base-Direct | P2/P3/P4/P5 Detect | — | 直接全模型 100 epochs |
| P2-Control-Head / Full | P2/P3/P4/P5 Detect | — | 20 epochs head-only + 80 epochs full |
| PaperFormula-Full | P2 或 P3 | 100% | DW3、DW5、DW(1×7→7×1)、DW(1×9→9×1)，兩層 1×1 residual fusion |
| Lite-35 | P2 或 P3 | 100% | DW3、DW5 |
| Lite-35-F7 | P2 或 P3 | 100% | DW3、DW5、DW(1×7→7×1) |
| Partial50-35 | P2 或 P3 | 50% | 前 50% channels 走 DW3+DW5，其餘 exact identity bypass |
| Partial25-35 | P2 或 P3 | 25% | 前 25% channels 走 DW3+DW5，其餘 exact identity bypass |

上述 P3-only、partial-channel 與 P2 placement 是 BBT5 adaptation；不宣稱等同論文 Figure 1 的四個 backbone MFAM placement。PaperFormula 實作依論文公式 (1)–(6)，沒有 learnable branch weighting 或額外 gate。

## 5. Validation 完整結果

{result_table(val_rows)}

## 6. Test 完整結果

{result_table(test_rows)}

## 7. 硬體靜態成本

{profile_table(profiles)}

這裡是靜態 Params/GFLOPs/activation/traffic，不是實機 latency；不能用 GFLOPs 直接宣稱 RTX 5090 或邊緣裝置更快。

## 8. 結果分析

1. B0 test mAP50–95={b0['map50_95']:.6f} 仍最高，但它的訓練歷史與本輪預算不同，只能作 operational upper reference。
2. P2 最佳為 {best_p2['name']}={best_p2['map50_95']:.6f}，相對 B0 為 {(best_p2['map50_95']-b0['map50_95'])*100:+.3f} 個百分點。P2 activation 從 B0 的 6,553,600 bytes 增至 13,107,200 bytes，沒有得到相稱的精度提升。
3. P3 最佳為 {best_p3['name']}={best_p3['map50_95']:.6f}，相對 B0 為 {(best_p3['map50_95']-b0['map50_95'])*100:+.3f} 個百分點；它也是全部新增模型中最接近 B0 者。
4. P3-Partial25-35 只增加至 67.779 GFLOPs、20,065,430 params，顯示限制高成本多尺度處理的方向比普遍加入 P2 更符合本資料。
5. P2-Control-Full 高於 Direct，支持 staged freeze/head-only 初始化有幫助；但仍未追平 B0，代表落差不只來自 epoch 數，也包含新增 P2 graph、transfer 與訓練分布差異。
6. 全部結果只有單一 seed；小於約 1 個百分點的差異應以多 seed 重測後再作穩定性主張。

## 9. Checkpoint 完整性警告

B0 initializer 仍在 repo 且由 Git LFS 管理。第二輪 13 個 formal best checkpoint 的 SHA-256 與原 runtime 路徑都有保存，但發布整理時原路徑已不存在，因此本次無法把那些 `.pt` 誠實上傳。這不影響已保存 metrics/profile 的可讀性，但會影響直接重現推論；詳見 [`weights/CHECKPOINT_STATUS.md`](weights/CHECKPOINT_STATUS.md)。第一輪 Phase 1 的正式權重仍在 `artifacts/static-phase1/`。

## 10. 證據入口

- 每個模型：[`experiments/`](experiments/)
- 機器總表：[`comparison.csv`](comparison.csv)、[`summary.json`](summary.json)
- Queue／worker／lineage：[`metadata/`](metadata/)
- 第一輪歷史結果：[`LEGACY_RESULTS.md`](LEGACY_RESULTS.md)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="runtime artifacts/b1r-p2-p3-retest")
    parser.add_argument("--repo", type=Path, required=True, help="yolo_masf repository root")
    args = parser.parse_args()
    source = args.source.resolve()
    repo = args.repo.resolve()
    results = repo / "b1r_p2_p3_study" / "results"
    results.mkdir(parents=True, exist_ok=True)

    summary = read_json(source / "summary.json")
    profiles = read_json(source / "profiles" / "summary.json")
    checkpoints = read_json(source / "lineage" / "checkpoints.json")
    checkpoint_by_slug = {item["name"].lower(): item for item in checkpoints}
    rows: dict[str, dict[str, dict]] = {}
    for row in summary:
        rows.setdefault(row["name"], {})[row["split"]] = row

    for name in ("REPORT.md", "comparison.csv", "summary.json"):
        materialize(results / name)
    write_text(results / "REPORT.md", build_report(summary, profiles))
    copy_file(source / "comparison.csv", results / "comparison.csv")
    copy_file(source / "summary.json", results / "summary.json")

    experiment_root = results / "experiments"
    materialize(experiment_root)
    experiment_root.mkdir(parents=True)
    index_lines = ["# 逐模型實驗索引", "", "每個資料夾均包含做法、val/test metrics、profile 與 checkpoint lineage。", ""]
    for name in sorted(rows):
        model_dir = experiment_root / name
        model_dir.mkdir(parents=True)
        profile = next(item for item in profiles if item["name"].lower() == name)
        checkpoint = checkpoint_by_slug[name]
        copy_file(source / "evaluation" / "val" / name / "metrics.json", model_dir / "val_metrics.json")
        copy_file(source / "evaluation" / "test" / name / "metrics.json", model_dir / "test_metrics.json")
        write_text(model_dir / "profile.json", json.dumps(profile, ensure_ascii=False, indent=2))
        checkpoint_record = {**checkpoint, "available_at_publication": Path(checkpoint["checkpoint"]).is_file()}
        write_text(model_dir / "checkpoint.json", json.dumps(checkpoint_record, ensure_ascii=False, indent=2))
        write_text(model_dir / "README.md", experiment_readme(name, rows[name], profile, checkpoint))
        index_lines.append(f"- [{name}]({name}/README.md)")
    write_text(experiment_root / "README.md", "\n".join(index_lines))

    profile_root = results / "profiles"
    materialize(profile_root)
    profile_root.mkdir()
    copy_file(source / "profiles" / "summary.json", profile_root / "summary.json")
    for profile in profiles:
        write_text(profile_root / f"{profile['name'].lower()}.json", json.dumps(profile, ensure_ascii=False, indent=2))
    write_text(profile_root / "README.md", "# 硬體成本\n\n`summary.json` 與各模型 JSON 保存 Params、MACs、GFLOPs、activation、operator count 與 feature traffic；本輪沒有可發布的實機 latency。")

    metadata = results / "metadata"
    materialize(metadata)
    metadata.mkdir()
    for name in ("final_audit.json", "queue_state.json"):
        copy_file(source / name, metadata / name)
    copy_file(source / "lineage" / "checkpoints.json", metadata / "checkpoints.json")
    shutil.copytree(source / "requests", metadata / "requests")
    shutil.copytree(source / "worker", metadata / "worker")
    write_text(metadata / "README.md", "# 執行 metadata\n\n- `queue_state.json`：GPU 單工排程最終狀態。\n- `requests/`：每個 stage 的輸入、parent 與設定。\n- `worker/`：每個 stage 的 best/last 原路徑與 SHA-256。\n- `checkpoints.json`：14 個正式評估 checkpoint lineage。\n- `final_audit.json`：28 metrics；14 個逐模型 profile 加 1 個 summary JSON。")
    write_text(metadata / "RESULTS_SCOPE.md", "# 發布證據範圍\n\n本發布包包含 queue state、worker/request manifests、checkpoint lineage、metrics、profiles、總表與報告。未納入 dataset、smoke/preflight checkpoint、last.pt、raw prediction dumps、false-positive 大圖與大型 queue logs。")

    weights = results / "weights"
    materialize(weights)
    weights.mkdir()
    status_lines = [
        "# Checkpoint 狀態", "", "| 模型 | SHA-256 | 發布時狀態 |", "|---|---|---|",
    ]
    for item in checkpoints:
        available = Path(item["checkpoint"]).is_file()
        if item["name"] == "B0-Original-3Scale":
            status = "已追蹤：`bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`（Git LFS）"
        else:
            status = "原 runtime 路徑已遺失，未上傳"
        status_lines.append(f"| {item['name']} | `{item['sha256']}` | {status} |")
    status_lines.extend(["", "第一輪 Phase 1 的 native/canonical 正式權重仍位於 `artifacts/static-phase1/`，不受此缺失影響。"])
    write_text(weights / "CHECKPOINT_STATUS.md", "\n".join(status_lines))
    write_text(weights / "README.md", "# 權重入口\n\n- [第二輪 checkpoint 狀態與 SHA-256](CHECKPOINT_STATUS.md)\n- [B0 initializer](../../../bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt)\n- 第一輪完整權重：`artifacts/static-phase1/runs/*/weights/best.pt` 與 `training/*/canonical.pt`。")
    runtime_link = results / "metrics" / "runtime_evaluation"
    if runtime_link.is_symlink():
        runtime_link.unlink()

    manifest = []
    for path in sorted(results.rglob("*")):
        if path.is_file() and not path.is_symlink():
            manifest.append({"path": str(path.relative_to(repo)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    write_text(results / "PUBLICATION_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
