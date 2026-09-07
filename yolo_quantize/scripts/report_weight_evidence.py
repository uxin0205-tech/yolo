"""重整既有 CPU/PTQ/QAT 證據，不載入模型、不執行 GPU。"""

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/full-model-audit-2026-09-07"
SOURCE = ROOT / "artifacts/reports/v36-qsilu-v35-parent-148-layer-9format-cpu-v1.json"
FORMATS = (
    "w8",
    "w7",
    "w6",
    "w5",
    "w4",
    "fixed-sd4",
    "exact-ternary",
    "twn-v3",
    "paper-twn-v2",
)


def main():
    data = json.loads(SOURCE.read_text())
    paths = data["summary"]["path_rankings"]
    assert len(paths) == len({r["path"] for r in paths}) == 148
    raw = {(r["path"], r["format_id"]): r for r in data["measurements"]}
    rows = []
    grouped = defaultdict(list)
    for site in paths:
        assert set(site["formats"]) == set(FORMATS)
        for fmt in FORMATS:
            result = site["formats"][fmt]
            measurement = raw[site["path"], result["cpu_format_id"]]
            dist = measurement["distribution"]
            rms = math.sqrt(dist["standard_deviation"] ** 2 + dist["mean"] ** 2)
            row = {
                "path": site["path"],
                "region": site["region"],
                "format": fmt,
                "granularity": measurement["granularity"],
                "elements": site["elements"],
                "nrmse": result["normalized_rmse"],
                "cosine": result["cosine"],
                "estimated_packed_bytes": result["packed_bytes"],
                "original_exact_zero_ratio": dist["zero_ratio"],
                "original_mean_absolute": dist["mean_absolute"],
                "original_maximum_absolute": dist["maximum_absolute"],
                "original_standard_deviation": dist["standard_deviation"],
                "max_to_rms_proxy": dist["maximum_absolute"] / rms if rms else 0,
                "quantized_zero_ratio": measurement["numeric"]["zero_ratio"],
            }
            assert all(
                math.isfinite(v) for v in row.values() if isinstance(v, (float, int))
            )
            rows.append(row)
            grouped[site["region"], fmt].append(row)
    assert len(rows) == 1332
    with (OUT / "weight-format-evidence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    regions = sorted({r["region"] for r in paths})
    summary = []
    for region in regions:
        summary.append(
            "| "
            + region
            + " | "
            + " | ".join(
                f"{statistics.median(r['nrmse'] for r in grouped[region, fmt]):.4f}"
                for fmt in FORMATS
            )
            + " |"
        )
    count = sum(
        s["formats"]["fixed-sd4"]["normalized_rmse"]
        < s["formats"]["w4"]["normalized_rmse"]
        for s in paths
    )
    table = (
        "| 區域 | "
        + " | ".join(FORMATS)
        + " |\n|---|"
        + "---:|" * len(FORMATS)
        + "\n"
        + "\n".join(summary)
    )
    with (OUT / "ptq-candidates.csv").open() as stream:
        ptq = list(csv.DictReader(stream))
    ptq_table = "| 階段／候選 | 判定 | 最差 mAP50 Δ（pp） | 最差 mAP50–95 Δ（pp） |\n|---|---|---:|---:|\n"
    ptq_table += "\n".join(
        f"| {r['candidate']} | {r['decision']} | {100 * float(r['worst_total_map50_delta']):.3f} | {100 * float(r['worst_total_map50_95_delta']):.3f} |"
        for r in ptq
    )
    with (OUT / "qat-metrics.csv").open() as stream:
        qat = [r for r in csv.DictReader(stream) if r["epoch"] == "1"]
    assert len(qat) == 16 and len(ptq) == 25
    qat_table = (
        "| 指標 | accepted | V36 epoch 1 | 總變化（pp） |\n|---|---:|---:|---:|\n"
    )
    qat_table += "\n".join(
        f"| {r['metric']} | {float(r['accepted']):.6f} | {float(r['value']):.6f} | {100 * float(r['total_delta']):+.3f} |"
        for r in qat
    )
    appendix = f"""# 已測數據附錄：權重格式、PTQ 與 QAT

由 `scripts/report_weight_evidence.py` 讀取既有產物生成。解釋與限制見[詳細分析](../../docs/reports/2026-09-07-weight-distribution-detailed-analysis.md)。本次不是新實驗。

## 十區 CPU NRMSE

以下是區內各 path NRMSE 的**中位數**，不是參數加權全區誤差，也不是 mAP 下降。低者僅表示權重重建较接近。Fixed-SD4 為靜態 optimal scale，不是 LS-SD4 QAT 結果。

{table}

148 路中，Fixed-SD4 的 CPU NRMSE 嚴格小於同為 per-output-channel W4 的路徑有 **{count} 路**。這是碼本擬合比較，不是任務勝率。三元粒度與 W4/SD4 不完全相同，不能將全部差異歸因碼本。

完整逐路徑／格式表：[weight-format-evidence.csv](weight-format-evidence.csv)，1332 筆。`original_exact_zero_ratio` 是原權重恰為零的比例，不是接近零比例；`quantized_zero_ratio` 是該格式投影後零比例。`max_to_rms_proxy` 是原權重 max / sqrt(std²+mean²)，只作離群程度提示，不等於分散度或近零密度。std 沿用既有統計定義。

## 25 個 PTQ 結果

全部是相對 accepted 的 **total** delta（乘100轉百分點），包含 activation 替換，不混用 incremental。原 PTQ 用早期 qSiLU parent，CPU 用 V35 learned parent；各特殊格式的 top4 paths 可能不同，uniform 又沿用累積 routes，不能當作同層同 parent 的公平格式比賽。`recover` 表示候選待恢復，不是精度通過。

{ptq_table}

## V36 epoch 1：16 項任務結果

epoch index 1 即第二個 epoch；此處因其最差 mAP50 總損失小於 epoch 2 而展示，並非宣稱已重載驗證 best_joint export。搜尋 split，非 formal。各列 Δ = value − accepted，正數是提升；mAP 數值為 0–1。

{qat_table}

worst mAP50 總下降 0.835 百分點、worst mAP50–95 下降 1.157 百分點，分别低於 1.5／4 百分點門檻。不同指標的改善不能抵銷別的指標超標。

## 重建與來源

先執行 `scripts/report_full_model_audit.py`，再執行 `scripts/report_weight_evidence.py`。僅 CPU 讀取既有 JSON/CSV；原始 checkpoint、data、plan 血緣見同目錄 audit.json；本附錄來源 hash 見 weight-evidence-provenance.json。estimated_packed_bytes 是分析估算，沒有硬體測速或實際 packing 驗證。
"""
    (OUT / "weight-evidence-tables.md").write_text(appendix)
    provenance = {}
    for source in (SOURCE, OUT / "ptq-candidates.csv", OUT / "qat-metrics.csv"):
        provenance[str(source.relative_to(ROOT))] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    (OUT / "weight-evidence-provenance.json").write_text(
        json.dumps(
            {
                "sources": provenance,
                "rows": len(rows),
                "sd4_lower_nrmse_than_w4_paths": count,
                "new_gpu_experiment": False,
            },
            indent=2,
        )
        + "\n"
    )
    matrix = [
        [statistics.median(r["nrmse"] for r in grouped[reg, fmt]) for fmt in FORMATS]
        for reg in regions
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    plot = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(FORMATS)), FORMATS, rotation=25, ha="right")
    ax.set_yticks(range(len(regions)), regions)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("V35 parent: CPU weight reconstruction error (not mAP)")
    fig.colorbar(plot, ax=ax, label="Median path NRMSE (lower is better)")
    fig.text(
        0.5,
        0.01,
        "Static scales; ternary granularities differ. Not a raw-weight distribution histogram.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"weight-format-nrmse.{ext}", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "regions": len(regions),
                "sd4_better_than_w4": count,
                "ptq": len(ptq),
                "qat_metrics": len(qat),
            }
        )
    )


if __name__ == "__main__":
    main()
