# YOLO Activation 研究

**Stage7 已完成並整理交付，RepConv 尚未開始。** 本輪比較 qSiLU、Hardswish、PolyShift、PolyQuality；各訓練 10 輪並完成正式驗證。qSiLU E8 固定作為下一階段 parent，PolyQuality E10 為綜合精度最佳比較模型。

直接使用請開啟 [handoff/](handoff/README.md)；完整數據與判斷見 [Stage7 結果](result/stage7/README.md)。交接含原 `.pt`、必要 runtime、載入入口、設定、SHA 與已通過的獨立 CPU 等價驗證；不需舊實驗目錄或資料集即可做 tensor 推論。

| 位置 | 用途 |
| --- | --- |
| `handoff/` | 指向唯一的 `result/stage7/handoff/`；qSiLU 與 PolyQuality 可攜交接 |
| `result/stage7/` | 本輪結果、四個最佳候選、圖表、小型數據與完整實驗證據 |
| `stage7/`、`run_stage7.sh` | 本輪研究實作、protocol、來源鎖與 queue 工具；queue 已完成，勿重複啟動 |
| `src/activation_lab/` | canonical activation 公式／局部定點 reference；optimize、quantize 等仍使用 |
| `scripts/`、`tests/` | 研究工具、交付產生器、清理程式與相關驗證 |
| `result/reference-full35/` | 舊 Full35 結果／設定的小型抽取與必要外部依賴說明 |
| `artifacts/runs/full35/` | 僅留外部仍依賴的四份舊權重及必要小型來源；不是本輪 parent |
| `release/weights/` | 既有 Full35 SiLU／qSiLU 分片交付，保留舊交付用途；本輪直接使用 handoff 的 `.pt` |
| `docs/`、`reports/`、`training/` | 數學與既有研究證據、配方來源；以 Stage7 作現行工作入口 |

本次直接清除約 5.25 GiB 舊 runtime 連結、預覽、快取與零樣本權重，資料夾由約 6.3 GiB 降至 1.1 GiB；不建立 archive／backup。正式資料與 48 個受保護檔案 SHA 不變，見 [清理清單](result/stage7/cleanup-20261103.json)。

```bash
# 專案根目錄執行：交接模型的 CPU smoke
.venv/bin/python -B yolo_activation/handoff/load_model.py --candidate qsilu_pq
# 只讀已完成狀態
bash yolo_activation/run_stage7.sh status
# 無 GPU／checkpoint 重畫
.venv/bin/python yolo_activation/result/stage7/redraw.py
```

共同 parent 為 Stage6 E1；B2 二值 QK、S1 Q10 scale／bias、D1 整數 PWL exp＋LUT256 reciprocal 皆繼承。任務模型的 Conv、QKV、P·V、projection 與 activation 仍是浮點 reference，尚無 RTL／HLS／板端實測。原獨立模型精度目標仍未全部達成，單 seed 結果不宣稱全域最優。

[研究 protocol](stage7/README.md) · [本輪工作紀錄](../docs/worklogs/2026-11-03-activation-stage7-handoff.md) · [BBAT5 v1 規範](../docs/agents/bbat5-datasets.md) · [數學推導](docs/research/full35-activation-mathematical-derivations.md) · [舊研究索引](result/reference-full35/README.md)
