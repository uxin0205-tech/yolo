# 第一輪 Phase 1 歷史紀錄

本頁把既有的 Phase 1 實驗納入目前 study 的結果索引。它與第二輪 B1R/P2/P3 重測分開保存；數值不可直接混排名，因為兩輪的模型集合、訓練流程與後處理批次不同。

## 歷史實驗口徑

- 資料：`bbt5-detect-baseline/dataset`，train/val/test 為 1,987/300/291 frames。
- 初始化：`bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`，為已看過 BBT5 的 pose-derived initializer，因此是資料暴露下的 operational ablation。
- 輸入 640、batch 16、seed 42、SGD、AMP；硬體 profile 為 RTX 5090、FP16、batch 1。
- B1 採 freeze backbone 0–10 共 10 epochs，再全模型 90 epochs；其餘正式變體依第一輪 pipeline 的 manifest 執行。
- AP 為 faster-coco-eval 的 COCO-style mAP50–95；Ball/Bat 欄位也是 AP50–95。

## 第一輪 val/test 指標

| 模型 | Val mAP | Test mAP | Test Ball AP | Test Bat AP | Test Ball FP | Test Bat FP |
|---|---:|---:|---:|---:|---:|---:|
| B0（參考） | 0.7508 | **0.7708** | 0.6784 | 0.8632 | 1,204 | 485 |
| B1 | 0.7081 | 0.7303 | 0.6126 | 0.8481 | 1,599 | 1,053 |
| M7 | 0.7079 | **0.7386** | **0.6324** | 0.8447 | 3,000 | 1,781 |
| M0 | **0.7108** | 0.7234 | 0.6113 | 0.8356 | 2,281 | 1,240 |
| M1 | 0.7095 | 0.7263 | 0.6162 | 0.8365 | 2,243 | 1,137 |
| M2 | 0.7047 | 0.7250 | 0.6089 | 0.8412 | 1,971 | 1,103 |
| M3 | 0.7020 | 0.7262 | 0.6143 | 0.8381 | 1,928 | 1,106 |
| P3M | 0.6990 | 0.7340 | 0.6248 | 0.8432 | 2,346 | 982 |
| SP2 | 0.7056 | 0.7170 | 0.5970 | 0.8370 | 6,908 | 986 |
| SP2P | 0.6971 | 0.7213 | 0.6047 | 0.8378 | 6,983 | 1,319 |

### 物件尺寸指標（第一輪 B0/B1）

| 模型 | Ball AP_S | Ball AP_M | Ball AP_L | Bat AP_S | Bat AP_M | Bat AP_L |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.4722 | 0.8752 | 0.8410 | 0.7182 | 0.8829 | 0.8664 |
| B1 | 0.4405 | 0.8282 | 0.7869 | 0.6939 | 0.8699 | 0.8085 |

上表的完整 AP50、precision、recall、missed、size/blur recall 與所有模型的尺寸欄位，請以原始報告及 `evaluation/{val,test}/<model>/metrics.json` 為準；此頁只放最常用的比較欄位，避免複製大型 JSON。

## 第一輪架構與結論

- **M7**：P2 全 channels 的 DW3、DW5、DW(1×7→7×1)，第一輪 test 最佳公平消融，但 false positives 明顯增加。
- **M0/M1/M2/M3**：P2 MFAM 的 full、DW3+DW5、1/2 channels、1/4 channels 對照；partial 能降低成本，但本輪未穩定超過 B1。
- **P3M**：只在 P3 放 MASF/MFAM，不含 9×9；test 比 B1 高 0.366 個百分點，是較平衡候選。
- **SP2/SP2P**：Selective P2 與 M2 組合；Ball recall 上升但 false positives 很高，且 RTX 5090 實測 latency 沒有下降。
- **B0**：既有三尺度權重，只能作資料暴露且訓練預算不同的 operational upper reference，不是公平勝者。

## 原始證據與對應路徑

| 類型 | 入口 |
|---|---|
| 第一輪完整中文報告 | [`EXPERIMENT_RESULTS_ZH.md`](../../EXPERIMENT_RESULTS_ZH.md) |
| 機器重建報告 | [`artifacts/static-phase1/report.md`](../../artifacts/static-phase1/report.md) |
| 第一輪產物索引 | [`artifacts/static-phase1/README.md`](../../artifacts/static-phase1/README.md) |
| 評估 metrics | [`artifacts/static-phase1/evaluation/`](../../artifacts/static-phase1/evaluation/) |
| 硬體 profile | [`artifacts/static-phase1/profiles/`](../../artifacts/static-phase1/profiles/) |
| M2/M3 選模與稽核 | [`selection.json`](../../artifacts/static-phase1/selection.json)、[`final_audit.json`](../../artifacts/static-phase1/final_audit.json) |

第一輪原始 checkpoints、CSV、metrics、profiles、selection 與 audit 保留；smoke/preflight checkpoint、cache 與可重建預覽圖不列入正式結果。
