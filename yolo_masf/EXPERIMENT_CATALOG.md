# MASF-YOLO 完整實驗目錄

這一頁回答「每一個實驗到底在做什麼」。它把 Phase 1 的探索實驗與第二輪 B1R/P2/P3 重測串成同一條研究脈絡；精確的所有欄位仍以各輪正式報告與機器可讀 metrics 為準。

> **共同限制**：所有模型使用 `bbt5-detect-baseline/dataset/` 的 BBT5 固定資料集，train/validation/test 為 1,987/300/291 個 unique frames。來源初始化權重 `yolo11m_bat_detect_init.pt` 已接觸 BBT5，因此以下是同源、資料暴露的操作性消融，不是無洩漏泛化估計。所有正式結果只有 seed 42。

## 先理解比較邏輯

```text
B0 三尺度既有權重（操作性參考，不是公平 baseline）
├── 加完整 P2 Detect head：B1 / B1R / Direct / Control
│   ├── 在 P2 加 MFAM：完整公式、簡化分支、Partial channels
│   └── 只讓 Ball 使用 P2：SP2 / SP2P
└── 保留三尺度 Detect，只在 P3 加 MFAM：P3M 與第二輪五種 P3 變體
```

- `P2`、`P3`：模組放置位置；P2 解析度較高但成本較大。
- `PaperFormula-Full`：DW3、DW5、factorized DW(1×7→7×1)、factorized DW(1×9→9×1)，全 channels，使用論文式的兩層 1×1 residual fusion。
- `Lite-35`：只保留 DW3、DW5。
- `Lite-35-F7`：DW3、DW5，再加 factorized 7×7。
- `Partial50/25-35`：只有 50%/25% channels 通過 DW3、DW5，其餘 exact identity bypass。
- `SP2`：Selective P2；只有 Ball 使用輕量 P2 prediction head，Bat 仍由 P3/P4/P5 預測。

## Phase 1：先探索架構方向

Phase 1 的正式設定為 640、batch 16、SGD、AMP、seed 42。B1 採 10 epochs 凍結 backbone 0–10，再全模型解凍 90 epochs；其他相依訓練與選模細節見[第一輪完整報告](EXPERIMENT_RESULTS_ZH.md)。

| 實驗 | 在做什麼、要回答什麼 | Test mAP50–95 | 結果怎麼看 |
|---|---|---:|---|
| **B0** | 使用既有 P3/P4/P5 三尺度 Detect 權重，不做本輪同預算重訓；提供「目前既有權重能到哪裡」的操作性上限。 | **0.7708** | 訓練歷史與預算不同，不能當公平消融勝者。 |
| **B1** | 加入標準 P2 Detect head，建立四尺度公平 baseline；先凍結 10 epochs、再全解凍 90 epochs。 | 0.7303 | 後續 P2 變體的主要公平比較基準。 |
| **M7** | 在 P2 全 channels 使用 DW3、DW5、factorized 7×7；測試不含 9×9 的多尺度分支。 | **0.7386** | Phase 1 公平組最佳，但 Ball FP 明顯增加。 |
| **M0** | 在 P2 全 channels 使用 DW3、DW5、DW7、DW9；測試大型 kernel 是否有效。 | 0.7234 | 9×9 沒有得到 test 支持，且成本最高之一。 |
| **M1** | 在 P2 全 channels 只使用 DW3、DW5；測試最簡化 MFAM。 | 0.7263 | 成本略降，但沒有超過 B1。 |
| **M2** | P2 只有 1/2 channels 經過 M1，其餘 bypass；測試部分通道是否更硬體友善。 | 0.7250 | validation 比 M3 高 0.002709，因此依預先規則選入 SP2P。 |
| **M3** | P2 只有 1/4 channels 經過 M1，其餘 bypass；進一步壓低成本。 | 0.7262 | latency 幾乎回到 B1；test 雖略高於 M2，不能事後用 test 改選。 |
| **P3M** | P2 保持 Identity，只在 P3 放 DW3、DW5、factorized 7×7；測試不用昂貴完整 P2 head 的替代方案。 | **0.7340** | 比 B1 高 0.0037，成為值得重測的 P3-only 方向。 |
| **SP2** | 只有 Ball 使用 hidden=32 輕量 P2 head，Bat 保留 P3/P4/P5；測試 selective high-resolution prediction。 | 0.7170 | Ball recall 提高，但 Ball FP=6,908，且 RTX 5090 實測反而更慢。 |
| **SP2P** | SP2 再結合 validation 選出的 M2 partial MFAM，並額外做自己的 10+90 epochs。 | 0.7213 | 比 SP2 略升，但 FP 與 latency 問題仍在；訓練預算也較高。 |

Phase 1 的完整 AP、precision/recall、FP、延遲與參數量在[第一輪正式報告](EXPERIMENT_RESULTS_ZH.md)，逐模型權重與證據在[第一輪 artifacts](artifacts/static-phase1/README.md)。

## 第二輪：修正 baseline 並公平重測 P2/P3

第二輪用固定資料與設定重新回答三件事：P2 baseline 是否因訓練方式而偏弱、MFAM 放 P2 或 P3 哪裡較合理、減少分支或只處理部分 channels 能否保留精度並降低成本。

### 訓練階段（不是額外排名模型）

| 階段 | 實際用途 |
|---|---|
| **B1R-A** | 修正版四尺度 P2/P3/P4/P5 baseline，保留 B0 的 P3–P5 head shape；凍結 backbone 0–10，訓練 10 epochs，`lr0=0.01`。 |
| **B1R-B** | 從 B1R-A best 初始化，解除全部凍結，以新的 90-epoch run 訓練，`lr0=0.001`；不是沿用 epoch counter 的 resume。 |
| **Smoke** | 每個 P2/P3 MFAM 候選先跑 3 epochs，只檢查 graph、loss、checkpoint 與顯存流程是否穩定；不列入科學結果。 |
| **Formal** | smoke 通過後，每個 MFAM 變體重新正式訓練 100 epochs；validation、test、profile 與 final audit 才是正式證據。 |

### Baseline 與訓練策略對照

| 實驗 | 在做什麼、要回答什麼 | Test mAP50–95 | 結果怎麼看 |
|---|---|---:|---|
| **B0 Original 3Scale** | 原始 P3/P4/P5 initializer；作為既有系統參考。 | **0.770812** | 仍非同預算公平 baseline。 |
| **B0-Fair（待執行）** | 保持與 B0 完全相同的三尺度 graph，從相同 initializer 出發，以 P3 formal 完全相同的全模型 100 epochs、`lr0=0.001` 與其餘訓練設定重訓。 | 尚未產生 | 已完成程式、契約與 request；尚未排程、未使用 GPU。它能修正 training budget 公平性，但不能消除 initializer data exposure。 |
| **P2 Base Direct** | 使用與 B1R 相同的四尺度 graph，直接全模型訓練 100 epochs、`lr0=0.001`；比較 direct training 與 staged fine-tuning。 | 0.739114 | 比第一輪 B1 好，但仍低於 B0。 |
| **P2 Control Head** | 先只訓練新增 P2 slot/head，凍結繼承的 backbone、neck 與 P3–P5 towers，20 epochs、`lr0=0.01`。 | 0.743269 | 單獨適配新 head 有幫助；它同時是下一階段的起點。 |
| **P2 Control Full** | 從 Control Head best 初始化，全模型解除凍結再訓練 80 epochs、`lr0=0.001`。 | **0.747240** | 第二輪最佳 P2 baseline，顯示 staged head adaptation 比直接訓練好，但仍未追上 B0。 |

### P2 MFAM：高解析特徵上的模組消融

以下五個 formal 變體都把 MFAM 放在 P2，並與 P2 Control-Full/Direct 比較。

| 實驗 | 實際改動與研究問題 | Test mAP50–95 | AP_S / AP_M / AP_L | 結果怎麼看 |
|---|---|---:|---:|---|
| [**P2 PaperFormula-Full**](b1r_p2_p3_study/results/experiments/p2-paperformula-full/README.md) | 全 channels、3/5/F7/F9 四分支與論文式雙 1×1 residual fusion；測完整公式。 | 0.743312 | .535394 / .868195 / .831479 | 接近 Control Head，但低於 Control Full；完整分支未帶來淨增益。 |
| [**P2 Lite-35**](b1r_p2_p3_study/results/experiments/p2-lite-35/README.md) | 只留 DW3+DW5；測最小雙分支。 | 0.739476 | .543006 / .847917 / .827503 | 簡化成本，但精度近 Direct、低於 Control Full。 |
| [**P2 Lite-35-F7**](b1r_p2_p3_study/results/experiments/p2-lite-35-f7/README.md) | DW3+DW5+factorized 7×7；檢查 F7 是否補回大感受野。 | 0.736631 | .530559 / .834063 / .823871 | 加 F7 沒有改善，且 Ball FP 最高。 |
| [**P2 Partial50-35**](b1r_p2_p3_study/results/experiments/p2-partial50-35/README.md) | 前 50% channels 經 DW3+DW5，其餘 exact bypass。 | 0.741984 | .549337 / .841836 / .827648 | 比 Lite-35 略好，但仍低於 P2 Control Full。 |
| [**P2 Partial25-35**](b1r_p2_p3_study/results/experiments/p2-partial25-35/README.md) | 前 25% channels 經 DW3+DW5，其餘 bypass；測更低成本比例。 | 0.742383 | .545146 / .861847 / .830756 | P2 partial 中最佳 overall，但未證明 P2 MFAM 優於 control。 |

### P3 MFAM：不增加完整 P2 Detect head

以下五個 formal 變體保留 B0 的三尺度 Detect，只在 P3 slot 改特徵；它們直接回答「是否能用低成本 P3 改造取代昂貴 P2」。

| 實驗 | 實際改動與研究問題 | Test mAP50–95 | AP_S / AP_M / AP_L | 結果怎麼看 |
|---|---|---:|---:|---|
| [**P3 PaperFormula-Full**](b1r_p2_p3_study/results/experiments/p3-paperformula-full/README.md) | P3 全 channels、3/5/F7/F9 與論文式 fusion。 | 0.750617 | .557967 / .871743 / .827096 | 高於所有 P2 MFAM，但成本高於 partial P3。 |
| [**P3 Lite-35**](b1r_p2_p3_study/results/experiments/p3-lite-35/README.md) | P3 只用 DW3+DW5。 | 0.727651 | .525245 / .826482 / .805815 | 簡化過度，本輪最差的 P3 變體。 |
| [**P3 Lite-35-F7**](b1r_p2_p3_study/results/experiments/p3-lite-35-f7/README.md) | P3 使用 DW3+DW5+factorized 7×7。 | 0.740361 | .537943 / .847150 / .825202 | F7 比 Lite-35 有效，但仍低於完整公式與 partial。 |
| [**P3 Partial50-35**](b1r_p2_p3_study/results/experiments/p3-partial50-35/README.md) | P3 前 50% channels 經 DW3+DW5，其餘 bypass。 | 0.743664 | .571497 / .851336 / .825575 | validation 很強，但 test 低於 Partial25；單 seed 下不可過度解讀。 |
| [**P3 Partial25-35**](b1r_p2_p3_study/results/experiments/p3-partial25-35/README.md) | P3 前 25% channels 經 DW3+DW5，其餘 bypass。 | **0.754951** | **.576821 / .877733 / .845235** | 新增模型中 overall、small-object AP 與成本折衷最佳；仍比 B0 低 0.015861。 |

第二輪完整的 validation/test、class AP、FP 與硬體成本在[正式結果報告](b1r_p2_p3_study/results/REPORT.md)，所有原始欄位在 [`comparison.csv`](b1r_p2_p3_study/results/comparison.csv)，每個模型的 metrics/profile/lineage 在[逐模型索引](b1r_p2_p3_study/results/experiments/README.md)。

## 這些實驗目前支持的結論

1. **B0 好不代表新增架構一定較差。** B0 的 initializer 已看過 BBT5，且訓練歷史與本次預算不同；它只能表示既有權重的操作性水準。
2. **重新訓練方式確實影響 P2 baseline。** Direct 0.739114，先訓練 head 再全解凍的 Control Full 為 0.747240；差距縮小，但仍未完全解釋 B0 的優勢。
3. **新增 P2 不會自動變好。** 高解析 head 增加特徵與候選框，也可能增加最佳化難度與誤報；本次所有 P2 MFAM 都未超過 Control Full。
4. **P3 Partial25-35 是目前最合理的新候選。** 它不增加完整 P2 Detect head，只處理 25% P3 channels，test mAP50–95 為 0.754951，第二輪新增模型中最高且 GFLOPs 僅 67.779。
5. **仍需多 seed 與乾淨 initializer。** 目前每個 formal 只有 seed 42，且共同 initializer data-exposed；結論只能用於本資料與本權重來源的操作性決策。

## 待執行：B0-Fair

B0-Fair 的實作與 request 已完成，但不會自動啟動 GPU。確認 GPU 空閒後，才依 [`configs/retest/requests/README.md`](configs/retest/requests/README.md) 的命令執行。完成後應使用與第二輪相同的 val/test、profile 與報告流程，主要公平比較 `B0-Fair`、`P3 PaperFormula-Full`、`P3 Partial50-35` 與 `P3 Partial25-35`。

## 完整證據入口

- [Phase 1 完整中文報告](EXPERIMENT_RESULTS_ZH.md)
- [Phase 1 正式 artifacts 與權重](artifacts/static-phase1/README.md)
- [第二輪完整執行流程](b1r_p2_p3_study/EXPERIMENT_PROCESS.md)
- [第二輪完整結果與分析](b1r_p2_p3_study/results/REPORT.md)
- [第二輪逐模型 metrics、profile、lineage](b1r_p2_p3_study/results/experiments/README.md)
- [第二輪 checkpoint 保存狀態](b1r_p2_p3_study/results/weights/CHECKPOINT_STATUS.md)
