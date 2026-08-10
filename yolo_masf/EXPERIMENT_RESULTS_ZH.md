# MASF-YOLO Phase 1 實驗結果與分析

更新日期：2026-08-10

## 1. 實驗狀態與口徑

- Pipeline ID：`cb7e0392cc32`
- 最終狀態：34 個階段全部完成，final audit 通過，錯誤數為 0。
- 資料：`bbt5-detect-baseline/dataset`，共 2,578 個 unique frames；train/val/test 為 1,987/300/291 frames。
- 物件數：Ball 為 2,696/414/503，Bat 為 4,178/685/590；group overlap 與 hash overlap 均為空。
- 輸入尺寸 640、batch 16、seed 42、SGD、AMP；profiling 為 RTX 5090、FP16、batch 1、100 warmups、1,000 iterations。
- 下表主要準確率採 `faster-coco-eval 1.7.2` 對輸出 predictions 計算的 COCO-style mAP，不與 `metrics.json` 內嵌的 Ultralytics fitness 欄位混用。
- 所有模型都繼承曾看過 BBT5 的 pose-derived initializer，因此結果是 data-exposed operational ablation，不是 leak-free generalization estimate。B0 另外禁止參與 M2/M3 選模。

## 2. 變體定義

| 模型 | 主要差異 |
|---|---|
| B0 | 既有三尺度 P3/P4/P5 detect 權重，未做本次同預算重訓 |
| B1 | P2 baseline；先凍結 backbone 0–10 共 10 epochs，再全解凍 90 epochs |
| M7 | P2 全 channels：DW3、DW5、DW(1×7→7×1) |
| M0 | P2 全 channels：DW3、DW5、DW7、DW9 |
| M1 | P2 全 channels：DW3、DW5 |
| M2 | P2 只有 1/2 channels 通過 M1 |
| M3 | P2 只有 1/4 channels 通過 M1 |
| P3M | P2 Identity；只在 P3 放 DW3、DW5、DW(1×7→7×1)，不含 9×9 |
| SP2 | 只有 Ball 使用 hidden=32 的輕量 P2 head；Bat 維持 P3/P4/P5 |
| SP2P | SP2 加上 validation 選出的 M2 partial-MFAM |

## 3. Validation 與 Test 結果

Test 的 Ball/Bat AP 均指 AP50–95。括號內為 test overall 相對 B1 的百分點差。

| 模型 | Val mAP50–95 | Test mAP50–95 | 相對 B1 | Ball AP | Ball AP50 | Bat AP | Bat AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0* | 0.7508 | 0.7708 | +4.048 | 0.6784 | 0.8876 | 0.8632 | 0.9891 |
| B1 | 0.7081 | 0.7303 | 基準 | 0.6126 | 0.8281 | 0.8481 | 0.9846 |
| M7 | 0.7079 | **0.7386** | **+0.823** | **0.6324** | **0.8847** | 0.8447 | 0.9864 |
| M0 | **0.7108** | 0.7234 | -0.689 | 0.6113 | 0.8373 | 0.8356 | 0.9832 |
| M1 | 0.7095 | 0.7263 | -0.399 | 0.6162 | 0.8301 | 0.8365 | 0.9830 |
| M2 | 0.7047 | 0.7250 | -0.528 | 0.6089 | 0.8255 | 0.8412 | 0.9831 |
| M3 | 0.7020 | 0.7262 | -0.410 | 0.6143 | 0.8285 | 0.8381 | 0.9781 |
| P3M | 0.6990 | **0.7340** | **+0.366** | **0.6248** | 0.8577 | 0.8432 | **0.9866** |
| SP2 | 0.7056 | 0.7170 | -1.330 | 0.5970 | 0.8161 | 0.8370 | 0.9766 |
| SP2P | 0.6971 | 0.7213 | -0.906 | 0.6047 | 0.8459 | 0.8378 | 0.9817 |

\* B0 是資料暴露且訓練預算不同的參考值，不能當成公平勝者。粗體只標示公平消融中值得注意的 test 結果；所有實驗只有單一 seed。

## 4. Test 操作性指標

| 模型 | Ball Precision | Ball Recall | Ball FP | Ball Missed | Bat Precision | Bat Recall | Bat FP | Bat Missed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0* | 0.2876 | 0.9662 | 1,204 | 17 | 0.5480 | 0.9966 | 485 | 2 |
| B1 | 0.2338 | 0.9702 | 1,599 | 15 | 0.3575 | 0.9932 | 1,053 | 4 |
| M7 | 0.1399 | 0.9702 | 3,000 | 15 | 0.2476 | 0.9932 | 1,781 | 4 |
| M0 | 0.1762 | 0.9702 | 2,281 | 15 | 0.3209 | 0.9932 | 1,240 | 4 |
| M1 | 0.1784 | 0.9682 | 2,243 | 16 | 0.3401 | 0.9932 | 1,137 | 4 |
| M2 | 0.1985 | 0.9702 | 1,971 | 15 | 0.3470 | 0.9932 | 1,103 | 4 |
| M3 | 0.2013 | 0.9662 | 1,928 | 17 | 0.3452 | 0.9881 | 1,106 | 7 |
| P3M | 0.1716 | 0.9662 | 2,346 | 17 | 0.3745 | 0.9966 | 982 | 2 |
| SP2 | 0.0667 | **0.9821** | 6,908 | 9 | 0.3712 | 0.9864 | 986 | 8 |
| SP2P | 0.0661 | **0.9821** | 6,983 | 9 | 0.3072 | 0.9915 | 1,319 | 5 |

## 5. 硬體結果

延遲是在同一張 RTX 5090 上實測；GFLOPs 與參數量較低不保證 latency 較低。

| 模型 | GFLOPs | Params (M) | Mean latency (ms) | P95 (ms) | 相對 B1 latency |
|---|---:|---:|---:|---:|---:|
| B0* | 67.646 | 20.055 | 3.729 | 3.827 | -19.26% |
| B1 | 87.431 | 20.551 | 4.619 | 4.770 | 基準 |
| M7 | 88.584 | 20.575 | 4.765 | 4.935 | +3.17% |
| M0 | 88.702 | 20.577 | 4.683 | 4.841 | +1.39% |
| M1 | 88.492 | 20.572 | 4.666 | 4.832 | +1.03% |
| M2 | 87.752 | 20.558 | 4.647 | 4.813 | +0.63% |
| M3 | 87.539 | 20.553 | 4.623 | 4.755 | +0.09% |
| P3M | 88.427 | 20.631 | 4.715 | 4.841 | +2.09% |
| SP2 | **80.528** | **20.416** | 4.820 | 4.943 | **+4.35%** |
| SP2P | 80.849 | 20.422 | 4.972 | 5.118 | **+7.66%** |

SP2 相對 B1 的 GFLOPs 減少 7.89%、參數減少 0.66%，但 mean latency 增加 4.35%。SP2P 的 GFLOPs 減少 7.53%，但 mean latency 增加 7.66%。目前只能宣稱理論計算量下降，不能宣稱在 RTX 5090 上更快。

## 6. M2/M3 選模與 SP2P

- M2 validation mAP50–95 為 0.704737，M3 為 0.702028，差 0.002709。
- 預定的效率等價門檻是 mAP 差不超過 0.002 且 Ball Recall 差小於 0.01；本次超過 mAP 門檻，因此依品質排序選 M2。
- M3 test mAP50–95 雖比 M2 高 0.001174，但 test 在 selection 凍結後才執行，不能回頭改選 M3；這是正確的防止 test leakage 行為。
- SP2P 相較 SP2：test overall 增加 0.00424、Ball AP 增加 0.00771，但延遲再增加 0.152 ms，Ball FP 也增加 75。組合 M2 沒有解決 Selective P2 的誤報問題。
- SP2P 繼承已完成的 SP2-B 與 M2，再額外做自己的 10+90 epochs，訓練預算高於單段變體；不可把差異全歸因於架構。

## 7. 結論

1. **公平消融中，M7 是目前 test overall 最佳模型**：比 B1 高 0.823 個百分點，Ball AP 高 1.979 個百分點；代價是 latency 增加 3.17%，而 Ball/Bat FP 分別增加 1,401/728。它適合列為準確率候選，但不是直接可部署的精度改善。
2. **P3M 是較平衡的第二候選**：test overall 比 B1 高 0.366 個百分點，Ball AP 高 1.218 個百分點，Bat FP 反而少 71；但 validation 比 B1 低 0.908 個百分點，必須用多 seed 確認是否為 split 波動。
3. **M0 的 9×9 分支沒有得到支持**：validation 雖最高，但 test 比 B1 低 0.689 個百分點，成本也是 MFAM 變體最高之一。
4. **M1/M2/M3 沒有超過 B1**：partial channel 能逐步降低額外成本，M3 幾乎回到 B1 latency，但本次準確率仍略低。
5. **SP2/SP2P 未達目前目標**：它們提高 Ball recall、降低理論 GFLOPs，卻造成極大量 Ball false positives，而且實測更慢。現階段不應選為部署模型。
6. **B0 只能當 operational upper reference**：它分數與速度都最高，但權重已看過 BBT5、訓練預算不同，不能用來宣稱架構勝過公平 baseline。

## 8. 建議下一步

1. 先以 B1、M7、P3M 做至少 3 seeds 重跑；只有 M7/P3M 的平均值與變異仍優於 B1，才確認改善。
2. 若部署重視誤報，優先分析 M7/P3M 的 false-positive 類型並做固定 confidence calibration；不要只看 AP。
3. SP2 路線若繼續，先降低 P2 auxiliary loss 權重、加強 P2 hard negatives，並評估固定的 P2 Ball confidence gate；在誤報與實測 latency 改善前，不再擴大 SP2P 結構。
4. 另做一組未看過 BBT5 的乾淨 initializer 實驗，才能估計真正的泛化能力。

原始證據位於：

- `artifacts/static-phase1/report.md`
- `artifacts/static-phase1/evaluation/{val,test}/<model>/metrics.json`
- `artifacts/static-phase1/profiles/<model>/profile.json`
- `artifacts/static-phase1/selection.json`
- `artifacts/static-phase1/final_audit.json`
