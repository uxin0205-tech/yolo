# 2026-09-16 完成狀態與驗證指標

## 狀態與驗證

四階段及統一評估於台北時間 2026-09-16 20:24:28 完成。Stage 1／2／3 分別完成 20／50／40 輪；Stage 4 完成 45 輪，第 15 輪最佳，連續 30 輪未改善而提早停止。GPU 使用率 0%。

檢查 queue-status、各階段 results.csv、comparison.json 及實際 checkpoint。final 的 best、last、last-resumable SHA256 分別與 Stage 4 原件一致；最後續訓 checkpoint 為第 45 輪且保有 optimizer。未重跑評估或啟動新訓練。

## 統一 960 評估

固定 bbat5-v1 formal val 683 張、932 個物件，batch 8。全部為各階段 best.pt；避免混用 640 訓練中評估與 960 最終評估。沒有獨立 test split。

| 階段 | Box Precision | Box Recall | Box mAP50 | Box mAP50–95 | Pose Precision | Pose Recall | Pose mAP50 | Pose mAP50–95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage1_adamw_640 | 90.44% | 85.95% | 88.86% | 58.33% | 91.68% | 87.12% | 90.52% | 90.23% |
| stage2_adamw_640 | 92.58% | 82.49% | 87.99% | 59.52% | 93.59% | 83.53% | 90.22% | 89.70% |
| stage3_adamw_960 | 91.74% | 88.01% | 89.91% | 61.32% | 92.43% | 88.67% | 90.90% | 90.85% |
| stage4_musgd_960 | 94.16% | 85.59% | 90.24% | 62.43% | 94.71% | 85.97% | 90.54% | 90.51% |

## 類別指標與取捨

| 階段 | 類別 | Box mAP50–95 | Pose mAP50–95 |
| --- | --- | --- | --- |
| stage3_adamw_960 | ball | 50.71% | 86.32% |
| stage3_adamw_960 | bat | 71.94% | 95.38% |
| stage4_musgd_960 | ball | 51.36% | 86.42% |
| stage4_musgd_960 | bat | 73.50% | 94.60% |

Stage 4 對 Stage 3 的 Box mAP50–95 增加 1.11 個百分點，Pose mAP50–95 減少 0.34 個百分點，綜合平均增加 0.39 個百分點。Stage 4 為原定平均分最高候選；重視 Pose 或召回可考慮 Stage 3。這是接續訓練比較，並非同起點控制實驗，不能把差異全歸因於 optimizer。

Stage 3 最佳在第 8 輪、Stage 4 最佳在第 15 輪；後段 loss 下降未帶來驗證改善，目前沒有直接加長相同設定的證據。球的 Box mAP50–95 為 51.36%，低於球棒的 73.50%，仍是主要待改善項目。Pose mAP 不等同像素定位誤差，目前未計算每個 keypoint 的像素誤差。

## 困難與解法

先前 Stage 3 batch 32 OOM，降至 16 後順利完成。本次查詢與檔案驗證困難：無。

## 未解事項與風險

指標用於 formal val 選模，不是獨立 test 成績；未進行不同 seed 重複試驗或像素誤差分析。各階段的 last.pt 與 best.pt 均保留，最終 best.pt 指向 Stage 4 最佳，最後 checkpoint 則對應第 45 輪。
