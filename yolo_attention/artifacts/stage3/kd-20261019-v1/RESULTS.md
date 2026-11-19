# 使用者覆核接續

2026-10-19 使用者接受第五輪小幅提升；已從正式 resume 匯出 EMA 並完整重載 val5000，八項 AP 完全一致。現行選定 AP50–95 **0.514620791**，交接見 [result/stage3](../../../result/stage3/README.md)。下文保留原先自動 +.001 門檻的判定與負結果，不改寫事前規則。

# Stage 3 KD 最終結果

2026-10-19 **16:48:05** 完成，五個 queue jobs 全部成功，總流程約 2 小時 45 分 36 秒。17:33 核對 GPU 已無 compute process，僅顯示系統約 611 MiB。

**KD 有小幅提升，但未達事前採用門檻，主線維持原 D1。** 第 5 輪是最高 overall AP 的 KD 結果：0.514189349 → **0.514620791**，增加 **0.000431442（0.04314 個百分點）**；事前要求至少 +0.001（0.1 個百分點）。八項 AP 各不下降超過 0.001 的條件已通過，唯一未通過的是 overall 最低提升門檻。不更改事前門檻，不追加訓練；Stage 4 未開始。

## 同口徑八項 AP

| 指標 | 原 D1 | KD 第 5 輪 | 差異 |
| --- | ---: | ---: | ---: |
| 整體 AP50 | 0.687536147 | 0.688156664 | +0.000620518 |
| 整體 AP50–95 | 0.514189349 | 0.514620791 | +0.000431442 |
| Person AP50 | 0.865018885 | 0.865282951 | +0.000264066 |
| Person AP50–95 | 0.643273092 | 0.643206570 | -0.000066522 |
| Ball AP50 | 0.716171605 | 0.719326365 | +0.003154760 |
| Ball AP50–95 | 0.512681031 | 0.513224007 | +0.000542975 |
| Bat AP50 | 0.721461724 | 0.724143671 | +0.002681948 |
| Bat AP50–95 | 0.486021483 | 0.487594089 | +0.001572606 |

全部使用相同 COCO2017 val5000、E2E FP32、conf=0.001、IoU=0.7、max_det=300 與官方 COCO bbox evaluator。各輪完整數據見 [comparison.csv](comparison.csv)，曲線見 [eight-ap.svg](eight-ap.svg)，核對紀錄見 [final-review.json](final-review.json)。第 2 輪 Bat AP50–95 曾達 0.490078565，但 overall 只有 0.513947140，不能把不同輪次的最佳欄位拼成一個模型。

## 採用模型與保留候選

主線交接仍是 [原 D1](../../../result/stage3/README.md)，checkpoint SHA `a691a600715b58bf7d1438e379022443fde784c95ae75bf13336da28cc728a11`。原始權重及固定常數沒有被 KD 覆蓋。`comparison.json` 的 `best_epoch=0`／`verified_best` 是通過事前門檻的選定模型，並不代表 KD 訓練沒有變化。

第五輪實際 live／EMA、optimizer、scaler 與配置保存在 `training/resume.pt`，SHA `41697db3c5ad1869add2074fd5bed59c37254b2b2db77a59a2c54dd5e373c2c3`，沒有刪除這份唯一 KD 權重。它不是普通 inference checkpoint；載入需使用 `stage3_checkpoint` 對應的 D1-KD 架構／manifest，再嚴格載入 `ema_state` 或 `live_state`。本輪 `training/epoch-00.pt` 是回退點，不能誤稱為已訓 KD 最佳。

第 5 輪 AP 來源是訓練結束時記憶體中的 EMA 完整 val5000。事後已以 CPU 嚴格重載最後 live／EMA，確認兩處 Attention、固定 S1 affine、PWL／D1 reciprocal、activation 與 provenance 契約相同，64×96／96×64 前向有限。**另一次獨立 GPU reload val5000 驗證的對象是選定的 D1，而不是未採用的第 5 輪 KD。** 兩者證據分開記錄，不把 CPU forward 稱為任務精度驗證。

## 訓練、部署與限制

本輪先參考 optimize 的 spatial attention transfer 及負結果，採單一原始 YOLO26m 教師、layer 16／19／22 特徵，GT bat 遮罩屬專案延伸。四個 train-only batches 校準 μ=1.274051769；兩個 AMP macro smoke 通過後，從原 D1 跑五輪，每輪完整 train118287／val5000，無額外 control 加訓。前四輪 overall 都略低於 D1，最後一輪小幅上升；不能據此斷言 KD 普遍有效或無效。

學生一直保留 B2 Binary Q/K、S1 Q10／bias、integer PWL exp 與 D1 reciprocal 前向；固定常數與 BN buffers 不變，教師不進學生 state／EMA／optimizer／匯出。僅 KD 損失的訓練運算使用浮點 normalization。部署結構與原 D1 相同，沒有額外教師推論成本；QKV／P·V／projection 等仍浮點，qSiLU 留待指定的 Activation 階段建立。沒有 RTL／HLS 或板端測速，不能宣稱實測硬體加速。

全部五輪 AMP retry=0；背景監測共 17 次，最大間隔約 600.11 秒。程式／參考 SHA 核對通過，optimize 未修改。既有 202＋10 項測試通過；本次收尾只改報告與狀態，未重跑無關測試或啟動 GPU。單 seed、無相同加訓量 control，無法分離 KD 與額外訓練的獨立效果，也不將微小 AP 差異當成統計優勢。

## 清理與重畫

已自動刪除兩份 smoke checkpoint，釋放 **277.53 MiB**，逐檔 SHA 見 [smoke-cleanup.json](smoke-cleanup.json)。必要正式 resume、原 D1、原始指標、來源 snapshot 與 CSV 保留，沒有 archive／backup。

曲線由 `python -m yolo_attention.cli stage3-kd plot --output ...` 重畫；資料僅需 `training/result.json`，不需 GPU 或 checkpoint。完整配方見 [KD 協定](../../../docs/stage3-recovery-kd.md)。
