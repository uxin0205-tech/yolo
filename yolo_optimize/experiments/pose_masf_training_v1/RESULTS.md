# B 組 Pose MASF：訓練、驗證與最終分析

5 epoch 與完整驗證已完成。A 組已取消；本次不自動替換正式模型、不再加訓。

## 結論

B5 通過事前工程參考門檻，可列為待人工確認候選，仍不能宣稱 MASF 獨立增準。

最佳 Pose 回合是 B4；它是驗證集選出的補充候選，主要比較仍固定 B5。

沒有 A5，B5−E2 包含額外訓練與 MASF 的共同效果。B5 關閉 α 只代表已訓模型的分支依賴，不能替代無 MASF 重訓對照。

## 完整精度比較

| AP50–95 (%) | 原 E2 | B5 | B5 α=0 | 最佳 Pose 回合 | B5−E2 (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| coco/box/map50_95 | 50.2215 | 50.2215 | 50.2215 | 50.2215 | +0.0000 |
| coco/person/box/map50_95 | 62.4625 | 62.4625 | 62.4625 | 62.4625 | +0.0000 |
| bbat/box/map50_95 | 60.8301 | 60.9473 | 60.9137 | 60.7904 | +0.1172 |
| bbat/pose/map50_95 | 88.6944 | 88.9071 | 88.9096 | 88.9101 | +0.2128 |
| bbat/ball/box/map50_95 | 49.0438 | 49.0353 | 48.9918 | 48.7592 | -0.0085 |
| bbat/ball/pose/map50_95 | 84.9048 | 85.3175 | 85.3206 | 85.3193 | +0.4127 |
| bbat/bat/box/map50_95 | 72.6164 | 72.8593 | 72.8355 | 72.8216 | +0.2430 |
| bbat/bat/pose/map50_95 | 92.4839 | 92.4968 | 92.4986 | 92.5009 | +0.0128 |

[完整 CSV](<comparison.csv>)；AP50、AP75、precision、recall 與 Float 數值保存在 analysis/summary.json 及逐回合 metrics.json。

## 訓練與 MASF 曲線

![B 組曲線](<figures/b-training-curves.png>)

[逐回合 CSV](<training-curves.csv>)包含 alpha live／EMA、Pose loss 及全部 AP50–95。0 是 native QK E2 起點，不是另一組新訓練。

## 訓練契約與驗證

只訓 Pose head＋Pose P3 MASF，其他 shared／Detect 參數与全部 BN running 固定；β=1、α=0 啟動，原生 QK＋PWL [-10,0] 20 段與 qSiLU 不變。

AdamW、5 epoch、warmup 1、physical batch16×累積8、每回合5964張／47更新／尾端76張。Pose head LR5e-6、context1e-5、alpha1e-4；詳細設定見 [execution-config.json](<execution-config.json>)。

每回合完整 COCO val5000／BBAT5 v1 val683；沒有重切或抽樣。E5 另從匯出權重重建獨立驗證並核對全指標；best 不同於 E5 時另重验並關 α。Float 與 BitTrue 分開保存。

最終 CPU 稽核逐回合核對續訓 state／EMA／匯出逐項相等、固定參數與 BN 不变、235 次總更新及完整來源 SHA。

## 成本與推論使用

完整雙 head Params：26,604,466；B5 inference 檔：102.156 MiB。

[相同新增 Pose MASF 結構的既有成本量測](<../pose_masf_priority_v1/RESULTS.md>)只作參考，本次未重測 MAC／FLOPs、peak memory、CPU／GPU latency、target latency 或 energy/frame，不混成新實測。

以 training_b.build() 重建、對 epochs/e5/ema.pt 使用 weights_only=True 取 state_dict 並 strict=True 載入，再 materialize 成 bittrue；權重不是官方 YOLO 可直接猜結構的通用 pickle。alpha-off 僅關 Pose gate，Detect MASF 不變。

## 來源與剩餘限制

固定 parent SHA：`4257ca9c471736aa97d0077097526b7863a9268b773913119b9969991d00b0f3`。所有五回合 checkpoint 的 SHA 與路徑見 [最終稽核](<artifacts/b-e5-seed1-v1/final-audit.json>)。

無獨立 test／多 seed 顯著性；沒有硬體實板或能耗驗證。原 Attention E2 後續仍暫停；後續 scale/bias 與 Rep 已完成，見 [0914 更新](<../../reports/update-0914/README.md>)，A 不執行；不自動推 Git、不覆寫舊 best。
