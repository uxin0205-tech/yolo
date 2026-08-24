# Results

## 目前報告

- `rtx5060ti-final-0824.md`：最終結論入口；完整表格與圖直接連到 `../final/reports/`。
- `rtx5060ti-final-0824.json`：最終 selection、8 次訓練、資源與同機 latency 的 machine-readable 摘要。
- `rtx5060ti-final-0824.csv`：8 次 B/C 訓練的一列一 run 摘要。
- `rtx5060ti-half-0822.md`：截至 2026-08-22、排除仍在跑的完整資料 Phase C 之階段性結論。
- `rtx5060ti-half-0822.json`：結論、來源 SHA、契約與 audit 狀態。
- `rtx5060ti-half-0822.csv`：A0、A2、30% B/C、100% B 的逐列 COCO 比較。
- `results-template.csv`：A0/A1/A2/T1–T3 最終交付欄位；尚缺正式 latency 與後續 tuning。
- `../final/reports/RTX5060TI_FINAL_0824.md`：完整結論、訓練圖、RAM／VRAM 與正式 FP16 latency。
- `../final/reports/FULL35_PARTIAL75_AP.md`：11 個保存候選在 COCO2017 與 BBT5 detect_dataset 的完整
  AP／P／R／F1 矩陣，以及 canonical COCO API 逐類 AP。

Full35／Partial75 accepted checkpoint 仍是各自 A2。30% 與完整資料的 B/C 八個 child 都沒有通過 gate；
不能把 rollback child 的 `best.pt` 當成 accepted checkpoint。完整資料 Phase C 已完成，不再是進行中項目。

歷史 capacity probes 與正式 accuracy runs 必須分開解讀。正式 comparable latency 已在同一張 RTX 5060 Ti、
相同 FP16 batch-1 設定下量測；Peak VRAM 仍只作容量診斷。
