# Results

## 目前報告

- `rtx5060ti-half-0822.md`：截至 2026-08-22、排除仍在跑的完整資料 Phase C 之階段性結論。
- `rtx5060ti-half-0822.json`：結論、來源 SHA、契約與 audit 狀態。
- `rtx5060ti-half-0822.csv`：A0、A2、30% B/C、100% B 的逐列 COCO 比較。
- `results-template.csv`：A0/A1/A2/T1–T3 最終交付欄位；尚缺正式 latency 與後續 tuning。
- `../final/reports/FULL35_PARTIAL75_AP.md`：9 個保存候選在 COCO2017 與 BBT5 detect_dataset 的完整 AP
  矩陣；包含後續補跑的 100% Phase B BBT5 結果與 canonical COCO API 逐類 AP。

目前 Full35／Partial75 accepted checkpoint 仍是各自 A2。30% B/C 與完整資料 B 都沒有通過 child gate；
不能把訓練 child 的 `best.pt` 當成正式 winner。完整資料 Phase C 的 Full35 嘗試未產生可驗證
checkpoint 就失敗，因此不屬於已完成比較；不得把不完整 epoch 追加成既有結論。

歷史 capacity probes 與正式 accuracy runs 必須分開解讀。Peak VRAM 僅作容量診斷；最終 comparable
latency 必須讓候選在同一張 GPU、相同設定下量測，並依 selector 契約使用 FP16 p50。
