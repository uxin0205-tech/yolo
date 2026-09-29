# Clean BBT5 正式證據

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../../docs/cleanup-v1/README.md)。


本目錄只保留可重現、稽核與比較本輪正式實驗所需的證據；smoke checkpoint、console log、訓練預覽圖、cache，以及可由正式預測重建的中間 JSON 已在 2026-08-16 清除。

- `training/`：28 個正式 stage；每組保留 `best.pt`、`last.pt`、`args.yaml` 與 `results.csv`。
- `evaluation/val/`：28 組 validation 的 `metrics.json`、`predictions.json` 與錯誤案例圖片。
- `evaluation/test/`：28 組 historical test 的同類證據；不得用於選模。
- `profiles/`：14 種實驗的參數量、GFLOPs、activation、GPU memory 與 FP16 latency。
- `audit/formal_runs.json`：checkpoint hash、lineage、參數與 fresh-process reload 稽核。
- `selection.json`：只依 validation 建立的選模凍結證據。
- `final_audit.json`：報告、指標數量與選模時序的最終一致性結果。
- `queue_state.json`、`worker/`：40 個 GPU jobs 的完成狀態及 28 個正式 worker 回條。
- `cleanup_manifest.json`：本次清理範圍與保留契約。

正式結論請從 [`clean_bbt5_study/results/REPORT.md`](../../clean_bbt5_study/results/REPORT.md) 閱讀。
