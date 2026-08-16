# Clean BBT5 正式證據

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
