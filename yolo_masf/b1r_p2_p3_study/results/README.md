# Study results

這裡是正式成果的穩定入口；檔案由 `artifacts/b1r-p2-p3-retest/` 的 runtime
產物連結而來，不重複複製大型 checkpoint。

- `REPORT.md`：完整中文報告。
- `comparison.csv` / `summary.json`：28 筆 val/test 統一比較。
- `metrics/`：每個模型的 AP、Ball/Bat、AP_S/M/L 與 diagnostics。
- `weights/`：formal best checkpoint 索引，不含 smoke 或 last checkpoint。
- `profiles/`：Params、GFLOPs、activation 與 feature traffic。
- `metadata/`：queue state、lineage、final audit、設定摘要。
