# Study results

這裡是正式成果的穩定入口；檔案由 `artifacts/b1r-p2-p3-retest/` 的 runtime
產物連結而來，不重複複製大型 checkpoint。

- `REPORT.md`：完整中文報告。
- `comparison.csv` / `summary.json`：28 筆 val/test 統一比較。
- `metrics/`：每個模型的 AP、Ball/Bat、AP_S/M/L 與 diagnostics。
- `weights/`：formal best checkpoint 索引，不含 smoke 或 last checkpoint。
- `profiles/`：Params、GFLOPs、activation 與 feature traffic。
- `metadata/`：queue state、lineage、final audit、設定摘要。

- `LEGACY_RESULTS.md`：第一輪 Phase 1 的歷史摘要與精確入口。
- `legacy/`：第一輪原始報告與產物索引；不複製大型 checkpoint。

結果分層：`REPORT.md`、`comparison.csv` 與 `summary.json` 是第二輪 B1R/P2/P3 重測；`LEGACY_RESULTS.md` 是第一輪 M0–M7、P3M、SP2/SP2P 與 B0/B1 紀錄。兩輪使用相同 BBT5 來源但訓練批次與模型集合不同，報告中的排名不跨輪合併。
