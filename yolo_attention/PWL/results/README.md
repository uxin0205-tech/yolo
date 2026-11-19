# PWL results

`pwl-score-analysis` 與 `pwl-compare` 已完成。此處包含：

- `REPORT.md`：range、COCO、誤差、硬體 proxy、限制與最終判斷。
- `comparison.csv`：Exact／Float／Q8.8 Bit-True paired comparison。
- `endpoint-table.csv`：選定 `[-10,0]` 的 21 個 UQ1.15 endpoint integers/hex/decoded values。
- `score-analysis.json`、`score-statistics.csv`、`score-histogram.csv`：雙 site/per-head 原始統計。
- `figures/`：score histogram 與 normalization mAP 圖。

完整 raw provenance 位於 `artifacts/runs/pwl-compare/`。所有硬體 cost 都明列為
analytical proxy，不是 FPGA/GPU latency、DSP、BRAM 或 energy 實測。
