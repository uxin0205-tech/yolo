# 程式模組

目前入口見[腳本操作說明](../../scripts/README.md)，研究安排見[現行計畫](../../docs/CURRENT_PLAN.md)。

| 類型 | 主要模組 |
| --- | --- |
| 格式／圖轉換 | `weight_formats`、`qat_weights`、`qat_graph`、`qat_projection`、`lsq_plus` |
| QAT | `qat_plan`、`qat_runtime`、`qat_schedule`、`qat_optimizer`、`qat_validation` |
| 評估／證據 | `mixed_policy_search`、`search_evidence`、`metric_gate`、`weight_sensitivity` |
| 狀態監測 | `blocking_monitor`；本輪 supervisor 位於 `scripts/` |
| 歷史 queue／adapter | 舊 progressive、qsilu lane、activation 與 search 模組仍供歷史契約／重建使用，不因名字舊直接刪除 |

程式模組的存在不代表相應實驗已執行，也不代表有硬體 kernel。不同量化格式與方法粒度的差異需要測試及報告揭露。
