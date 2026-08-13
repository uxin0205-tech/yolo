# 執行 metadata

- `queue_state.json`：GPU 單工排程最終狀態 `formal_complete`。
- `requests/`：每個 stage 的 family、variant、source、parent 與設定。
- `worker/`：每個 stage 的 best/last 原路徑與 SHA-256。
- `checkpoints.json`：14 個正式評估模型的 checkpoint lineage。
- `final_audit.json`：原始稽核結果；28 metrics，profile 計數 15 是 14 個逐模型 JSON 加 1 個 summary JSON。
- `RESULTS_SCOPE.md`：GitHub 發布與排除範圍。

原路徑保留是為了 provenance；路徑不存在不代表 metrics 被重算，checkpoint 實際可用狀態請看 [`../weights/CHECKPOINT_STATUS.md`](../weights/CHECKPOINT_STATUS.md)。
