# artifacts/runs

本目錄每個 leaf 是一個 immutable run：

~~~text
<run-id>/
├── manifest.json
├── variant.yaml
├── training.yaml
├── checkpoints/
├── metrics/queue-result.json
├── profiles/analytical.json
├── exports/
├── logs/
└── ultralytics/
~~~

原始 training run 含 `results.csv`、`best.pt` 與 `last.pt`。2026-08-15 經使用者核准清理後，所有 run 保留 CSV/metrics/config/profile provenance，但只保留 `h-scr`、`w-dir`、`v1-br`、`n1-shift`、`d1-shared-10` 的 `best.pt`，所有 `last.pt` 已刪除。Evaluate-only variant 由 `../../queue/generated/` 追溯。run 不可覆寫或手改為成功；清理明細見 `../../../reports/CLEANUP.md`。
