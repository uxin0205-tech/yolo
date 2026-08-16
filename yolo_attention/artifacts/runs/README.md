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

原始 training run 含 `results.csv`、`best.pt` 與 `last.pt`。2026-08-15 主線清理後只保留 `h-scr`、`w-dir`、`v1-br`、`n1-shift`、`d1-shared-10` 的 `best.pt`。2026-08-16 完成的 `bdcn-v2-learn` 與 `bdcn-v3-learn` 另保留 best/last，供 defect provenance 與安全 postprocess retry 使用。Evaluate-only variant 由 `../../queue/generated/` 追溯。

run 不可覆寫或手改為成功；清理明細見 `../../../reports/CLEANUP.md`。
