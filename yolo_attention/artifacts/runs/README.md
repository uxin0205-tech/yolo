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

原始 training run 含 `results.csv`、`best.pt` 與 `last.pt`。最終清理後只保留 `v1-br`、`n1-shift`、`bdcn-v3-learn` 的 `best.pt`；所有 `last.pt`、階段中間權重與 V2 負結果權重均已刪除。CSV、metrics、config、profile 與 manifest provenance 仍保留；evaluate-only variant 由 `../../queue/generated/` 追溯。

已清理 run 視為 archived completed result，不再支援原地 retry。run 不可覆寫或手改為成功；清理明細見 `../../../reports/CLEANUP.md`。
