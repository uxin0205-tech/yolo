# artifacts/runs

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../../docs/cleanup-v1/README.md)。


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
