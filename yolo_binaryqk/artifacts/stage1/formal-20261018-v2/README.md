# 第 1 階段 v2 正式隊列

2026-10-18 12:46 啟動，23:10 完成並停止。輸入為同一份原始 YOLO26m 與完整 COCO 資料；B1／B2／B3 各完成 10 epochs Attention 初篩，選定 B2；其全模型恢復 5 epochs 無改善，依 patience 停止。未進入第 2 階段。

`supervisor.json` 是目前程序狀態，`monitor.jsonl` 每 600 秒記錄 GPU／進度；各 run 的 `status.json`、`progress.jsonl` 記錄 batch 邊界進度。`source.json` 鎖定研究程式，避免隊列中途默默換實作。

已提供 [結果與分析](RESULTS.md)、[完整比較 JSON](comparison.json)、[CSV](comparison.csv)、[曲線](curves.svg)、[handoff](handoff.json) 與 [CPU 交付驗證](handoff-verification.json)。最終 `recovery/epoch-00.pt` 保留 B2 初篩第 10 epoch 的相同模型狀態，新增 phase 交接血緣；不是恢復訓練改善後的模型。Checkpoint／必要 resume 僅本機保存，不自動提交 Git；小型數據與報告可用於交付。來源、預算、停止條件與限制見 [protocol](../../../docs/stage1-protocol.md)。
