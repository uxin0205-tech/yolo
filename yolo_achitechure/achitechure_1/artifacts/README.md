# Artifacts

Generated checkpoints, runs, profiles, validation outputs, and curves belong here and are ignored. The committed
2026-08-17 record preserves the historical contention-related OOM; the 2026-08-18 record and profile JSON files
show that both unchanged fixed-batch probes passed during that check。2026-08-19 的 RTX 5060 Ti preflight 與
AMP physical batch-16 capacity probe 已通過；development probes 使用 hardware-tagged 檔名，且未覆寫
RTX 5090 報告。這些 probe 不是正式 accuracy training。Peak VRAM 只保留作容量診斷，不參與架構排名。

目前完成的 `fraction=0.3` B／C、`fraction=1.0` B／C queue state 分別位於
`artifacts/queues/fraction03-*`、`fraction10-phase-b-*`、`fraction10-phase-c-*`。每個正式 run 的
`resource-telemetry.jsonl` 會記錄定期 RAM/VRAM snapshot 與 epoch cleanup 前後狀態；這些 checkpoint、
validation、curve 與 telemetry 都是 runtime artifacts，不加入 Git。經稽核的精簡結果與 SHA 請見
`../results/rtx5060ti-final-0824.*`；完整訓練圖與精度證據已複製到 `../final/reports/`。

2026-08-22 報告依使用者要求不刪除、不搬移也不重新整理本目錄。30% Full35 B 曾在斷電後接續，其
telemetry 有一段 NUL padding；原始檔保留作鑑識，報告已明確記載，不應以文字編輯器重存覆蓋。

2026-08-24 收尾同樣沒有刪除 runtime artifacts。完整資料 Phase C 最終 queue status 為 `completed`；
Full35／Partial75 均未發生 OOM，且都由 gate rollback 到 A2。
