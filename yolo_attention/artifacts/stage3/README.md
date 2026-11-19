# 整合研究第 3 階段

**已完成並停止：D1 官方 AP50–95 0.514189349，無加訓。** 選定交付在 [result/stage3](../../result/stage3/README.md)，9 個完整 AP、3 個診斷排除；GPU 已釋放。[恢復／KD 判斷](../../docs/stage3-recovery-kd.md) 已完成，未啟動 KD 或第 4 階段。

從 `yolo_binaryqk/result/stage2/epoch-05.pt` 的 S1 Q10 出發，在 `yolo_attention` 比較 N1～N9 normalization 與 D1～D3 完整 PWL／reciprocal。實作前來源、SHA、數值門檻與預算見 [reference-review.json](reference-review.json)。

此處保存 CPU 定點 reference、真實資料診斷、完整 AP、運算分析及交付。單一 CLI 為 `python -m yolo_attention.cli`，GPU 工作沿用 QueueExecutor；舊研究隊列不重跑。小型 JSON／CSV／SVG／文件可提交 Git，權重及資料不提交。完成第 3 階段後停止。

- [數值 reference](cpu-reference.json)、[CPU 驗證](cpu-verification.json)。
- [正式評估](formal-20261019-v1/README.md)、[啟動](formal-launch.json)、[即時進度](formal-20261019-v1/progress.json)、[背景狀態](formal-20261019-v1/supervisor.json)。
- [重複程式清理與相容驗證](cleanup-core-duplicates.json)。
- [舊產物實際清理](cleanup-final-delivery.json)、[歷史小型證據索引](historical-evidence-index.json)。
