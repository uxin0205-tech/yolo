# 第五階段交接

本輪 HOG 不採用，主線保留 AP 0.514620791 的原 parent。完整訓練一輪後觸發球棒精度停止條件，訓練後剝除與全量重載驗證已完成。

- [正式結果](../../artifacts/stage5/formal-20261021-v1/RESULTS.md)
- [選定模型、負結果與 SHA](handoff.json)
- [八項 AP](metrics.csv)
- [差值圖](delta.svg)

selected.pt／selected.json 連到既有 Stage3 主線，使用 stage3 loader，不額外複製權重。依使用者後續要求，rejected-epoch01.pt 與 HOG resume 已刪除；manifest、SHA、八項 AP 與驗證紀錄保留，handoff 的 candidate／resume 為 null，不能直接續跑此候選。原 MASF 結果全部保留。第六階段尚未啟動。

見 [逐檔清理清單](../../artifacts/stage5/formal-20261021-v1/resource-cleanup-20261021.json) 與 [中文清理紀錄](../../../docs/worklogs/2026-10-21-hog-resource-cleanup.md)。
