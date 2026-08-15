# Artifact 清理紀錄

日期：2026-08-15

使用者明確核准後，永久刪除可重建的訓練中間產物：

- `artifacts/invalidated/` 中除 README 外的 rewind 歷史。
- worker logs。
- 17 個 `last.pt`。
- 12 個非 winner `best.pt`。
- 2 個 calibration checkpoint。
- 289 張 Ultralytics curve、batch、labels 與 confusion-matrix 圖。
- Python／pytest／Ruff／egg-info cache。
- 非正式研究權重 `yolo26n.pt`。

專案由約 3.9 GB 降到 371 MB，約釋放 3.5 GB。保留所有正式 metrics、manifest、variant/training YAML、analytical profile、`results.csv`、queue state/events、正式報告、COCO 設定與 `weights/yolo26m.pt`。

保留的階段 checkpoint：

- `h-scr`：architecture screening winner。
- `w-dir`：recovery winner。
- `v1-br`：A0 parent。
- `n1-shift`：queue-selected A-FINAL。
- `d1-shared-10`：BDCN learned-codebook winner。

A-FINAL 清理後 SHA-256 仍為 `6bd4b25ea36bfe453f437cec6647d7d7dbab745e1e0035b1610928a454e63ce1`；官方 `yolo26m.pt` 仍為 `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`。

`TRAINING_AUDIT.md` 的 17 個 checkpoint load/finite 結論是在清理前完成；清理後只保留上述五個 checkpoint。其他 run 仍保留足以支撐報告的數值與 provenance，但不能直接從本機恢復其權重。
