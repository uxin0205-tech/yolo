# Artifact 清理紀錄

日期：2026-08-16

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

## BDCN V2/V3 後續整理

完成 5 個 defect-fix jobs 後，再刪除 V2/V3 的 Ultralytics batch、curve、labels、confusion-matrix 圖，以及 Python／pytest／Ruff cache 與 patch 備份。以下兩個新 training runs 保留 `best.pt` 與 `last.pt`：

- `bdcn-v2-learn`：保留失敗案例的 unconstrained table 權重，供 defect provenance 使用。
- `bdcn-v3-learn`：保留 anchored table 的 best/final 權重，供 V3-R1 與重現使用。

這兩組 `last.pt` 是 BDCN append branch 的明確例外，因 queue 的 postprocess-only retry contract 需要同時核對 `best.pt`、`last.pt` 與完整 `results.csv`。它們不是冒充的新結果，也沒有覆寫舊 run。整理後專案約 609 MB；增加量主要是四個約 55.3 MiB 的 BDCN checkpoints 與完整 queue provenance。

最新 BDCN 結果、成本與 checkpoint 用途見 [BDCN_V3_REPORT.md](BDCN_V3_REPORT.md)。
