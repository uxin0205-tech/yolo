# 權重索引

此資料夾是交付包的唯一正式權重入口。`bittrue/` 是所有 AP 排名實際使用的 Bit-True PWL checkpoint；`float/` 是可接續訓練或重新 materialize 的 best checkpoint。

請先查 `index.json`（完整 machine-readable metadata）或 `index.csv`（快速篩選）。每筆都含架構、phase、fraction、seed、gate 狀態、parent、用途、SHA256 與主要 COCO／BBT5 指標。

Ultralytics `last.pt` 只用於同一 run 的斷電恢復，不是可發布候選，因此未重複複製到本資料夾；其原始路徑與 resume 紀錄保存在 `../reports/training/*/training-complete.json`。
