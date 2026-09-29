# 權重索引

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../../../docs/cleanup-v1/README.md)。


此資料夾是交付包的唯一正式權重入口。`bittrue/` 是所有 AP 排名實際使用的 Bit-True PWL checkpoint；`float/` 是可接續訓練或重新 materialize 的 best checkpoint。

請先查 `index.json`（完整 machine-readable metadata）或 `index.csv`（快速篩選）。每筆都含架構、phase、fraction、seed、gate 狀態、parent、用途、SHA256 與主要 COCO／BBT5 指標。

Ultralytics `last.pt` 只用於同一 run 的斷電恢復，不是可發布候選，因此未重複複製到本資料夾；其原始路徑與 resume 紀錄保存在 `../reports/training/*/training-complete.json`。
