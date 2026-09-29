# 資料處理模組

`labels.py` 嚴格解析標註；`grouping.py` 防止相鄰幀／增強 siblings 洩漏；`split.py` 產生 seed 42 固定 group split；`export.py` 轉 COCO；`audit.py` 建 manifest。輸入固定為 `bbt5-detect-baseline/dataset/` 且只讀，鎖定輸出保存在 `artifacts/locked-bbt5-dataset/`。目前 unique frames 2,578，train/val/historical test 為 1,987/300/291，group/hash overlap 為空。來源資料與鎖定清單皆保留，不因移除舊實驗結果而刪除。
