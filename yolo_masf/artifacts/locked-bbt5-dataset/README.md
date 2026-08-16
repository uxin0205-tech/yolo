# Locked BBT5 Dataset Evidence

本目錄保存本輪 clean BBT5 實驗實際使用的資料切分與可追溯證據，不是另一份影像資料集。

- `data.yaml`：統一驗證時使用的 locked dataset view。
- `train.txt`、`val.txt`、`test.txt`：固定影像清單；`test` 僅作 historical test，禁止參與訓練與選模。
- `manifest.json`、`audit.json`：資料清單、雜湊與稽核結果。
- `dataset_profile.json`：資料集統計摘要。
- `*.coco.json`：統一 COCO 指標計算所用標註。
- `*_hist.png`：bbox 與特徵層尺度分布圖。

本輪 locked dataset SHA-256 為 `6e16c975941dbae2af174a2eb4b5424bffd4736c74aad56d424805da019b8fbc`。原始資料仍位於 `bbt5-detect-baseline/dataset/`；本目錄只保存可重現實驗所需的索引、標註轉換與統計證據。

完整實驗方法、逐 seed 指標與結論請見 [`clean_bbt5_study/results/REPORT.md`](../../clean_bbt5_study/results/REPORT.md)。
