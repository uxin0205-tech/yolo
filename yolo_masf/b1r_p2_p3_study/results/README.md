# 第二輪正式結果

本目錄是 GitHub 可直接瀏覽的正式發布包，不依賴本機 `artifacts/b1r-p2-p3-retest/`。

| 內容 | 入口 |
|---|---|
| 完整中文報告 | [REPORT.md](REPORT.md) |
| 逐模型做法與精確結果 | [experiments/](experiments/README.md) |
| 所有 val/test CSV | [comparison.csv](comparison.csv) |
| 所有 val/test JSON | [summary.json](summary.json) |
| 靜態硬體成本 | [profiles/](profiles/README.md) |
| Queue、request、worker、audit | [metadata/](metadata/README.md) |
| Checkpoint 狀態 | [weights/](weights/README.md) |
| 發布檔案 SHA-256 | [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) |
| 第一輪歷史摘要 | [LEGACY_RESULTS.md](LEGACY_RESULTS.md) |

正式比較使用 14 個模型 × val/test=28 份 metrics；profile 共 14 份逐模型 JSON；final audit 的 15 是 14 個模型加 1 個 summary JSON。Raw predictions、false-positive 圖與 queue logs 不作為本目錄的 GitHub 瀏覽依賴。
