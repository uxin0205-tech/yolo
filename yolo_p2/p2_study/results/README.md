# 正式成果索引

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../../docs/cleanup-v1/README.md)。


此目錄是已接受的 YOLO11m-P2 實驗封存，不由重新訓練排程覆寫。

- [REPORT.md](REPORT.md)：完整中文實驗報告。
- [comparison.png](comparison.png)：A0/A1/A2 指標與速度比較圖。
- [comparison.csv](comparison.csv)：絕對值與相對 A0 差異。
- [summary.json](summary.json)：機器可讀摘要，包含 seed 0。
- [weights/](weights/README.md)：四個有效 checkpoint。
- [metrics/](metrics/README.md)：COCO、benchmark 與正式訓練歷史。
- [metadata/](metadata/README.md)：環境、排程狀態、設定與 SHA-256 manifest。

如需重跑，輸出應寫入 `../artifacts`；確認結果後再人工封存，避免覆寫本目錄。
