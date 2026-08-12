# 正式成果索引

此目錄是已接受的 YOLO11m-P2 實驗封存，不由重新訓練排程覆寫。

- [REPORT.md](REPORT.md)：完整中文實驗報告。
- [comparison.png](comparison.png)：A0/A1/A2 指標與速度比較圖。
- [comparison.csv](comparison.csv)：絕對值與相對 A0 差異。
- [summary.json](summary.json)：機器可讀摘要，包含 seed 0。
- [weights/](weights/README.md)：四個有效 checkpoint。
- [metrics/](metrics/README.md)：COCO、benchmark 與正式訓練歷史。
- [metadata/](metadata/README.md)：環境、排程狀態、設定與 SHA-256 manifest。

如需重跑，輸出應寫入 `../artifacts`；確認結果後再人工封存，避免覆寫本目錄。
