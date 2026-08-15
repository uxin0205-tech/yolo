# 正式報告索引

本目錄整理目前已完成的 YOLO26m Hardware-Friendly Attention 演算法模擬結果。原始證據仍以 `../artifacts/queue/queue.json`、`../artifacts/queue/events.jsonl` 與各 run artifact 為準；本目錄不覆寫原始結果。

- [REPORT.md](REPORT.md)：完整結果、方法比較、運算量與研究結論。
- [TRAINING_AUDIT.md](TRAINING_AUDIT.md)：17 個正式 training run 的 checkpoint、epoch、數值與 provenance 稽核。
- [COMPUTE_AND_SIZE.md](COMPUTE_AND_SIZE.md)：公式、修正後運算量、memory proxy、參數與 checkpoint 大小。
- [CLEANUP.md](CLEANUP.md)：中間產物永久清理與保留 checkpoint。
- [comparison.csv](comparison.csv)：主要候選的機器可讀比較表。
- [compute_and_size.csv](compute_and_size.csv)：修正後逐方法 selected operation counts。
- [checkpoint_sizes.csv](checkpoint_sizes.csv)：實際 checkpoint/tensor bytes。
- [figures/](figures/)：主線、normalization、accuracy-cost 與大小 SVG。
- [summary.json](summary.json)：最終 winner、baseline、限制與稽核摘要。

閱讀順序建議：先看 `TRAINING_AUDIT.md` 判斷資料是否可信，再看 `REPORT.md` 的研究比較；需要自行畫圖或重算表格時使用 CSV/JSON。

目前主線已完成至 A-FINAL；Optional Phase Q 仍鎖住，沒有執行 PTQ、QAT、INT5/6/8、ternary 或 SD4。這裡的 mAP 是目前 repository evaluator 的 Ultralytics internal metric，不是假稱為官方 COCO API AP。analytical profile 是固定 shape 假設下的 proxy，不是 RTX 5090、FPGA latency 或 energy 實測。

本目錄應提交 Git；原始 Ultralytics 中間圖與 queue logs 已依核准刪除，保留的五個階段 checkpoint 仍在 `../artifacts/runs/`。
