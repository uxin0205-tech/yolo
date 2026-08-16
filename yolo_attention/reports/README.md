# 正式報告索引

本目錄整理目前已完成的 YOLO26m Hardware-Friendly Attention 演算法模擬結果。原始證據仍以 `../artifacts/queue/queue.json`、`../artifacts/queue/events.jsonl` 與各 run artifact 為準；本目錄不覆寫原始結果。

- [REPORT.md](REPORT.md)：完整結果、方法比較、運算量與研究結論。
- [BDCN_V3_REPORT.md](BDCN_V3_REPORT.md)：BDCN 缺陷、V2/V3 每一步做法、正式結果、成本與最終判定。
- [TRAINING_AUDIT.md](TRAINING_AUDIT.md)：17 個正式 training run 的 checkpoint、epoch、數值與 provenance 稽核。
- [COMPUTE_AND_SIZE.md](COMPUTE_AND_SIZE.md)：整合原始 Attention、A-FINAL、全模型占比、逐項節省、memory proxy、參數與 checkpoint 大小。
- [operation_savings.csv](operation_savings.csv)：原始／最終／差值的機器可讀運算量表。
- [CLEANUP.md](CLEANUP.md)：中間產物永久清理與保留 checkpoint。
- [comparison.csv](comparison.csv)：主要候選的機器可讀比較表。
- [compute_and_size.csv](compute_and_size.csv)：修正後逐方法 selected operation counts。
- [checkpoint_sizes.csv](checkpoint_sizes.csv)：實際 checkpoint/tensor bytes。
- [figures/](figures/)：主線、normalization、accuracy-cost 與大小 SVG。
- [summary.json](summary.json)：最終 winner、baseline、限制與稽核摘要。
- [bdcn_v3_results.csv](bdcn_v3_results.csv)：BDCN V2/V3 的機器可讀 metrics、diagnostics 與成本。

閱讀順序建議：先看 `TRAINING_AUDIT.md` 判斷資料是否可信，再看 `REPORT.md` 的研究比較；需要自行畫圖或重算表格時使用 CSV/JSON。

目前主線與 BDCN V3 branch 均已完成（57/57 queue jobs succeeded）；Optional Phase Q 仍鎖住，沒有執行 PTQ、QAT、INT5/6/8、ternary 或 SD4。這裡的 mAP 是目前 repository evaluator 的 Ultralytics internal metric，不是假稱為官方 COCO API AP。analytical profile 是固定 shape 假設下的 proxy，不是 RTX 5090、FPGA latency 或 energy 實測。

本報告中的 `d1-*-10` 是當時實際完成的 staged 5+5 歷史 runs；目前程式已簡化為 `d1-shared`、`d1-pattn`、`d1-phead` 各一次完整 10 epochs。報告保留舊名稱以維持 provenance，不把既有量測改寫成新版流程的結果。

本目錄應提交 Git；原始 Ultralytics 中間圖、queue logs、所有 last 與階段中間 checkpoint 已刪除，只保留 V1-BR、N1-SHIFT、BDCN-V3-LEARN 三個必要 best checkpoints。
