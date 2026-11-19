# Activation smoke machine-readable證據

2026-09-03現行v5 mAP50 contract：

- `weight-ptq-qsilu-remaining-nine-regions-w8-diagnostic-v5.json`：其餘九區W8 output diagnostic，9/9結構通過，SHA-256 `4122d301a0dbd279559435527ccdcc4ed2e2785091b0865bd6029c9caceff73b`。
- `v5-qsilu-remaining-nine-regions-w8-search-v1.json`：九區完整mAP50搜尋，8格green、`backbone_attention_safe` recover，SHA-256 `6df5efd6e0094538b7b66bc068246c2cd525e90a2557819fa280606ef74b4665`。
- `v5-qsilu-backbone-early-w8-map50-regate-v1.json`：驗證舊raw metrics SHA後，以八項mAP50和`-0.015`總門檻重新判定W8；activation替換計入total，decision=`green`。
- `v5-qsilu-backbone-early-bit-search-v1.json`：W8–W4完整COCO／BBAT5 search-val bit curve；只有W8通過，W7 recover，W6–W4 reject，SHA-256 `fd4442f34df0077243aadd4b6d5855cbf37e1cb215a490e5b43c970191772d9b`。
- `weight-ptq-qsilu-backbone-early-w7-w4-grid-diagnostic-v4.json`與`weight-ptq-qsilu-backbone-early-w4-exact-diagnostic-v4.json`：GPU output diagnostic，不是winner判定。

2026-09-03凍結的v4 GPU bridge：

- `weight-ptq-qsilu-backbone-early-w8-diagnostic-v3.json`：固定manifest下的一格W8 output diagnostic，`diagnostic_pass`但沒有selection claim。
- `v4-qsilu-backbone-early-w8-search-v1.json`：同一契約accepted／matched／candidate八指標search validation，decision=`green`，SHA-256 `738f5103656deebcb248ad6e7d0770caa662fa191c90882f45f8f78e47bc6ad2`。它不是formal winner或訓練結果。

- `activation-smoke-v2.json`：30格完整原始結果，SHA-256 `3c9301adaa1937f50bdd0c059f7d8000ea4699ce229e98c8ddb387651340c1f1`。
- `activation-smoke-v2-summary.csv`：30列摘要，SHA-256 `e5d76871d3119e681cdcb7707dc2c4a6a24ad3e35c54815c666c6c72a2f51ab6`。

這些檔案是報告與圖表的source evidence，不是checkpoint。每格的matched reference為相同activation function、quantization disabled、FP32 weight輸出；不能跨activation把proxy直接當絕對mAP排名。

同目錄的`weight-ptq-*-v1`是尚未公開於上述activation publication的本機歷史PTQ證據；其限制與血緣另見`artifacts/manifests/v0-lineage-reconciliation-v1.yaml`。

V2 CPU-only全層靜態分析：

- `weight-format-analysis-qsilu-pq-a8-main-v2.json`：active qSiLU＋A8，2,960筆。
- `weight-format-analysis-hardswish-a8-main-v3.json`：active uniform Hardswish＋A8，2,960筆。
- `weight-format-analysis-poly-shift-a8-main-v2.json`：active poly_shift＋A8，2,960筆。
- `weight-format-analysis-poly-quality-a8-main-v2.json`：歷史 `poly_quality`＋A8，2,960筆；2026-08-31後不再進future matrix。

現行三份active JSON合計8,880筆，歷史`poly_quality`另有2,960筆；四份都包含W8／W7／W6／W5／W4、Fixed SD4與Paper-TWN。現行hash、彙總與限制見`artifacts/manifests/v1-v3-cpu-delivery-v2.yaml`，舊三parent事實仍見v1；它們是weight reconstruction，不是mAP validation。
