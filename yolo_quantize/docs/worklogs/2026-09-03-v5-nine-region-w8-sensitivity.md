# 2026-09-03 V5 九區 W8 敏感度

## 變更內容與原因

- 新增[`weight-region-sensitivity-plan-v5.yaml`](../../configs/experiments/weight-region-sensitivity-plan-v5.yaml)，只授權qSiLU＋A8其餘九區的isolated W8 diagnostic。
- 新增[`v5-qsilu-remaining-nine-regions-w8-search-v1.yaml`](../../configs/experiments/v5-qsilu-remaining-nine-regions-w8-search-v1.yaml)，只授權九個candidate validation，重用SHA驗證的accepted／matched。
- 完成九區diagnostic與完整mAP50 search，建立accuracy／balanced／hardware isolated Pareto角色。
- 更新v5總計畫，下一階段改為累積W8 policy與attention-safe layer拆分。

原因是`backbone_early` bit curve只回答一區，不能外推到backbone、neck或head。使用者要求從backbone一路分析到head並做敏感度，因此本輪完整覆蓋剩餘九區，同時維持八項mAP50與activation-inclusive總門檻。

## 驗證方式與結果

- list-only解析9/9 cell，authorization、checkpoint、diagnostic manifest及region inventory一致。
- GPU diagnostic 9/9為`diagnostic_pass`、same structure、all finite；沒有用proxy直接選winner。
- 完整5,000張COCO val＋600張BBAT5 search-val後，8格green、1格recover。
- 唯一recover是`backbone_attention_safe`，COCO box total delta `-0.017922`；W8 incremental `-0.006346`。
- isolated accuracy role為MASF W8（worst `-0.010988`）；balanced／hardware role為neck W8（worst `-0.011310`、全模型weight proxy `1.3703×`）。
- search artifact SHA-256 `6df5efd6e0094538b7b66bc068246c2cd525e90a2557819fa280606ef74b4665`；diagnostic artifact SHA-256 `4122d301a0dbd279559435527ccdcc4ed2e2785091b0865bd6029c9caceff73b`。
- GPU完成後為440 MiB used、0% utilization；沒有留下訓練程序。

## 遇到的困難及解法

1. Ultralytics逐class progress輸出很長，終端片段會截斷個別cell完成行。解法：以runner的atomic JSON、每格raw metrics SHA與最終gate作唯一結論來源，不從console抄值。
2. 部分head region對非目標task輸出完全不變，容易誤讀成「絕對安全」。解法：仍對每格完整跑兩個task與八項指標，並只把它解讀為路徑隔離證據。
3. `backbone_attention_safe`參數少但比大區敏感。解法：不依參數量外推；此區改做layer/group拆分，保護Binary Q/K與attention PWL既有邊界。

## 未解事項或風險

- 8格green只是isolated結果；組合誤差可能非線性，必須逐步累積重跑。
- qSiLU＋A8本身已消耗大部分COCO box總budget，combined policy可用裕度很小。
- attention-safe拆層、mixed bit、Fixed／LS-SD4與ternary task mAP尚未完成。
- QAT fold-aware有效權重grid、matched sham與optimizer HPO尚未開始。
- formal、多seed、packed export及硬體實測尚未完成。
