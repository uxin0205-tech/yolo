# V5 qSiLU＋A8 九區 W8 敏感度報告

日期：2026-09-03。

## 結論

除已完成的`backbone_early`外，其餘九個deployment region都已完成isolated W8 diagnostic與完整COCO／BBAT5 search validation。八格為`green`，只有`backbone_attention_safe`為`recover`。

這一輪最重要的發現是：參數量和task sensitivity不是同一件事。`neck`有33個module、8,142,848個weights，isolated W8最差總下降只有`-0.011310`，是目前最佳容量候選；`backbone_attention_safe`只有7個module、919,808個weights，卻使COCO box總下降到`-0.017922`，無法通過`-0.015`。因此後續不能用「層小／權重小」直接決定bit或SD4，必須依每層task sensitivity與分布共同路由。

isolated Pareto角色為：

- accuracy：`masf` W8，最差總下降`-0.010988`。
- balanced：`neck` W8，最差總下降`-0.011310`，全模型weight storage proxy `1.3703×`。
- hardware：`neck` W8；在所有通過硬門檻的isolated候選中容量收益最大。

這些是**單區候選**，不是可直接相加的全模型policy。下一階段會從matched qSiLU＋A8開始逐區累積，每次重新跑八項mAP50。

## 實驗契約

- Activation parent：qSiLU＋LSQ+ A8，checkpoint SHA-256 `7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e`。
- Weight：每次只量化一個region，per-output-channel W8、`mse_grid_v1`、zero-point 0、full two's-complement range。
- Reference：accepted Full35 SiLU／FP32與matched qSiLU＋A8／FP32 weight均重用已驗證raw metrics，沒有重跑或換資料。
- Calibration：固定COCO train 32＋BBAT5 train 32，fresh calibration identity與舊reference逐observer核對。
- Search：完整5,000張COCO val與canonical固定600張BBAT5 search-val；COCO person與BBAT ball／bat box／pose共八項mAP50。
- Gate：每項total delta `>= -0.015`，且包含activation替換；W8 incremental每項`>= -0.01`。
- 本輪無訓練、無formal validation、無新增augmentation或noise。

## 九區結果

| Region | Modules | Weights | 最差total mAP50 | 最差W8 incremental | 全模型weight proxy | 判定 |
|---|---:|---:|---:|---:|---:|---|
| backbone_deep | 22 | 8,126,464 | -0.011696 | -0.002784 | 1.3694× | green |
| backbone_attention_safe | 7 | 919,808 | **-0.017922** | -0.006346 | 1.0314× | recover |
| neck | 33 | 8,142,848 | -0.011310 | -0.000415 | **1.3703×** | green |
| MASF | 3 | 74,240 | **-0.010988** | +0.000039 | 1.0024× | green |
| neck_attention_safe | 5 | 395,520 | -0.011555 | -0.000171 | 1.0132× | green |
| Detect one2one tower | 18 | 1,390,592 | -0.011492 | -0.000378 | 1.0483× | green |
| Detect one2one predictor | 6 | 62,208 | -0.011439 | -0.000172 | 1.0021× | green |
| Pose one2one tower | 24 | 2,238,464 | -0.011576 | -0.000217 | 1.0801× | green |
| Pose one2one predictor | 9 | 3,456 | -0.011576 | -0.000008 | 1.0001× | green |

所有最差total項目都是COCO box。這主要因matched qSiLU＋A8相對accepted在COCO box已下降`-0.011576`，剩餘硬門檻裕度只有`0.003424`；因此即使weight incremental很小，total仍可能失敗。`backbone_attention_safe`的weight incremental為`-0.006346`，使total越過門檻。

## 如何使用這些結果

### 累積W8

累積順序不按單區delta相加，而採兩條preregistered chain：

1. accuracy chain：MASF → predictor小區 → neck-attention → towers → backbone/neck大區。
2. compression chain：neck → backbone_deep → pose tower → detect tower →其餘小區。

每加一區都重新validation。只要八項任一低於`-0.015`，該節點不再沿同方向擴展；保留上一個通過節點。`backbone_attention_safe`不進整區累積，只進layer/group拆分。

### Lower bits與特殊格式

- `backbone_early`整區W7已recover、W6–W4已reject；不能整區繼續壓。
- `backbone_attention_safe`連W8都recover，優先保護或拆層，不直接測整區W4／SD4。
- neck與backbone_deep的容量收益最大，但必須先測累積交互作用；單區green不保證兩者相加仍green。
- Fixed SD4、LS-SD4、Paper-TWN／TTQ只進低敏感layer，不以「weight較小」作唯一條件。

## 證據

- 九區diagnostic：[`weight-ptq-qsilu-remaining-nine-regions-w8-diagnostic-v5.json`](../../artifacts/reports/weight-ptq-qsilu-remaining-nine-regions-w8-diagnostic-v5.json)，SHA-256 `4122d301a0dbd279559435527ccdcc4ed2e2785091b0865bd6029c9caceff73b`。
- 九區search：[`v5-qsilu-remaining-nine-regions-w8-search-v1.json`](../../artifacts/reports/v5-qsilu-remaining-nine-regions-w8-search-v1.json)，SHA-256 `6df5efd6e0094538b7b66bc068246c2cd525e90a2557819fa280606ef74b4665`。
- Diagnostic plan SHA-256 `d5cb91f906b1fa18b47a3e1e129c6f899720cadf988315722776415bff3b8838`。
- Search plan SHA-256 `3783deb7607f7c61d77771dab359eef54fa3de1883748963e71c7cc24720a01e`。

## 限制

- 目前只有isolated PTQ，不是combined policy。
- search-val結果不是formal結果，也沒有多seed或QAT。
- fake-quant/dequantized weights尚不等於native integer kernel；壓縮率只含code與scale metadata，不是實測latency／power。
- attention、MASF與head的integer boundary lowering仍需在finalists前完成。
