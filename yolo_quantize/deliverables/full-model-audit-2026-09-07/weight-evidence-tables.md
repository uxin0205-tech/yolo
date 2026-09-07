# 已測數據附錄：權重格式、PTQ 與 QAT

由 `scripts/report_weight_evidence.py` 讀取既有產物生成。解釋與限制見[詳細分析](../../docs/reports/2026-09-07-weight-distribution-detailed-analysis.md)。本次不是新實驗。

## 十區 CPU NRMSE

以下是區內各 path NRMSE 的**中位數**，不是參數加權全區誤差，也不是 mAP 下降。低者僅表示權重重建较接近。Fixed-SD4 為靜態 optimal scale，不是 LS-SD4 QAT 結果。

| 區域 | w8 | w7 | w6 | w5 | w4 | fixed-sd4 | exact-ternary | twn-v3 | paper-twn-v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| backbone_attention_safe | 0.0087 | 0.0174 | 0.0348 | 0.0688 | 0.1318 | 0.1810 | 0.4986 | 0.4753 | 0.5103 |
| backbone_deep | 0.0129 | 0.0258 | 0.0510 | 0.0964 | 0.1694 | 0.1839 | 0.5341 | 0.5222 | 0.5432 |
| backbone_early | 0.0116 | 0.0233 | 0.0462 | 0.0893 | 0.1623 | 0.1729 | 0.5727 | 0.5567 | 0.5983 |
| detect_one2one_predictor | 0.0077 | 0.0148 | 0.0304 | 0.0644 | 0.1112 | 0.1654 | 0.4648 | 0.5243 | 0.5302 |
| detect_one2one_tower | 0.0035 | 0.0078 | 0.0476 | 0.0900 | 0.1642 | 0.1077 | 0.6431 | 0.5762 | 0.7614 |
| masf | 0.0035 | 0.0072 | 0.0150 | 0.0314 | 0.0670 | 0.1564 | 0.4896 | 0.3671 | 0.4963 |
| neck | 0.0146 | 0.0289 | 0.0577 | 0.1113 | 0.1888 | 0.1814 | 0.5669 | 0.5480 | 0.5914 |
| neck_attention_safe | 0.0096 | 0.0194 | 0.0389 | 0.0800 | 0.1461 | 0.1751 | 0.5584 | 0.4966 | 0.5726 |
| pose_one2one_predictor | 0.0049 | 0.0100 | 0.0202 | 0.0436 | 0.0875 | 0.1714 | 0.4260 | 0.4253 | 0.4277 |
| pose_one2one_tower | 0.0068 | 0.0136 | 0.0272 | 0.0530 | 0.1012 | 0.1861 | 0.4458 | 0.4220 | 0.4498 |

148 路中，Fixed-SD4 的 CPU NRMSE 嚴格小於同為 per-output-channel W4 的路徑有 **41 路**。這是碼本擬合比較，不是任務勝率。三元粒度與 W4/SD4 不完全相同，不能將全部差異歸因碼本。

完整逐路徑／格式表：[weight-format-evidence.csv](weight-format-evidence.csv)，1332 筆。`original_exact_zero_ratio` 是原權重恰為零的比例，不是接近零比例；`quantized_zero_ratio` 是該格式投影後零比例。`max_to_rms_proxy` 是原權重 max / sqrt(std²+mean²)，只作離群程度提示，不等於分散度或近零密度。std 沿用既有統計定義。

## 25 個 PTQ 結果

全部是相對 accepted 的 **total** delta（乘100轉百分點），包含 activation 替換，不混用 incremental。原 PTQ 用早期 qSiLU parent，CPU 用 V35 learned parent；各特殊格式的 top4 paths 可能不同，uniform 又沿用累積 routes，不能當作同層同 parent 的公平格式比賽。`recover` 表示候選待恢復，不是精度通過。

| 階段／候選 | 判定 | 最差 mAP50 Δ（pp） | 最差 mAP50–95 Δ（pp） |
|---|---|---:|---:|
| special-backbone-exact-ternary | reject | -99.301 | -99.334 |
| special-backbone-fixed-sd4 | recover | -3.836 | -5.415 |
| special-backbone-paper-twn-v2 | reject | -99.408 | -99.408 |
| special-backbone-twn-v3 | reject | -99.408 | -99.408 |
| special-head-exact-ternary | reject | -9.631 | -15.302 |
| special-head-fixed-sd4 | recover | -2.479 | -4.102 |
| special-head-paper-twn-v2 | reject | -6.110 | -12.279 |
| special-head-twn-v3 | recover | -2.003 | -3.809 |
| special-neck-exact-ternary | reject | -11.211 | -13.057 |
| special-neck-fixed-sd4 | recover | -1.657 | -4.053 |
| special-neck-paper-twn-v2 | reject | -14.204 | -14.933 |
| special-neck-twn-v3 | recover | -2.124 | -4.115 |
| uniform-backbone-w4 | reject | -94.978 | -96.346 |
| uniform-backbone-w5 | reject | -46.931 | -45.956 |
| uniform-backbone-w6 | reject | -10.133 | -15.489 |
| uniform-backbone-w7 | reject | -4.464 | -8.052 |
| uniform-head-w4 | reject | -4.186 | -7.145 |
| uniform-head-w5 | reject | -4.339 | -6.866 |
| uniform-head-w6 | reject | -4.263 | -7.715 |
| uniform-head-w7 | reject | -4.248 | -7.551 |
| uniform-neck-w4 | reject | -4.840 | -8.224 |
| uniform-neck-w5 | reject | -4.186 | -7.302 |
| uniform-neck-w6 | reject | -4.379 | -7.698 |
| uniform-neck-w7 | reject | -4.353 | -7.079 |
| final-full-coverage-policy | reject | -4.249 | -6.903 |

## V36 epoch 1：16 項任務結果

epoch index 1 即第二個 epoch；此處因其最差 mAP50 總損失小於 epoch 2 而展示，並非宣稱已重載驗證 best_joint export。搜尋 split，非 formal。各列 Δ = value − accepted，正數是提升；mAP 數值為 0–1。

| 指標 | accepted | V36 epoch 1 | 總變化（pp） |
|---|---:|---:|---:|
| bbat/ball/box/map50 | 0.958383 | 0.971654 | +1.327 |
| bbat/ball/box/map50_95 | 0.779040 | 0.772524 | -0.652 |
| bbat/ball/pose/map50 | 0.959489 | 0.971654 | +1.217 |
| bbat/ball/pose/map50_95 | 0.959489 | 0.971485 | +1.200 |
| bbat/bat/box/map50 | 0.994078 | 0.992302 | -0.178 |
| bbat/bat/box/map50_95 | 0.893845 | 0.888314 | -0.553 |
| bbat/bat/pose/map50 | 0.994078 | 0.992476 | -0.160 |
| bbat/bat/pose/map50_95 | 0.994078 | 0.992166 | -0.191 |
| bbat/box/map50 | 0.976230 | 0.981978 | +0.575 |
| bbat/box/map50_95 | 0.836442 | 0.830419 | -0.602 |
| bbat/pose/map50 | 0.976784 | 0.982065 | +0.528 |
| bbat/pose/map50_95 | 0.976784 | 0.981826 | +0.504 |
| coco/box/map50 | 0.670719 | 0.662372 | -0.835 |
| coco/box/map50_95 | 0.498022 | 0.486455 | -1.157 |
| coco/person/box/map50 | 0.839410 | 0.834406 | -0.500 |
| coco/person/box/map50_95 | 0.620381 | 0.611734 | -0.865 |

worst mAP50 總下降 0.835 百分點、worst mAP50–95 下降 1.157 百分點，分别低於 1.5／4 百分點門檻。不同指標的改善不能抵銷別的指標超標。

## 重建與來源

先執行 `scripts/report_full_model_audit.py`，再執行 `scripts/report_weight_evidence.py`。僅 CPU 讀取既有 JSON/CSV；原始 checkpoint、data、plan 血緣見同目錄 audit.json；本附錄來源 hash 見 weight-evidence-provenance.json。estimated_packed_bytes 是分析估算，沒有硬體測速或實際 packing 驗證。
