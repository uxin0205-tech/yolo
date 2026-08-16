# YOLO26 Binary Attention PWL validation

固定架構：Hadamard MDB Binary QK → PoT Scale → Decomposed 2D Bias → Normalization
→ $PV+PE(V)$。三個方法使用同一 checkpoint、COCO2017 val、imgsz、batch、seed 與 evaluator；
本實驗沒有訓練，也沒有修改 QK、scale、bias、V 或 projection。

## 執行條件與資料

- 模型：YOLO26m，parent checkpoint `artifacts/runs/v1-br/ultralytics/weights/best.pt`。
- 資料：COCO2017 `val2017`，5,000 images / 36,335 instances。
- evaluator：Ultralytics 8.4.90 internal metric，`imgsz=640`、`batch=16`、`seed=0`。
- 兩個 site 同時替換並量測：`model.10.m.0.attn`、`model.22.m.0.1.attn`。
- 本地 mAP 不等同 canonical COCO API AP，只作同 evaluator 的 paired comparison。

## Range 結論

- Exact Softmax 中 $u<-8$ 的平均 probability mass：`0.0011551845`。
- Gate：`0.001`；選定範圍：`[-10,0]`。
- segments：`20`，$\Delta=0.5$。
- Bit-True endpoints：`21 × 16-bit = 336 bits`
  （42 bytes）。

| site | min | mean | std | score < -8 | score < -10 | Exact mass at score < -8 |
|---|---:|---:|---:|---:|---:|---:|
| model.10.m.0.attn | -25.7156 | -5.3131 | 2.5655 | 14.8204% | 5.0826% | 0.086376% |
| model.22.m.0.1.attn | -33.8770 | -6.0841 | 3.1259 | 24.6163% | 11.4976% | 0.144661% |

| site | head | mean | std | score < -8 | Exact mass at score < -8 |
|---|---:|---:|---:|---:|---:|
| model.10.m.0.attn | 0 | -4.3514 | 2.0179 | 4.8438% | 0.021959% |
| model.10.m.0.attn | 1 | -4.5403 | 2.1272 | 6.4804% | 0.031043% |
| model.10.m.0.attn | 2 | -6.2501 | 2.7218 | 25.3718% | 0.154827% |
| model.10.m.0.attn | 3 | -6.1107 | 2.6994 | 22.5857% | 0.137674% |
| model.22.m.0.1.attn | 0 | -3.9160 | 1.7943 | 2.5707% | 0.010240% |
| model.22.m.0.1.attn | 1 | -6.6947 | 3.1204 | 30.2967% | 0.180386% |
| model.22.m.0.1.attn | 2 | -7.2903 | 3.3699 | 38.9912% | 0.208591% |
| model.22.m.0.1.attn | 3 | -6.4352 | 2.8498 | 26.6065% | 0.179427% |

site 2 的 score 分布較寬，約 24.62% score 小於 -8，tail probability mass 約
0.14466%；個別 head 最高達 0.20859%。因此論文起始設定 `[-8,0]` 會在少數 head
造成集中 clipping。保留 $\Delta=0.5$、擴至 `[-10,0]` 只增加 4 個 segments/endpoints，
同時保留直接 shift/mask indexing。

## `[-8,0]` 與 `[-10,0]` 的數值比較

| range | Float exp MAE | Float P MAE | Float PV MAE | Bit-True exp MAE | Bit-True P MAE | Bit-True PV MAE |
|---|---:|---:|---:|---:|---:|---:|
| [-8,0] | 0.000772866 | 4.0950478e-05 | 0.0034244211 | 0.00079257363 | 4.1402463e-05 | 0.0033684827 |
| [-10,0] | 0.00072923469 | 2.9813434e-05 | 0.0018029676 | 0.00074924581 | 3.0326078e-05 | 0.0017198875 |

擴大至 `[-10,0]` 後，Float PV MAE 約降低 47.4%，Bit-True PV MAE 約降低 48.9%。
所以本模型採 20 segments，而不是 16 segments。

## COCO 與數值結果

| 方法 | mAP50-95 | mAP50 | mAP75 | 相對 Exact | P MAE | P max | PV MAE | PV cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact | 0.506658 | 0.677103 | 0.554941 | 0 | 0 | 0 | 0 | 1 |
| Float PWL | 0.506730 | 0.676997 | 0.555058 | +0.000073 | 2.9813434e-05 | 0.017094672 | 0.0018029676 | 0.99999998 |
| Q8.8 Bit-True | 0.506737 | 0.677048 | 0.555104 | +0.000079 | 3.0326078e-05 | 0.011443079 | 0.0017198875 | 0.99999998 |

Bit-True 相對 Float PWL 的額外 mAP 差：`+0.000007`。
三個差值都遠小於 0.001 AP；微小正差不解讀為 PWL 提升精度，只能判定此 checkpoint
與 evaluator 下未觀察到可辨識的精度損失，因此不需要重新訓練。

## Hardware interpretation

PWL 把逐 score 的 transcendental `exp` 換成：clip、offset、segment shift、fraction mask、
兩個 endpoint read，以及一次 integer multiply/shift/add interpolation。$\Delta=0.5$ 時 index
直接由 Q8.8 的 bit slice 取得。這是規則、固定容量且容易 pipeline 的 datapath。

在兩 site、每 site 假設 400 tokens、4 heads、$d_k=32$、$d_v=64$ 的 analytical proxy 下：

| 項目 | Exact | PWL | 差異 |
|---|---:|---:|---:|
| normalization proxy ops | 6,406,400 | 3,846,400 | -39.96% |
| 整個 attention-path proxy | 91,910,400 | 89,350,400 | -2.79% |
| dense PV proxy | 81,920,000 | 81,920,000 | 不變 |

PWL 後 dense $PV$ 仍占此 attention-path proxy 約 91.68%，normalization 約占 4.30%。
所以 PWL 的主要價值是移除難 pipeline 的 transcendental `exp`、讓資料路徑固定且只需
42-byte endpoint table，不是大幅降低整個 YOLO26m 的 FLOPs。這些數字是固定 shape
與權重假設的演算法 proxy，不是 GPU/FPGA latency、DSP、BRAM 或 energy 實測。

但 PWL 只近似 numerator weight，沒有消除每列 denominator：仍需 row sum 與 reciprocal/divider；
之後的 dense $PV$ 也完全保留。指定論文同樣使用 serialized restoring divider。因此剩餘主要成本仍是
$PV$ 與 denominator，而不是 endpoint table。此處成本是演算法/儲存 breakdown，不是 FPGA 實測。

## Bit-True scope

論文支持 Q8.8、範圍、segments、17×16-bit endpoints 與 interpolation；但沒有發布 endpoint
整數、endpoint Q-format、rounding、saturation 與完整 intermediate widths。本專案明定 UQ1.15
endpoint、round-to-nearest for nonnegative shifted score、signed 16-bit saturation、floor clipping、
truncating `>>7` interpolation 與 `u=0` endpoint special case。因此這是 paper-parameterized project
bit-true reference，不是作者 RTL 的 faithful reproduction。denominator 目前仍是 exact float reference。

## Diagnostics defect 與修正

第一次 score job 的額外 PV diagnostic 在 Ultralytics AMP context 內被 autocast 成 FP16，
導致累加值出現 NaN；COCO output、score、P 與 mAP 本身仍有限。正式 compare job 已在自己的
immutable `score_analysis/` 重新量測，強制 PV matmul 使用 FP32、reduction 使用 FP64，
並對所有 non-finite 值 fail closed。最終 CSV/圖表只使用修正後 artifact。

## 最終回答

1. `[-8,0]` 不建議作最終設定；overall tail mass 0.11552% 超過 0.1% gate，且 site/head 不均勻。
2. 16 segments / $\Delta=0.5$ 不足；`[-10,0]`、20 segments、同一 $\Delta$ 較合適。
3. Float PWL 對 Exact 近乎無損：mAP50-95 差 `+0.000073`。
4. Q8.8 Bit-True 沒有可辨識的額外損失：相對 Float 差 `+0.000007`。
5. 硬體優勢是將 exp 轉為固定 clip/shift/mask/LUT/interpolation datapath，table 僅 42 bytes。
6. 主要成本仍是 dense $PV$，denominator 的 row sum + reciprocal/divider 也尚未移除。

原始數據：`score-analysis.json`、`score-statistics.csv`、`score-histogram.csv`；
RTL endpoint：`endpoint-table.csv`；
整理比較：`comparison.csv`；圖表：`figures/score-histogram.svg` 與
`figures/normalization-map.svg`。完整 queue provenance 位於
`artifacts/runs/pwl-compare/`。
