# Full35 全非 SiLU CPU Q3：11-region Zero-shot Sensitivity

> 狀態：5/5 candidate-region cases。這是 zero-shot validation，沒有 optimizer、backward、epoch 訓練或權重更新。

## 實驗契約

- 最終 policy 的 190 個 activation sites 均不得為 SiLU。
- 基底權重：已完成的 qSiLU 10-epoch short-recovery `best_joint.pt`。
- 裝置：CPU；backend：bit-true weights；資料比例 100%，不抽樣、不重新切分。
- 資料：COCO2017 與 Canonical BBAT5 v1。
- 主表四項 Δ 相對同裝置 uniform recovered qSiLU；正式 gate 仍相對同裝置 accepted SiLU 稽核八項。

## CPU Baseline

| Metric | accepted SiLU | uniform recovered qSiLU | qSiLU Δ |
| --- | ---: | ---: | ---: |
| COCO | 0.498460 | 0.495313 | -0.003147 |
| Person | 0.620334 | 0.619684 | -0.000650 |
| BBAT box | 0.629451 | 0.625257 | -0.004193 |
| BBAT pose | 0.902102 | 0.901469 | -0.000633 |

## 圖表

![相對 uniform qSiLU 的 11-region sensitivity](nonsilu-q3-11-region-local.png)

![相對 accepted SiLU 的正式 gate 視圖](nonsilu-q3-11-region-global.png)

灰底 region 是 training-only graph；其替換結果不代表部署推論節省。

## Region 結果

### Hardswish

| Region | 類型 | COCO Δ | Person Δ | BBAT box Δ | BBAT pose Δ | 八項最差 global Δ | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `backbone_attention` | 待執行 | — | — | — | — | — | — |
| `backbone_deep` | 待執行 | — | — | — | — | — | — |
| `backbone_early` | 待執行 | — | — | — | — | — | — |
| `detect_one2many` | 待執行 | — | — | — | — | — | — |
| `detect_one2one` | 待執行 | — | — | — | — | — | — |
| `masf` | 待執行 | — | — | — | — | — | — |
| `neck` | deployment | -0.014313 | -0.012533 | -0.010052 | -0.004725 | -0.017755 | 未通過 |
| `neck_attention` | deployment | -0.000039 | +0.000081 | +0.000116 | +0.000084 | -0.008581 | 通過 |
| `pose_flow` | training-only | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.008810 | 通過 |
| `pose_one2many` | training-only | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.008810 | 通過 |
| `pose_one2one` | deployment | +0.000000 | +0.000000 | -0.122989 | -0.005033 | -0.150325 | 未通過 |

## 證據邊界

- 這是逐 region 一次只替換一區的一階 zero-shot sensitivity，不能取代多區 mixed policy 完整驗證。
- CPU 與 GPU 不保證 bitwise 相同；本報告所有 delta 都使用同一 CPU backend baseline，避免跨裝置相減。
- `leaky_relu_1_8` 在本階段以 float graph 的精確 1/8 係數執行；真正 INT8／RTL bit-exact 屬於後續量化階段。
- 運算量欄位是解析 proxy，不是 latency、功耗或面積實測。

機器可讀結果：[`summary.json`](summary.json)／[`rows.csv`](rows.csv)。
