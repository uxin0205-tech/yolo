# RTX 5060 Ti HALF 階段性結論（2026-08-22）

> 此文件保留 `fraction=1.0` Phase B 完成當下的時間截面。後續 Phase C 嘗試未產生可驗證 checkpoint
> 就失敗；9 個保存候選在 COCO2017 與 BBT5 detect_dataset 的補齊報告請見
> [`../final/reports/FULL35_PARTIAL75_AP.md`](../final/reports/FULL35_PARTIAL75_AP.md)。

本報告在 `fraction=1.0` Phase B 完成後截斷。正在執行的 Phase C（physical batch 8、`nbs=16`、
workers 6、patience 7）只記錄為進行中，不使用任何未完成 epoch 或 checkpoint 下結論。

## 結論

1. 目前正式 gate 對 Full35、Partial75 都保留各自從 commit `7f0bd61` 匯入的 accepted A2 checkpoint。
   已完成的 Phase B／C child 沒有一個能取代 A2。
2. `fraction=0.3` 是 Phase B 退步的重要原因，但不是唯一原因。改成完整 COCO train split 後，Full35
   與 Partial75 Phase B 分別比 30% run 回升 `0.003998`、`0.005723` mAP50-95；相對各自 A2 仍低
   `0.002889`、`0.002745`，所以兩者依 rollback gate 都回到 A2。
3. `fraction=0.3` Phase C 沒有改善正式 COCO：Full35／Partial75 相對 A2 分別下降 `0.009244`、
   `0.009493` mAP50-95。這兩次 C 都從 gate 接受的 A2 開始，不是從已拒絕的 B child 開始。
4. 自訂 ball／bat valid 上，30% Phase C 的 baseball-bat AP50-95 有局部提升，但 sports-ball AP50-95
   沒有提升；同一 checkpoint 的完整 COCO mAP 又下降。因此這是資料域偏向的訊號，不是整體模型變好。
5. A2 邊界上，Partial75 的 COCO mAP50-95 比 Full35 高 `0.000363`，同時 fused Params／GFLOPs 較低。
   它是目前較合理的工程候選，但尚缺同一張正式 GPU、相同設定的 FP16 p50 latency，不能依完整 selector
   契約宣告正式 architecture winner。
6. 相對不可變 A0，Full35 A2 為 `-0.000346`，Partial75 A2 為 `+0.000018` mAP50-95；沒有足夠證據
   宣稱 P3 MASF 帶來實質整體 accuracy 改善。A0 未重訓，因此 A1/A2 與 A0 的差異也混合了架構與 staged
   retraining 效果。

## 驗證範圍與契約

| 項目 | 固定值 |
|---|---|
| 環境 | Python 3.12.13、PyTorch 2.11.0+cu128、Ultralytics 8.4.90 |
| GPU | NVIDIA GeForce RTX 5060 Ti，16 GiB |
| Dataset | COCO2017 train 118,287／val 5,000；imgsz 640；seed 0；deterministic |
| Phase B | physical batch 16、`nbs=16`、workers 6、patience 4、最多 10 epochs |
| 已完成的 30% Phase C | physical batch 8、`nbs=16`、workers 6、patience 9、最多 70 epochs |
| Validation | 完整 COCO val、batch 8、workers 6、Bit-True PWL |
| 資源保護 | job 啟動 RAM 8 GiB；訓練中 RAM 1.5 GiB／VRAM 1 GiB fail closed |

`fraction=1.0` Phase B 對照使用同一對 accepted A2 Float checkpoints、相同 Phase B 超參數與完整 validation；
主要受控變因是 train fraction。兩個 Full35 B 都曾由 `last.pt` 接續，queue 與
`training-complete.json` 留有 resume SHA／epoch，可避免把斷電後重跑誤當成新實驗。

## 正式 COCO2017 val

表中的 ball／bat 是 COCO class 32／34 AP50-95。Delta 一律相對同架構 A2。

| 架構／階段 | Train fraction | mAP50-95 | mAP50 | mAP75 | Ball AP | Bat AP | Delta vs A2 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0 immutable | — | 0.506737 | 0.677048 | 0.555104 | 0.513702 | 0.492753 | — | immutable |
| Full35 A2 | — | 0.506391 | 0.676290 | 0.554714 | 0.510611 | 0.489142 | 0 | retained |
| Full35 B | 0.3 | 0.499504 | 0.671043 | 0.546932 | 0.508897 | 0.477993 | -0.006887 | rollback |
| Full35 C | 0.3 | 0.497147 | 0.670737 | 0.544152 | 0.509981 | 0.473009 | -0.009244 | rollback |
| Full35 B | 1.0 | 0.503503 | 0.674369 | 0.551728 | 0.516057 | 0.467347 | -0.002889 | rollback |
| Partial75 A2 | — | 0.506754 | 0.677117 | 0.554577 | 0.515439 | 0.486951 | 0 | retained |
| Partial75 B | 0.3 | 0.498286 | 0.670894 | 0.546344 | 0.503613 | 0.466489 | -0.008468 | rollback |
| Partial75 C | 0.3 | 0.497262 | 0.669546 | 0.545059 | 0.513031 | 0.473828 | -0.009493 | rollback |
| Partial75 B | 1.0 | 0.504009 | 0.675549 | 0.550950 | 0.511516 | 0.463356 | -0.002745 | rollback |

30% 與完整資料的 Phase B 差異如下：

| 架構 | B 30% | B 100% | 回升 | 仍低於 A2 |
|---|---:|---:|---:|---:|
| Full35 | 0.499504 | 0.503503 | +0.003998 | -0.002889 |
| Partial75 | 0.498286 | 0.504009 | +0.005723 | -0.002745 |

完整資料讓 Phase B 收斂到更接近 A2，但 baseball-bat AP 仍明顯下降；Full35 的 sports-ball AP 雖比 A2
高 `0.005447`，整體與 bat 指標不足以通過 gate。不能只用單一類別的改善取代整體模型。

## 自訂 ball／bat valid

資料來自 `/home/uxin0/yolo/original/pose/dataset` 的 pose bounding boxes，轉成 COCO80 class 32／34 detection
view 後驗證。Valid 有 567 images、301 ball instances、484 bat instances；固定 imgsz 640、batch 8、
workers 6。以下皆為 Bit-True PWL。

| Checkpoint | mAP50-95 | Ball AP50-95 | Bat AP50-95 |
|---|---:|---:|---:|
| A0 | 0.405256 | 0.318082 | 0.492430 |
| Full35 A2 | 0.399448 | 0.314137 | 0.484759 |
| Full35 B, fraction 0.3 | 0.386937 | 0.308668 | 0.465206 |
| Full35 C, fraction 0.3 | 0.412511 | 0.309676 | 0.515345 |
| Partial75 A2 | 0.401554 | 0.313672 | 0.489437 |
| Partial75 B, fraction 0.3 | 0.379778 | 0.310864 | 0.448691 |
| Partial75 C, fraction 0.3 | 0.411398 | 0.312046 | 0.510749 |

相對各自 A2，Full35／Partial75 C 的 bat AP50-95 分別提高 `0.030587`、`0.021312`，但 ball 分別下降
`0.004461`、`0.001626`。此外 valid 中有 93/567 images（16.4%）的原始 COCO ID 出現在 COCO train2017；
這組數字適合同資料集的相對比較，不是獨立泛化成績。

## 成本、時間與資源

Params／GFLOPs 由實際 checkpoint 在 imgsz 640 以 Ultralytics `model.info()` fuse 後量測；不是估算值。

| 架構 | Fused Params | GFLOPs | 相對 A0 Params | 相對 A0 GFLOPs |
|---|---:|---:|---:|---:|
| A0 | 20,412,144 | 68.2 | — | — |
| Full35 | 20,487,153 | 69.1 | +75,009 | +0.9 |
| Partial75 | 20,418,609 | 68.3 | +6,465 | +0.1 |

30% Phase C 的兩步 AMP official-loss capacity probe：Full35 peak `5,328,045,056` bytes、Partial75 peak
`5,165,796,352` bytes；兩者皆以 batch 8 × accumulate 2 通過。這是容量診斷，不是 inference latency。

完成 run 的 telemetry 摘要如下。`min RAM`／`min free VRAM` 是訓練程序內定期觀測下限；`max torch peak`
是 allocator peak。所有 run 都高於 1.5 GiB RAM／1 GiB VRAM 的 fail-closed 門檻，且沒有 OOM。

| Run | 完成 epochs | Best epoch | Stop | 記錄 elapsed | Min RAM | Min free VRAM | Max torch peak |
|---|---:|---:|---|---:|---:|---:|---:|
| Full35 B, 30% | 5/10 | 1 | early stopping | 0.732 h* | 7.590 GiB | 6.719 GiB | 5.745 GiB |
| Partial75 B, 30% | 5/10 | 1 | early stopping | 1.750 h | 7.633 GiB | 5.426 GiB | 5.451 GiB |
| Full35 C, 30% | 10/70 | 1 | early stopping | 5.216 h | 8.658 GiB | 4.367 GiB | 5.252 GiB |
| Partial75 C, 30% | 10/70 | 1 | early stopping | 5.159 h | 8.527 GiB | 5.318 GiB | 5.099 GiB |
| Full35 B, 100% | 9/10 | 5 | early stopping | 9.333 h* | 7.214 GiB | 2.429 GiB | 5.745 GiB |
| Partial75 B, 100% | 7/10 | 3 | early stopping | 7.275 h | 7.440 GiB | 6.822 GiB | 5.448 GiB |

`*` Full35 B 曾接續 `last.pt`；`training-complete.json` 的 elapsed 只涵蓋最後一次 trainer invocation，不能
當成完整 wall-clock。30% Full35 B 的 telemetry 在斷電接續點有 461 個 NUL padding bytes；去除 padding
後 141 筆 JSON snapshot 均可解析，checkpoint、metrics、manifest 與 gate 雜湊驗證不受影響。本次依指示
不修改或清理 runtime artifact。

## 稽核、限制與下一步

- 已重新計算所有列入報告的 Bit-True checkpoint SHA256，並與各 `metrics.json` 一致；所有 metrics 與
  training CSV 均為 finite，完成 run 的 CSV epoch 數與 `training-complete.json` 一致。
- `fraction=0.3` queue state 只彙整最後一次 gate，沒有列出四個 child 的總表；本報告改由各 child 的
  manifest、training-complete、metrics、lineage 與 checkpoint 交叉驗證。這是 queue 彙整的可追溯性缺口，
  不是結果缺失。
- 自動測試：`48 passed`。測試以 CPU 隔離執行，沒有干擾正在跑的 Phase C。
- 尚缺正式 same-GPU FP16 p50 inference latency、A0 的同版 canonical AP_S/M/L／recall 重測，以及
  `fraction=1.0` Phase C 完成結果；因此不宣告整個研究完成，也不執行 T1/T2/T3。
- 後續只需等待目前 Phase C 完成後，以相同 audit 流程新增結果；不得回寫或改動本報告的截斷結論。

機器可讀資料位於 `results/rtx5060ti-half-0822.json`，比較列位於
`results/rtx5060ti-half-0822.csv`。Raw runtime artifacts 依 `.gitignore` 保留在 `artifacts/`，不納入 Git。
