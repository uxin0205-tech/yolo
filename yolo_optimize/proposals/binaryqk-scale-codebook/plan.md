# BinaryQK 少量 scale／codebook：條件式最小實驗計畫

> 2026-09-08 仍為Q0後的條件式，B4無runtime selector、A8動態上界；本次禁GPU且沒有replay/kernel結果。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

本計畫只在[BinaryQK 精度恢復主線](<../binaryqk-accuracy-recovery/plan.md>)完成 fixed-PoT site
isolation、matched QAT，以及必要的單一 KD／STE-window後，仍有明顯缺口且診斷指向
magnitude／scale時啟動。方向結論見[README](<README.md>)，資料流見
[架構圖](<architecture-report.md>)。

## 一、預先登錄的問題

只回答兩個問題：

1. 完全沒有 per-image scale運算的 `B4-GROUP`，能否比 C0保留更多 score fidelity，同時在目標
   backend仍保有 BinaryQK 的 latency／throughput收益？
2. 若 B4不夠，`A8-TOKEN` 能提供多大的 token-magnitude fidelity上界；其 runtime reduction、
   index與variable-shift成本是否在部署預算內？

不以本方向重新回答 site 10／site 22哪一處較敏感，也不重跑已完成的 global dynamic／SHEAD／PoT
消融。

## 二、唯一 control 與候選

| ID | 設計 | runtime scale assignment | 初始角色 |
|---|---|---:|---|
| `C0-FIXED` | 現行 two-basis per-head/basis fixed PoT | 無 | 唯一 control |
| `B4-GROUP` | 4 fixed channel groups × 8 channels | 無 | production-first |
| `A8-TOKEN` | per-token 8-level PoT、3-bit index | 有 | fidelity ceiling |

首輪固定 `G=4`、`K=8`，不做超參數 sweep。K=4／G=8只有在 cached replay清楚顯示 capacity
不足，而且不新增完整訓練 arm的前提下，才可作離線補充。

## 三、Phase 0：工程與血緣 gate

開始 replay 前必須全部通過：

1. parent、checkpoint digest、commit、site policy、basis、normalization與 evaluator全部凍結。
2. 固定 validation sample IDs與 cached Q/K來源；C0、B4、A8必須讀相同 tensors。
3. C0 fixed path在 dynamic reduction前 early-return，修改前後 output逐元素 bit-true一致。
4. sign zero policy、packing、XNOR-popcount、signed-dot轉換與 padding測試通過。
5. B4／A8只改 coefficient granularity與對應 datapath；不得同時改 threshold、basis、bias、site或 KD。
6. `1/sqrt(32)`、gamma與 coefficient的 fold／round規則寫入 config並由 parity test鎖定。
7. 全部結果標示 lineage；Full35多任務結果不能和較早單任務 YOLO26直接相減。

任一項失敗就先修工程問題，不執行完整 validation或訓練。

## 四、Phase 1：cached Q/K replay

用固定 subset 對 C0、B4、A8各跑一次 deterministic replay，不啟動 GPU 長訓練。

每個 site／head／basis保存：

- FP score與候選 score的 cosine、NRMSE、KL、entropy。
- top-k overlap與 pairwise ranking agreement。
- Q/K magnitude分布、token dispersion、sign ratio與 STE saturation。
- B4每組 coefficient／partial-count分布。
- A8 code usage、index entropy、clipping率與 exponent-sum分布。
- bit-true output digest與 analytical operation／storage counts。

固定計數基準：

| 項目 | C0 | B4 | A8 |
|---|---:|---:|---:|
| fixed combined-coefficient slots | 16 | 64 | 0 activation-fixed slots |
| runtime activation indices/image | 0 | 0 | 12,800 |
| magnitude abs/image | 0 | 0 | 409,600 |
| reduction adds/image | 0 | 0 | 396,800 |
| basis／partial terms上界 | 2,560,000 | 10,240,000 | 2,560,000 |
| variable pair shifts上界 | 0 | 0 | 2,560,000 |

表中 C0／B4的「0 magnitude operation」指正確的 fixed inference，不包括 calibration，也不包括
目前 eager reference算完又丟掉的軟體冗餘。

### Replay gate

- B4只有在主要 fidelity指標一致優於 C0，且沒有單一 site/head嚴重退化時才進 kernel gate。
- A8只有在明顯優於 C0/B4、且 token dispersion與 ranking改善方向一致時才保留；否則證明增加
  runtime scale granularity不值得。
- 若 C0、B4、A8差異都小，直接停止本方向；不要用 training把無效表示力差異硬拉成另一條 lineage。
- replay只可作篩選，不能把 score改善宣稱為 detection AP改善。

## 五、Phase 2：target-kernel gate

### B4

至少 prototype：

~~~text
4 × 8-channel partial popcount
    → 4 個 fixed PoT shift
    → fixed shift-add accumulation
~~~

量測 p50/p95 latency、throughput、peak workspace、packing／masking成本與數值 parity。Python eager
時間或 analytical BitOPs不算 target證據。

### A8

只有 replay保留 A8時才 prototype：

~~~text
per-token abs-sum / >>5
    → 8-level selector / 3-bit index
    → Q-index + K-index / exponent LUT
    → tiled variable shift + accumulation
~~~

不得只量 binary matmul而排除 reduction、selector、index traffic與 epilogue。若 index或 pair-scale
matrix需要 materialize，也要計入 peak memory；優先使用 tiled fusion但仍須報完整 end-to-end成本。

### Kernel停止條件

- 候選相對 FP QK沒有目標裝置效益，或相對 C0失去預先登錄的 BinaryQK latency／throughput預算：
  停止，不進完整 validation。
- parity、padding或 export任一失敗：回到工程修正，不用近似輸出去跑 mAP。
- B4與A8都通過時，只選 fidelity／latency Pareto frontier上的一個候選。

## 六、Phase 3：最多一個新完整 validation

以同 checkpoint、同 validation protocol比較：

| Arm | 用途 |
|---|---|
| `C0-FIXED` | 重用同 lineage正式 control metrics |
| `SCALE-CAND` | B4或A8的唯一 kernel winner |

必須同時報 overall AP50-95、AP50、AP75、AP_S/M/L、正式任務關鍵類別、score fidelity與 target
profile。只有 `SCALE-CAND - C0 >= +0.001` AP50-95，且關鍵類別與硬體 gate都不退化，才進
matched QAT。未過立即標記本方向 `rejected`，不改 K/G、不疊 threshold或新 basis。

若目的是論文式歸因，可另加同 checkpoint `DYN-GLOBAL` evaluation；產品選型預設不做。沒有
這個 control時，不得宣稱 A8收益全部來自 token granularity。

## 七、Phase 4：通過後才做 matched QAT

只新增兩個同 seed arms：

| Arm | scale設計 | trainable scope | 用途 |
|---|---|---|---|
| `C0-QAT` | C0 fixed PoT | 與候選完全相同 | 排除一般 recovery gain |
| `CAND-QAT` | 唯一 B4或A8 winner | 與 C0相同 | 測 scale設計的額外收益 |

兩臂使用相同 parent、seed、資料、augmentation、optimizer、LR、epoch、patience與 evaluator。
第一個 paired seed的增益未達 `+0.001` 或硬體 gate失效就停止；通過才補到3個 paired seeds。
不在這一階段新增第二種 KD、threshold sweep或 basis sweep。

## 八、資料、輸出與狀態契約

- 若使用 BBAT5 detection，只能使用不可變的
  `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 與正式 `configs/detect.yaml`；不得重切、
  抽樣或修改影像／標註。
- 每個 replay／kernel／validation保存 config、commit、parent digest、sample IDs、tensor digest、
  metrics、operation trace、target資訊與 failure reason。
- 實驗開始後在本資料夾新增 `results.md`；大型 tensors、checkpoint、trace與 binary artifacts
  留在正式 artifacts目錄，只由相對連結引用。
- 狀態只可依證據由 `proposed` 改為 `ready`、`running`、`validated`或`rejected`。

## 九、最少工作量總結

~~~text
條件未成立                         → 0 jobs，本方向保持 proposed
條件成立                           → 3 個 cached replays：C0 / B4 / A8
replay無 winner                     → 停止
有 winner但 kernel不過              → 停止
kernel通過                          → 1 個新完整 candidate validation
validation未達 +0.001               → 停止
validation通過                      → 2 個 matched QAT jobs（首 paired seed）
首 seed通過                         → 再補4個 jobs，完成3 paired seeds
~~~

返回[方向說明](<README.md>)或[優化方向索引](<../README.md>)。
