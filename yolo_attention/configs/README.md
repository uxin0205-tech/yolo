# configs

設定分成兩個互不耦合的 seam：

~~~text
configs/
├── variants/   # p0、I/H/T5、BDCN D0/D1/D2/R0/R1/R2、Q1-L3A
├── training/   # screening、recovery、normalization、BDCN codebook、bias、Q2
└── evaluation/ # 全部 baseline/zero-train/final run 共用 COCO evaluator
~~~

`variants/` 只控制 basis、bias、scale、normalization 與 bit width，由 `VariantConfig` 驗證。`training/` 只控制 weights、data、epochs、batch、device、optimizer 與 LR，由 `TrainingRecipe` 驗證。

`q1-l3a.yaml` 是 Optional Phase Q 的 Hadamard + Decomposed bias 範例，不是主線必跑設定；只有量化另行核准後，才依 A-FINAL 複製成正式 Q1。

`normalization-recovery.yaml` 是 N1 的 5-epoch attention-only schedule。N0 是 zero-train 評估，不能使用這份 recipe。N0/N1 variant 必須從實際 A0 winner 複製後只改 `normalization` 與其專用參數，不能用固定 Identity template 偷換 architecture winner。

`bdcn-codebook.yaml` 是 D1 三個候選共用的完整 10-epoch codebook-only schedule。`bdcn-codebook-seed1.yaml` 只在排名落入 tie band 時，讓 winner 從共同 D0 parent 用 seed 1 再完整跑 10 epochs；兩者都只解凍 `BDCNCodebookBank`。`bdcn-d1-pattn.yaml`／`phead.yaml` 比較共享粒度；`bdcn-d2-1p.yaml`／`2p.yaml` 比較 codebook 投影；`bdcn-r0-div.yaml`、`r1-rlut.yaml`、`r2-pshift.yaml` 比較 denominator。這些模板目前以 Identity 為可執行 reference；正式 run 必須從實際 A0 複製其 basis、bias 與 scale，不得偷換 parent。

`bdcn-codebook-stable.yaml` 是 BDCN v3 的 5-epoch、LR `5e-6` codebook-only recipe；演算法設定位於 `BCND/configs/`，不在 training YAML 重複。

N0 的 `normalization` 對照如下：

| Run | YAML 值 | 額外欄位 |
|---|---|---|
| N0-EXACT | `exact` | 無 |
| N0-LUT | `lut` | `score_step`、`score_min` |
| N0-PWL | `piecewise_linear` | `pwl_segments: 16` |
| N0-SHIFT | `power_of_two` | project approximation |
| N0-HSIG | `hard_sigmoid` | 無 |
| N0-RELU | `relu` | `relu_margin` |
| N0-MK1/3/5 | `multimax` | `multimax_top_k: 1/3/5` |

進入 N1 時，再設 `normalization_progressive: true` 與 `normalization_transition_epochs: 5`。`integer_lut` 不屬於 N0；它會產生 U8 P，只保留給 Optional Phase Q。

檢查設定：

~~~bash
python -m yolo_attention.cli validate-config configs/variants/h-screen.yaml
~~~

禁止寫入 API key。正式 run 會把兩份 YAML 複製進 immutable artifact directory。

Queue 動態產生的 variant 不寫回本資料夾，而放在 `artifacts/queue/generated/<job-id>/variant.yaml`。它必須由 winner configuration 透過 `dataclasses.replace()` 派生，保留 basis、bias、scale 等未指定欄位；禁止用 committed template 的預設 Identity 偷換 winner。
