# Binary Distance-Codebook Normalization 研究支線設計

> 日期：2026-08-11
> 狀態：accepted
> 定位：proposed contribution hypothesis，不宣稱世界首次。

## 1. 目標與研究位置

BDCN 不取代既有 Exact/LUT/PWL/PoT/HardSigmoid/ReLU/Multimax screening，而是在共同 A0 checkpoint 後新增 proposed-method 支線。最終 A-FINAL 必須在 A0 Exact、既有 normalization winner 與 BDCN winner 之間公平選擇。

~~~text
A0
├─ Existing N0/N1 normalization branch
└─ Proposed BDCN branch
      ├─ D0 fixed exponential LUT reference
      ├─ D1 learnable monotonic-codebook ablation
      └─ D2 PoT projection and one recovery run
              ↓
Compare A0 / existing winner / BDCN winner
              ↓
           A-FINAL
~~~

所有 BDCN 實驗同時作用於：

- `model.10.m.0.attn`
- `model.22.m.0.1.attn`

不使用 KD，不量化 P/V/projection，不改 `PV+PE(V)`、output projection、FFN、residual、C2PSA/C3k2 或 Detect。

## 2. 演算法定義

對完整 Binary QK score（包含 scale 與 relative bias）做 row-wise max subtraction：

$$
d_{hij}=\max_k S_{hik}-S_{hij}\geq0.
$$

使用固定等距 bucket：

$$
b_{hij}=\operatorname{clip}\left(\operatorname{round}\left(\frac{d_{hij}}{\Delta}\right),0,L-1\right).
$$

第一版固定 $L=16$；$\Delta$ 只由 A0 calibration statistics 決定，不學習 bucket boundary。查詢正值、單調遞減 codebook：

$$
w_{hij}=C_h[b_{hij}],
\qquad
C_h[0]=1\geq C_h[1]\geq\cdots\geq C_h[15]>0.
$$

輸出保持正規化：

$$
P_{hij}=\frac{w_{hij}}{\sum_k w_{hik}}.
$$

Histogram dataflow 僅是等價重排：

$$
\sum_jw_{hij}=\sum_{l=0}^{15}n_{hi}[l]C_h[l].
$$

因此 direct sum 與 histogram 必須 bit-equivalent；histogram 只比較硬體 proxy，不列為獨立 accuracy 方法或 novelty。

## 3. 精簡實驗漏斗

### D0：最接近 prior-art 的控制組

| ID | Codebook | 訓練 | 目的 |
|---|---|---:|---|
| D0-IDX | 固定 $C[l]=\exp(-l\Delta)$ | 0 | 對照 IntAttention/IndexSoftmax-style fixed LUT |

D0 與現有 N0-LUT 共用相同 A0、bucket range 與 evaluator；若公式完全一致，只保留一份執行結果並用兩個名稱交叉引用，避免重複跑。

### D1：學習範圍 ablation

| ID | Codebook sharing | Trainable scope | Epochs |
|---|---|---|---:|
| D1-SHARED | 兩處、所有 heads 共用一表 | codebook only | 5 |
| D1-PATTN | 每個 Attention 一表 | codebook only | 5 |
| D1-PHEAD | 每個 Attention、每個 head 一表 | codebook only | 5 |

三者從相同 A0、相同 exponential initialization 開始。若較複雜版本相對下一個簡單版本的 mAP50–95 增益小於 0.001，選擇較簡單版本。D1 winner 相對 A0 loss 超過 0.01 時，停止 BDCN，不進 D2。

### D2：硬體表示

對單一 D1 winner 做零訓練投影：

| ID | 表示 | 一般乘法 |
|---|---|---|
| D2-FP | floating learned codebook | 需要或 LUT 儲存較寬 |
| D2-1P | $C[l]=2^{-a_l}$ | shift only |
| D2-2P | $C[l]=2^{-a_l}+2^{-c_l}$ | two shifts + add |

只允許一個硬體候選進入 5-epoch probability-level PMP recovery：

$$
P_{mix}=(1-\rho)P_{exact}+\rho P_{BDCN},
\qquad \rho:0\rightarrow1.
$$

優先順序是：先選通過 0.01 gate 的最低成本候選；若 D2-1P/2P 都未通過但 D2-FP 通過，只讓 D2-2P recovery，因其表示能力較高。不得把所有 PoT 版本都做完整訓練。

### D3：Denominator 與 fused value aggregation

BDCN 不是天然 division-free。令

$$
D_{hi}=\sum_l n_{hi}[l]C_h[l],
$$

則 exact normalization 仍需要 $1/D_{hi}$：

| ID | 方法 | 定位 |
|---|---|---|
| R0-DIV | 浮點 exact reciprocal | accuracy reference |
| R1-RLUT | leading-one normalization + reciprocal LUT | 主硬體方案 |
| R2-PSHIFT | denominator 投影到最近 $2^e$ 後 shift | division-free diagnostic |

只有 R1 未達 0.01 gate 時才增加一次 Newton–Raphson correction。R2 必須量測 row sum 偏離 1 的程度，不得稱 exact normalization。

硬體 reference 同時依 bucket 累加 Value：

$$
G_{hi}[l]=\sum_{j:b_{hij}=l}V_{hj},
$$

$$
N_{hi}=\sum_l C_h[l]G_{hi}[l],
\qquad O_{hi}=N_{hi}\widehat{D_{hi}^{-1}}.
$$

R0 direct $P\times V$ 與 fused grouped path 必須數值等價；它是 dataflow implementation，不另列 accuracy novelty。

## 4. A-FINAL 決策

最後只比較三個 parent：

1. A0 Exact Softmax。
2. 既有 N1 normalization winner。
3. BDCN D2 + D3 winner。

主要 gate 是相對 A0 的 COCO mAP50–95 loss 不超過 0.01。通過者再比較：

- codebook/LUT bytes。
- exp、一般乘法、shift、add、compare 數量。
- denominator accumulation 與 reciprocal 成本。
- histogram buffer/control 成本。
- P row-sum、KL divergence、top-k agreement。
- 兩個 Attention sites 各自的 score-distance/bucket occupancy。
- 端到端 latency proxy；未上板前不得宣稱 FPGA 實速。

$PV$ 仍為 dense floating path，除非最後另外選到 Multimax；BDCN 本身不得宣稱降低 $PV$ MAC。

## 5. Claim 邊界

可使用：

> 本研究提出一個針對 Binary QK 的 per-head learnable monotonic distance-codebook normalization，並研究其 two-term PoT 投影所形成的精度—硬體成本折衷。

不可使用：

- 首次提出 distance-bucket LUT Softmax。
- 首次使用 learnable LUT 或 per-head normalization。
- 首次使用 histogram denominator。
- 首次提出 two-term PoT 表示。

最接近的 prior art、來源與必要比較見 `../../research/bdcn-prior-art.md`。投稿前仍需依最終公式重做 citation graph、IEEE/ACM/Google Scholar 與專利檢索。

## 6. 程式邊界（核准後才實作）

- 新增獨立 `bdcn.py`，封裝 bucket calibration、monotonic codebook、PoT projection、reciprocal 與 grouped-value reference。
- `normalization.py` 只透過共同 `scores -> P` factory 接線，不塞入 BDCN 內部細節。
- `VariantConfig` 增加 BDCN enum 與明確欄位；不以 magic strings 控制 sharing/projection。
- `workflow.py` 新增 D0/D1/D2 steps，量化 Q phase 仍保持 optional。
- 測試涵蓋 row normalization、monotonic constraint、sharing parameter count、PoT projection、direct/histogram equivalence、兩個 Attention integration。

本規格已由使用者核准；正式 `plan.md`、workflow 與程式修改依對應 implementation plan 執行。
