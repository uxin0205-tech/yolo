# YOLO26m Hardware-Friendly Binary Attention：主線研究計畫

> 狀態：程式骨架與 CPU reference 已完成；尚未開始 GPU baseline、正式訓練或量化搜尋。
> 模型：官方 Ultralytics `yolo26m.pt`。
> 資料：COCO2017 train2017／val2017，`imgsz=640`。
> 範圍：先做 PyTorch 與 bit-true 模擬；不做上板、HLS、Vivado 或 Vitis-AI。
> 限制：不使用 knowledge distillation（KD）。

## 1. 研究目標

本研究只重新設計 YOLO26m 兩個 Attention 的 **QK score 與 normalization path**，目標是在保留 global attention 功能的前提下，降低 QK 計算及 Softmax 的硬體成本，並量測實際精度損失。

正式 replacement 固定同時作用於：

- `model.10.m.0.attn`：C2PSA 內的 PSABlock Attention。
- `model.22.m.0.1.attn`：C3k2 內的 PSABlock Attention。

少任一處即視為 implementation failure；只改單一 site 僅可作故障診斷，不能列為正式結果。

核心問題：

1. Binary QK 能否在 COCO2017 上維持可接受的偵測精度？
2. Identity、Hadamard matched-dual basis（MDB）與 BinaryQK T5 residual basis，哪一種精度／規則性最好？
3. Direct binary training 與 Progressive Blend，哪一種 recovery 方法較有效？
4. Dense 2D 或 Decomposed 2D relative bias 能否補回 binary score 遺失的位置資訊？
5. Binary score 的離散性是否足以支援 U8 normalized probability 與 integer LUT Softmax？
6. Multimax Top-K normalization 能否在維持 normalized attention 的同時降低 Softmax 與 \(P\times V\) 成本？
7. **可選延伸**：只有主研究完成且另行核准後，才研究 INT8 reference、非標準 3/5/6-bit、三元權重與 SD4；哪些候選需要 QAT？

### 1.1 保留與修改範圍

只修改：

- Q/K binary basis。
- XNOR + popcount score。
- magnitude／basis fusion scale。
- relative position bias。
- exact Softmax、LUT/PWL/PoT/HardSigmoid/ReLU 近似，或 Multimax sparse normalization。

完整保留：

- Q/K/V projection 的功能。
- global token-to-token attention。
- \(P \times V\)。
- \(PE(V)\) 及 \(PV+PE(V)\)。
- Attention output projection。
- PSABlock FFN 與兩個 residual。
- C2PSA 的 CSP split、bypass、concat 與 `cv2`。
- YOLO26 Detect 的 end-to-end／dual-head 介面。

不採用：

- \(Q \odot K\)。
- local key prior。
- 以 local element-wise product 取代 global \(QK^T\)。
- 沒有 row normalization 的 raw HardSigmoid。
- KD。

### 1.2 正式版本與可選延伸

- **V1**：Q/K 為方法本身的 1-bit representation；projection、Softmax、\(P\)、\(V\)、\(PV\) 先維持浮點。
- **A-FINAL**：完成 basis、training schedule、scale、relative bias 與 normalization 選擇後的最終演算法模型；它是本研究的正式終點。
- **Q-EXT（可選）**：只有 A-FINAL 完成且使用者／老師另行核准後，才從 A-FINAL 複製 quantization parent。INT8 只作 bit-true reference，不預先限定最終量化只能是 8-bit。

「V1 不做 INT8」不代表取消 binary Q/K。

## 2. YOLO26m Attention 邊界

研究資料流：

~~~text
YOLO26m C2PSA input X
  │
  ├─ cv1 → split → bypass a ──────────────────────────────────────┐
  │                                                               │
  └─ active b → PSABlock                                          │
                ├─ Q projection → Q ─┐                             │
                ├─ K projection → K ─┼→ Binary QK path → P         │
                └─ V projection → V ───────────────────┐           │
                                                       ↓           │
                                                    P × V          │
                                                       ├─ + PE(V)  │
                                                       ↓           │
                                                  Projection       │
                                                       ↓           │
                                     x₁ = x + Attention(x)         │
                                     x₂ = x₁ + FFN(x₁)             │
                                                       ↓           │
                                concat(a, x₂) ─────────────────────┘
                                                       ↓
                                                      cv2
~~~

以 token-major 記法：

$$
P=\operatorname{Softmax}(S)
$$

$$
O_{attn}=PV+PE(V)
$$

$$
O'=\operatorname{Proj}(O_{attn})
$$

PSABlock：

$$
x_1=x+\operatorname{Attention}(x)
$$

$$
x_2=x_1+\operatorname{FFN}(x_1)
$$

C2PSA：

$$
(a,b)=\operatorname{split}(cv1(X))
$$

$$
Y=cv2\!\left(\operatorname{concat}(a,\operatorname{PSABlocks}(b))\right)
$$

Ultralytics source 可能因 tensor layout 寫成 \(VP^T\)；它和 token-major 的 \(PV\) 是同一個 value aggregation。

## 3. Projection interface 與 P0 等價性

《No Attention, No Problem》只作為 modular projection interface 的設計靈感。本研究不採用它的 \(Q\odot K\) 或 normalization。

官方 QKV projection 使用 `Conv(dim, h, 1, act=False)`，實際是：

~~~text
Conv2d(bias=False) → BatchNorm2d → Identity
~~~

演算法介面可拆成：

$$
Q=BN_Q(Conv_Q(X)),\quad
K=BN_K(Conv_K(X)),\quad
V=BN_V(Conv_V(X))
$$

從 fused QKV 切分時，必須一起切：

- Conv weight \(W\)。
- BN affine parameters \(\gamma,\beta\)。
- BN running mean \(m\)。
- BN running variance \(v\)。
- `num_batches_tracked`。

原 Conv2d 為 `bias=False`，因此沒有 convolution bias 可直接切分。

官方 forward 先把 fused output reshape 成
\([B,h,2d_k+d_v,N]\)，再於每個 head 內切成 Q/K/V。因此實際 channel
順序是逐 head 的 \([Q_h,K_h,V_h]\)，不是三段連續的
\([Q_{all};K_{all};V_{all}]\)。modular projection 必須依 head gather
正確 channel indices；P0 比較時則把三路輸出重新 interleave 成官方
fused layout。

### 3.1 BN fold

bit-true、PTQ 與 export 前，將最新 checkpoint 的 Conv+BN fold：

$$
W_c^{fold}
=
\frac{\gamma_c}{\sqrt{v_c+\epsilon}}W_c
$$

$$
b_c^{fold}
=
\beta_c-\frac{\gamma_c m_c}{\sqrt{v_c+\epsilon}}
$$

固定流程：

~~~text
latest checkpoint
  → eval()
  → read latest BN parameters/statistics
  → Conv+BN fold
  → folded/unfolded equivalence check
  → PTQ / bit-true / export
~~~

不能使用 P0 時期 fold 出來的參數代表訓練後 checkpoint。

### 3.2 P0 validation gates

| Gate | 必須成立的條件 |
|---|---|
| P0-A | fused QKV Conv+BN 與 split Q/K/V Conv+BN 的 Q、K、V 等價 |
| P0-B | Conv+BN 與 folded Conv 等價，fold 後 split 仍等價 |
| P0-C | 完整 Attention、PSABlock、C2PSA 輸出等價，包含 \(PE(V)\)、FFN、residual 與 CSP branch |

P0 任一項失敗就停止後續 COCO 實驗。P0 是 implementation validation，不是演算法貢獻。

split projection 與 fused projection 的理論 MAC 相同：

$$
C_{in}(C_Q+C_K+C_V)HW=C_{in}C_{QKV}HW
$$

split 也不保證 memory traffic 較低。未來上板才比較一次讀取 \(X\) 的 fused engine 與共享／不共享 input buffer 的 parallel Q/K/V engine。

## 4. Layer 1：Binary QK 候選

共同記號：

- \(h\)：head index。
- \(i,j\)：query/key token index。
- \(d_k\)：每個 head 的 Q/K channel dimension。
- \(\tau=d_k^{-1/2}\)：attention score normalization。
- \(z_{ij}^{(r)}\)：第 \(r\) 路 binary dot score。
- \(\mu_Q,\mu_K\)：sign binarization 的 magnitude scale。
- \(\gamma_r\)：basis fusion coefficient。
- \(\rho_X^{(r)}\)：T5 residual reconstruction coefficient。

binary similarity：

$$
z_{ij}
=
2\,\operatorname{Popcount}
\left(
\operatorname{XNOR}(B_{Q,i},B_{K,j})
\right)-d_k
$$

### 4.1 I：Identity-only

$$
B_Q^{(I)}=\operatorname{sign}(Q),\qquad
B_K^{(I)}=\operatorname{sign}(K)
$$

$$
S_{bhij}^{I}
=
\tau\gamma_I
\mu_{Q,bh}^{(I)}
\mu_{K,bh}^{(I)}
z_{hij}^{(I)}
$$

I 是一條 binary QK 的 screening control。

### 4.2 H：Identity + Hadamard MDB

第一路使用原始 channel；第二路以相同 Hadamard transform 混合 Q/K channel：

$$
Q^{(H)}=\frac{QH}{\sqrt{d_k}},
\qquad
K^{(H)}=\frac{KH}{\sqrt{d_k}}
$$

$$
B_Q^{(H)}=\operatorname{sign}(Q^{(H)}),
\qquad
B_K^{(H)}=\operatorname{sign}(K^{(H)})
$$

training 使用 normalized Hadamard，避免 activation 大量落在 clipped STE window 外。對非零輸入：

$$
\operatorname{sign}
\left(
\frac{QH}{\sqrt{d_k}}
\right)
=
\operatorname{sign}(QH)
$$

因此 inference sign path 不必真的除以 \(\sqrt{d_k}\)；需要的常數可吸收到 scale。

magnitude estimator 使用每個 sample/head、跨 token 與 channel 的 mean absolute：

$$
\mu_{Q,bh}^{(r)}
=
\frac{1}{Nd_k}
\sum_{i,c}|Q_{bhic}^{(r)}|
$$

$$
\mu_{K,bh}^{(r)}
=
\frac{1}{Nd_k}
\sum_{j,c}|K_{bhjc}^{(r)}|
$$

其中 \(r\in\{I,H\}\)。完整 score：

$$
S_{bhij}^{H}
=
\tau
\left[
\gamma_I\mu_Q^{(I)}\mu_K^{(I)}z_{hij}^{(I)}
+
\gamma_H\mu_Q^{(H)}\mu_K^{(H)}z_{hij}^{(H)}
\right]
$$

### 4.3 T5：Residual matched basis

第一個 binary approximation：

$$
\rho_{X,bhi}^{(1)}
=
\frac{1}{d_k}\sum_c|X_{bhic}|
$$

$$
U_1=\rho_X^{(1)}\operatorname{sign}(X)
$$

第二路近似 residual：

$$
R=X-U_1
$$

$$
\rho_{X,bhi}^{(2)}
=
\frac{1}{d_k}\sum_c|R_{bhic}|
$$

$$
U_2=\rho_X^{(2)}\operatorname{sign}(R)
$$

Q/K 各得到兩路，只計算 matched pairs：

$$
S_{bhij}^{T5}
=
\tau
\left[
\gamma_1\rho_{Q,bhi}^{(1)}\rho_{K,bhj}^{(1)}z_{hij}^{(1)}
+
\gamma_2\rho_{Q,bhi}^{(2)}\rho_{K,bhj}^{(2)}z_{hij}^{(2)}
\right]
$$

不計算 \(Q_1K_2^T\) 或 \(Q_2K_1^T\)。T4 full cross-basis 不進入主實驗，避免把 binary QK 次數由 2 增加到 4。

### 4.4 三個候選的角色

| 方法 | Binary QK 次數 | 第二路來源 | 主要風險 |
|---|---:|---|---|
| I | 1 | 無 | 表達能力最低 |
| H | 2 | 固定 Hadamard add/sub | 尚需驗證 COCO recovery |
| T5 | 2 | 第一個 binary approximation 的 residual | per-token dynamic coefficient 較複雜 |

I/H/T5 必須使用相同 projection、exact Softmax、外層 C2PSA、optimizer、augmentation、seed 與訓練長度。

## 5. Layer 2：Scale 與 Relative Bias

### 5.1 Scale ablation

Scale 不與 I/H/T5 screening 同時變動。architecture winner 形成 formal V1 後才測：

| ID | Scale | 用途 |
|---|---|---|
| V1-DYN | 原始 dynamic magnitude/reconstruction coefficient | 浮點精度上界 |
| V1-SHEAD | calibration 後固定 per-head/per-basis coefficient \(\widehat\kappa_h^{(r)}\) | integer score 主候選 |
| V1-P2 | 將 \(\widehat\kappa_h^{(r)}\) 量化為最近的 \(2^k\) | shift-only 候選 |

執行方式：

1. V1-DYN 是 formal V1 reference。
2. V1-SHEAD 先以相同 COCO val pass 累積 per-head/per-basis coefficient，freeze 後再跑一次正式 evaluation；calibration pass 的 metric 不列入比較。
3. V1-P2 使用同一 calibration 流程，freeze 時量化為最近的 ^kcalibrated checkpoint 必須保存，供後續 bias branch 載入。
4. 候選相對 V1-DYN 掉點超過 0.001 時，只有預計送入 A0 的一個候選可補 3 epochs recovery。

### 5.2 Dense／Decomposed 2D bias

Bias 加在 basis score 合併後、Softmax 前：

$$
S_h(i,j)=S_h^{bin}(i,j)+B_h(i,j)
$$

令：

$$
\Delta y=y_i-y_j,\qquad
\Delta x=x_i-x_j
$$

Dense 2D：

$$
B_h^{dense}(i,j)=T_h[\Delta y,\Delta x]
$$

Decomposed 2D：

$$
B_h^{decomp}(i,j)
=
T_h^y[\Delta y]+T_h^x[\Delta x]
$$

以 `max_bias_size=32`、4 heads 計：

$$
P_{dense}=4\times63\times63=15{,}876
$$

$$
P_{decomp}=4\times(63+63)=504
$$

兩者統一 zero-init，並從同一 formal V1 checkpoint 分支：

| ID | 實驗 | 訓練 |
|---|---|---:|
| V1-B0 | 不加 bias 的時間控制組 | 5 epochs |
| V1-BD | Dense 2D bias | 5 epochs |
| V1-BR | Decomposed 2D bias | 5 epochs |

若 V1-BD 與 V1-BR 收益差異小於 0.001，優先選參數較少的 V1-BR。

## 6. Layer 3：Normalization 與延後量化

### 6.1 V1 exact Softmax

$$
m_i=\max_j S_{ij}
$$

$$
P_{ij}
=
\frac{\exp(S_{ij}-m_i)}
{\sum_k\exp(S_{ik}-m_i)}
$$

V1 保留：

$$
P_{ij}\ge0,\qquad
\sum_jP_{ij}=1
$$

I/H/T5 與 Direct/Progressive 比較時都使用 exact Softmax，避免同時改兩個核心因素。

### 6.2 Binary-Score-Aware Integer LUT Softmax

若 \(d_k=32\)，每個 raw binary dot score 只有 33 種：

$$
z_{ij}^{(r)}
\in
\{-32,-30,\ldots,30,32\}
$$

但 scale、fusion coefficient 與 bias 會使最終 score 不再天然是整數。因此若未來核准 Optional Phase Q，必須先建立共同 score grid。

令 \(\Delta_s\) 為 score step：

$$
\bar\kappa_h^{(r)}
=
\operatorname{round}
\left(
\frac{\kappa_h^{(r)}}{\Delta_s}
\right)
$$

$$
B_{hij}^{int}
=
\operatorname{round}
\left(
\frac{B_{hij}}{\Delta_s}
\right)
$$

$$
s_{hij}^{int}
=
\operatorname{clip}
\left(
\sum_r\bar\kappa_h^{(r)}z_{hij}^{(r)}
+B_{hij}^{int},
s_{min},
s_{max}
\right)
$$

row-wise max subtraction：

$$
u_{ij}^{int}
=
s_{ij}^{int}-\max_j s_{ij}^{int}
$$

exp LUT、row sum 與 reciprocal LUT：

$$
e_{ij}^{int}=LUT_{\exp}(u_{ij}^{int})
$$

$$
E_i=\sum_j e_{ij}^{int}
$$

$$
\widehat r_i=LUT_{recip}(E_i)
$$

U8 probability：

$$
P_{ij}^{U8}
=
\operatorname{clip}
\left(
\operatorname{round}
\left(
\frac{255\,e_{ij}^{int}}{E_i}
\right),
0,255
\right)
$$

### 6.3 Row-sum correction 分級

| ID | 方法 | 定位 |
|---|---|---|
| L3-A | 不做 exact correction；量測 \(\sum_jP_{ij}^{U8}\) 誤差 | 主方案 |
| L3-B | \(\delta=255-\sum_jP_{ij}^{U8}\)，加到最大 probability 元素 | L3-A 不合格時才跑 |
| L3-C | largest-remainder correction | software upper bound，不作主硬體方案 |

L3-A 要回報 row sum 為 254、255、256 的比例及 mean/max absolute error。

### 6.4 N0/N1 Normalization screening

Normalization 必須在 architecture、Direct/Progressive、scale 與 relative bias 都選定，形成共同 A0 checkpoint 後才比較。所有 N0 run 同時替換兩個 Attention，且不得量化 P、V 或 projection。

| ID | 方法 | 訓練 | 定位 |
|---|---|---:|---|
| N0-EXACT | exact Softmax | 0 | 同一 A0 reference |
| N0-LUT | score-indexed exp LUT + reciprocal | 0 | binary-score-aware dense path |
| N0-PWL | 16-segment piecewise-linear exp + normalize | 0 | multiplier/comparator trade-off |
| N0-SHIFT | project PoT exp + normalize | 0 | shift-oriented approximation |
| N0-HSIG | normalized HardSigmoid | 0 | low-cost dense diagnostic |
| N0-RELU | shifted ReLU + row normalization | 0 | 最低成本 dense diagnostic |
| N0-MK1 | floating Multimax Top-1 | 0 | aggressive sparse diagnostic |
| N0-MK3 | floating Multimax Top-3 | 0 | sparse accuracy/cost candidate |
| N0-MK5 | floating Multimax Top-5 | 0 | sparse accuracy/cost candidate |

所有候選必須滿足：

$$
P_{ij}\ge 0,\qquad \sum_jP_{ij}=1.
$$

PoT 是本專案的軟體 reference approximation，不預先宣稱為特定 Shiftmax 論文的 bit-true reproduction。N0 相對同一 A0 的 COCO mAP50–95 loss 必須不超過 0.01；通過後依 accuracy、normalization cost、memory/dataflow 與可規則化程度最多保留兩個候選。

N1 只對保留候選各做 5 epochs attention-only recovery，使用共同 probability-level progressive mix：

$$
P_{mix}=(1-\rho)P_{exact}+\rho P_{candidate},
\qquad \rho:0\rightarrow1.
$$

這個 $\rho$ 和 score-level Progressive Blend 的 $\lambda$ 是不同位置的 training trick，報告必須分開。N1 winner 與 A0 exact 一起比較精度與硬體成本，最後選 A-FINAL。

### 6.4.1 Multimax 的來源與公式

Multimax 來自陳法諭 2025 年碩士論文第 3.6.1 節。它不是三元權重量化，而是另一種 sparse normalization：每列只保留 Top-K score，論文支援

$$
K\in\{1,3,5\}.
$$

為避免和本計畫以 \(N\) 表示 token 數混淆，本計畫一律使用 \(K\) 表示 Multimax 保留數量。對排序後 score：

$$
s_1\ge s_2\ge\cdots\ge s_K,
$$

以相鄰 score difference 計算 sigmoid ratio，並遞迴分配剩餘機率。以 \(K=3\) 為例：

$$
a_1=\sigma(s_1-s_2),\qquad p_1=a_1,
$$

$$
a_2=\sigma(s_2-s_3),\qquad p_2=(1-p_1)a_2,
$$

$$
p_3=1-p_1-p_2.
$$

因此仍有：

$$
\sum_{j=1}^{K}p_j=1,
$$

但未進 Top-K 的位置全部為 0。Binary QK 的 score 與 score difference 都是離散值，因此後續可把 sigmoid 實作為 difference-indexed LUT；fixed-point 時最後一項直接使用 remainder，可令 U8 row sum 恰為 255。

N0-MK1/3/5 使用 floating sigmoid，先隔離 Top-K 與遞迴 normalization 本身的誤差；此時不加入 integer LUT、P/V quantization 或 ternary/SD4。只有被選入最多兩個 N1 候選時才做 PMP。完整論文分析見 `docs/research/ternary-accelerator-section-3.6.1.md`。

### 6.5 BDCN：Binary Distance-Codebook Normalization 研究分支

BDCN 是 A0 上的 normalization/dataflow 候選，不取代 I/H/T5 architecture screening。它利用 binary score 的離散性，把 row-max distance 量化成有限 bucket：

$$
d_{ij}=\max_k S_{ik}-S_{ij},
$$

$$
b_{ij}=\operatorname{clip}\left(\operatorname{round}\left(\frac{d_{ij}}{\Delta_d}\right),0,L-1\right),
$$

$$
w_{ij}=C_{b_{ij}},\qquad D_i=\sum_jw_{ij}.
$$

其中 $C_0=1$ 且 $C_l$ 單調非增。dense reference 為

$$
P_{ij}=w_{ij}\widehat{D_i^{-1}}.
$$

主要 dataflow 不 materialize $P$，而是先依 bucket 聚合 value：

$$
G_{il}=\sum_{j:b_{ij}=l}V_j,
$$

$$
O_i=\left(\sum_{l=0}^{L-1}C_lG_{il}\right)\widehat{D_i^{-1}}.
$$

這個重排在 R0 exact reciprocal 時與 $PV$ 數學等價，並完整保留 $PE(V)$、output projection、FFN、兩個 residual 與 C2PSA/C3k2 外層。它移除 dense probability tensor，但不代表運算免費；報告必須另列 bucket comparison/value add、$L$ 組 codebook weighting 與每 query row 的 reciprocal。

軟體 AMP 模擬中，bucket partial sum、codebook weighting 與 reciprocal 統一使用 FP32 accumulator，完成 normalized output 後才轉回 V dtype。這是避免 FP16 partial sum 在正規化前溢位的 reference 規則；硬體報告則另外根據 value range 與最大 bucket occupancy 計算 accumulator guard bits。任何 training batch 出現 NaN/Inf 必須立即停止，不得繼續產生 checkpoint 或 mAP。

| 階段 | Run | 內容 | 訓練 |
|---|---|---|---:|
| D0 | D0-IDX | fixed exponential table；確認 distance indexing 本身 | 0 |
| D1 | D1-SHARED / D1-PATTN / D1-PHEAD | global、每 Attention、每 head 的 learned monotonic table | seed 0，各一次完整 10 ep codebook-only |
| D2 | D2-FP / D2-1P / D2-2P | winner table 保持 float、投影成一項或兩項 power-of-two | 0；最多一個候選補 5 ep |
| R0 | R0-DIV | exact division/reciprocal numerical reference | 0 |
| R1 | R1-RLUT | leading-one normalization + reciprocal LUT | 0，主要硬體候選 |
| R2 | R2-PSHIFT | $D_i\rightarrow2^{\operatorname{round}(\log_2D_i)}$，shift-only | 0，division-free diagnostic |

R1 並沒有消除 normalization；它把通用除法改成每列一次 reciprocal-LUT lookup 與乘法。R2 才能稱為 division-free，但會破壞 exact row sum，因此只用來量測精度／成本邊界。nearest-PoT denominator 的 row sum 理論範圍是 $[2^{-1/2},2^{1/2}]$；只要輸出與 artifact 仍為有限值，低 mAP 必須保留為負結果，不得被 queue 當成 worker 故障，也不得阻塞只依 R0/R1 判斷的 denominator selection。若 R1 相對 R0 的 mAP loss 超過 0.002 或 row-sum max error 超過 0.01，才加入一次 Newton reciprocal correction；否則不增加實驗。

D2 與 R0/R1/R2 都必須從 D1 winner checkpoint 載入同一組 learned codebook parameters，再套用 float/1-PoT/2-PoT projection 或 denominator variant。variant reconfiguration 不得重新使用 exponential initializer；若 codebook kind、sharing、levels 或 step 不相容，必須明確建立新研究分支，不能靜默搬移 state。

D1 三個 sharing 候選依序排程，各自從共同 D0 parent 進行一次完整 10-epoch seed-0 codebook-only training，再直接比較三個結果。若前兩名差小於 0.001，才從共同 D0 parent 對 winner 補一個完整 10-epoch seed 1；seed 1 只作重現性報告，D2 固定沿用 seed-0 winner，避免 best-seed selection bias。既有 artifacts 的 staged 5+5 結果維持其原始名稱與方法註記，不回寫成新版單次訓練。

D1 sharing 以與最佳者差小於 0.001 時選較簡單者；D2/R1/R2 相對 A0 的 mAP loss gate 為 0.01。所有 run 同時替換 YOLO26m 兩個 Attention，使用相同 A0、COCO evaluator 與未修改模組；除明列的 seed 1 confirmation 外均固定 seed 0。BDCN core 和 IntAttention/IndexSoftmax 接近，因此不得主張 distance-indexed normalization 為本研究首次提出；可檢驗的研究點是 Binary-QK 情境、跨兩個 Attention 的 learned sharing、PoT table projection、denominator ablation 與 fused bucket-$PV$ 組合。完整 prior-art 見 `docs/research/bdcn-prior-art.md`。

### 6.6 Optional Phase Q：延後量化、PTQ 與 QAT

量化不是主研究必做項目。A-FINAL 的完整 COCO evaluation、運算量報告與方法結論完成後，主研究即可結束。只有使用者或老師明確決定繼續 Phase Q，才啟動正式 bit-width／format 搜尋；沒有新核准時，Q0/Q1/Q2、TW0/TW1 與 SD4-0 全部保持 dormant。

即使決定執行 Phase Q，以下條件也必須全部完成：

1. I/H/T5 architecture winner 已選定。
2. Direct／Progressive recovery 已完成並產生 formal V1。
3. scale、relative bias 與 Multimax 是否保留已決定。
4. 最新 checkpoint 已重新 `eval()`、BN fold 並通過等價性驗證。

本研究有兩類不同的「QAT」概念：

- Binary Q/K 的 `sign` 需要 STE 訓練；這是 V1 必要的 binary-aware training。
- projection、activation、P/V、PE/FFN 等數值量化的 QAT 是最後階段，目前尚未執行。
- uniform integer quantization 先做 PTQ，PTQ 不合格才做 QAT。
- 三元權重與 SD4 是結構化 weight format，不假設單純 PTQ 可恢復精度；正式採用時需另立 QAT schedule。

目前已實作的 INT8 reference 漏斗只作 optional extension 備案：

| ID | 內容 | 訓練 |
|---|---|---:|
| Q0 | 選定 parent；projection/P/V 做 W8A8/U8/S8，仍用 exact Softmax | PTQ |
| Q1 | Q0 加 integer score grid 與 LUT Softmax | 無 |
| Q2 | Q1 的 no-KD INT8 recovery | 條件式 5 epochs |

Q0/Q1/Q2 是驗證 bit-true 資料流的 reference，不是最終 bit-width 結論。正式量化研究必須支援中間 bit-width。第一輪 coarse sweep 使用：

$$
b\in\{8,6,5,4\}.
$$

若某 tensor group 在 6-bit 首次不合格，才補 7-bit；若 4-bit 仍合格，才向下補 3-bit。2-bit 只作壓力測試，16-bit 只作 numerical ceiling，且必須分清 INT16 與 FP16。量化器本身仍須接受任意 `bits >= 2`，不可只寫死上述候選。不得一次展開 weight bit × activation bit × layer × format 的完整 Cartesian product，而採分層敏感度漏斗：

1. **Q-SENS**：固定其他 tensor，逐一量化 projection weights、projection activations、\(P\)、\(V\)、score/LUT、PE/output projection 與 FFN；使用固定 calibration subset 量測每個 \(b\) 的誤差。
2. **Q-PARETO**：依 mAP loss、model bits、estimated memory traffic 與硬體 primitive，保留最多三個候選。
3. **Q-COMB**：只對上述候選做組合量化與完整 COCO val。
4. **Q-QAT**：PTQ 規格合理但未達門檻時，只對最佳一至兩個候選做 no-KD QAT；不為每個 bit-width 各跑一輪完整訓練。

量化格式至少分三族，不可只用 bit 數命名：

| Format family | Weight 表示 | Activation | 初始定位 |
|---|---|---|---|
| Uniform integer | signed symmetric per-channel weight | per-tensor／per-channel 候選 | PTQ 起步；支援任意 2–16 bit |
| Ternary weight | \(\{-\alpha,0,+\alpha\}\) | bit-width 另選 | 結構化低位元；另立 QAT |
| SD4 | 4-bit signed-digit weight codebook | bit-width 另選 | multiplier-less weight 候選；另立 QAT 與硬體成本模型 |

老師所說的 SD4 在目前硬體量化脈絡下，最可能指 Hsieh、Liu、Chiueh 於 2021 年提出的 Signed-Digit-4／FloatSD4 weight format，但縮寫並不唯一。必須先向老師確認論文，再從全文鎖定 codebook、scale、encoding、gradient 與支援算子；確認前只列 backlog，不猜數值集合。SD4 不等同一般 INT4，也不等同三元權重。

ternary 與 SD4 不加入 uniform bit-width 曲線：

- **TW0**：先只量化 Q/K/V 與 output projection weights，activation 維持 A8，採 QAT。
- **TW1**：TW0 達標後，activation 才按 6→5→4-bit 降低，最多選一組進 QAT。
- **TW-ALL**：只有 Attention-local ternary 已證明可恢復，才考慮 PE／FFN／其餘 YOLO26 weights；不直接從全模型開始。
- **SD4-0**：老師確認來源且取得完整 codebook 後才建立，否則保持 backlog。

三元權重論文採用 FP32 → ternary-weight/FP32-activation → ternary-weight/INT4-activation 的 staged QAT，並加入 progressive quantization。這可列為 TW branch 的候選 schedule，但需和 Direct QAT 做單一對照，不能未經 YOLO26 實驗便視為必要步驟。

PTQ 順序：

1. MinMax calibration。
2. 不合格才改 Percentile calibration。
3. 以 per-layer sensitivity 找主要掉點層。
4. 檢查 clipping、scale、LUT range、overflow 與 row normalization。
5. 規格合理但 INT8 Q1 相對 parent 仍掉超過 0.002，才跑 reference Q2。

低於 8-bit、三元權重與 SD4 的 QAT schedule 暫不在本階段定案。候選確定後，再比較 Direct QAT 與 staged/progressive QAT；後者可採先 weight、後 activation，或 progressive quantization mask，但不能直接假設 Transformer 論文的 schedule 適用 YOLO26。

主線使用 Custom PyTorch fake quant 與 NumPy/PyTorch bit-true reference。量化器 API 必須以 `bits: int` 接收任意 bit-width，不為 INT5／INT6 寫死不同實作。外部量化框架最多選一套交叉驗證 standard Conv，不同框架不形成新的主實驗。

## 7. 正式實驗漏斗

### 7.1 固定 baseline

| ID | 模型 | 訓練 | 輸出 |
|---|---|---:|---|
| B26-FP | untouched `yolo26m.pt` | 無 | COCO2017 val；default one-to-one/e2e 為 primary，one-to-many 為 auxiliary |

官方公開值只作 sanity check。所有正式掉點都以 pinned environment 中實測的 B26-FP e2e 為基準；不同 head mode 不得混算。

### 7.2 10-epoch architecture screening

| ID | 架構 | Schedule | Epochs |
|---|---|---|---:|
| I-SCR | Identity-only | Direct，\(\lambda=1\) | 10 |
| H-SCR | Identity + Hadamard MDB | Direct，\(\lambda=1\) | 10 |
| T5-SCR | residual matched basis | Direct，\(\lambda=1\) | 10 |

screening 規則：

- 三者都從同一份通過 P0 的 checkpoint 開始。
- 不使用 KD。
- 只解凍兩個目標 Attention module；其餘 backbone、neck、Detect 與 PSABlock FFN 凍結。
- 三者使用 exact Softmax、相同 optimizer、LR、batch、augmentation 與 seed 0。
- 本輪只比較 architecture；不加入 relative bias、INT8 或 Progressive Blend。
- 以 COCO mAP50–95 選 winner。
- 差異小於 0.001 時，以額外運算較少、coefficient 較規則、bit-true 誤差較小者優先。
- 若結果接近 evaluator 波動，再補 seeds 1/2；否則不增加 screening runs。

zero-shot replacement 不列正式實驗，只在 forward 或 training 異常時作 diagnostic。

### 7.3 Winner-only 20–40 epoch recovery

winner 從相同 P0 checkpoint 分成：

| ID | Schedule | Epochs |
|---|---|---:|
| W-DIR | epoch 0 起 \(\lambda=1\)，全程 binary score | 20–40 |
| W-PROG | 前 10 epochs 令 \(\lambda:0\rightarrow1\)，之後 pure binary | 20–40 |

Progressive score：

$$
S_{blend}
=(1-\lambda)S_{FP}+\lambda S_{bin}
$$

recovery 採 full-model fine-tuning；W-DIR 與 W-PROG 必須使用相同資料、optimizer、LR 與 early-stop 規則：

- 至少訓練 20 epochs。
- 最多訓練 40 epochs。
- epoch 20 後以 validation mAP、patience 5 early stop。
- 保存 best validation checkpoint，不使用最後一個 epoch 取代 best。

mAP 較高者成為 formal V1。差異小於 0.001 時優先 W-DIR，因其訓練路徑較簡單。

### 7.4 V1 add-ons、Normalization 與 A-FINAL

formal V1 形成後：

1. 做 V1-DYN → V1-SHEAD → 條件式 V1-P2。
2. 從相同 formal V1 分支做 V1-B0／V1-BD／V1-BR。
3. 依 mAP、scale 成本與 bias 參數選出 algorithm parent A0。
4. 從同一 A0 做 N0-EXACT/LUT/PWL/SHIFT/HSIG/RELU/MK1/MK3/MK5 zero-train screening。
5. 通過 0.01 gate 後依 accuracy/cost 最多保留兩個候選，各做 N1 5-epoch attention-only PMP。
6. 從同一 A0 依序做 BDCN D0 → D1 三候選各完整 10 epochs → conditional winner seed 1 → D2 projection → R0/R1/R2；只讓通過 gate 的 BDCN winner 留下。
7. 依 mAP、normalization 成本與 dataflow，從 A0 exact、N1 winner、BDCN winner 選 A-FINAL。
8. 對 A-FINAL 執行完整 COCO val、運算量、latency proxy、memory 與 error analysis。
9. 記錄 A-FINAL 的完整 provenance，不覆寫 formal V1；完成研究報告後主線結束。
10. 只有另行核准 Optional Phase Q 時，才複製 A-FINAL 為 quantization parent，重新 `eval()`、BN fold 並通過等價性測試。

scale 的條件式 3-epoch recovery、bias 的 5-epoch branches 與 N1 都只解凍兩個目標 Attention module。Q2 則採 full-model fake-quant fine-tuning。

### 7.5 完整流程

~~~text
B26-FP
   ↓
P0-A / P0-B / P0-C
   ↓
10 ep Direct screening
   ├─ I-SCR
   ├─ H-SCR
   └─ T5-SCR
        ↓
 architecture winner
        ↓
20–40 ep recovery
   ├─ W-DIR
   └─ W-PROG
        ↓
    formal V1
        ↓
scale / bias add-ons → A0
        ↓
N0 normalization zero-train screening
   ├─ EXACT / LUT / PWL / SHIFT / HSIG / RELU
   └─ Multimax K=1 / 3 / 5
        ↓（通過 0.01 gate，最多兩個）
N1 probability-level PMP recovery，5 ep
        ↓
BDCN: D0 → D1 sharing，三候選各完整 10 ep codebook-only
      → conditional winner seed 1，完整 10 ep from D0
      → D2 FP / 1-PoT / 2-PoT
      → R0 exact / R1 reciprocal LUT / R2 PoT shift
        ↓
select A-FINAL: A0 exact / N1 winner / BDCN winner
        ↓
A-FINAL full COCO evaluation
accuracy / operations / latency proxy / memory / errors
        ↓
MAIN RESEARCH COMPLETE
        ↓
是否另行核准 Optional Phase Q？
   ├─ 否 → STOP；量化不執行
   └─ 是
        ↓
clone A-FINAL → quantization parent
        ↓
Q-SENS: 8→6→5→4-bit；條件式補 7/3-bit
        ↓
Q-PARETO → Q-COMB → optional PTQ/QAT
~~~

## 8. 驗證、指標與停止條件

### 8.1 必要測試

| 層級 | 測試 |
|---|---|
| Projection | fused/split Conv+BN、folded/unfolded、channel order、BN buffers |
| Integration | Attention、\(PE(V)\)、projection、PSABlock residual/FFN、C2PSA CSP |
| Binary | sign、XNOR、popcount、binary dot、zero policy |
| Hadamard | transform reference、Q/K matched transform、normalized training sign equivalence |
| Bias | zero-init 等價、offset 邊界、non-zero gradient |
| Softmax | exact row sum、LUT 無 NaN/Inf、L3-A/B/C row-sum 規則 |
| Multimax | stable Top-K/index、K=1/3/5、遞迴 sum=1、U8 remainder=255、sparse \(PV\) 等價 |
| BDCN | bucket 邊界、monotonic table、sharing、1/2-PoT、R0/R1/R2 row error、fused bucket-$PV$ 與 dense $PV$ 等價 |
| Quantization | 任意 2–16 bit range、rounding、saturation；U8 \(P\)、S8 \(V\)、S32 \(PV\) reference |
| Smoke | 一個 train batch、一個 val batch、backward、EMA、save/reload |

Smoke test 可使用固定 COCO 子集，但不能當正式 mAP。

### 8.2 正式指標

必報：

1. COCO mAP50–95。
2. AP50、AP75、APs、APm、APl。
3. 相對 B26-FP 或明確 parent 的 mAP loss。
4. parameters、FP MAC、packed XNOR-popcount、Hadamard add/sub、LUT lookup；BDCN 另列 bucket-value add、codebook weighting、reciprocal rows 與省下的 materialized-$P$ entries。
5. PyTorch latency 與 peak memory；只作軟體相對比較。
6. attention output cosine similarity。
7. quantization saturation、clipping、overflow、effective bits 與 row-sum error。
8. Multimax Top-K agreement、index correctness、稀疏度與 sparse \(PV\) memory read。

只有 mAP 明顯掉點時才加入 score entropy、top-k agreement、Spearman correlation 等 diagnostics。

### 8.3 門檻

- P0 任一 gate 失敗：停止，先修正 projection／BN fold。
- formal V1 相對 B26-FP e2e：目標 mAP loss 不超過 0.0035。
- V1-BD／V1-BR：只相對相同訓練時間的 V1-B0 判斷收益。
- V1-SHEAD／V1-P2：相對 V1-DYN 掉點超過 0.001 才允許單一候選補 3 epochs。
- N0 候選相對 A0 掉點超過 0.01：淘汰該候選，不進入 N1。
- 所有 N1 候選都不具可接受的 accuracy/cost Pareto：A-FINAL 固定使用 A0 exact Softmax。
- BDCN R1 相對 R0 row-sum max error 超過 0.01 或 mAP loss 超過 0.002：才允許加入一次 Newton reciprocal correction。

只有 Optional Phase Q 核准後才使用下列門檻：

- Q1-L3A 相對明確 quantization parent：目標 mAP loss 不超過 0.002。
- Q1 超過 0.002：先修 PTQ／LUT 規格；確認合理後才跑 Q2。

## 9. 運算量報告

下列先列單一 YOLO26m Attention block 的估算；正式模型有兩個相同 reference shape 的目標 sites，因此兩處 Attention path 合計值為表列值的 2 倍。這仍不代表整個 YOLO26m 的端到端實際加速。

固定近似：

$$
N=20\times20=400
$$

$$
h=4,\qquad d_k=32,\qquad d_v=64
$$

原始 FP QK：

$$
hN^2d_k
=
4\times400^2\times32
=
20.48\text{M MAC}
$$

PV：

$$
hN^2d_v
=
40.96\text{M MAC}
$$

Multimax sparse PV（不含 Top-K selection 與 index/gather control）：

$$
\operatorname{MAC}_{PV}^{multi}=hNKd_v.
$$

| Multimax | sparse PV | sigmoid LUT／row | 說明 |
|---|---:|---:|---|
| \(K=1\) | 0 probability MAC；0.1024M V elements selected | 0 | 直接選取對應 V row |
| \(K=3\) | 0.3072M MAC | 2 | dense PV 的 0.75% |
| \(K=5\) | 0.512M MAC | 4 | dense PV 的 1.25% |

全 block 的 sigmoid LUT 次數為 \(hN(K-1)\)：\(K=3\) 為 3,200 次，\(K=5\) 為 6,400 次。這些數字只描述 normalization 後的 sparse PV，不包含 streaming Top-K comparator、index buffer、V gather 或不規則記憶體存取。

BDCN fused bucket-$PV$ 在 $L=16$ 時：

$$
\operatorname{lookup}_{score}=hN^2=0.64\text{M},
$$

$$
\operatorname{add}_{bucket-V}=hN^2d_v=40.96\text{M},
$$

$$
\operatorname{op}_{codebook-V}=hNLd_v=1.6384\text{M},
$$

$$
\operatorname{rows}_{reciprocal}=hN=1{,}600.
$$

它以 bucket-value accumulation 取代 40.96M dense $PV$ MAC，並避免 materialize $hN^2=0.64$M 個 probability entries；但仍有 40.96M 次 value 累加與 bucket control。D2-1P 可把 codebook weighting 變成 shift，D2-2P 變成兩次 shift 加法；R1 每列做一次 reciprocal-LUT lookup，並對 $hNd_v=0.1024$M 個 output elements 乘 reciprocal。這是 primitive count，不可直接解讀成端到端 latency 或 energy 改善。

dual-basis 32-bit packed QK：

$$
2hN^2\frac{d_k}{32}
=
1.28\text{M packed word operations}
$$

Hadamard Q/K：

$$
2hNd_k\log_2d_k
=
0.512\text{M add/sub}
$$

主要項目：

| 模組 | 約略運算／參數 |
|---|---:|
| Q/K/V 1×1 Conv | 52.43M MAC |
| output 1×1 Conv | 26.21M MAC |
| \(PE(V)\) depthwise 3×3 Conv | 0.92M MAC |
| \(PV\) | 40.96M MAC |
| 原 FP QK | 20.48M MAC |
| dual binary QK | 1.28M packed word ops |
| Hadamard | 0.512M add/sub |
| PSABlock FFN | 104.86M MAC，保留 |
| C2PSA `cv1+cv2` | 209.72M MAC，保留 |
| Dense 2D bias | 15,876 parameters；約 0.64M lookup/add |
| Decomposed 2D bias | 504 parameters；約 1.28M lookup/add |
| exp LUT | 約 0.64M lookups |
| reciprocal LUT | \(hN=1{,}600\) lookups |
| Multimax Top-K | comparator/index 成本依架構另報，不併入 MAC |

解讀限制：

- Binary MDB 本身只降低 QK，不會降低 \(PV\)、projection、FFN 或 C2PSA `cv1/cv2`；只有採用 Multimax sparse path 才可能降低 \(PV\)。
- split Q/K/V 不改變 projection MAC。
- packed word operation 不能直接和 GPU FP MAC 當作相同時間單位。
- wall-clock 另以 PyTorch profiler 量測；不能從此表宣稱 FPGA speedup。
- memory traffic 同時報告 shared input buffer 與 \(X\) 被讀三次兩種模型。

## 10. 執行清單

- [ ] 鎖定 Ultralytics commit、`yolo26m.pt` hash、環境版本、COCO evaluator 與 e2e head mode。
- [x] 執行本地 B26-FP internal-metric baseline；canonical COCO JSON reference 待最終重測。
- [x] 實作 split Conv+BN 與最新 checkpoint BN fold 的程式骨架。
- [x] 以單元測試通過 P0-A、P0-B、P0-C；正式 checkpoint 仍須於訓練後重驗。
- [x] 實作 I、H、T5 float reference 與 XNOR-popcount reference。
- [x] 通過 binary、Hadamard、integration unit tests 與 CPU smoke test。
- [x] 建立並執行 persistent single-worker queue；加入 failure/retry、selection rewind/archive、動態 winner graph、共同 evaluator 與 analytical profile contract。
- [x] Queue 只包含 A-FINAL 前的主線；Optional Phase Q 維持 locked，不自動排入。
- [x] 執行 I/H/T5 各 10 epochs Direct screening；Hadamard 勝出。
- [ ] 重新執行 W-DIR 與 W-PROG recovery；舊結果因 custom parent state 未完整載入而封存。
- [ ] 重新選 formal V1 並保存共同 evaluator 結果。
- [ ] 執行 scale 與 bias add-ons，選定 algorithm parent A0。
- [ ] 從 A0 執行完整 N0 normalization screening；通過 gate 後最多兩個候選做 N1 PMP 5 epochs。
- [ ] 選定 A-FINAL，完成 full COCO evaluation 與 accuracy、運算量、latency proxy、memory、scale、bias、normalization error 報告。
- [ ] A-FINAL 報告完成即視為主研究完成；量化不作為完成條件。

Optional Phase Q（目前不執行；需使用者／老師另行核准）：

- [ ] 複製 A-FINAL 為唯一 quantization parent，重新 eval、BN fold 與等價性驗證。
- [ ] 執行 Q-SENS：8→6→5→4-bit；只在 gate 觸發時補 7/3-bit，2/16-bit 只作邊界參考。
- [ ] 只對 Q-PARETO 最多三個候選做 Q-COMB；只對最佳一至兩個不合格候選做 QAT。
- [ ] ternary 只開 TW0→條件式 TW1；SD4 只在老師確認原論文與完整格式後建立。
- [ ] 保留 Q0/Q1/Q2 作 INT8 bit-true reference，建立 integer score grid、LUT 與 overflow report。
- [ ] 使用者確認演算法結果後，再另立上板計畫。

## 11. 實驗產物

每個 run 至少保存：

- config snapshot。
- Git revision 與套件版本。
- base checkpoint hash 與 parent run ID。
- seed、trainable scope、head mode 與 COCO split。
- best checkpoint 與 best epoch。
- accuracy metrics。
- profiler 與 operation report。
- BN-fold equivalence report。
- PTQ/QAT encoding、LUT、rounding 與 saturation report。

大型 checkpoint、COCO 資料與 profiler trace 不提交 Git；詳細目錄規則見 `artifacts/README.md`。

## 12. 來源

研究與架構：

- No Attention, No Problem
  Paper: https://arxiv.org/abs/2607.13106
  HTML: https://arxiv.org/html/2607.13106
- BinaryAttention
  Paper: https://arxiv.org/abs/2603.09582
  GitHub: https://github.com/EdwardChasel/BinaryAttention
- Ultralytics YOLO26
  Documentation: https://docs.ultralytics.com/models/yolo26/
  Architecture: https://docs.ultralytics.com/guides/yolo-architecture/
  Model configs: https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26
  Attention/C2PSA source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py
  Conv+BN source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py

Integer Softmax／量化：

- Multimax / ternary-weight accelerator thesis
  Thesis: https://doi.org/10.6342/NTU202504546
  Local analysis: `docs/research/ternary-accelerator-section-3.6.1.md`
- SD4 multiplier-less CNN accelerator
  Paper: https://doi.org/10.1109/JETCAS.2021.3116044
  NTU record: https://scholars.lib.ntu.edu.tw/entities/publication/c1fea85e-53e2-44dc-8045-346f6b81ecb0
- Mixed-precision details and source audit
  Local analysis: `docs/research/deferred-mixed-precision-quantization.md`
- AdaBits
  Paper: https://openaccess.thecvf.com/content_CVPR_2020/html/Jin_AdaBits_Neural_Network_Quantization_With_Adaptive_Bit-Widths_CVPR_2020_paper.html
- HAQ
  Paper: https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf
- Ternary Weight Networks
  Paper: https://arxiv.org/abs/1605.04711

- I-ViT / Shiftmax
  Paper: https://arxiv.org/abs/2207.01405
  GitHub: https://github.com/zkkli/I-ViT
- FQ-ViT / Log-Int-Softmax
  Paper: https://arxiv.org/abs/2111.13824
  GitHub: https://github.com/megvii-research/FQ-ViT
- PyTorch AO QAT: https://docs.pytorch.org/ao/stable/workflows/qat.html
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- COCO: https://cocodataset.org/
