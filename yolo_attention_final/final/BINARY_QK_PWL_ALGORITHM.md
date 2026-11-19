# 創新與演算法設計：YOLO26m Binary Q/K＋Bit-True PWL Attention

## 1. 文件定位

本文是本 workspace 關於創新、演算法與硬體資料路徑的單一權威文件。內容以
production source、正式 configs 與 `pwl-final-best.pt` checkpoint contract 為準，
適合交給研究合作方、模型工程師、硬體工程師或作為論文／技術報告的設計基礎。

這裡的「創新」是指本專案已實作並整合的技術組合與工程貢獻，不宣稱每個 primitive
都是全球首次提出。若要提出專利或學術 first-claim，仍需另做完整 prior-art search。

固定模型是官方 YOLO26m、`scale=m`、COCO 80 classes，只替換兩個 Attention site：

- `model.10.m.0.attn`
- `model.22.m.0.1.attn`

## 2. 本專案的創新點

| 創新 | 解決的問題 | 本專案做法 | 已落地位置 |
| --- | --- | --- | --- |
| Deterministic Binary Q/K | 浮點 QKᵀ 乘法成本高，零值符號可能不一致 | `x >= 0 -> +1`，以 XNOR＋popcount 等價式計分 | `binary_basis.py` |
| Identity＋Hadamard 雙 binary basis | 單一 sign basis 表達力有限 | 同時計算原座標與 Walsh–Hadamard 座標的 binary similarity | `BinaryScore` |
| Per-head power-of-two scale | binary score 仍需要幅度校正 | 校正每個 head／basis 係數，再投影為 signed power-of-two | `BinaryScore.set_fixed_coefficients` |
| Decomposed 2D bias | Binary Q/K 容易失去空間相對位置 | 使用 y/x 兩個一維 relative bias table 相加 | `relative_bias.py` |
| Bit-True PWL exponential numerator | exact exp 不利於固定點硬體 | Q8.8 score、20 segments、UQ1.15 endpoints、整數 interpolation | `BitTruePiecewiseLinearSoftmax` |
| Float-PWL／Bit-True 雙路徑 | rounding、cast、indexing 無法穩定反向傳播 | Float-PWL＋STE 訓練；Bit-True materialization 做 gate／交付 | variants＋evaluation |
| Fail-closed 雙-site conversion | 部分替換模型可能靜默產生錯誤結果 | 必須同時且只存在兩個指定 Attention sites | `integration.py` |
| Bit-True gate＋rollback | surrogate 指標可能不代表部署路徑 | child 用完整 Bit-True val；退化超過 0.001 就 rollback | final backend／selection |
| Immutable queue＋provenance | 長訓練容易失去 parent、設定與結果血緣 | typed queue、atomic state、hash、manifest、append-only events | queue／artifact modules |

### 2.1 核心貢獻不是單一替換

本設計不是只把 softmax 換成一條 PWL，也不是只把 Q、K 做 sign。核心貢獻是把下列鏈條
放進同一個可訓練、可驗證、可交付的模型 contract：

```text
Binary Q/K
  + identity／Hadamard 多 basis
  + fixed power-of-two scale
  + spatial relative bias
  + Float-PWL training surrogate
  + Bit-True PWL deployment reference
  + Bit-True full-validation gate
  + checkpoint／queue provenance
```

### 2.2 可對外描述的簡短版本

> 本方法在 YOLO26m 的兩個 Attention sites 中，以 deterministic Binary Q/K、
> identity／Walsh–Hadamard 雙 basis 與 per-head power-of-two coefficients 近似 QK score，
> 再以 Q8.8-to-UQ1.15 的 20-segment Bit-True PWL exponential numerator 取代 exact exp。
> 訓練使用同 knots 的可微分 Float-PWL surrogate，正式選擇則回到 Bit-True checkpoint
> 完整驗證，並以 fail-closed gate 與 rollback 防止 surrogate improvement 被誤判為部署改善。

## 3. 模型整合與資料流

輸入 feature map：

```text
X: [B, C, H, W]
N = H × W
```

官方 fused QKV projection 依 per-head channel layout 精確拆成三條 Conv＋BN paths：

```text
Q, K: [B, heads, key_dim, N]
V:    [B, heads, head_dim, N]
```

完整 forward：

```text
X
├─ Q/K/V projection
├─ Binary Q/K score
│  ├─ identity binary basis
│  └─ Walsh–Hadamard binary basis
├─ decomposed 2D relative-position bias
├─ Float-PWL（training）或 Bit-True PWL（gate／inference）
├─ V × Pᵀ
├─ 原 YOLO positional encoding residual
└─ output projection
```

Value path、positional encoding 與 output projection 保留官方 YOLO26 行為；主要替換集中在
Q/K score 與 probability normalization。

## 4. Binary Q/K 演算法

### 4.1 Deterministic sign

對 Q、K 每個元素：

```text
sign(x) = +1, x >= 0
          -1, x < 0
```

零固定映射為 `+1`，避免 Python reference、checkpoint 與硬體 reference 在零值上分歧。

Float-PWL training variant 使用 clipped STE：

```text
d sign(x) / dx ≈ 1, |x| <= 1
                 0, otherwise
```

Bit-True evaluation／delivery variant 關閉 STE，只執行 deterministic forward。

### 4.2 XNOR＋popcount 等價內積

對長度 `D = key_dim` 的 binary vectors，令相同 bit 數量為 `M`：

```text
z(q, k) = 2M - D
```

它與 `sign(q) · sign(k)` 完全等價。production Python reference 使用 boolean equality
與 count 表達 XNOR＋popcount 語意；目前尚未宣稱已完成 bit-packed kernel。

### 4.3 Identity／Hadamard 雙 basis

```text
z_I = binary_dot(sign(Q), sign(K))
z_H = binary_dot(sign(H(Q)), sign(H(K)))
S   = α_I × z_I + α_H × z_H
```

`H(·)` 是沿 `key_dim` 的 fast Walsh–Hadamard transform，該維度必須是 2 的冪次。
Identity basis 保留原座標的 sign similarity；Hadamard basis 先混合 channel，再提供第二個
互補的 binary similarity。

### 4.4 Per-head fixed power-of-two coefficient

每個 head、每個 basis 都有固定 coefficient。校正值會投影到最近的 signed power-of-two：

```text
α_fixed = sign(α) × 2 ^ round(log2(|α|))
```

因此固定 coefficient multiplication 可映射成 sign control 與 shift。固定 scale inference
只需要 Hadamard output 的 sign；正的 normalization constant 不改變 sign，所以可以省略。

### 4.5 Decomposed 2D relative-position bias

```text
bias((y_i,x_i),(y_j,x_j))
  = table_y[y_i - y_j] + table_x[x_i - x_j]
```

相較 dense 2D table，decomposed 表示把垂直與水平位移拆成兩個一維表；它補回 Binary Q/K
本身不直接編碼的空間關係。

## 5. Bit-True PWL 演算法

### 5.1 Row centering

對每列 score：

```text
c_j = S_j - max_k(S_k)
```

因此 `c_j <= 0` 且 row maximum 固定為 0，再將範圍限制到 `[-10, 0]`。

### 5.2 固定數值規格

| 欄位 | 規格 |
| --- | --- |
| score format | signed Q8.8 |
| score range | `[-10, 0]` |
| segments | 20 |
| segment width | `0.5` |
| endpoints | 21 |
| endpoint format | unsigned UQ1.15 |
| endpoint bits | 16 |
| endpoint storage | 每個 site 336 bits；兩個 sites 共 672 bits |

第 `i` 個 knot 與 endpoint：

```text
x_i = -10 + 0.5i, i = 0...20
Y_i = round(exp(x_i) × 2^15)
```

### 5.3 Q8.8 indexing

```text
q = round(c × 2^8)
q = clamp(q, -2560, 0)
z = q + 2560
```

每個 segment 寬度：

```text
0.5 × 256 = 128 = 2^7
```

因此：

```text
segment_index = z >> 7
fraction      = z & 127
```

### 5.4 整數線性 interpolation

```text
Y = Y_i + ((fraction × (Y_(i+1) - Y_i)) >> 7)
```

`q = 0` 時直接使用最後一個 endpoint，避免 index 超出範圍。lookup、saturation、
segment indexing 與 interpolation 都重現預定 integer datapath。

### 5.5 Probability normalization

```text
P_j = (Y_j / 2^15) / Σ_k(Y_k / 2^15)
```

目前 numerator 是 Bit-True；denominator、row sum 與 reciprocal 仍是 exact float reference。
所以本版本不能宣稱全整數 softmax、division-free normalization 或完成硬體部署。

## 6. Training／deployment 雙路徑

Bit-True 路徑包含 rounding、integer cast、clamp 與離散 indexing，不能提供穩定的一般梯度。
本專案把 training 與 formal selection 分開：

| 階段 | Binary sign | Normalization | 用途 |
| --- | --- | --- | --- |
| Training | clipped STE | differentiable Float-PWL | optimizer backward |
| Diagnostics | clipped STE／float | Float-PWL | 訓練觀察，不作正式 winner |
| Gate／selection | deterministic | Bit-True PWL | 完整 COCO2017 val |
| Delivery／inference | deterministic | Bit-True PWL | 正式 checkpoint |

Float-PWL 與 Bit-True PWL 使用相同 `[-10,0]` range、20 segments 與 knots，降低 train／deploy
semantic gap；正式結論仍只採 Bit-True metrics。

## 7. Attention 輸出

得到 `P: [B, heads, N, N]` 後：

```text
A = V × Pᵀ
Y = output_projection(A + positional_encoding(V))
```

此處的 P/V aggregation 仍是 software reference，尚未完成 P/V 全整數 datapath。

## 8. Production source 對照

| 演算法／安全機制 | Source |
| --- | --- |
| 完整 Attention forward | `yolo_attention/attention.py` |
| Binary sign、XNOR＋popcount、Hadamard | `yolo_attention/binary_basis.py` |
| Float-PWL、Bit-True PWL | `yolo_attention/normalization.py` |
| Q/K/V projection layout | `yolo_attention/projection.py` |
| decomposed 2D bias | `yolo_attention/relative_bias.py` |
| YOLO26 兩-site fail-closed conversion | `yolo_attention/integration.py` |
| Float-PWL training Adapter | `yolo_attention/training.py` |
| parameter／BN buffer scope | `yolo_attention/scopes.py` |
| Bit-True materialization／evaluation | `yolo_attention/final_evaluation.py` |
| phase gate／final winner policy | `yolo_attention/final_selection.py` |
| immutable queue／provenance | `queue_*.py`、`artifacts.py` |
| training／delivery variants | `configs/variants/*.yaml` |

## 9. 已驗證的內容

- 正式 checkpoint 是 YOLO26m、scale m、80 classes。
- 兩個且只有兩個指定 HardwareFriendlyAttention sites。
- 正式 normalization 是 Bit-True PWL，`use_ste=false`。
- score range `[-10,0]`、20 segments、21 endpoints、每 site 336 endpoint bits。
- LR sweep／staged recovery queue 共 19 jobs，均已完成。
- Formal winner mAP50-95：`0.5067369`。
- Best-observed seed-0 mAP50-95：`0.5069387`。
- Best-observed 相對 audited parent：`+0.0001944`。

最後一項改善低於預先定義的 `+0.001` gate，且沒有 three-seed mean，因此正式 winner
保留 zero-train checkpoint。這是 negative result 的誠實保留，不把 evaluator noise 包裝成可靠改善。

## 10. Claim 邊界與後續工作

已實作／可重現：

- Binary Q/K software reference。
- identity／Hadamard 雙 basis。
- fixed power-of-two coefficients。
- decomposed relative bias。
- Float-PWL training 與 Bit-True PWL numerator。
- 完整 queue、gate、rollback、checkpoint hash 與 provenance。

尚未完成／不可宣稱：

- bit-packed XNOR／popcount kernel。
- Bit-True denominator／reciprocal。
- P/V/projection 端到端整數化。
- HLS、RTL、FPGA／ASIC 上板。
- 實測 latency、功耗、面積或記憶體流量。
- canonical COCO API gate。
- three-seed mean/std。
- 對每一項 primitive 的全球首創性判定。
