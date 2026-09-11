# BinaryQK scale／codebook：目前、候選與改後架構圖

> 2026-09-08 仍為Q0後的條件式，B4無runtime selector、A8動態上界；本次禁GPU且沒有replay/kernel結果。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

本報告以 terminal 可讀格式呈現 C0目前架構、B4部署候選與 A8精度上界。執行 gate見
[計畫](<plan.md>)，證據來源見[證據索引](<evidence.md>)。

## 1. 統一符號

~~~text
S = 2 attention sites
B = 2 bases（identity + Hadamard）
H = 4 heads
N = 400 tokens
D = 32 channels/head
batch = 1
~~~

~~~text
token pairs       = S × H × N²       = 1,280,000
basis terms       = S × B × H × N²   = 2,560,000
token-scale slots = S × B × H × 2 × N = 12,800 / image
~~~

## 2. 目前 C0：fixed per-head／basis PoT

### 2.1 calibration只做一次

~~~text
Q [B,H,D,N] ──> abs ──> mean(D,N) ──> mu_q [B,H,1,1] ─┐
                                                        ├─> combined coefficient
K [B,H,D,N] ──> abs ──> mean(D,N) ──> mu_k [B,H,1,1] ─┘
                                                                    │
                             多張 calibration images 累積／平均 <────┘
                                                                    │
                                                                    ▼
                                                PoT round → c_fixed[h,basis]
                                                寫入 checkpoint／artifact
~~~

### 2.2 正式 inference

~~~text
Q ──> sign ──> packed q_bits ─┐
                               ├─> XNOR-popcount → signed dot z[i,j] ─┐
K ──> sign ──> packed k_bits ─┘                                      │
                                                                       ▼
                    c_fixed[site,head,basis] ──> fixed shift／multiply
                                                                       │
identity basis ────────────────────────────────────────────────────────┤
Hadamard basis ────────────────────────────────────────────────────────┤
                                                                       ▼
                                      basis sum → relative bias → normalization → P·V
~~~

~~~text
16 fixed slots = 2 sites × 4 heads × 2 bases
runtime magnitude reduction = 0
runtime scale selector       = 0
activation scale index       = 0
~~~

目前每一個 head/basis的所有 image、query token與 key token共用同一個 fixed coefficient。它保留
sign與整個 head的平均 magnitude，但捨棄 token-pair與 channel-group的 magnitude差異。

### 2.3 eager reference目前多算了什麼

~~~text
目前軟體：
Q/K → dynamic abs/mean → 判斷 scale_mode=fixed → 丟掉 dynamic值 → 讀 c_fixed

應改成：
                    ┌─ fixed且已校正 ──> 直接讀 c_fixed
Q/K → scale_mode ───┤
                    └─ dynamic/calibration ──> abs/mean
~~~

Hadamard有 Q、K、Q_h、K_h 四個 magnitude reductions/site；兩個 sites共八次無效 reduction。
early-return只移除白算，輸出必須 bit-true相同，所以它是工程修正，不是精度候選。

## 3. 為何單一 fixed coefficient不能修回全部 score

令：

~~~text
q_i = mu_q b_qi + r_qi,    b_qi = sign(q_i)
k_j = mu_k b_kj + r_kj,    b_kj = sign(k_j)
~~~

則：

~~~text
q_i^T k_j
  = mu_q mu_k b_qi^T b_kj
  + mu_q b_qi^T r_kj
  + mu_k r_qi^T b_kj
  + r_qi^T r_kj
~~~

C0主要保留第一項；後三項依 token pair `(i,j)` 改變。單一 head-level fixed coefficient只能做整體
重縮，不能任意恢復每一對 score差值，因此 top-k ranking、entropy與 softmax probability都可能改變。

但既有 global dynamic只比 fixed PoT高 `0.000505537` mAP，表示「每張圖換一個 global scale」
不是約 `0.011` 缺口的主因；不能因此跳過 site isolation與 QAT/KD主線。

## 4. 預計候選 B4：fixed 4-group PoT

### 4.1 單一 basis／head資料流

~~~text
q_bits / k_bits [D=32]
           │
           ├─ group0 [ 0: 8] → partial popcount → fixed PoT shift ─┐
           ├─ group1 [ 8:16] → partial popcount → fixed PoT shift ─┤
           ├─ group2 [16:24] → partial popcount → fixed PoT shift ─┤─> add → score
           └─ group3 [24:32] → partial popcount → fixed PoT shift ─┘
~~~

~~~text
64 fixed slots = 2 sites × 4 heads × 2 bases × 4 groups
每個 slot可直接保存 exponent，或保存3-bit index指向8-entry shared PoT LUT

runtime magnitude reduction = 0
runtime selector            = 0
activation index            = 0
~~~

公式：

~~~text
S_B4[s,h,i,j]
  = sum_basis sum_group
      2^e[s,h,basis,group]
      · binary_dot(q_bits[s,h,basis,i,group],
                   k_bits[s,h,basis,j,group])
~~~

### 4.2 B4換來什麼、付出什麼

~~~text
C0：每個 D=32 basis dot → 1 個完整 word popcount → 1 個 fixed coefficient
B4：每個 D=32 basis dot → 4 個 partial popcounts  → 4 個 fixed coefficients
~~~

不同 group權重無法從一次 total popcount還原，所以 naïve basis terms由 `2.56M`增加到
`10.24M`，另需 shift-add合併。B4的價值是完全固定、控制面簡單，**不是** operation count比較少。
是否能部署必須由目標 backend的 lane-wise popcount、packing與融合 shift-accumulate實測決定。

## 5. 條件式候選 A8：per-token 8-level PoT

### 5.1 每張圖都會執行的資料流

~~~text
Q token [D=32] ─┬─> abs-sum → >>5 → 8-level selector → q_index[3 bit] ─┐
                 └─> sign → packed q_bits ──────────────────────────────┤
                                                                          │
K token [D=32] ─┬─> abs-sum → >>5 → 8-level selector → k_index[3 bit] ─┤
                 └─> sign → packed k_bits ──────────────────────────────┤
                                                                          ▼
                                XNOR-popcount → signed dot z[i,j]
                                                                          │
q_index + k_index → exponent-sum LUT → variable shift/alignment ─────────┤
                                                                          ▼
                                               basis sum → bias → normalization
~~~

codebook values可離線固定，但 q/k index由當前 activation決定。因此 A8仍是 dynamic：

~~~text
每張圖 magnitude abs       = 409,600
每張圖 reduction adds      = 396,800
每張圖 3-bit indices       = 12,800 = 最小 4,800 B
pair exponent/variable shift上界 = 2,560,000
~~~

若 8 個 exponent連續，8×8個 Q/K組合只產生15種 exponent sums，可用 index add或小 LUT；這移除
一般乘法，卻不會移除 activation reduction、selector、index traffic與 pair-level alignment。

### 5.2 A8保留的資訊

~~~text
mu_q[i] = mean_d |q[d,i]|
mu_k[j] = mean_d |k[d,j]|

S_A8[i,j]
  ≈ quant_PoT(mu_q[i]) · quant_PoT(mu_k[j])
    · <sign(q_i), sign(k_j)> / sqrt(D)
~~~

A8可恢復不同 token的 magnitude排序，比 C0更接近 token-wise score scaling；但它仍沒有重建
residual cross terms，也不能保證 detection AP上升。

## 6. 三種架構放回兩個 attention sites

### 6.1 目前

~~~text
Backbone P5
     │
     ▼
site 10：C0 fixed-PoT BinaryQK
     │
     └─> Neck → P3 ─────────────────────────────────────────────────────┐
                  └─> P4 ───────────────────────────────────────────────┤
                         └─> site 22：C0 fixed-PoT BinaryQK → P5 ──────┤
                                                                          ↓
                                                             Detect([P3,P4,P5])
~~~

### 6.2 若 B4通過

~~~text
Backbone P5
     │
     ▼
site 10：B4 fixed-group BinaryQK
     │      └─ 4-way partial popcount + fixed shifts
     └─> Neck → P3 ─────────────────────────────────────────────────────┐
                  └─> P4 ───────────────────────────────────────────────┤
                         └─> site 22：B4 fixed-group BinaryQK → P5 ────┤
                                                                          ↓
                                                             Detect([P3,P4,P5])

每張圖 scale reduction／selector：0
~~~

### 6.3 若只有 A8能提供足夠 fidelity

~~~text
Backbone P5
     │
     ▼
site 10：A8 per-token BinaryQK
     │      └─ runtime magnitude → 3-bit indices → variable shifts
     └─> Neck → P3 ─────────────────────────────────────────────────────┐
                  └─> P4 ───────────────────────────────────────────────┤
                         └─> site 22：A8 per-token BinaryQK → P5 ──────┤
                                                                          ↓
                                                             Detect([P3,P4,P5])

每張圖 scale reduction／selector：有
~~~

這三張圖只表示 scale方案。實際部署仍應沿用 Q0 site-isolation winner；若最後只二值化一個 site，
上面的計數須按實際 site數重新計算，不能沿用 two-site上界。

## 7. `1/sqrt(32)` 不能直接稱為整數 PoT

~~~text
1 / sqrt(32) = 2^-2.5 = sqrt(2) · 2^-3
~~~

它不是整數 exponent shift。若要求純 integer shift，只能：

1. 把 attention scale、gamma與 Q/K coefficient離線合併後重新 PoT量化，接受 rounding error；或
2. 保留共同的 fixed `sqrt(2)` multiplier；或
3. 在下一個固定線性 score quantizer前吸收共同常數，但必須證明不跨越 clipping、rounding、
   saturation或 normalization邊界。

任一作法都要由 bit-true parity與 target export鎖定，不能只因 codebook是 PoT就宣稱整條
attention multiplication-free。

## 8. 最終比較

| 架構 | magnitude粒度 | 每圖重算 | 主要優點 | 主要風險 |
|---|---|---:|---|---|
| C0 | fixed head/basis | 否 | 最簡單、已有正式結果 | token/channel差異消失 |
| B4 | fixed channel group | 否 | 無 selector，可固定 datapath | partial popcount上界4倍 |
| A8 | dynamic token/head/basis | 是 | 最可能保留 token magnitude | reduction、index、variable shift昂貴 |

推薦順序仍是：先完成 Q0 fixed-PoT site isolation／QAT／KD；真的剩下 magnitude殘差時，再以
C0/B4/A8 replay判斷。B4是 deployment-first候選，A8是 fidelity ceiling，兩者都不是已驗證 winner。

返回[方向說明](<README.md>)。
