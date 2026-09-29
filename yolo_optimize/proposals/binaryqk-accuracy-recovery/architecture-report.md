# BinaryQK：目前、預計與條件式修改架構

> 2026-09-08 總計畫S6/S7為準；Full35 Float不等於FP-QK，新parent重做site screens，Q/K hardware_frozen需核對；單任務W-DIR LR不能搬成joint已驗證recipe。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

> 狀態：機理分析／尚未修改 production model／尚未以完整 validation AP 證實
> 方向 ID：`OPT-BINARYQK-ACCURACY-RECOVERY`

在 `yolo_optimize` 目錄可用下列命令於 terminal 重看：

~~~bash
sed -n '1,420p' optimizations/binaryqk-accuracy-recovery/architecture-report.md
~~~

## 0. 一句話判斷

不是把 BinaryQK 全部拿掉，也不是再多訓練幾個 epochs。現在最可能的瓶頸是：

~~~text
每個 head/basis 只有一個由 global calibration 得到的 fixed PoT scale
                    +
兩個 attention sites 同時 binary
                    +
pairwise ranking 被 sign approximation 改變
                    ↓
       W-DIR recovery 後仍差 0.010540 mAP
~~~

因為硬體目標不希望每張圖重算 scale，而且既有 V1-DYN/SHEAD/P2 已完成，所以預計先保留 fixed PoT
做 single-site validation。per-token/head 降為條件式 accuracy ceiling；只有 fixed-PoT site winner
經 direct QAT 後仍有 ranking/KL 問題，才加入一種 FP-teacher loss。

## 1. A：目前 YOLO26 的兩-site BinaryQK 資料流

~~~text
Backbone P5 feature
       │
       ▼
model.10.m.0.attn：BinaryQK-10（fixed per-head/basis PoT）
       │
       └─> Neck top-down ──> layer16：P3 ────────────────────────────────────────────┐
                                  │                                                   │
                                  └─> downsample → layer19：P4 ──────────────────────┤
                                                        │                             │
                                                        └─> downsample                │
                                                               │                      │
                                                               ▼                      │
                                                model.22.m.0.1.attn：BinaryQK-22      │
                                                               │                      │
                                                               └─> layer22：P5 ───────┤
                                                                                      ↓
                                                                 Detect([P3, P4, P5])
~~~

site 10 是上游 attention；它的誤差可以一路進入 P3、P4、P5。site 22 更靠近 P5 Detect feature。
兩處同時 binary 時，最終 gap 不能直接歸因到其中任一處。

## 2. A：目前單一 site 內部怎麼算

現行 V1-BR／A-FINAL 分成 calibration 與 inference 兩段：

~~~text
calibration only：
Q/K [B,H,D,N]
   └─> abs → mean(D,N) → mu_q / mu_k [B,H,1,1]
       └─> gamma · mu_q · mu_k / sqrt(D)
           └─> 對 calibration observations 取平均
               └─> round 到最近的 2^k
                   └─> c_fixed [H,2 bases]

正式 inference：
Q,K ─> sign ─> identity XNOR-popcount z_i ──× c_fixed[h,identity] ─┐
  └─> normalized Hadamard ─> sign ─> z_h ───× c_fixed[h,hadamard] ┤
                                                                  ▼
                                                    z_i·c_i + z_h·c_h
                                                                  │
                                            + decomposed relative bias
                                                                  │
                                       Exact(V1-BR) / SHIFT(A-FINAL)
                                                                  │
V ────────────────────────────────────────────────────────────────┤
                                                                  ▼
                                                          P·V + PE(V)
~~~

`gamma` 目前按 basis 共用，不是每個 head 各一個；它在 calibration 產生 dynamic coefficient 時被
納入，正式 fixed-scale inference則直接使用 `c_fixed`。A-FINAL checkpoint 的實際 coefficients：

~~~text
site 10：head 0..3 的 [identity, hadamard] 全部是 [0.25, 0.25]
site 22：head 0 是 [0.125, 0.25]；head 1..3 是 [0.25, 0.25]
~~~

A-FINAL 在後續改用 SHIFT normalization；但 Exact/PWL/SHIFT 的 zero-train AP 差都不到 `0.001`，
不是約 `0.011` gap 的主因。

### 2.1 目前 fixed-PoT scale 公式

對一個 sample/head：

~~~text
calibration：
mu_q = mean_{d,i} |q[d,i]|，mu_k = mean_{d,j} |k[d,j]|
c_dyn[h,r] = gamma[r] · mu_q · mu_k / sqrt(D)
c_fixed[h,r] = 2^round(log2(mean_calibration(c_dyn[h,r])))

z_ij = <sign(q_i), sign(k_j)>
     = 2 · popcount(XNOR(q_bits_i, k_bits_j)) - D

inference：
S_current[i,j] = c_fixed[h,r] · z_ij
~~~

identity與 Hadamard basis 各自校正一個 fixed coefficient，最後相加。shape 是 `[1,H,1,1]`，所以
同一個 head/basis 的所有 images、query tokens、key tokens共用同一 scale。

## 3. 為什麼 fixed global-derived scale 會失真

### 3.1 dot-product 誤差推導

令：

~~~text
q_i = mu_q b_qi + r_qi,    b_qi = sign(q_i)
k_j = mu_k b_kj + r_kj,    b_kj = sign(k_j)
~~~

則 full-precision score：

~~~text
S_fp[i,j]
= [mu_q mu_k b_qi^T b_kj
   + mu_q b_qi^T r_kj
   + mu_k r_qi^T b_kj
   + r_qi^T r_kj] / sqrt(D)
~~~

BinaryQK 只保留第一項，所以：

~~~text
Delta S[i,j] = S_bin - S_fp

= -[mu_q b_qi^T r_kj
    + mu_k r_qi^T b_kj
    + r_qi^T r_kj] / sqrt(D)
~~~

這三個 residual terms 都依 token pair `(i,j)` 改變。單一 `gamma`、bias 常數或 temperature 最多重縮
整列 logits，不能任意修回每一對 token 的差值與排序。

### 3.2 softmax 為什麼會放大 ranking 問題

令 `p=softmax(s)`，小擾動的一階式：

~~~text
Delta p ≈ [diag(p) - p p^T] Delta s
~~~

同一列加常數不影響 softmax；但 BinaryQK 的 `Delta s_ij` 是 pair-specific，所以 top-k、entropy 與
probability mass 都會改。這也解釋為何「換一個 normalization approximation」不會自動修復 ranking。

### 3.3 本地 probe 符合這個症狀

~~~text
site 10：global top-10 0.4206 → per-token 0.5119
         global KL     0.5431 → per-token 0.4651

site 22：global top-10 0.4962 → per-token 0.5952
         global KL     0.8810 → per-token 0.6188
~~~

兩個 sites 的 binary attention 都比 FP 更平；per-token scale 同時改善排序 overlap 與 KL。但這個
probe 比較的是 `global dynamic` 與 `per-token dynamic`，不是正式 fixed-PoT A-FINAL。樣本也只有
兩張，因此它只是在 hardware-friendly recovery 失敗時重新開啟成對完整 validation 的理由，不是
首輪工作或最後結論。

## 4. B：條件式 accuracy ceiling——兩站 per-token/head scale

### 4.1 修改後單一 site 資料流

~~~text
Q [B,H,D,N] ─┬─> abs → mean(D) → mu_q [B,H,1,N] ── transpose ─┐
              └─> sign ─> packed q_bits ──────────────────────┤
                                                              │
K [B,H,D,N] ─┬─> abs → mean(D) → mu_k [B,H,1,N] ─────────────┤
              └─> sign ─> packed k_bits ──────────────────────┤
                                                              ▼
                         z[i,j] = XNOR-popcount(q_bits_i,k_bits_j)
                                                              │
                         pair_scale[i,j] = mu_q[i] · mu_k[j]  │
                                                              │
                                                              ▼
                      S_token[i,j] = gamma · pair_scale[i,j] · z[i,j]/sqrt(D)
                                                              │
                                  relative bias → normalization → P·V + PE(V)
~~~

修改前／後核心差異：

~~~text
原本：mean(D,N) → 每 sample/head 一個 scale
建議：mean(D)   → 每 sample/head/token 一個 scale
~~~

公式：

~~~text
mu_q[i] = mean_d |q[d,i]|
mu_k[j] = mean_d |k[d,j]|

S_token[i,j]
  = gamma · mu_q[i] · mu_k[j]
    · <sign(q_i),sign(k_j)> / sqrt(D)
~~~

Hadamard basis 也用 transformed Q/K 各自的 per-token magnitude；不能只改 identity basis，否則兩個
bases 的 coefficient granularity 不一致。

### 4.2 修改後整體資料流

~~~text
Backbone P5
     │
     ▼
site 10：BinaryQK(per-token/head)
     │
     └─> Neck → P3 ─────────────────────────────────────────────────────────────┐
                  └─> P4 ───────────────────────────────────────────────────────┤
                         └─> site 22：BinaryQK(per-token/head) → P5 ────────────┤
                                                                                 ↓
                                                                    Detect([P3,P4,P5])
~~~

這不是首輪 production架構；只有 fixed-PoT hybrid/QAT/KD仍失敗時才開啟，而且要同時通過 AP 與
target latency。以 `B=1,H=4,N=400`、兩站估算：

~~~text
Q/K token scales：2 sites × 2(Q,K) × 4 heads × 400 tokens = 6,400
pair-scale epilogue：2 sites × 4 heads × 400 × 400 = 1,280,000 operations/image

以上是共用一套 token scale的下限；若 identity / Hadamard各自獨立：
Q/K token scales = 12,800；pair-scale epilogue = 2,560,000 operations/image
~~~

實際 kernel 應在 tiled epilogue 即時計算，避免 materialize完整 pair-scale matrix；沒有 fused kernel
profile 前，不能說這個架構一定比 FP 快。

### 4.3 少量 scale／codebook 架構已獨立

8-level per-token PoT與 fixed 4-group PoT涉及另一個主要因果問題及獨立硬體 gate，詳細圖不在本
主線重複維護。請見
[OPT-BINARYQK-SCALE-CODEBOOK 架構圖](<../binaryqk-scale-codebook/architecture-report.md>)，其中並列：

- 目前 C0 fixed per-head／basis PoT。
- B4 fixed 4-group、無 dynamic selector資料流。
- A8 per-token 8-level PoT、每張圖需 reduction／index資料流。

## 5. C：實用第一候選——fixed-PoT hybrid

免訓練完整 validation 先比較：

~~~text
C1：只讓 site 10 binary

Backbone P5
     │
     ▼
site 10：BinaryQK ─> Neck → P3/P4 ─> site 22：FP QK → P5
                                            │
                                            └─> Detect([P3,P4,P5])
~~~

~~~text
C2：只讓 site 22 binary

Backbone P5
     │
     ▼
site 10：FP QK ─> Neck → P3/P4 ─> site 22：BinaryQK → P5
                                            │
                                            └─> Detect([P3,P4,P5])
~~~

決策不是預先指定 site 10 或 site 22。哪個 Binary-only 候選保留較高 AP，才讓哪個 site 使用 1-bit；
敏感 site 保留 FP。hybrid 能減少 accuracy loss，但 binary coverage 也下降，必須報真實比例與 latency。

## 6. D：訓練期才存在的 FP teacher

只有 B/C winner 經 direct QAT 後仍有明確 ranking/KL gap，才加入：

~~~text
                         frozen final FP teacher
input ─────────────────────────┬─> FP attention map / ranking ────────┐
                               │                                      │
                               └─> BinaryQK student ─> detection loss │
                                                   │                  │
                                                   └─ KD loss <───────┘

deployment：只保存 BinaryQK student；teacher 與 KD branch 全部移除
~~~

~~~text
ranking 尚可、KL/entropy 差 → attention-map KL
top-k/pairwise order 差      → ranking-aware loss
~~~

兩者只選一個。teacher 不增加 inference graph，但會增加訓練記憶體與時間。

## 7. E：clipped STE 的條件式修正

目前 train-time sign：

~~~text
forward： b = sign(x)
backward：db/dx ≈ 1{|x| ≤ 1}
~~~

本次兩張圖 probe 的每-head saturation 約 35%–54%；sign positive ratio 約 44%–57%，沒有嚴重
sign collapse。若正式 probe 重現高 saturation，可用正值 head-wise scale只改 gradient window：

~~~text
c_h = softplus(raw_c_h) > 0
b_h = sign(x_h / c_h) = sign(x_h)

effective STE window：|x_h| ≤ c_h
~~~

它不改 inference bits，理論上可在 export 移除。threshold `sign(x-tau_h)` 會改 bits；只有 sign ratio
明顯偏斜時才值得測，現在不是第一順位。

## 8. 原本與候選差異總表

| 項目 | A：目前 global both | B：per-token both | C：hybrid | D：+ teacher KD |
|---|---|---|---|---|
| Binary sites | 10 + 22 | 10 + 22 | 只留一站 | 繼承 B/C |
| magnitude | fixed per head/basis PoT | per token/head dynamic | fixed per head/basis PoT | 繼承 B/C |
| score ranking 補償 | 無直接機制 | 保留 token magnitude | 移除敏感 site 誤差 | training loss 約束 |
| 新 binary products | 0 | 0 | 減少 | 0 |
| inference side cost | O(BH) scale | O(BHN) scale + N² epilogue | FP site + binary site | 無 teacher cost |
| 現有 AP 證據 | gap 約 0.011 | 只有 2-image fidelity | 尚無完整 AP | YOLO11/論文間接證據 |
| 首輪角色 | baseline | 條件式 accuracy ceiling | deployment-first | 條件式第三步 |

## 9. 最小決策圖

~~~text
V1-BR retained binary checkpoint
            │
            ├─> ISO-10-BIN：10 binary / 22 FP ────┐  2 次 validation-only
            └─> ISO-22-BIN：10 FP / 22 binary ────┘  保留 fixed PoT
                                                   │
                                                   ▼
                                          只選 1 個候選
                                                   │
                   final same-lineage FP parent ───┤
                                                   ├─> FP-CTRL direct recovery
                                                   └─> BQK-CAND direct QAT
                                                              │
                                  ┌───────────────────────────┴────────────────────┐
                                  │                                                │
                           首 seed 未過                                       首 seed 通過
                                  │                                                │
                                  ▼                                                ▼
                                停止                           ranking/KL 仍差才加 1 個 KD arm
                                                                                   │
                                                                                   ▼
                                                                     補至 3 paired seeds
                                                                                   │
                                                                                   ▼
                                                               bit-true + target latency gate

若以上 fixed-PoT 路徑仍失敗，且接受 dynamic hardware成本：

V1-BR ─┬─> DYN-GLOBAL ─┐
       └─> TOKEN-BOTH ──┴─> 只以 TOKEN-BOTH - DYN-GLOBAL 歸因 token scale
~~~

## 10. 最終建議

目前最值得實作的第一版不是 per-token、dual-basis、threshold 或新的 softmax，而是：

~~~text
deployment-first：保留 fixed PoT，只讓較不敏感的 site 使用 BinaryQK
recovery：上述唯一 winner + direct QAT；仍有 ranking gap 才加單一 teacher loss
accuracy ceiling：前兩步失敗後，才成對測 global dynamic / per-token dynamic
~~~

是否最終採 both-binary 或 hybrid 必須由完整 AP 與 target-device latency共同決定。現在可以合理說
「fixed-PoT site isolation 最值得先測」；per-token fidelity改善不能直接等同已補回 `0.011` mAP，
也不能等同已獲得硬體加速。

返回[方向說明](<README.md>)、[執行計畫](<plan.md>)或[優化方向索引](<../README.md>)。
