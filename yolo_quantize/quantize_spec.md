---
title: YOLO26m LS-SD4/LSQ+ 量化研究與 Codex Master Spec
working_method_name: FALSQ-YOLO26
core_baseline: Learnable-Scale SD4 weights + LSQ+ activations
research_extensions: Fusion-relative scale coupling; optional O2M-to-O2O assignment-aware distillation
status: Research hypothesis and engineering contract; not a novelty claim
source_scope: Visible project conversation, uploaded PDFs, and public sources; private chat /c URL content was not readable
language: zh-TW
date: 2026-08-27
target_task: YOLO26m Detect
primary_checkpoint: runs/detect/optim_1280/weights/best.pt
primary_dataset_yaml: Finale_version.v6i.yolov11/data.yaml
primary_image_size: 1280
---

# YOLO26m LS-SD4/LSQ+ 量化研究與 Codex Master Spec

> 本文件是交給 Codex 的完整研究工程契約。Codex 應把它視為「可逐階段執行、可測試、可消融、可匯出」的規格，而不是一次把所有想法塞進模型。
>
> **目前收斂後的核心方法不是籠統的「SD4 + LSQ」。** 應精確定義為：
>
> - **Weight：Learnable-Scale SD4（LS-SD4）**。SD4 保留 signed power-of-two 非均勻 codebook；每層或每個輸出通道額外學一個正的 range scale `s_w`，讓 codebook 適應實際 weight distribution。
> - **Activation：LSQ+**。針對 YOLO26 中 SiLU 類 activation 的 signed、skewed distribution，使用 learnable asymmetric scale/offset；原始 PACT 僅保留為控制組。
> - **Research extension A：Fusion-aware shared scale + relative integer exponent**。在 neck fusion group 內，把自由的 per-branch scale 收斂成共享 base scale 與有限整數相對 exponent，提供 shift-compatible alignment。
> - **Research extension B：YOLO26 O2M→O2O assignment-aware distillation**。只在基線穩定後測試，且必須和既有 quantization self-teaching 工作區分。
>
> 暫定整體名稱：**FALSQ-YOLO26 = Fusion-Aware Learnable-Scale SD4 Quantization for YOLO26**。這只是工作名稱，不是「首次提出」或創新性結論。
>
> **第一優先順序：先證明 fixed SD4 < LS-SD4，並確認 LS-SD4/LSQ+ 可以穩定 QAT、bit-exact export，再進入 fusion coupling 與 distillation。**
>
> 資料範圍說明：本版已整合此專案中可見的量化討論、使用者上傳的 LSQ/PACT/YOLOX accelerator PDF與公開文獻。先前提供的 `chatgpt.com/c/...` 私人對話網址只顯示登入頁，未能讀取其正文；該對話若有額外內容，尚未被假設性地寫入。

---

## 0. Codex 執行規則

你現在是此 repository 的 quantization research engineer。請依照以下規則工作：

1. **先檢查 repository，再改程式。** 不得假設使用者的 Ultralytics 版本、YOLO26m YAML、Detect head、loss API 或 checkpoint 結構與官方 main branch 完全相同。
2. 預設模型權重為 `runs/detect/optim_1280/weights/best.pt`，資料集為 `Finale_version.v6i.yolov11/data.yaml`，最終驗證影像尺寸為 `1280`。路徑不存在時，先依 basename 搜尋並把替代路徑記錄在 manifest，不要靜默更改。
3. **不要直接修改 site-packages 或整份 fork Ultralytics。** 量化程式放在獨立 package，透過明確的 module replacement、wrapper 或小型 adapter 接入。
4. Forward hook 可用於一次性的統計、shape tracing 與 sensitivity 分析；**不可把 hook 當成主要量化執行機制**。
5. 每一階段都必須先新增測試，再執行小型 smoke test，最後才跑完整訓練。
6. 每個實驗必須保存 config、git commit、套件版本、seed、checkpoint、metrics、量化政策與匯出資訊。
7. 不得把 fake-quant PyTorch 的執行時間宣稱成 INT4、SD4 或 FPGA 的實際速度。
8. 沒有硬體 kernel 時，只能報告 packed model size、BOPs、shift-add count、requantization count、memory traffic estimate 等 proxy；不得宣稱真實 speedup。
9. 任何結論都要和至少一個公平基線比較。方法新增多個元件時，必須逐項消融。
10. 不要一次實作全部內容。先完成 Phase 0，確認基線可重現後再進入 Phase 1。

每完成一個 phase，回報格式固定為：

```text
Phase:
Changed files:
Commands executed:
Tests:
Measured results:
Assumptions:
Known failures or risks:
Recommended next phase:
```

---

## 1. 研究情境、核心問題與成功標準

### 1.1 已知情境

- 模型：YOLO26m Detect。
- 最佳 checkpoint：`runs/detect/optim_1280/weights/best.pt`。
- Dataset YAML：`Finale_version.v6i.yolov11/data.yaml`。
- 最終 input size：1280。
- 研究方向：低位元 QAT、SD4／power-of-two weight、LSQ+ activation、mixed precision、fusion scale alignment、YOLO26 dual-head 的量化訓練。
- 部署導向：FPGA／ASIC 類 shift-friendly datapath；同時保留 ONNX Q/DQ 或一般 INT8 backend 的可靠基線。
- 後續可延伸到 YOLO26m Pose 或 detect/pose 多任務模型；本文件先完成 Detect 的乾淨基線。

### 1.2 本研究真正要回答的問題

1. 固定 SD4 codebook 在 YOLO26m 各 stage 的準確率損失與 codebook 使用情形如何？
2. 在 SD4 外加入 learnable positive range scale `s_w`，能否穩定提升 mAP、降低 clipping／normalized quantization error？
3. `s_w` 應採 layer-wise、per-output-channel、group-wise，還是 power-of-two constrained？
4. YOLO26 的 SiLU activation 是否需要 LSQ+ asymmetric quantization，且相對 PACT、symmetric LSQ 的收益如何？
5. Backbone、neck、head 對 SD4 的敏感度是否不同；是否需要 evidence-based mixed precision？
6. Neck 多尺度 fusion 是否因 branch scale mismatch 產生額外 clipping、requantization 與 accuracy loss？
7. 能否以 shared base scale + relative integer exponent 接近自由 per-channel/per-branch scale 的準確率，同時降低 arbitrary multiplier 數量？
8. YOLO26 的 training-only one-to-many branch 能否協助低位元 one-to-one inference branch，而不增加 inference cost？
9. 最終方法是否能在 fake-quant、integer simulator、packed format 與實際硬體上保持一致？

### 1.3 核心研究假設

```text
H1: fixed SD4 的主要限制之一是 codebook range 無法適應每層 weight distribution。
H2: Learnable-Scale SD4 可透過 task loss 學到更合適的 range，優於 fixed/max-based SD4。
H3: per-channel free scale 是 accuracy upper bound，但會增加 per-channel requantization 成本。
H4: shared base scale + relative integer exponent 可在較低硬體成本下逼近 free scale 的效果。
H5: LSQ+ 比原始 PACT 更適合保留 SiLU 的小幅負值與非對稱正負範圍。
H6: YOLO26 neck fusion 與 inference-time O2O head 是低位元量化的特殊敏感位置。
```

### 1.4 成功標準

- **工程成功**：可重現 FP baseline；完成 W8A8、W4A8、fixed SD4W/A8、LS-SD4W/A8；checkpoint 可載入、resume、驗證與匯出。
- **量化成功**：LS-SD4 相對 fixed SD4 有可重複收益，且不是只因訓練更久或 first/last layer policy 不公平。
- **研究成功**：shared-scale relative-exponent 或 assignment-aware branch distillation 在公平基線與 3 seeds 下提供可解釋增益。
- **部署成功**：fake quant、integer simulator、packed SD4 decoder 與 export 結果一致；硬體成本或測得 latency/resource 明確改善。

### 1.5 非目標

- 不把「把幾個既有 quantizer 串起來」包裝成創新。
- 不用 PyTorch fake-quant latency 代替真實 INT/SD4 kernel latency。
- 不在 Phase 0 前先改模型或假設官方 YOLO26 結構和本地 checkpoint 相同。
- 不把 per-channel arbitrary floating scale 稱為 multiplierless 部署。

## 2. 最終方法決策：核心基線、研究擴充與創新界線

### 2.1 最終建議架構

```text
YOLO26m
  ├─ Weights
  │    ├─ reliable baseline: symmetric uniform LSQ W8/W4
  │    ├─ SD4 baseline: fixed or statistic-derived SD4 scale
  │    └─ core candidate: Learnable-Scale SD4 (LS-SD4)
  │
  ├─ Activations
  │    ├─ main baseline: asymmetric LSQ+ A8
  │    ├─ aggressive: LSQ+ A6/A4
  │    └─ control: symmetric LSQ / PACT / dual-bound clipping
  │
  ├─ Neck fusion research extension
  │    └─ shared base scale + branch-relative integer exponent
  │
  └─ YOLO26 head research extension
       └─ assignment-aware O2M → O2O quantization distillation
```

第一個值得完整訓練與寫報告的組合是：

```text
LS-SD4 weight + LSQ+ activation
```

但這只是**核心工程候選**。它的價值在於改善 fixed SD4 對 layer distribution 的適應性；單獨的 learnable scale 並不足以支持強創新主張。

### 2.2 各元件的角色必須分清楚

| 元件 | 解決的問題 | 不應混淆成 |
|---|---|---|
| SD4 codebook | 4-bit signed PoT weight；乘法可映射為 sign/shift | uniform INT4 |
| Learnable `s_w` | 調整整個 SD4 codebook 的實數 range | 新 bit-width 或新 codebook |
| LSQ | uniform quantizer 的 learnable step size與梯度尺度 | SD4 nearest-code rule |
| LSQ+ | signed/skewed activation 的 asymmetric scale/offset | 原始 `[0, alpha]` PACT |
| PoT-constrained scale | 簡化 output rescale／requantization | 自動讓所有卷積完全無乘法 |
| Relative exponent | 在 shared base 下表達 branch/channel 相對尺度 | 自由 global exponent offset |
| Mixed precision | 對敏感 stage 保留較高精度 | 單獨的創新 |
| O2M→O2O KD | 改善低位元 inference head 的訓練訊號 | 一般 self-distillation 的首次提出 |

### 2.3 建議優先級

1. **必做基線**：FP、INT8 PTQ、LSQ W8A8、LSQ W4A8。
2. **核心驗證**：fixed SD4W/A8 vs LS-SD4W/A8。
3. **scale 粒度消融**：layer-wise vs per-channel vs PoT-constrained vs grouped。
4. **stage sensitivity**：backbone、neck、head 分開量化。
5. **研究 extension A**：neck fusion shared base + relative integer exponent。
6. **研究 extension B**：assignment-aware O2M→O2O distillation。
7. 最後才組合 FALSQ + KD、跑多 seeds、做 hardware/export validation。

### 2.4 創新性分級

| 想法 | 工程價值 | 暫定創新性 | 必須怎麼證明 |
|---|---:|---:|---|
| LSQ 套到 YOLO26 | 高 | 低 | 作可靠 uniform baseline |
| LSQ+ activation | 高 | 低 | 比較 symmetric LSQ/PACT，證明適合 SiLU |
| fixed SD4W/A8 | 高 | 低 | 建立 shift-weight baseline |
| **LS-SD4：SD4 + learnable `s_w`** | 高 | 低～中 | fixed/max/MSE scale 消融；多 seeds；不可只看 weight MSE |
| per-channel LS-SD4 | 高 | 低 | 當 accuracy upper bound；報 scale metadata 成本 |
| PoT-constrained LS-SD4 scale | 中高 | 中低 | accuracy vs shift-only rescale trade-off |
| stage-wise INT6/SD4/W8 | 高 | 低 | 既有 mixed-precision 先例很多 |
| **fusion-group shared scale + relative exponent** | 高 | 中 | 對照 independent scale、shared scale、TQT、soft alignment；報 requant count |
| **assignment-aware YOLO26 O2M→O2O KD** | 中高 | 中 | 對照 generic KD、no-KD、GHOST 類 self-teaching；證明 inference 無額外成本 |
| FALSQ + bit-exact deployment | 很高 | 取決於結果 | 必須有實際 kernel/FPGA 或可信 integer/synthesis evidence |

### 2.5 必須承認的先行研究風險

- LSQ、LSQ+、PACT、TQT、APoT、learnable clipping/scale、PoT quantization、mixed precision 都已有成熟先例。
- SD4 原論文已提出 4-bit signed power-of-two weight 與 multiplier-less accelerator；因此 SD4 本身不是新方法。
- GABFusion 已研究低位元 feature fusion 的 gradient imbalance；本研究不可只用「fusion-aware」作貢獻名稱。
- PTQ4SNN（2026-08）已使用形如 `s_target,c = s_source,c * 2^k_c` 的 shift-compatible scale bridge；因此「shared/reference scale + integer exponent」這個裸公式也不能宣稱首次提出。
- GHOST 已把 hybrid quantization 與 one-to-one self-teaching 用於 object detection；因此「量化 + self-teaching」本身也不是新概念。

較有機會成立的論述必須縮小為：

> 在 YOLO26 的 SD4 weight QAT 與跨尺度 neck fusion 中，如何以可學的 nonuniform range scale、fusion-group scale coupling、bit-exact shift alignment，以及 YOLO26 dual-head assignment 對齊，取得準確率與部署成本的聯合改善。

### 2.6 禁止的表述

在完整查重、消融與硬體驗證前，不得寫：

```text
首次提出 learnable SD4 scale
首次提出 relative exponent quantization
首次提出 one-to-one self-teaching quantization
完全 multiplierless
零額外硬體成本
不掉精度
真正 INT4/SD4 加速
state of the art
```

## 3. 必須正確理解的量化方法

## 3.1 Uniform affine quantization

通用形式：

```text
q      = clamp(round(x / s + z), q_min, q_max)
x_hat  = (q - z) * s
```

- `s > 0`：step size／scale。
- `z`：zero point。
- symmetric quantization 通常 `z = 0`。
- asymmetric quantization 允許非零 `z`，較適合 signed 且 skewed activation。
- QAT forward 使用 `x_hat`；真正 integer inference 使用 `q`、integer accumulator 與 requantization。

## 3.2 LSQ

LSQ 的核心不是只做 rounding，而是把每一層的 step size `s` 設成可訓練參數，直接由 task loss 更新。

對 signed weight：

```text
Qn = -2^(b-1)
Qp =  2^(b-1) - 1
q  = clamp(round(w / s), Qn, Qp)
w_hat = q * s
```

原始建議初始化：

```text
s_init = 2 * mean(abs(v)) / sqrt(Qp)
```

LSQ 的 step-size gradient 要做尺度校正：

```text
g_weight     = 1 / sqrt(N_weight * Qp)
g_activation = 1 / sqrt(N_feature * Qp)
```

實作必須包含 `grad_scale()` 與 `round_pass()` 的 STE；不能只把 `s` 直接丟進普通 `torch.round()` 後期待它自然學好。

重要觀念：LSQ 最佳化的是 task loss，不保證最小化 weight/activation MSE。這也是為什麼單純用 histogram MSE 選 scale 不等同於 LSQ。

## 3.3 PACT

原始 PACT activation：

```text
y = 0,             x < 0
y = x,        0 <= x < alpha
y = alpha,         x >= alpha
```

再把 `[0, alpha]` 均勻量化到 `2^b` 個 levels。`alpha` 為 learnable clipping parameter，通常每層共享一個 alpha，並加 regularization 避免 dynamic range 過大。

限制：

- 原始 PACT 是 non-negative activation quantizer。
- YOLO26 的 SiLU activation 會有負值，直接用 PACT 會把負 activation 截成 0。
- 可做的控制版本：
  - 把 activation 改成 ReLU/HSwish 再用 PACT；這同時改架構，必須單獨報告 FP baseline。
  - Dual-bound PACT：學習 `alpha_low < 0` 與 `alpha_high > 0`。
  - 直接使用 LSQ+ affine activation，通常更乾淨且更容易 integer export。

## 3.4 LSQ+

LSQ+ 是此研究對 SiLU activation 的主要基線。它對 signed、skewed activation 學習 asymmetric range：

```text
q     = clamp(round((x - beta) / s), q_min, q_max)
x_hat = q * s + beta
```

其中 `s > 0` 與 offset `beta` 都可學。LSQ+ 論文同時討論 signed/unsigned integer range；對 SiLU 類 activation，第一個主 baseline 建議使用：

```text
q_min = 0
q_max = 2^b - 1
beta  = learnable, usually negative
```

如此仍可使用全部 `2^b` 個 unsigned codes 表達包含小幅負值的 activation range，而不是把一半 codes預留給很窄的負區間。

#### LSQ+ 的部署方式

不要強迫把連續 `beta` 不加驗證地四捨五入成 integer zero point。至少實作並比較兩種 export contract：

1. **Offset folded into convolution bias**，適用於 symmetric weight／SD4 weight：

```text
w_hat = s_w * q_w
x_hat = s_x * q_x + beta

sum(w_hat * x_hat)
  = s_w*s_x * sum(q_w*q_x)
  + beta*s_w * sum(q_w)
```

第二項可在 compile/export 時預先計算成每個 output channel 的 bias correction。對 per-channel `s_w` 仍可逐 output channel計算。

2. **Affine integer zero-point approximation**：

```text
z_int = clamp(round(-beta / s_x), q_min, q_max)
beta_export = -z_int * s_x
```

這條路較符合一般 backend，但會引入 `beta` rounding gap。必須重新評估 export parity與 mAP，不能假設等價。

LSQ+ 的 activation `s,beta` 應使用數個 representative batches做 MSE-based initialization，以降低低位元 QAT 的 run-to-run variance。

對 **uniform QAT baseline** 的建議：

- Weight：per-output-channel symmetric LSQ。
- Activation：per-tensor unsigned-range asymmetric LSQ+。
- Fusion boundary：先測 LSQ+；FREQ 版本優先使用 zero-centered signed symmetric boundary quantizer，降低 shift alignment與 offset folding複雜度。
- Sigmoid等本來非負且 bounded 的 activation，可用 unsigned symmetric/affine quantization，不必強制學負 offset。

## 3.5 TQT

TQT 學習 clipping threshold，但把 quantization scale 約束為 power-of-two：

```text
log2_s = round_ste(theta)
s      = 2 ** log2_s
```

關鍵區分：

- TQT 讓 **scale 是 2 的次方**。
- 量化後的值仍是均勻 integer levels。
- MAC 仍然是 integer multiply-accumulate；主要簡化的是 requantization／rescaling。

因此 TQT 不等同於 SD4。

## 3.6 SD4 與 Learnable-Scale SD4（LS-SD4）

### 3.6.1 SD4 hardware codebook

依 SD4 原始 signed 4-bit PoT 表示與使用者上傳的 accelerator 論文，可將硬體值集合寫成：

```text
C_hw = {0, ±1, ±2, ±4, ±8, ±16, ±32, ±64}
```

使用者上傳論文中的 bit mapping 為：

| 4-bit code | decoded value |
|---|---:|
| `0000` | `+1` |
| `0001` | `+2` |
| `0010` | `+4` |
| `0011` | `+8` |
| `0100` | `+16` |
| `0101` | `+32` |
| `0110` | `+64` |
| `0111` | `0`（positive-zero code） |
| `1000` | `-1` |
| `1001` | `-2` |
| `1010` | `-4` |
| `1011` | `-8` |
| `1100` | `-16` |
| `1101` | `-32` |
| `1110` | `-64` |
| `1111` | `0`（negative-zero code） |

它有 15 個不同數值，但 4-bit encoding 有 16 個 codes。Exporter、packer 與 decoder 必須明確記錄：

```text
code -> sign -> exponent/value
positive zero code
negative zero code
canonical zero code for export
```

不可只把 codebook 存成 Python set，否則會失去 bit-level encoding。

### 3.6.2 建議使用 normalized training codebook

為避免 `s_w` 的語意混亂，訓練時使用 normalized codebook：

```text
C_norm = C_hw / 64
       = {0, ±2^-6, ±2^-5, ±2^-4, ±2^-3, ±2^-2, ±2^-1, ±1}
```

`C_hw` 與 `C_norm` 表示相同的離散形狀，只差一個固定常數。此時 `s_w` 可明確解釋為該 layer/channel 的最大正規化 range：

```text
w_hat ∈ s_w * C_norm
```

對硬體輸出，normalized code 可無損轉回原始 SD4 code：

```text
c_hw = 64 * c_norm
w_hat = s_w * c_norm
      = (s_w / 64) * c_hw

hardware_scale = s_w / 64
```

因此 manifest 必須保存 `training_range_scale=s_w` 與 `hardware_scale=s_w/64`，並測試兩條 decode path 完全一致。

### 3.6.3 Fixed SD4 baseline

固定或統計式 scale：

```text
s_w = max(abs(w))
# 或 percentile / MSE grid search 選出，但訓練中固定
```

Forward：

```text
u      = clamp(w / s_w, -1, 1)
c_hard = nearest_code(u, C_norm)
w_hat  = s_w * c_hard
```

這是必要 baseline。沒有 fixed SD4，就不能知道 learnable `s_w` 的真正收益。

### 3.6.4 Learnable-Scale SD4

將 positive range scale 設為可學：

```text
s_w = softplus(rho_w) + eps
u = w / s_w
u_clip = clamp(u, -1, 1)
c_hard = nearest_code(u_clip, C_norm)
c_ste = u_clip + stop_gradient(c_hard - u_clip)
w_hat = s_w * c_ste
```

說明：

- `s_w` 調整的是整個 nonuniform SD4 codebook 的實數範圍。
- 因 SD4 levels 非均勻，`s_w` 應稱為 **range scale** 或 codebook scale，不要不加說明地稱為 LSQ uniform step size。
- full-precision master weights 保留並更新；forward/backward 使用 fake-quantized weights。
- nearest-code boundary 與 clipping 行為必須有單元測試。
- 在未飽和區，這個 STE 對 scale 的局部梯度包含 `c_hard - u`；在飽和區，clamp 使 `u` 路徑梯度為 0，scale 梯度由端點 code 主導。這與 LSQ 的 scale-gradient 直覺一致，但 transition geometry 由 SD4 非均勻 boundaries 決定。

### 3.6.5 Scale granularity

必須支援下列模式：

```text
layer_wise:
  shape = [1, 1, 1, 1]

per_output_channel:
  shape = [C_out, 1, 1, 1]

grouped_output_channel:
  one scale per G output channels

fusion_group_shared:
  one base scale shared by selected branch/layer group
```

研究角色：

- `layer_wise`：主要工程 baseline，metadata 最少。
- `per_output_channel`：accuracy upper bound；不可直接稱 multiplierless，因為 output requantization 可能需每通道 multiplier。
- `grouped_output_channel`：accuracy/hardware compromise。
- `fusion_group_shared`：FREQ 的基礎。

### 3.6.6 Scale initialization

至少實作與比較：

1. **Max initialization**：`s = max(abs(w))`。
2. **Percentile initialization**：例如 99.9%、99.99%，範圍由 config 控制。
3. **MSE grid search**：搜尋 `s`，使 `||w - Q_SD4(w;s)||^2` 最小。
4. **Alternating MSE scale fitting**：在 hard assignments `c_i` 固定時，使用閉式更新：

```text
s_star = sum_i(w_i * c_i) / max(sum_i(c_i^2), eps)
```

再重新 nearest-code assignment，重複 5–20 次或直到收斂。此方法通常比只用 max 更強，應和 grid search 比較。

5. **Mean-absolute heuristic**：作 LSQ-inspired control，但不可盲目把原始 `Qp=64` 帶入，因 normalized SD4 並非 129-level uniform quantizer。

MSE 初始化只負責起點；最終 `s_w` 仍由 detection task loss 學習。LSQ 已顯示 task-optimal scale 不一定是 MSE-optimal，因此報告必須同時記錄 task metric 與 quantization error。fixed-SD4 baseline 也應使用最好的 MSE fitting，避免把 learnable method 和過弱的 max baseline 比較。

### 3.6.7 Scale gradient 與穩定化

第一版至少提供兩種可切換版本：

```text
A. autograd STE:
   u_clip = clamp(u, -1, 1)
   c_ste = u_clip + stopgrad(c_hard - u_clip)
   w_hat = s_w * c_ste

B. LSQ-style gradient-scaled STE:
   same forward
   backward gradient to s_w multiplied by g
```

建議 gradient scale 先使用：

```text
g = 1 / sqrt(N_w * K_pos)
K_pos = number of positive nonzero magnitudes = 7
N_w = number of weights controlled by one scale parameter
```

並消融：

```text
no grad scaling
1/sqrt(N_w)
1/sqrt(N_w * K_pos)
learned-scale LR multiplier only
```

不要把 uniform LSQ 的 `Qp` 公式未經驗證直接套到 SD4。若要做更精確的 boundary-sensitive gradient，需明確推導 nearest-code transition boundary，並用 finite-difference/gradient sanity test 驗證；先不要在第一版過度複雜化。

穩定化要求：

- `s_w = softplus(rho) + eps` 或 `exp(rho)`；不得直接 clamp Parameter 而破壞 optimizer state。
- qparam 不使用 weight decay。
- qparam LR 單獨設定，初始建議 `{0.01, 0.1, 1.0} × base_lr` 小型 sweep。
- 記錄 `s_w`、gradient norm、clipping ratio、codebook occupancy、zero ratio。
- scale collapse、爆炸或全部 codebook collapse 時立即停止 full run。

### 3.6.8 Codex 參考演算法（不是可直接複製的最終程式）

```python
# Shapes and distributed behavior must be implemented and tested explicitly.
# This pseudocode only fixes the mathematical contract.

def ls_sd4_fake_quant(weight, rho, levels, grad_factor, eps=1e-8):
    # positive range scale; shape may be scalar, [Cout,1,1,1], or grouped
    scale = softplus(rho) + eps
    scale = grad_scale(scale, grad_factor)

    u = weight / scale
    u_clip = clamp(u, -1.0, 1.0)

    # Exact nearest-level lookup with deterministic midpoint tie-breaking.
    index = argmin(abs(u_clip[..., None] - levels), axis=-1)
    hard = gather(levels, index)

    # Hard value in forward, identity gradient through u_clip in backward.
    code_ste = u_clip + stop_gradient(hard - u_clip)
    dequant_weight = scale * code_ste

    return dequant_weight, index, scale
```

最低要求：

- `levels` 只存 distinct normalized values；bit encoder另以 lookup table把 value index映射成 canonical 4-bit code。
- midpoint tie-breaking 必須固定，例如 ties-to-smaller-magnitude，並寫入 manifest。
- zero 必須 export為 canonical zero code；negative-zero只用於 decode compatibility，除非硬體另有需求。
- `grad_factor` 依每個 scale控制的 weight數量計算，不是永遠用整層總數。
- 不能在每個 forward用 Python loop掃 codebook；使用 broadcast/vectorized distance或預先推導 boundaries。

### 3.6.9 PoT-constrained outer scale

硬體導向版本：

```text
theta_w = learnable real parameter
k_w = round_ste(theta_w)
s_w = 2 ** k_w
```

或以 shared floating base scale 搭配 integer relative exponent：

```text
s_i = s_group * 2 ** k_i
```

區分：

- `s_w = 2^k`：外層 rescale 可用 shift，但表達自由度較低。
- `s_i = s_group * 2^k_i`：保留一個 shared continuous base，分支/通道差異由 shift 表示。
- arbitrary per-channel `s_w`：預期準確率最好，但硬體 metadata 與 requantization 成本最高。

### 3.6.10 SD4 inference 的正確硬體敘述

卷積核心中：

```text
activation * SD4_code
```

可映射成 sign change、zero skip 與 bit shift。但完整 layer output 仍包含 activation scale、weight scale、bias、accumulator 與 output requantization。對 activation integer scale `s_x` 與 SD4 hardware scale `s_w_hw=s_w/64`：

```text
acc_int = sum(q_x * c_hw)
real_conv_output = acc_int * s_x * s_w_hw
```

Bias 應量化到近似 `s_x * s_w_hw` 的 accumulator domain；輸出再以 `M = s_x*s_w_hw/s_y` fixed-point requantize。per-channel `s_w_hw` 會使 `M` 成為 per-output-channel vector。只有在這些 scale 被 fold、固定點化或 PoT 限制，且有對應 kernel/datapath 時，才能宣稱 multiplier reduction。

## 3.7 APoT

APoT 的 level 是多個 power-of-two term 的和，不只是單一 power-of-two：

```text
level = ±(2^e1 + 2^e2 + ...)
```

它比單一 PoT codebook 更密，能較好貼合 bell-shaped／long-tailed distribution，但一個乘法可能需要多個 shift-add。比較 SD4 與 APoT 時，不能只比較 bit 數；必須同時報告每個非零 weight 的平均 shift-add 數。

## 3.8 Mixed precision

Mixed precision 可以依 stage、layer 或 channel 指派不同格式，例如：

```text
Stem:       W8A8
Backbone:   W4A8 or INT6/A8
Neck:       SD4W/A8
O2O head:   W8A8 initially, then W4A8
O2M head:   FP16/W8A8 during training only
Postprocess: FP16/FP32
```

但 mixed precision 本身不是創新。要成為研究貢獻，bit/codebook assignment 必須來自可解釋的 sensitivity 或 hardware objective，並和 uniform precision 公平比較。

---

## 4. 硬體負擔與部署判斷

| 方法 | Training 額外負擔 | Inference 表示 | 主要硬體成本／限制 |
|---|---|---|---|
| LSQ W8/W4 | learnable scale、fake quant | 標準 integer weight | integer MAC + requant scale |
| LSQ+ A8/A4 | scale + offset、observer | asymmetric integer activation | zero-point handling、requant |
| PACT | learnable clipping alpha | nonnegative clipped activation | 原始形式不保留 SiLU 負值 |
| Fixed SD4 | nearest-code fake quant | 4-bit sign/exponent/zero code | shift/sign datapath；scale 仍要處理 |
| **LS-SD4 layer-wise** | 每層一個 learnable range scale | SD4 codes + one scale/layer | scale 可 fold 成 layer requant factor |
| **LS-SD4 per-channel** | 每輸出通道一個 scale | SD4 codes + scale vector | 可能需 per-channel fixed-point multipliers；metadata 增加 |
| PoT-scale SD4 | STE exponent | SD4 codes + integer exponent | shift-only outer scale；accuracy 可能下降 |
| shared-base + relative exponent | group scale + integer offsets | one base + small exponent vector | shift align、bit growth、round/saturation |
| APoT | codebook/STE 較複雜 | 多個 PoT term | 每個 weight 多次 shift-add |
| Mixed precision | policy/search/debug | 多格式 weights | decoder/mode switch/control overhead |
| KD | teacher forward/loss | 無 teacher inference cost | training memory/time only |

### 4.1 必須報告的硬體 proxy

沒有實際 kernel/FPGA 時，至少報告：

```text
actual packed weight bytes
scale/zero-point metadata bytes
SD4 code occupancy and zero ratio
integer MAC count
shift count
shift-add count
arbitrary requant multiplier count
per-channel requant multiplier count
estimated accumulator bit growth
post-shift saturation count
activation traffic and peak activation bytes
```

### 4.2 關於 learnable `s_w` 的直接回答

加入 learnable `s_w` **很可能比 fixed SD4 好**，因為它允許每個 layer/channel 的 SD4 range 由 task loss 調整。但必須區分：

```text
accuracy perspective:
  per-channel free s_w usually gives highest flexibility

hardware perspective:
  layer-wise s_w is cheaper
  grouped/shared s_w is compromise
  PoT s_w or relative exponent is most shift-friendly
```

因此主實驗不能只做一個版本。至少要比較 fixed、layer-wise learnable、per-channel learnable、PoT-constrained、group-shared 五種 scale policy。

### 4.3 禁止的硬體推論

- PyTorch GPU 上 SD4 fake quant 比 FP 慢，不代表硬體無效；fake quant 含 nearest-code search。
- PyTorch fake quant 變快，也不代表真實 SD4 kernel 變快。
- 只有 4-bit storage 不等於 4-bit compute。
- 有 SD4 codebook 不等於整個 network 完全無 multiplier。
- 有 per-channel learnable scale 不等於 scale 在 inference 消失；必須證明已 fold 或 fixed-point 化。

## 5. Scale 與 exponent 的可辨識性：哪些有效、哪些只是重參數化

### 5.1 自由 scale + global exponent offset 為冗餘

假設 codebook：

```text
C(s, o) = {0, ±s * 2^(o+i) | i in E}
```

若 `s` 是任意正實數，令：

```text
s_prime = s * 2^o
```

則：

```text
C(s, o) = C(s_prime, 0)
```

因此同一 layer 同時學自由 `s` 與共同 exponent offset `o` 不增加表示能力，只造成參數不可辨識、optimizer 漂移與論述風險。

### 5.2 有意義的 scale/exponent 組合

1. **固定 `s`，學 exponent**：純 PoT range selection。
2. **`s = 2^k`**：只學整數/近整數 exponent，rescale 可 shift。
3. **group-shared continuous base + relative integer exponent**：

```text
s_i = s_g * 2^k_i
```

4. **固定 reference branch/channel**：`k_ref = 0`，消除 gauge ambiguity。
5. **有限 exponent span**：`k_i ∈ [k_min, k_max]`，對應實際 shifter 與 bit growth。
6. **學 exponent range/gaps**：改變 codebook 幾何，而不是整體平移；此方向較複雜，先不作第一版。

### 5.3 Scale 粒度的研究梯度

```text
free per-channel scale
    ↓ accuracy upper bound, highest metadata/requant cost

grouped scale
    ↓ compromise

layer-wise scale
    ↓ cheap baseline

shared base + relative exponent
    ↓ shift-compatible relative alignment

pure PoT scale
      lowest arbitrary multiplier cost, least flexibility
```

### 5.4 Relative exponent 也不是裸公式創新

2026 年 PTQ4SNN 已使用 channel-wise Unified Scale Bridge：目標 scale 等於來源 scale 乘上 `2^k`，以保留 shift-compatible conversion。雖然其場景是 SNN membrane state PTQ，和 YOLO26/SD4/fusion QAT 不同，但它足以否定「`s_i=s_g2^k_i` 公式首次提出」的表述。

本研究若要形成貢獻，焦點必須是：

- YOLO26 neck 的實際 fusion graph/group discovery。
- SD4 weight range 與 activation fusion scale 的聯合設計。
- Concat/Add 前 bit-exact shift-round-saturate。
- arbitrary requant multiplier 減少與 mAP/clipping trade-off。
- 相對 independent LSQ+/TQT/shared-scale/GABFusion controls 的公平消融。

## 6. Research Extension A：Fusion-Relative Scale Coupling（FREQ）

## 6.1 研究動機與定位

YOLO neck 會融合不同解析度、語意深度與 activation distribution 的 feature。一般 QAT 常讓每個分支獨立學 scale，fusion 前再用一般 requantization，可能造成：

- 分支 dynamic range 不匹配。
- 低 bit 下某一支大量 clipping，另一支 resolution 浪費。
- Add/Concat 前需要多個 arbitrary fixed-point multipliers。
- fake-quant graph 與實際 FPGA shift datapath 不一致。

FREQ 的目標不是一般的 feature reweighting，而是把 fusion-boundary scale 關係限制成可 bit-exact 實作的 dyadic relation。

先行研究風險：GABFusion 已研究低位元 fusion 的 gradient imbalance；PTQ4SNN 已研究 `scale × 2^k` 型 scale bridge。因此 FREQ 只能作為**YOLO26 + SD4 + neck fusion + deployment constraint 的特定設計與驗證**，不可宣稱裸概念首次提出。

## 6.2 第一版研究範圍

先只對 neck 的跨尺度 fusion 節點實作：

- top-down path 的 Concat。
- bottom-up path 的 Concat。
- 若實際 YOLO26m graph 有 Add fusion，納入同一 group。
- 暫時不對每個 backbone residual/CSP 內部 Concat 都套 FREQ，避免一次改動過大。

Codex 必須從實際 checkpoint/model graph 產生 fusion manifest，不得硬編 layer index。

## 6.3 數學定義

對 fusion group `g`，學習一個共享 base scale：

```text
s_g = softplus(rho_g) + eps
```

對每個輸入分支 `b`，學習有限範圍整數 offset：

```text
o_gb = clamp(round_ste(phi_gb), -O_max, O_max)
```

分支 step size：

```text
Delta_gb = s_g * 2^o_gb
```

為消除 gauge ambiguity，固定一個 reference branch：

```text
o_g,ref = 0
```

或使用等價約束 `sum_b o_gb = 0`；第一版固定 reference branch 較容易實作與匯出。

Fusion boundary activation 使用 zero-centered signed quantization：

```text
q_gb = clamp(round(x_gb / Delta_gb), q_min, q_max)
x_hat_gb = Delta_gb * q_gb
```

其他非 fusion activation 繼續用 LSQ+ asymmetric quantization。

## 6.4 Shift-only alignment

選擇 fusion reference exponent `o_ref = 0` 後，將每個分支對齊到 base scale `s_g`：

```text
q_aligned_gb = shift_round_sat(q_gb, o_gb)
```

概念上：

```text
Delta_gb * q_gb = s_g * (2^o_gb * q_gb)
```

因此只需整數左移／右移，再做 rounding 與 saturation。實作必須模擬：

- arithmetic right shift 的 rounding mode。
- left shift bit growth。
- target bit-width clamp。
- saturation count。
- zero preservation。

對 Concat，可把各分支 shift-align 後 concatenation。對 Add，兩支必須對齊到相同 scale 才能相加。

可選的第二種部署模式：對 Concat，把 branch-relative scale fold 到下一層 convolution weight；只有在硬體支援 per-input-group scaling 時才使用。預設仍採 shift-align，因為可驗證性較高。

## 6.5 Regularization 與硬體成本

總共加入：

```text
L_offset = mean(abs(o_gb))
L_sat    = mean(saturation_rate_after_alignment)
L_span   = max_b(o_gb) - min_b(o_gb)
```

硬體 proxy：

```text
C_hw = lambda_shift * total_shift_distance
     + lambda_sat   * saturation_count
     + lambda_req   * arbitrary_requant_multiplier_count
     + lambda_mem   * estimated_activation_bytes
```

訓練 loss 可使用：

```text
L = L_task + lambda_offset * L_offset
           + lambda_span   * L_span
           + lambda_sat    * L_sat_surrogate
```

注意：真實 saturation count 不可微。第一版可用超出可表示範圍的 soft overflow penalty 作 surrogate，並另外記錄 hard saturation metric。

## 6.6 FREQ 的必要消融

至少比較：

1. 每分支獨立 LSQ+ scale，fusion 時一般 requantization。
2. 所有分支強制共用同一 scale。
3. 獨立 scale + soft log2 scale alignment：`abs(log2(s_i)-log2(s_j))`。
4. TQT：各分支 scale 獨立，但各自為 PoT。
5. **FREQ：shared base + relative integer exponent offsets。**
6. FREQ 去掉 offset regularization。
7. FREQ 不做 saturation-aware penalty。

只有 FREQ 同時在 mAP、clipping、requantization multiplier count 或硬體成本上優於 1–4，才值得成為主要貢獻。

---

## 7. Research Extension B：Branch-aware O2M-to-O2O Distillation（BD）

## 7.1 YOLO26 專屬動機與先行研究限制

官方 YOLO26 end-to-end Detect head 使用 dual-head design：訓練時同時包含 one-to-many 與 one-to-one prediction，推論採 one-to-one path；本地版本仍必須由 Codex 實際檢查。這提供一個 training-only teacher 的量化研究機會：

- one-to-many branch 可維持 FP16/W8A8，提供較密集 supervision。
- one-to-one inference branch 與共享 trunk 使用目標低位元。
- one-to-many branch 在 fuse/export 時移除，理論上不增加 inference model cost。

但 GHOST 已把 object detection、hybrid quantization 與 one-to-one self-teaching 結合。因此本研究不可宣稱「one-to-one self-teaching 首次用於量化偵測」。可能的差異點只在：

- YOLO26 原生 O2M/O2O dual-head，而非額外複製完整 teacher network。
- O2M 與 O2O assignment 不同，需依 ground-truth assignment 聚合與匹配。
- Progressive Loss/STAL/實際 loss API 可能影響 KD 權重，需要 model-specific integration。
- export 後必須驗證 O2M 全部移除並重算 inference-only bytes/latency。

若 assignment metadata 無法可靠取得，或 generic KD 已同樣有效，BD 應降級為輔助工程技巧，不作主貢獻。

## 7.2 Precision policy

第一版：

```text
Shared backbone/neck: target quantization policy
O2O head:             target quantization policy
O2M head:             FP16 or W8A8, training only
Decode/postprocess:   FP16/FP32
```

控制組：

- O2M 與 O2O 都量化。
- O2M FP16，但不做 KD。
- O2M FP16 + ordinary logits KD。
- O2M FP16 + assignment-aware KD。
- 外部 frozen FP32 teacher + feature KD。

## 7.3 Assignment-aware teacher aggregation

對每個 ground-truth object `j`：

- 從 one-to-many assigner 取得正樣本集合 `T_j`。
- 從 one-to-one assigner 取得唯一 matched prediction `k_j`。
- 以 teacher confidence、IoU 或 assignment quality 建立權重：

```text
alpha_i = softmax(quality_i / tau_a),  i in T_j
```

聚合 teacher class probability／logit：

```text
p_bar_j = sum_i alpha_i * p_teacher_i
```

聚合 teacher box：

```text
b_bar_j = sum_i alpha_i * b_teacher_i
```

student 為 one-to-one match `k_j`：

```text
L_kd_cls = sum_j KL(p_bar_j || p_student_kj)
L_kd_box = sum_j [L1(b_bar_j, b_student_kj) + GIoU(...)]
```

若直接平均 box 導致幾何不合理，可改用最高 quality teacher box，或在 decoded xyxy 空間做加權平均。三種聚合方式應做小型消融。

禁止：

- 直接以 raw tensor index 對齊 O2M/O2O。
- 把 teacher branch 的 gradient 回傳進 teacher。
- 在未檢查官方 Progressive Loss/STAL 實作前，覆蓋原始 loss 權重。

## 7.4 Feature distillation

對 trunk 量化造成的誤差，O2M head 只看量化後 feature 可能不夠。可加入 frozen FP32 checkpoint 作外部 teacher，對 P3/P4/P5 或 neck output 做 feature KD：

```text
L_feat = sum_l normalized_mse(norm(F_student_l), norm(F_teacher_l))
```

或使用 channel-wise attention map：

```text
A(F) = mean_channel(abs(F))
```

外部 teacher KD 是已知基線，不是主要創新；它的作用是分辨 FREQ/BD 的改善是否只是一般 KD 效果。

## 7.5 Branch-aware KD 的必要消融

1. 無 KD。
2. 外部 FP32 teacher logits KD。
3. 外部 FP32 feature KD。
4. O2M high-precision branch，不做 KD。
5. O2M→O2O naive index KD。
6. O2M→O2O assignment-aware class KD。
7. assignment-aware class + box KD。
8. assignment-aware KD + FREQ。

只有 6/7 相對 2/3/5 有穩定改善，才能支持「YOLO26 branch-aware」的論點。

---

## 8. 完整 Training Objective

最終 loss 只能逐階段加入，不得從第一個 run 同時打開所有項目。

```text
L_total = L_yolo26_task
        + lambda_scale_reg * L_scale_reg
        + lambda_codebook * L_codebook_usage_optional
        + lambda_freq * L_freq
        + lambda_kd * L_branch_kd
```

### 8.1 LS-SD4 本身

第一版通常不需額外 quantization MSE loss，直接由 task loss 學 `s_w`。可選 regularizer：

```text
L_scale_reg = mean((log2(s_w) - stopgrad(log2(s_init)))^2)
```

只在 scale 明顯漂移或 collapse 時使用，且需消融。不要強迫 `s_w` 靠近初始化而抹掉 task-driven learning。

可選 codebook occupancy penalty 只用於診斷／避免 collapse，不能預設每個 code 都應均勻使用。

### 8.2 FREQ

```text
L_freq = lambda_offset * mean(abs(k_i))
       + lambda_span   * exponent_span
       + lambda_overflow * soft_overflow_penalty
       + lambda_requant * arbitrary_multiplier_surrogate
```

### 8.3 Branch KD

```text
L_branch_kd = lambda_cls * L_cls_kd
            + lambda_box * L_box_kd
            + lambda_feat * L_feat_kd_optional
```

啟用順序：

```text
1. L_task only, fixed SD4
2. L_task only, LS-SD4
3. LS-SD4 + FREQ
4. LS-SD4 + BD
5. LS-SD4 + FREQ + BD
```

每增加一項都必須有 config diff、ablation 與單獨 checkpoint。

## 9. 建議量化政策

### 9.1 第一輪可靠基線

```text
Stem:                    W8A8 LSQ/LSQ+
Backbone conv:           W8A8 → W4A8 uniform baseline
Neck conv:               W8A8 → W4A8 → SD4W/A8
O2O head internal conv:  W8A8 first; later W4A8 or LS-SD4/A8
O2M head:                FP16/W8A8 training-only control
Final prediction conv:   W8A8 initially
SiLU:                    operation kept FP16/FP32; quantize its output
Attention/softmax/div:   FP16/FP32 initially
Decode/top-k/postprocess: FP16/FP32
Accumulator:             INT32 in integer simulation
```

Activation 主 baseline：

```text
per-tensor asymmetric LSQ+ A8
```

Weight 主候選：

```text
layer-wise LS-SD4 first
per-channel LS-SD4 as upper bound
```

### 9.2 為什麼先 layer-wise LS-SD4

- 只有一個 scale/layer，較容易 debug、export 與 fold。
- 能直接回答 learnable range 是否優於 fixed SD4。
- per-channel scale 增益若很小，就沒有必要增加硬體複雜度。
- per-channel 只在 layer-wise 已穩定後做 accuracy upper-bound 實驗。

### 9.3 低位元推進順序

```text
FP baseline
→ uniform W8A8
→ uniform W4A8
→ fixed SD4W/A8
→ layer-wise LS-SD4W/A8
→ per-channel LS-SD4W/A8 upper bound
→ PoT/group-shared scale variants
→ stage-wise mixed precision
→ neck FREQ boundary A8
→ FREQ boundary A6/A4
→ optional O2M→O2O KD
→ hardware-cost-aware final policy
```

不要從 FP 直接跳到全網 SD4W/A4 + FREQ + KD；否則無法定位失敗來源。

### 9.4 Stage-wise 預設假設

使用者上傳的 YOLOX accelerator 論文顯示不同 stage 對 SD4/低位元的敏感度不同，且最後採 backbone INT6、neck SD4、head ternary、activation INT8。這只能作 prior baseline，不可直接套到 YOLO26。Codex 必須重做 YOLO26 的 layer/stage sensitivity，再決定 mixed policy。

## 10. Repository 設計

建議新增：

```text
quantization/
  __init__.py
  common.py
  observers.py
  ste.py
  uniform_lsq.py
  asymmetric_lsq.py
  pact.py
  tqt.py
  sd4.py
  learnable_sd4.py
  scale_constraints.py
  fusion_relative.py
  wrappers.py
  inject.py
  policies.py
  sensitivity.py
  losses.py
  branch_distill.py
  export.py
  packing.py
  integer_sim.py
  diagnostics.py

scripts/
  inspect_yolo26.py
  calibrate_quant.py
  train_qat.py
  evaluate_quant.py
  sweep_quant.py
  run_sensitivity.py
  export_quant.py
  verify_integer.py

configs/quant/
  w8a8_lsqplus.yaml
  w4a8_lsqplus.yaml
  w4a4_lsqplus.yaml
  sd4w_a8_fixed.yaml
  ls_sd4_layer_a8.yaml
  ls_sd4_channel_a8.yaml
  ls_sd4_pot_scale_a8.yaml
  tqt_w4a8.yaml
  mixed_stage.yaml
  branch_kd.yaml
  freq.yaml
  freq_bd.yaml

outputs/quant/
  manifests/
  calibration/
  sensitivity/
  experiments/
  exports/
  reports/

tests/quantization/
  test_ste.py
  test_lsq.py
  test_lsqplus.py
  test_pact.py
  test_tqt.py
  test_sd4.py
  test_learnable_sd4.py
  test_scale_constraints.py
  test_fusion_relative.py
  test_module_injection.py
  test_checkpoint_roundtrip.py
  test_dual_head.py
  test_export_parity.py
  test_integer_sim.py
```

若 repository 已有類似結構，整合而不是重複建立。

---

## 11. 核心 API 規格

## 11.1 Quantizer state machine

每個 quantizer 至少支援：

```python
class QuantizerBase(nn.Module):
    enabled: bool
    observer_enabled: bool
    fake_quant_enabled: bool
    initialized: bool

    def enable_observer(self) -> None: ...
    def disable_observer(self) -> None: ...
    def enable_fake_quant(self) -> None: ...
    def disable_fake_quant(self) -> None: ...
    def freeze_qparams(self) -> None: ...
    def export_qparams(self) -> dict: ...
```

State 必須被 `state_dict` 保存，checkpoint reload 後行為一致。

## 11.2 Weight quantizer

介面：

```python
weight_quantizer(weight: Tensor) -> Tensor
```

支援：

- symmetric uniform LSQ：per-tensor／per-output-channel，W8/W6/W4/W3/W2。
- fixed SD4：max/percentile/MSE initialized scale。
- Learnable-Scale SD4：layer-wise、per-output-channel、grouped。
- optional PoT-constrained outer scale。
- export mode：回傳或保存 SD4 code tensor、scale tensor、encoding policy。

建議 class：

```python
class LearnableSD4WeightQuantizer(QuantizerBase):
    normalized_codebook: Tensor
    rho: nn.Parameter
    scale_granularity: str
    grad_scale_mode: str
    scale_constraint: str

    def positive_scale(self) -> Tensor: ...
    def encode(self, weight: Tensor) -> Tensor: ...
    def dequantize(self, codes: Tensor) -> Tensor: ...
    def forward(self, weight: Tensor) -> Tensor: ...
    def export_qparams(self) -> dict: ...
```

`encode()` 的 hard codes 必須與 `forward()` 中的 hard nearest-code 完全一致。不可在 export 另外寫一套近似規則。

## 11.3 Activation quantizer

介面：

```python
activation_quantizer(x: Tensor) -> Tensor
```

支援：

- asymmetric LSQ+。
- symmetric boundary quantizer。
- PACT control implementation。
- calibration observer。
- clipping rate、zero ratio、SNR、min/max/percentile logging。

## 11.4 Quantized convolution wrapper

優先實作明確 wrapper，而不是 global hook：

```python
class QuantConv2d(nn.Module):
    fp_conv: nn.Conv2d
    weight_quantizer: QuantizerBase
    input_quantizer: QuantizerBase | None
    output_quantizer: QuantizerBase | None
```

必須保留：

- stride、padding、dilation、groups。
- bias。
- dtype/device。
- state_dict key mapping。
- `requires_grad` 狀態。
- depthwise convolution。

當所有 quantizer disabled 時，輸出必須與原始 module 在 tolerance 內一致。

## 11.5 Module injection

`inject.py` 必須：

1. 依 module type 與完整 module path 產生政策。
2. 明確排除不量化模組。
3. 區分 backbone、neck、O2M head、O2O head、final prediction conv。
4. 產生替換前後 manifest。
5. 支援 restore original modules 或重新載入 checkpoint。
6. 不依賴脆弱的數字 layer index。

MQBench、QTool、VP-QOD 可作為 quantizer 與 detector integration 的參考，但第一版不要把整個 Ultralytics model 強制轉成 torch.fx graph。YOLO26 的動態 train/eval、end-to-end dual-head 與 export path 可能使 FX rewrite 脆弱。

---

## 12. Fusion group discovery

`inspect_yolo26.py` 必須輸出：

```json
{
  "model_class": "...",
  "ultralytics_version": "...",
  "checkpoint": "...",
  "end2end": true,
  "reg_max": 1,
  "modules": [],
  "fusion_groups": [],
  "one2many_paths": [],
  "one2one_paths": [],
  "prediction_layers": [],
  "attention_or_special_ops": []
}
```

Fusion group 每筆至少包含：

```json
{
  "name": "neck.concat_0",
  "op": "concat",
  "input_module_paths": ["...", "..."],
  "input_shapes": [[1, 128, 160, 160], [1, 128, 160, 160]],
  "output_shape": [1, 256, 160, 160],
  "candidate_for_freq": true
}
```

可用一次 forward hook 收集 producer/output shape，但量化執行需改成明確 module/wrapper。

---

## 13. YOLO26 dual-head 接入規格

Codex 必須檢查實際 Detect head 是否具有：

```text
one2many
one2one
end2end
fuse()
```

並確認：

- training forward 回傳的 dict 結構。
- one2one 是否由 detached features 計算。
- validation 使用哪一支 prediction。
- fuse/export 是否真的移除 one2many module。
- checkpoint 中兩支 head 的 key naming。

量化 policy 必須能指定：

```yaml
head_policy:
  one2many:
    weight_bits: 16
    activation_bits: 16
    quantize: false
    training_only: true
  one2one:
    weight_bits: 8
    activation_bits: 8
    quantize: true
  final_prediction_conv:
    weight_bits: 8
    activation_bits: 8
```

完成 export 後，必須 assertion：

```text
one2many parameters absent from exported inference model
```

並重新計算 inference-only parameter bytes，不能把 training-only O2M branch 算入部署模型。

---

## 14. BN、activation 與 export

### 14.1 BatchNorm

第一版流程：

1. 載入 FP checkpoint。
2. Observer warmup 時允許 BN 更新少量 steps，或完全沿用 pretrained running stats。
3. QAT 開始後 freeze BN running mean/var。
4. 匯出前 fold Conv+BN。
5. 比較 fold 前 fake-quant 與 fold 後 inference output。

若 export gap 過大，再實作 on-the-fly folded-weight QAT；不要一開始就增加複雜度。

### 14.2 SiLU

第一版保留 SiLU 本身為 FP16/FP32 operator，量化 SiLU output 作為下一層輸入。之後再做：

- SiLU LUT／piecewise approximation。
- HSwish 替換。
- hardware-friendly activation co-design。

Activation 替換會改變 FP architecture，必須有獨立 FP baseline，不能把其結果混在純量化方法內。

### 14.3 Export

至少支援兩條路：

1. **標準 uniform baseline**：ONNX Q/DQ 或 backend 可接受的 INT8 graph。
2. **研究格式**：SD4/FREQ custom manifest + packed binary + integer simulator。

SD4/FREQ 不可假裝成標準 INT4 ONNX 後宣稱 backend 會使用 shift kernel。若無 custom op/kernel，必須明確標示為 simulation/export artifact。

---

## 15. 建議 Config Schema

```yaml
experiment:
  name: ls_sd4_layer_a8
  seed: 0
  output_dir: outputs/quant/experiments/ls_sd4_layer_a8_seed0
  tags: [yolo26m, qat, sd4, lsq_plus]

model:
  checkpoint: runs/detect/optim_1280/weights/best.pt
  task: detect
  imgsz: 1280
  dataset: Finale_version.v6i.yolov11/data.yaml
  trust_checkpoint_graph_over_official_assumptions: true

quantization:
  enabled: true
  accumulator_bits: 32
  first_last_bits: 8
  keep_fp16_patterns:
    - "*softmax*"
    - "*decode*"
    - "*postprocess*"

  weight:
    method: ls_sd4       # lsq | fixed_sd4 | ls_sd4 | tqt | apot
    bits: 4
    symmetric: true
    granularity: layer_wise  # layer_wise | per_output_channel | grouped
    group_size: null

  activation:
    method: lsq_plus
    bits: 8
    symmetric: false
    granularity: per_tensor
    zero_point_mode: learned_offset
    export_offset_mode: bias_fold  # bias_fold | integer_zero_point

sd4:
  normalized_codebook: true
  hardware_values: [0, 1, 2, 4, 8, 16, 32, 64]
  canonical_zero_code: positive_zero
  initialization: mse_alternating  # max | percentile | mse_grid | mse_alternating | mean_abs
  percentile: 99.9
  mse_grid_min_ratio: 0.25
  mse_grid_max_ratio: 1.25
  mse_grid_steps: 101
  learnable_scale: true
  scale_parameterization: softplus
  scale_eps: 1.0e-8
  scale_constraint: free     # free | power_of_two | shared_relative
  grad_scale_mode: nw_kpos   # none | nw | nw_kpos
  qparam_lr_multiplier: 0.1
  qparam_weight_decay: 0.0
  module_patterns: []
  exclude_patterns: []

mixed_precision:
  enabled: false
  policy_source: sensitivity_manifest
  stage_defaults:
    stem: w8a8
    backbone: ls_sd4_a8
    neck: ls_sd4_a8
    one2one_head: w8a8
    one2many_head: fp16
    prediction_conv: w8a8

freq:
  enabled: false
  fusion_scope: neck_cross_scale_only
  activation_bits: 8
  boundary_symmetric: true
  base_scale_init: mse
  offset_min: -3
  offset_max: 3
  reference_branch: 0
  rounding_mode: nearest_even
  saturation_mode: clamp
  lambda_offset: 1.0e-4
  lambda_span: 1.0e-4
  lambda_overflow: 1.0e-3

branch_distillation:
  enabled: false
  teacher: one2many
  teacher_precision: fp16
  student: one2one
  assignment_aware: true
  aggregation: quality_weighted
  temperature: 2.0
  lambda_cls: 0.5
  lambda_box: 0.5
  external_fp_teacher: null
  lambda_feature: 0.0

training:
  epochs: 50
  batch: auto
  optimizer: inherit_or_explicit
  base_lr: inherit
  observer_batches: 64
  freeze_observer_epoch: 5
  freeze_bn_epoch: 1
  amp: true
  resume: false

reporting:
  log_layer_stats: true
  log_scale_trajectory: true
  log_codebook_usage: true
  log_fusion_offsets: true
  log_saturation: true
  save_integer_manifest: true
  save_resolved_config: true
```

所有 `inherit`、`auto` 與 module pattern 都必須在執行時解析成實際值並寫入 `resolved_config.yaml`。每個 checkpoint 同時保存 quantizer state、policy manifest 與 codebook version。

## 16. 實作 Phases 與 Gate

## Phase 0：環境、實際模型 graph 與 FP baseline

### 任務

1. 找到 repository root、checkpoint、dataset YAML。
2. 記錄 git commit、Python、PyTorch、CUDA、Ultralytics、ONNX、GPU。
3. 載入實際 checkpoint；輸出完整 module path/type/parameters。
4. 確認 Detect head、`end2end`、`reg_max`、one2many/one2one path。
5. 掃描 Concat/Add fusion groups、attention、softmax、decode/postprocess。
6. 在 `imgsz=1280` 重跑 FP validation。
7. 保存固定 validation inputs 與 FP outputs。
8. 測試 train/eval/fuse/export，確認 inference path。

### 產物

```text
outputs/quant/manifests/environment.json
outputs/quant/manifests/model_architecture.json
outputs/quant/manifests/fusion_groups.json
outputs/quant/reports/fp_baseline.md
outputs/quant/reference/fp_inputs.pt
outputs/quant/reference/fp_outputs.pt
```

### Gate

- checkpoint 可載入，FP mAP 可重現或差異有解釋。
- quantization module 尚未注入。
- train/eval/fuse/export 行為已理解。

未通過不得進入 Phase 1。

## Phase 1：Quantizer primitives 與 unit tests

### 任務

實作：

- STE round、gradient scaling。
- uniform LSQ weight quantizer。
- LSQ+ activation quantizer。
- PACT control、TQT control。
- SD4 normalized codebook、hard encoder/decoder。
- fixed SD4 quantizer。
- LS-SD4 layer-wise/per-channel/grouped scale。
- max/percentile/MSE-grid/alternating-MSE initialization。
- PoT scale constraint。

### Gate

- 所有 unit tests 通過。
- quantizer disabled 與 identity/parity 一致。
- scale 始終正且 checkpoint roundtrip 一致。
- SD4 hard values/codes 屬於指定 encoding。
- fake quant、encode、decode 三者一致。
- AMP 下無 NaN/Inf。

## Phase 2：Uniform W8A8/W4A8 可靠基線

### 任務

1. module replacement 與 policy manifest。
2. calibration/initialization。
3. LSQ W8 + LSQ+ A8 QAT。
4. LSQ W4 + LSQ+ A8 QAT。
5. first/last W8 與全 W4 對照。
6. 標準 INT8 ONNX Q/DQ export baseline。

### Gate

- W8A8 接近 FP baseline。
- W4A8 穩定，無 scale collapse。
- fake-quant、ONNX/integer simulator 在 tolerance 內。

## Phase 3：Fixed SD4 與 Learnable-Scale SD4 核心驗證

### 任務

1. fixed max-scale SD4W/A8。
2. fixed MSE-scale SD4W/A8（grid與alternating fitting取較佳者）。
3. layer-wise LS-SD4W/A8。
4. 比較 scale initialization 與 gradient scale variants。
5. 保存 scale trajectory、clipping、SQNR、codebook occupancy、zero ratio。
6. 產生 bit-exact SD4 pack/unpack 與 manifest。

### Gate

- layer-wise LS-SD4 至少穩定收斂。
- exporter/simulator 可重現 fake quant。
- 若 LS-SD4 不優於 fixed SD4，先分析 scale gradient/initialization，不進入 FREQ。

## Phase 4：Scale granularity、constraint 與 accuracy/hardware frontier

### 任務

1. per-output-channel LS-SD4：accuracy upper bound。
2. grouped-output-channel LS-SD4：至少 2–3 個 group size。
3. PoT-constrained outer scale。
4. layer-wise vs channel-wise metadata/requantization cost。
5. 比較 TQT W4、可選 APoT W4。

### Gate

- 所有 scale policy 使用同一訓練 budget 與 precision policy。
- 報告 accuracy、packed bytes、metadata bytes、requant multiplier count。
- 選出 `accuracy upper bound` 與 `deployment candidate` 兩個 checkpoint。

## Phase 5：YOLO26 layer/stage sensitivity 與 mixed precision

### 任務

1. one-layer-at-a-time 或 group-wise perturbation。
2. 分析 backbone/neck/O2O head/final conv 的 sensitivity。
3. 比較 uniform W4、all-SD4、evidence-based mixed policy。
4. 將使用者上傳論文的 INT6/SD4/ternary policy作 prior-style control，但重新適配 YOLO26。

### Gate

- policy 完全由 manifest/config 定義。
- 不得人工挑結果後才補解釋。
- 最終 mixed policy 至少在一個成本軸優於 uniform baseline。

## Phase 6：Neck FREQ shared-base relative exponent

### 任務

1. 從 graph manifest 建立 neck fusion groups。
2. shared base scale + reference-fixed relative offsets。
3. bit-exact shift-round-saturate。
4. 比較 independent scale、shared scale、soft log2 alignment、TQT、FREQ。
5. 記錄 offset、span、clipping、saturation、arbitrary requant count。
6. 可選最小 GABFusion-style learnable weighting control；必須標示 reimplementation。

### Gate

- offset 不無意義地全部卡邊界。
- shift simulator 與 fake quant 對齊。
- FREQ 在 accuracy 或硬體 proxy 中有可重複優勢；否則保留為 negative result。

## Phase 7：Optional YOLO26 O2M→O2O assignment-aware distillation

### 任務

1. 取得實際 O2M/O2O assignment metadata。
2. no-KD parity；O2M high-precision no-KD control。
3. generic logit/feature KD。
4. assignment-aware class KD、box KD。
5. 和外部 FP teacher、GHOST-style same-location/self-teaching control比較。
6. 確認 export 後 O2M 全部移除。

### Gate

- teacher 全部 detach，empty-positive/unmatched 不產生 NaN。
- inference-only graph/bytes 不包含 O2M。
- 若 assignment-aware 不優於 generic KD，降級為 training trick。

## Phase 8：Final combination、multi-seed 與部署驗證

### 任務

1. 選 Phase 3–7 最佳 deployment candidate。
2. 跑 LS-SD4、FREQ、BD、FREQ+BD 四組公平消融。
3. 關鍵組至少 3 seeds，報 mean ± std。
4. 產生 mAP vs packed bytes/BOPs/latency/resource Pareto curve。
5. 實際目標硬體測試；沒有硬體時完成 integer simulator、synthesis-ready manifest與限制說明。
6. 撰寫論文級報告、negative results、prior-art risk。

## 17. 實驗矩陣

### 17.1 必做主矩陣

| ID | Weight | Activation | Scale/Fusion | KD | 目的 |
|---|---|---|---|---|---|
| E0 | FP32/FP16 | FP32/FP16 | 原始 | 無 | FP baseline |
| E1 | INT8 PTQ | INT8 PTQ | backend default | 無 | 工業 PTQ baseline |
| E2 | LSQ W8 | LSQ+ A8 | independent | 無 | QAT sanity |
| E3 | LSQ W4 | LSQ+ A8 | independent | 無 | uniform low-bit baseline |
| E4 | fixed SD4, max scale | LSQ+ A8 | layer scale fixed | 無 | 最簡 SD4 baseline |
| E5 | fixed SD4, MSE scale | LSQ+ A8 | layer scale fixed | 無 | 好的 fixed-scale baseline |
| E6 | **LS-SD4 layer-wise** | LSQ+ A8 | free learnable scale | 無 | 核心工程候選 |
| E7 | LS-SD4 per-channel | LSQ+ A8 | free scale vector | 無 | accuracy upper bound |
| E8 | LS-SD4 grouped | LSQ+ A8 | grouped scale | 無 | compromise |
| E9 | LS-SD4 PoT scale | LSQ+ A8 | `s=2^k` | 無 | shift-only outer scale |
| E10 | TQT/uniform W4 | LSQ+/TQT A8 | PoT scale | 無 | PoT uniform control |
| E11 | APoT W4 optional | LSQ+ A8 | additive PoT | 無 | denser shift-add control |
| E12 | evidence-based mixed | LSQ+ A8 | stage-wise | 無 | deployable mixed policy |

### 17.2 Fusion 矩陣

| ID | Base weight policy | Fusion boundary | Scale relation | 目的 |
|---|---|---|---|---|
| F0 | E6/E12 | LSQ+ independent | arbitrary per-branch | normal baseline |
| F1 | E6/E12 | symmetric | all branches one shared scale | hard shared control |
| F2 | E6/E12 | LSQ+/symmetric | soft log2 alignment regularizer | soft control |
| F3 | E6/E12 | TQT | independent PoT scales | PoT control |
| F4 | E6/E12 | symmetric | **shared base + relative integer exponent** | FREQ |
| F5 | E6/E12 | FREQ | no offset/span regularization | regularizer ablation |
| F6 | E6/E12 | FREQ | no overflow penalty | saturation ablation |

### 17.3 Distillation 矩陣

| ID | Student | Teacher | Alignment | 目的 |
|---|---|---|---|---|
| D0 | E6/E12 | none | none | no-KD baseline |
| D1 | E6/E12 | O2M FP16 | none | high-precision O2M only |
| D2 | E6/E12 | external FP model | generic feature/logit | standard KD |
| D3 | E6/E12 | O2M FP16 | naive same-index | negative/simple control |
| D4 | E6/E12 | O2M FP16 | assignment-aware cls | proposed component |
| D5 | E6/E12 | O2M FP16 | assignment-aware cls+box | full BD |
| D6 | E6/E12 | same-location/self-teaching | GHOST-style control | prior-art control |

### 17.4 Final combination

| ID | Weight/Activation | Fusion | KD | 目的 |
|---|---|---|---|---|
| C0 | best LS-SD4/LSQ+ | independent | none | final base |
| C1 | same | FREQ | none | fusion contribution |
| C2 | same | independent | BD | KD contribution |
| C3 | same | FREQ | BD | full candidate |

Screening 可用 1 seed；E0、E3、E5、E6、E7、E12、F4、D5、C0–C3 的最終結果至少 3 seeds。所有比較需固定資料 split、augmentation、epoch、optimizer budget、first/last policy與 validation path。

### 17.5 最關鍵的研究比較

```text
E5 vs E6:
  learnable scale 是否真的優於好的 fixed/MSE scale？

E6 vs E7:
  per-channel 自由度能增加多少 accuracy？

E7 vs E8/F4:
  能否用 grouped/shared+shift relation 接近 upper bound？

F0 vs F4:
  是否減少 requant multiplier/clipping，而非只多參數？

D2/D6 vs D5:
  YOLO26 assignment-aware O2M→O2O 是否有特定價值？
```

## 18. Training schedule 建議

### 18.1 Calibration

- 使用 representative validation/training subset，關閉 mosaic、mixup 與強 augmentation。
- 起始值：64 batches 或 256–1024 images，依 GPU memory 調整。
- 同時保存 min/max、percentile、histogram、negative ratio、clipping rate。
- LSQ+ 使用 MSE initialization；uniform LSQ 使用 paper initialization作對照；SD4/LS-SD4 使用 max、percentile、MSE-grid initialization 消融。

### 18.2 QAT stage

建議：

```text
Epoch 0: observer + fake quant warmup
Epoch 1: freeze BN running stats
Epoch 5: freeze observer statistics; scale/offset 仍可學習
Final 5–10 epochs: 降低 base LR，固定 bit policy
```

實際 epoch 依 baseline recipe 調整，建議先 30–50 epochs screening，最終方法 50–100 epochs。

### 18.3 Optimizer parameter groups

至少分成：

1. 原模型 weights。
2. quantizer scale/offset/alpha。
3. LS-SD4 range scale parameters。
4. FREQ exponent logits/parameters。
5. optional KD adapters。

規則：

- quantizer parameters 不使用 weight decay。
- qparam LR 初始可用 base LR 的 0.1，並做 `{0.01, 0.1, 1.0}` 小型 sweep。
- LS-SD4 range scale 初始用 base LR 的 0.1；做小型 sweep。
- exponent offset 先用比 range scale 更低的 LR。
- 若使用 MuSGD 或 Ultralytics 特殊 optimizer，確認自訂 scalar parameter 是否被正確分類；不確定時對 qparams 使用獨立 AdamW/SGD group。

### 18.4 Progressive quantization

可用於穩定：

```text
W8A8 -> W4A8 -> W4A4
uniform W4A8 -> fixed SD4W/A8 -> LS-SD4W/A8
```

但 progressive schedule 只算訓練技巧。報告時要和 direct QAT 比較，以免把額外訓練時間忽略。

---

## 19. Diagnostics

每個 quantized layer 至少記錄：

```text
scale / zero point / clipping bounds
weight or activation bit-width
quantizer type
min/max/mean/std
negative ratio
clipping ratio
zero ratio
SQNR or normalized MSE
gradient norm of qparams
codebook occupancy
SD4 exponent histogram
SD4 hard-code histogram and duplicate-zero policy
LS-SD4 scale initialization/final value/trajectory
scale granularity and metadata bytes
FREQ branch offsets
shift distance
post-shift saturation rate
```

偵測任務層級記錄：

```text
mAP50-95
mAP50
precision / recall
per-class AP
AP_small / AP_medium / AP_large, if dataset supports
box loss / cls loss / auxiliary losses
O2M and O2O metrics separately where possible
confidence calibration
```

部署層級記錄：

```text
raw parameter count
FP checkpoint bytes
actual packed weight bytes
activation peak bytes
BOPs
integer MAC count
shift-add count
arbitrary requant multiplier count
memory traffic estimate
batch-1 latency
throughput
FPGA LUT / FF / BRAM / DSP / frequency / power, if available
```

---

## 20. 必要測試

### 20.1 通用數學與 gradient

- `round_pass` forward 等於 round，backward 等於 identity。
- `grad_scale` forward 不變，backward 乘指定係數。
- positive scale parameterization 永遠大於 `eps`。
- LSQ+ asymmetric endpoint、offset gradient正確。
- LSQ+ bias-fold公式與 direct dequant convolution parity。
- integer-zero-point export模式需量測 beta rounding gap。
- PACT 在 `<0`、範圍內、`>alpha` 的 forward/backward 正確。
- TQT/PoT scale 永遠是 `2^integer`。
- AMP、CPU/GPU dtype/device、DDP scalar parameter broadcast smoke test。

### 20.2 SD4 / LS-SD4

- normalized codebook 與 hardware codebook 的 encode/decode 對應正確。
- 16 個 bit codes、15 個 distinct values、canonical zero policy 明確。
- hard nearest-code 輸出全部位於 codebook。
- boundary midpoint 的 tie-breaking 固定且有測試。
- `forward hard code == export encode code`。
- `w_hat=s_w*c_norm=(s_w/64)*c_hw` 在所有 codes上數值一致。
- pack/unpack roundtrip bit-exact。
- layer-wise、per-channel、grouped scale broadcasting 正確。
- max/percentile/MSE-grid/alternating-MSE initialization可重現，且 alternating objective不增。
- `s_w` backward 非零且方向 sanity 正確。
- no grad scale／`1/sqrt(N)`／`1/sqrt(N*K_pos)` 行為可切換。
- scale checkpoint roundtrip、resume optimizer state 一致。
- disabled quantizer output 與原 weight parity。

### 20.3 FREQ

- reference branch offset 固定 0。
- 其他 offset 在指定整數範圍。
- `s_g * 2^k * q` 與 shift-aligned integer representation 一致。
- arithmetic right-shift rounding mode bit-exact。
- left-shift overflow、saturation、zero preservation 可計數。
- checkpoint reload 後 group mapping/base scale/offset 一致。

### 20.4 Model integration

- quantizer disabled 時 model output parity。
- train/eval mode 與 YOLO26 dual-head輸出結構不被破壞。
- loss backward 完成且 critical parameters 有 gradient。
- O2M/O2O precision policy正確。
- fuse/export 後 one2many 被移除。
- FP、W8A8、fixed SD4、LS-SD4 checkpoint均可 resume。

### 20.5 Export/integer

- uniform Q/DQ graph通過 ONNX checker。
- fake quant vs ONNX output parity；LSQ+ bias-fold與integer-zero-point兩條路分開驗證。
- SD4 binary + manifest 可獨立 decode。
- scale metadata shape、dtype、endianness、version正確。
- FREQ manifest完整描述 group/base scale/offset/rounding/bit-width。
- integer simulator vs PyTorch fake quant parity。
- final detection decode/postprocess parity 在指定 tolerance內。

## 21. Failure conditions 與停止規則

遇到以下情況先停止 full training，回到最小重現：

- FP baseline 無法重現。
- quantizer disabled 仍改變 model output。
- scale 變成 NaN/Inf/0。
- W8A8 即大幅掉點。
- export 與 fake quant 差異超過 tolerance。
- SD4 codebook collapse 到單一 code 或極端 zero ratio。
- LS-SD4 scale 相對初始化暴增/趨近 eps，或 qparam gradient長期為 0/Inf。
- FREQ offset 全部卡在 `offset_min/max`。
- post-shift saturation 持續高於預設門檻，例如 1%，且無下降趨勢。
- KD loss 遠大於原始 loss 或 unmatched assignment 導致不穩定。
- 報告的速度只來自 PyTorch fake quant，而非真正 integer/custom kernel。

---

## 22. 結果判定門檻

### Tier S：可作強研究與部署成果

- 相對 FP：mAP50-95 drop ≤ 1.0 absolute，或在相同硬體成本下顯著優於所有公平 quantization baselines。
- 實際 packed weight compression ≥ 4×，且 scale metadata 已計入。
- 有實際 kernel/FPGA/ASIC 測得 latency、DSP/LUT/energy改善。
- LS-SD4、FREQ、BD 的增益在 3 seeds 下穩定，完整消融成立。
- fake quant、integer simulator、export/hardware path bit-exact 或誤差可解釋。

### Tier A：可信研究候選

- LS-SD4 明確優於 fixed/MSE SD4，且收益可重複。
- mAP50-95 drop ≤ 2.0 absolute。
- FREQ 降低 arbitrary/per-channel requant multipliers、clipping或 saturation，且 accuracy trade-off合理。
- export/integer simulator/packed format完整。

### Tier B：工程成果

- W8A8/W4A8/SD4 QAT穩定、可重現、可匯出。
- 只有 accuracy 改善但沒有部署驗證，或只有 hardware proxy；可作中間結果，不宣稱完成硬體加速。

### 不足以成為主貢獻

- 只完成 W8A8。
- 只顯示 `.pt` 檔案變小。
- 只把 SD4、learnable scale、LSQ+ 串起來，沒有 fixed/MSE/per-channel/PoT消融。
- 只用 weight reconstruction MSE證明 scale好，沒有 detection task metric。
- per-channel scale提升 accuracy，但未計 scale metadata與requant成本。
- FREQ只是更多參數，沒有減少 arbitrary scale conversion。
- BD沒有超過 generic KD/GHOST-style control。
- 速度數字只來自 PyTorch fake quant。

## 23. 最終產物

Codex 最後應產生：

```text
outputs/quant/manifests/environment.json
outputs/quant/manifests/model_architecture.json
outputs/quant/manifests/quantized_model.json
outputs/quant/reports/fp_baseline.md
outputs/quant/reports/quantization_baselines.md
outputs/quant/reports/ls_sd4_scale_ablation.md
outputs/quant/reports/novelty_ablation.md
outputs/quant/reports/hardware_cost.md
outputs/quant/sensitivity/layer_sensitivity.csv
outputs/quant/experiments/ablation.csv
outputs/quant/experiments/pareto.csv
outputs/quant/exports/best_quantized.pt
outputs/quant/exports/best_uniform_qdq.onnx
outputs/quant/exports/sd4_weights.bin
outputs/quant/exports/sd4_manifest.json
outputs/quant/exports/scale_metadata.bin
outputs/quant/exports/freq_manifest.json
outputs/quant/exports/integer_validation.json
reproduce_quantization.sh
```

`novelty_ablation.md` 必須清楚分成：

```text
Known method reproduction
Engineering adaptation
New hypothesis
Experimental support
Negative results
Remaining prior-art risk
Deployment limitations
```

---

## 24. Codex 第一個實際任務

收到此文件後，**只執行 Phase 0**：

1. 找到 repository root、checkpoint 與 dataset YAML。
2. 建立 `scripts/inspect_yolo26.py`，但先不要改 model class。
3. 產生 environment/model/fusion/head manifest。
4. 重跑 FP validation at imgsz 1280。
5. 測試 model 的 train/eval/fuse/export 行為。
6. 建立 `outputs/quant/reports/fp_baseline.md`。
7. 回報任何與本文件假設不一致的地方，例如：
   - checkpoint 不是官方 YOLO26m。
   - `end2end=False`。
   - `reg_max` 不是 1。
   - 沒有 one2many/one2one。
   - module path 與官方不同。
   - dataset YAML 路徑失效。

Phase 0 完成前，不要新增 LSQ、PACT、SD4 或 FREQ module。

---

## 25. 文獻與 GitHub 查閱清單

Codex 應優先閱讀原論文、官方 repository 與本專案已上傳的 PDF；不要只依賴第三方 blog。

### 25.1 核心量化

1. **Learned Step Size Quantization (LSQ)**, ICLR 2020  
   https://arxiv.org/abs/1902.08153  
   重點：learnable uniform step size、transition-sensitive STE、`1/sqrt(N*Qp)` gradient scaling。使用者已上傳 PDF。

2. **LSQ+: Improving Low-Bit Quantization through Learnable Offsets and Better Initialization**  
   https://arxiv.org/abs/2004.09576  
   重點：Swish/H-Swish/Mish 類 signed、skewed activation；learnable asymmetric offset與 MSE initialization。

3. **PACT: Parameterized Clipping Activation for Quantized Neural Networks**  
   https://arxiv.org/abs/1805.06085  
   重點：learnable upper clipping `alpha`；原始形式為 `[0, alpha]`，只作控制組。使用者已上傳 PDF。

4. **Trained Quantization Thresholds (TQT)**  
   https://arxiv.org/abs/1903.08066  
   重點：PoT threshold/scale、fixed-point deployment。

5. **Additive Powers-of-Two Quantization (APoT)**  
   https://arxiv.org/abs/1909.13144  
   官方實作：https://github.com/yhhhli/APoT_Quantization

### 25.2 SD4 與 shift-based weight

6. **A Multiplier-Less Convolutional Neural Network Inference Accelerator for Intelligent Edge Devices**  
   IEEE document 9551204, DOI 10.1109/JETCAS.2021.3116044  
   https://ieeexplore.ieee.org/document/9551204/  
   重點：4-bit SD4 weight、shift/add datapath、accelerator。SD4 本身不可當新方法。

7. 使用者上傳：**Design and Implementation of a Multi-Precision Deep Learning Accelerator for YOLOX-Based Object Detection in Synthetic Aperture Radar Images**  
   重點：A8、backbone INT6、neck SD4、head ternary；stage-wise QAT；unified SD4 decode；FPGA resource/memory-bound analysis。

8. **DeepShift / shift-based networks**  
   https://arxiv.org/abs/1905.13298  
   用於理解 PoT/shift weight 的先行研究範圍。

### 25.3 Object detector quantization

9. **AQD: Towards Accurate Quantized Object Detection**  
   https://arxiv.org/abs/2007.06919

10. **HQOD: Harmonious Quantization for Object Detection**  
    查閱 ICME 2024 論文與 VP-QOD reference implementation。

11. **Reg-PTQ: Regression-specialized PTQ for Fully Quantized Object Detector**  
    CVPR 2024；用於 box regression quantization baseline與分析。

12. **GHOST: Guided Hybrid Quantization ... via One-to-one Self-teaching**  
    https://arxiv.org/abs/2301.00131  
    https://github.com/icey-zhang/GHOST  
    重要：量化 + one-to-one self-teaching已有先例，BD 必須明確區分。

13. **VP-QOD**  
    https://github.com/Menace-Dragon/VP-QOD  
    支援 detector PTQ/QAT、LSQ、TQT、AQD、HQOD、YOLOX；README明確指出沒有處理硬體 backend alignment。

### 25.4 Fusion 與 scale bridge 重疊風險

14. **GABFusion: Rethinking Feature Fusion for Low-Bit Quantization of Multi-Task Networks**  
    https://arxiv.org/abs/2511.05898  
    重點：gradient-aware balanced fusion與 attention alignment。FREQ 必須聚焦 scale relation/shift alignment，不是 feature weighting。

15. **PTQ4SNN: Membrane-Aware Post-Training Quantization for Spiking Neural Networks**  
    https://arxiv.org/abs/2608.07066  
    重點：channel-wise scale bridge `s_mem,c = s_w,c * 2^k_c`。雖然領域不同，但否定 relative exponent裸公式的首次性。

### 25.5 YOLO26 官方來源

16. **Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models**  
    https://arxiv.org/abs/2606.03748

17. **Ultralytics Detect head**  
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py

18. **Ultralytics Conv/SiLU implementation**  
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py

注意：本地 checkpoint、Ultralytics版本或自訂 YAML 可能與 main不同；以 repository 實際 graph 為準。

### 25.6 實作框架

19. **MQBench**  
    https://github.com/ModelTC/MQBench  
    參考 LSQ/TQT/observer/backend configuration；不要未測試就把整個 YOLO26 強制改成 FX pipeline。

20. **PyTorch quantization / ONNX QDQ official docs**  
    只用於 uniform baseline與 export contract。

### 25.7 使用外部程式的原則

- 先確認 license。
- 保留 attribution與來源 commit。
- 不把舊 PyTorch/TensorFlow code直接貼入現代 Ultralytics。
- 借用 quantizer時仍需重新驗證 STE、shape、checkpoint、AMP、DDP、export。
- 投稿前再做一次 2026-08-27 之後的 systematic search。

## 26. 保守且可辯護的研究論述

在實驗完成前，只能使用以下表述：

> 本研究首先建立 YOLO26m 的 uniform QAT 與 fixed-SD4 基線，並研究 Learnable-Scale SD4，使 SD4 非均勻 power-of-two weight codebook 的實數 range 由 detection task loss調整。進一步地，我們評估在 neck fusion group 中以 shared base scale 與有限整數 relative exponent 取代自由 branch scale，是否能在維持準確率的同時降低 arbitrary requantization與提供 bit-exact shift alignment。作為選用擴充，我們亦研究 YOLO26 training-only one-to-many branch 對量化 one-to-one inference branch 的 assignment-aware distillation。方法的創新性與實用性須由先行研究比對、scale/fusion/KD消融、integer export與硬體測試共同驗證。

### 26.1 最合理的論文貢獻候選

```text
Contribution 1:
  一套適用於 YOLO26 的 bit-exact SD4 QAT/export framework，
  含 normalized codebook、learnable range scale、packing與 integer simulator。

Contribution 2:
  YOLO26 stage/fusion sensitivity 的系統性研究，
  以及從 per-channel free scale 收斂到 shared-base relative-exponent 的 accuracy-cost frontier。

Contribution 3 (conditional):
  neck fusion-specific scale coupling 在 mAP、clipping、requant multiplier與硬體成本上有穩定優勢。

Contribution 4 (optional/conditional):
  YOLO26 O2M→O2O assignment-aware KD 超過 generic/GHOST-style controls，
  且 export 無額外 inference branch。
```

### 26.2 可能的 negative result 也要保存

- learnable `s_w` 不優於好的 MSE fixed scale。
- per-channel scale收益不足以抵消 metadata/requant成本。
- PoT scale造成明顯 accuracy loss。
- FREQ只有成本下降但 accuracy變差，或反之。
- assignment-aware KD不優於 generic KD。

這些結果可縮小設計空間，不能刪除或只報最佳組。

## 27. 最終研究與 Codex 執行優先順序

```text
1. Phase 0：FP baseline、實際 graph、fusion/head manifest
2. Phase 1：LSQ、LSQ+、SD4 encoder/decoder、LS-SD4 unit tests
3. Phase 2：W8A8、W4A8 uniform QAT
4. Phase 3：fixed SD4 vs layer-wise LS-SD4
5. Phase 4：per-channel/grouped/PoT scale ablation
6. Phase 5：stage sensitivity與 mixed precision
7. Phase 6：neck FREQ
8. Phase 7：optional O2M→O2O KD
9. Phase 8：final combination、3 seeds、export/hardware
10. Pose或 detect+pose延伸
```

若時間有限，保留的最低完整研究主線為：

```text
FP
+ W4A8 LSQ/LSQ+ baseline
+ fixed MSE-SD4W/A8
+ layer-wise LS-SD4W/A8
+ per-channel upper bound
+ one個 hardware-constrained scale variant
+ bit-exact packing/integer validation
```

只有上述主線成立，再把 FREQ 或 BD加入主論文。

---

## 28. 可直接貼給 Codex 的啟動指令

```text
請先完整閱讀 CODEX_YOLO26M_LS_SD4_LSQPLUS_MASTER_SPEC.md。

你負責在現有 YOLO26m Detect repository 中建立可重現、可匯出、可消融的量化研究框架。本文件是研究工程執行契約；不得自行省略 gate，也不得一次實作所有 phase。

本輪只執行 Phase 0。禁止先新增 LSQ、LSQ+、PACT、TQT、SD4、LS-SD4、FREQ 或 distillation module。

Phase 0 任務：
1. 確認 repository root、git status與 Python環境。
2. 尋找並驗證：
   - runs/detect/optim_1280/weights/best.pt
   - Finale_version.v6i.yolov11/data.yaml
3. 記錄 Python、PyTorch、CUDA、Ultralytics、ONNX、GPU、git commit。
4. 載入實際 checkpoint；不得假設和官方 main完全相同。
5. 建立 scripts/inspect_yolo26.py，輸出：
   - module path/type/parameter count
   - Detect head class
   - end2end與reg_max
   - one2many與one2one paths
   - neck Concat/Add fusion groups與tensor shapes
   - attention、softmax、decode、postprocess特殊運算
6. 在 imgsz=1280 重跑 FP validation，保存 mAP50-95、mAP50、precision、recall、per-class AP。
7. 保存固定 validation inputs與原始 FP outputs，供後續 parity test。
8. 分別測試 training forward、evaluation forward、end-to-end inference、fuse、export。
9. 確認 fuse/export後 one-to-many head是否移除。
10. 產生：
   outputs/quant/manifests/environment.json
   outputs/quant/manifests/model_architecture.json
   outputs/quant/manifests/fusion_groups.json
   outputs/quant/reports/fp_baseline.md
   outputs/quant/reference/fp_inputs.pt
   outputs/quant/reference/fp_outputs.pt

不得直接修改 site-packages。Forward hook本階段只可用於shape/statistics tracing，不能作後續量化執行機制。

若 checkpoint、dataset path、head結構或官方假設不一致，不要靜默修正；把實際結果與差異寫入 manifest/report。

完成後固定回報：
Phase:
Changed files:
Commands executed:
Tests:
Measured results:
Assumptions:
Known failures or risks:
Recommended next phase:

Phase 0 gate通過前，不得進入 Phase 1。
```
