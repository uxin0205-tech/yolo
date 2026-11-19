> [!IMPORTANT]
> 這是早期以 YOLO11 為背景整理的歷史研究筆記，只用於追溯 BinaryQK、No Attention 與硬體友善化思路。目前正式模型已改為 **YOLO26m**；可執行的研究流程、實驗代號、公式與判定條件一律以 [plan.md](plan.md) 為準。本文中的 YOLO11 結構與舊實驗數字不得直接當作現行 implementation spec。

# YOLO11 C2PSA 硬體友善 Binary Global Attention 研究設計報告

## Executive summary 與交付檔案

完整可交付 Markdown 報告已產生，內容包含 A0–A5 研究矩陣、Matched Dual-Basis 定義、七類 Softmax/normalization 方案、QAT/STE/distillation、PTQ calibration、FPGA dataflow、XNOR-popcount core、LUT sizing、U4/PoT \(P\)、INT32 PV、硬體風險、Codex 任務清單、命令介面、十二週 Gantt 與資源預估。

**[下載完整研究設計報告 Markdown](sandbox:/mnt/data/yolo11_c2psa_hardware_friendly_binary_attention_research_plan_zhTW.md)**

本研究的核心結論是：**不建議直接照 *No Attention, No Problem* 把 YOLO11 的 \(QK^T\) 換成 \(Q\odot K\)，也不建議只把 BinaryAttention 的 Binary QK 接到原始 FP Softmax 後就停止硬體化。** BinaryAttention 已證明可只保留 Q/K 的 sign，以 bitwise XNOR/popcount 取代 QK 浮點 dot product，並透過 scale、learnable bias、QAT/self-distillation，以及 U8 \(P\)+S8 \(V\) 降低精度損失。citeturn16view0turn14view1 *No Attention* 則從另一個角度指出，YOLO11 C2PSA 的 reshape、transpose、matrix multiplication 與 Softmax 對 DPUCZDX8G 並不原生友善，因此把 attention 改寫成 1×1 Conv、scaled \(q\odot k\)、HardSigmoid 和 gated fusion。citeturn15view1 使用者提供的論文原文第 4 頁 Figure 3 亦直接畫出了這個替換。fileciteturn0file0

因此本研究把問題重新定義成：

\[
\boxed{
\text{保留 global }Q_i\leftrightarrow K_j
+
\text{Binary QK}
+
\text{integer normalization}
+
\text{low-bit PV}
}
\]

而不是：

\[
\boxed{
QK^T\rightarrow Q\odot K
}
\]

最終建議主方案為：

\[
\boxed{
\text{Matched Dual-Basis Binary QK}
+
\text{Quantized Relative Bias}
+
\text{Score-Indexed LUT Softmax}
+
\text{Reciprocal Normalization}
+
U4/U8\ P
+
S8\ V
}
\]

其中 **Matched Dual-Basis 是本研究新定義的延伸，不是 BinaryAttention 原論文已有方法**；此次查核 BinaryAttention 論文與官方 repository 時，原方法的明確核心為 scaled 1-bit Q/K、bias、8-bit \(P/V\)、QAT 與 self-distillation。citeturn16view0turn14view1

## 研究定位與 attention layer 邊界

Ultralytics 官方架構文件目前將 YOLO11 的 C2PSA 放在 SPPF 之後、FPN 之前；C2PSA 是 CSP 結構，其 active branch 使用 PSABlock attention。官方文件也明確區分 YOLO11 的 C2PSA 與後面的 Detect/DFL，因此本研究可以只修改 attention，而讓 DFL、NMS、Neck 等保持不變。citeturn13search2turn18view3 正式實驗應固定 Ultralytics commit SHA，因官方 `main` repository 是持續更新的 canonical implementation。citeturn13search1turn13search4

本報告把 attention layer 拆成以下完整資料路徑：

\[
X
\rightarrow
Q/K/V\ Projection
\rightarrow
Binaryization
\rightarrow
Scale
\rightarrow
Relative\ Bias
\rightarrow
Score
\rightarrow
Normalization
\rightarrow
P\ Quantization
\rightarrow
PV
\rightarrow
Output\ Projection.
\]

其中硬體政策為：

| Attention 子模組 | 研究預設 |
|---|---|
| Q/K/V projection | W8A8 1×1 projection，S32 accumulation 後 requantize S8 |
| Q/K binaryization | 1-bit sign + bit packing |
| Scale | dynamic、fixed per-head、PoT per-head 三組消融 |
| Bias | dense 2D、separable 2D、3-bit bias-level |
| Score | XNOR + popcount；dual-basis 用兩路 popcount + shift-add |
| Normalization | Normalized HardSigmoid 或 LUT/reciprocal |
| \(P\) | 先 U8，再 U4，最後 PoT |
| \(V\) | channel-wise S8 |
| PV | U8/U4×S8→S32；PoT \(P\) 則 shift+accumulate |
| Output projection | W8A8 1×1 projection |
| BN | inference 前 fold；無 BN 的 projection 不人為加入 |
| Activation | Q/K/V/output 保持 Identity；gating 才使用 HardSigmoid/ReLU/clip |

*No Attention* 的 PTQ deployment 也採用 Conv-BN fusion、calibration scale/zero-point，再轉 INT8 並編成 `.xmodel`；其目標平台是 ZCU104 + DPUCZDX8G。citeturn16view1

## 主研究方法與 A0–A5

完整報告定義六組主實驗，這樣可以把「Binary QK、dual basis、Softmax 移除、global relation 是否保留、\(P\) 低位元化」各自造成的效果拆開，而不會一次改五件事情後無法解釋結果。

| ID | 方法 | Q/K | Relation | Normalization | \(P/V\) | 研究用途 |
|---|---|---:|---|---|---|---|
| **A0** | Original YOLO11 | FP | Global \(QK^T\) | FP Softmax | FP/FP | 精度與 attention fidelity 上界 |
| **A1** | No-Attention attention-only | INT8 | Local \(Q\odot K\) | HardSigmoid gate | gate/S8 | Stock-DPU baseline |
| **A2** | BinaryAttention-aligned | 1b | Global | Reference Softmax | U8/S8 | 隔離 Binary QK 效果 |
| **A3** | Matched Dual-Basis BQK | 1b×2 | Global | Reference Softmax | U8/S8 | 隔離 dual-basis 效果 |
| **A4** | MDB + Normalized HSigmoid | 1b×2 | Global | HSigmoid + row normalize | U8→U4/S8 | 去 exp 的折衷方案 |
| **A5** | MDB + Indexed-LUT Softmax | 1b×2 | Global | LUT + histogram + reciprocal | U8/U4/PoT + S8 | **主 FPGA proposal** |

BinaryAttention 的 binary score 可以由 Hamming similarity 表示；對 \(d_k\) 維的 \(\{-1,+1\}\) Q/K，

\[
z_{ij}
=
2\operatorname{Popcount}
(
\operatorname{XNOR}(B_{Q,i},B_{K,j})
)
-d_k.
\]

原論文進一步用 scale 與 learnable bias：

\[
S_{ij}
=
\frac{\mu_q\mu_k}{\sqrt{d_k}}z_{ij}+b_{ij},
\]

並將 Softmax 後的 coefficient 量成 U8、\(V\) 做 channel-wise S8。citeturn16view0

本研究提出的 **Matched Dual-Basis** 則定義為：

\[
R_h^{(0)}=I,
\qquad
R_h^{(1)}=\mathcal B_h,
\]

\[
B_{Q,h}^{(r)}
=
sign(Q_hR_h^{(r)}),
\qquad
B_{K,h}^{(r)}
=
sign(K_hR_h^{(r)}).
\]

同一個 basis \(r\) 對 Q、K 使用相同或 tied transform，因此稱為 matched。score 為：

\[
S_{ij}
=
\alpha_{h,0}z_{ij}^{(0)}
+
\alpha_{h,1}z_{ij}^{(1)}
+
b_{\Delta y,\Delta x}^{(h)}.
\]

硬體首選的第二 basis 不是 arbitrary dense FP matrix，而是 Walsh-Hadamard/butterfly 類 add-sub transform；初始 basis-0 為 identity，basis-1 則從多個固定 Hadamard-mask candidates 中，以 teacher score correlation/top-k fidelity 挑選每 head 最佳候選。第二階段才測 learnable sparse butterfly，並將係數限制成 \(0,\pm2^k\)。這一部分是報告中明確標示的**研究提案**，不歸因於 BinaryAttention。

A4 則利用 *No Attention* 的 HardSigmoid 思想，但不像 A1 只做 independent gate，而是補上：

\[
g_{ij}=HardSigmoid(S_{ij}),
\]

\[
P_{ij}
=
\frac{g_{ij}}
{\sum_k g_{ik}+\epsilon}.
\]

如此仍保留：

\[
\sum_jP_{ij}\approx1
\]

和 global \(i\leftrightarrow j\) aggregation。這可以直接回答一個很重要的研究問題：

> *No Attention* 的精度成本主要來自 Softmax→HardSigmoid，還是來自更劇烈的 \(QK^T\rightarrow Q\odot K\)？

A1、A3、A4 的對照可以把兩者分開。

## Softmax 硬體化與 A5 核心設計

A5 的關鍵不是設計一顆「通用 FP exponential accelerator」，而是利用 Binary QK 的**離散性**。當 \(d_k=32\) 時，單基底 popcount 只有 33 種可能值；\(d_k=64\) 時只有 65 種。因此 fixed scale 的單基底甚至可以直接：

\[
Popcount
\rightarrow
LUT_{\exp}
\]

而不是：

\[
Popcount
\rightarrow FP\ score
\rightarrow FP\ exp.
\]

16-bit entry 的單基底 ROM 只需要：

\[
33\times16=528\ bits
\]

或：

\[
65\times16=1040\ bits.
\]

即使加 8-level relative-bias index，也只有 528 B/head（\(d_k=32\)）或 1040 B/head（\(d_k=64\)）。dual-basis 不應直接建 \((d+1)^2\) 的巨大二維 LUT；報告因此提出「兩路 score 先用 PoT scale shift-add，再合成 32/64/128-level scalar score bin，最後查單一 exp LUT」。

主 A5：

\[
s_{ij}
=
(z^{(0)}_{ij}\ll k_0)
+
(z^{(1)}_{ij}\ll k_1)
+
b^q_{ij},
\]

\[
u_{ij}=Q_{\rm bin}(s_{ij}),
\]

\[
E_{ij}
=
LUT_{\exp}[u_{ij}-u_{i,\max}],
\]

\[
Z_i=\sum_jE_{ij},
\qquad
R_i\approx LUT_{\rm reciprocal}(Z_i),
\]

\[
P^q_{ij}
=
Q_{U8/U4}(E_{ij}R_i).
\]

報告特別提出一個適合 binary attention 的 **row-histogram normalizer**：第一遍不存整個 attention row，而是累積 64/128 個 score bins 的 histogram，直接由

\[
Z_i
=
\sum_k
hist_i[k]LUT_{\exp}[k-m_i]
\]

得到 denominator。第二遍重新算一次便宜的 XNOR-popcount，直接產生 \(P\) 並串流進 PV。如此用廉價的 binary recomputation 換掉 \(N\times N\) attention matrix 的 DDR write/read。

這個方向也與現有 integer-only Transformer 研究相符：I-ViT 的 Shiftmax 以 integer arithmetic/bit shifting 處理非線性 Softmax；FQ-ViT 的 Log-Int-Softmax則進一步研究 4-bit attention map；兩者均提供「不必回 FP Softmax」的實證先例。citeturn12academia2turn12academia1 另外，2026 年 FPGA-specific 工作以 16-segment PWL natural exponential 做 BRAM-free approximate Softmax，在 Zynq-7020 上報告完整 attention-row core 為 1444 LUT、77 DSP、0 BRAM；這提供 A5 的 PWL 對照組，但不能直接把該資源數字外推到 ZCU104/MDB architecture。citeturn16view2

報告比較的 normalization / \(P\) 方案共七類：

| 方法 | RAM | DSP | 延遲 | 精度風險 | 建議 |
|---|---|---|---|---|---|
| Normalized HardSigmoid | 極低 | 0–低 | 低 | 中 | A4 |
| Popcount-indexed LUT | 極低 | 低 | 低 | 低–中 | **A5 首選** |
| Integer LUT + Reciprocal | 低 | 低–中 | 中 | 低 | A5 reference |
| PWL exp | 可 0 BRAM | 中–高 | 規則 | 低–中 | FPGA 對照 |
| Shift/PoT Softmax | 極低 | 0–低 | 極低 | 中 | aggressive |
| ReLU-normalize | 極低 | 0–低 | 極低 | 中–高 | 最低成本 ablation |
| PoT \(P\) | 極低 | PV 可近 0 | 極低 | 中–高 | A5 最後階段 |

PoT \(P\) 不被報告錯誤地稱為 Softmax replacement，而被定義為 normalization **之後**的 coefficient encoding：

\[
P_q\in\{0,2^{-1},2^{-2},\ldots\}.
\]

如此：

\[
P_qV
=
V\gg k
\]

可以把 PV multiplier 改成 shift+accumulate；但由於 row sum 可能偏離 1，報告要求額外量測 row-sum error 並測 residual correction。

## 訓練、量化、實驗與硬體資源規劃

BinaryAttention 官方工作採用 QAT/self-distillation，而官方 repo 的 quick-start 也明確提供 `--attn-quant`、`--attn-bias`、`--pv-quant` 以及 optional teacher distillation 介面，因此本研究的 A2–A5 以 QAT 為主、PTQ 只負責 deployment calibration 較合理。citeturn14view1turn16view0 A1 則保留 *No Attention* 的 PTQ deployment philosophy，作 stock-DPU 對照；該論文採 Conv-BN fuse、calibration、INT8、XIR/.xmodel compilation。citeturn16view1

總 loss 建議：

\[
\mathcal L
=
\mathcal L_{det}
+
\lambda_{KL}\mathcal L_{attn-KL}
+
\lambda_o\mathcal L_{out}
+
\lambda_s\mathcal L_{score}
+
\lambda_d\mathcal L_{basis-div}
+
\lambda_p\mathcal L_{PoT}.
\]

Teacher 是 frozen A0；A2–A5 全部從同一 A0 checkpoint 分岔。除了 mAP，還必須量：

\[
D_{KL}(P_{A0}\Vert P_v),
\]

top-k overlap（k=1,4,8,16）、attention entropy、score Spearman correlation、row-sum error、output cosine、binary tie-rate、dual-basis correlation，以及 small-object AP\(_S\)。這是為了避免「mAP 看似沒掉，但 attention 已經 collapse 成近 uniform distribution」而無法觀察。

建議正式統計規模是：pilot 各 variant 先 1 seed；finalists 再跑 3 seeds。Attention fidelity 至少 2,000 images；PTQ/calibration 做 256/512/1024/2048 sensitivity，主結果採 1024–2048；FPGA benchmark 使用 200 warm-up + 2,000 measured frames，重複 5 runs並報 median/P95/P99。這些是**本研究的實驗規劃值**，不是既有論文規定。

硬體上，不應只報 DPU FPS。*No Attention* 已展示 DPU architecture 從 B512 擴到 B4096 時 DPU FPS會提升，但 E2E throughput 仍受到其他系統部分限制，因此其研究特別同時報 DPU latency、E2E latency、power 與資源。citeturn18view1 其 ZCU104 resource baseline 為：

| DPU | LUT | BRAM | URAM | DSP | Power |
|---|---:|---:|---:|---:|---:|
| B2304 | 45.1k | 104 | 38 | 438 | 7.165 W |
| B3136 | 50k | 123 | 42 | 566 | 7.864 W |
| B4096 | 55.8k | 142 | 46 | 710 | 8.760 W |

這些是該論文 Vivado 報告的 DPU 基準，不是本研究 custom core 的預測。citeturn18view1 完整報告因此建議先以 B2304/B3136 做 hybrid DPU+custom-PL，讓 routing、BRAM、DSP 保有較多 headroom，再以 B4096 作 throughput upper bound。

本研究 custom core 的**前期 engineering budget** 定為約 +10k–25k LUT、+4–24 BRAM36、0–8 URAM；U8/U4 PV 目標 +16–64 DSP，PoT-PV 則目標壓到 +0–16 DSP。這些不是已驗證 resource claim，而是 HLS design-space sweep 的 acceptance window，最終必須由 post-route Vivado report 取代。

時程規劃為十二週：版本/基線 → A1 → A2 → A3 MDB → A4 → A5 → QAT/U4/PoT → HLS → DPU/PL integration → hardware benchmark → Pareto/report。GPU 計畫預留約 200–400 A100/H100-equivalent GPU-hours，但完整報告要求先以 A0 一個 epoch 的實測 throughput重新估算，而不把這個預算視為固定訓練時間。

## 來源、官方程式碼與重現要求

**BinaryAttention — CVPR 2026 / arXiv / 官方 GitHub。** CVPR 官方頁面列出 Chaodong Xiao、Zhengqiang Zhang、Lei Zhang，並提供官方 code repository；arXiv formulation 清楚定義 scaled binary Q/K、bias enhancement 與 U8 \(P\)/S8 \(V\)。citeturn13search6turn16view0

- Paper: `https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_BinaryAttention_One-Bit_QK-Attention_for_Vision_and_Diffusion_Transformers_CVPR_2026_paper.html`
- arXiv: `https://arxiv.org/abs/2603.09582`
- Official GitHub: `https://github.com/EdwardChasel/BinaryAttention`

**No Attention, No Problem。** 原論文定義 YOLO11 DPU-aware attention approximation，並使用 ZCU104、DPUCZDX8G 與 Vitis/Vivado 2022.2 flow；其 attention 修改就是 Q/K/V 1×1 Conv、\(q\odot k\)、HardSigmoid 與 gated fusion。citeturn15view1turn16view1 本次 exact-title/arXiv-ID 檢索未找到可驗證的作者官方 GitHub，因此完整 Markdown 報告明確標註為 **「無官方 repo」**，而不是把第三方重現冒充官方實作。使用者提供的論文 PDF亦作為主要來源。fileciteturn0file0

- Paper: `https://arxiv.org/abs/2607.13106`
- HTML: `https://arxiv.org/html/2607.13106`
- Official code: **截至 2026-08-09 查無可驗證官方 repo**
- Upstream Ultralytics: `https://github.com/ultralytics/ultralytics`
- Upstream Vitis-AI: `https://github.com/Xilinx/Vitis-AI`

Vitis-AI 官方 repository 將其定位為 AMD/Xilinx adaptable platform 的 AI inference development stack；目前 release notes 顯示 Vitis-AI 5.0 對 Zynq UltraScale+ DPUCZDX8G 沒有新的 DPU IP/reference-design 更新，因此要重現 *No Attention* 時，應固定與其 Vivado/Vitis 2022.2 deployment flow相容的環境，而不是任意用最新版替代。citeturn13search0turn18view2

補充的 Softmax/quantization primary references：

- I-ViT / Shiftmax: `https://arxiv.org/abs/2207.01405`，code `https://github.com/zkkli/I-ViT`。citeturn12academia2
- FQ-ViT / Log-Int-Softmax: `https://arxiv.org/abs/2111.13824`，code `https://github.com/megvii-research/FQ-ViT`。citeturn12academia1
- FPGA 16-segment PWL Softmax: `https://arxiv.org/abs/2607.01798`。citeturn16view2
- I-BERT integer-only nonlinear precedent: `https://arxiv.org/abs/2101.01321`。citeturn12academia3

**完整 42k+ 字元、可直接交給 Codex 拆實作任務的 Markdown 版本： [下載研究設計報告](sandbox:/mnt/data/yolo11_c2psa_hardware_friendly_binary_attention_research_plan_zhTW.md)**
