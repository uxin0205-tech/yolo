# Natural-exponential PWL FPGA attention weighting：原論文規格核對

日期：2026-08-16

## 結論先行

論文 *Approximate Attention Weighting for Sustainable FPGA-Based Vision Transformer Inference* 確實支持以下 starting point：先做 row-wise max centering，直接近似自然指數 $e^u$，在 $[-8,0]$ 使用 16 個等寬區段、區段寬度 $0.5$、17 個 16-bit endpoint，輸入採 Q8.8，並以線性插值產生未正規化權重。低於 $-8$ 的輸入 clip 到左端點，$u=0$ 明確使用最後一個 endpoint。[arXiv v2 §3.1–3.2](https://arxiv.org/html/2607.01798v2#S3)

但是，論文**沒有完整公開可直接照抄的 bit-true 規格**。它沒有列出 17 個量化後的 endpoint 整數、endpoint 的唯一 fixed-point format、rounding mode、中間乘積／累加器位寬或每一步 saturation 規則。論文只說 endpoint 使用 fixed-point fractional format，並以「若使用 Q0.16」描述 $u=0$ endpoint 的 saturation。因此本專案提出的 shift/mask interpolation 可以是與論文相容的 project bit-true specification，但不能標成逐位重製作者 RTL。[arXiv v2 §3.2](https://arxiv.org/html/2607.01798v2#S3.SS2)

更重要的是，論文直接把 $u<-8$ 的 contribution 稱為 negligible，卻沒有提供各模型的 centered-score distribution、clipping ratio 或 clipped tail probability mass。原文的 Imagenette／ViT 結果不能證明 $[-8,0]$ 適合 Binary-QK YOLO26；目前計畫先量兩個 Attention site、各 head 的分布與 tail mass，是必要驗證，不是重複論文工作。[arXiv v2 §3.1](https://arxiv.org/html/2607.01798v2#S3.SS1)、[§5.1](https://arxiv.org/html/2607.01798v2#S5.SS1)

## 1. 原論文可以確認的數學與數值設定

### Stable score 與 normalization

論文使用：

$$
u_j=s_j-\max_k s_k\leq 0,
$$

$$
\widetilde p_j=\frac{\widetilde e(u_j)}{\sum_k\widetilde e(u_k)}.
$$

它只以 PWL 替換 exact $e^{u_j}$；denominator accumulation、normalization 與 value accumulation 仍然存在。[arXiv v2 Eq. (1), §3.1](https://arxiv.org/html/2607.01798v2#S3.SS1)

### Uniform PWL

論文固定 16 個 uniform segments：

$$
b_i=-8+0.5i,\qquad i=0,\ldots,16,
$$

$$
y_i=e^{b_i},
$$

$$
\widetilde e(x)=y_i+\alpha(y_{i+1}-y_i),
\qquad x\in[b_i,b_{i+1}),\quad \alpha\in[0,1).
$$

$x=0$ 例外地指定為最後一個 endpoint。輸入 $x<-8$ 會 clip 到 $-8$；由 stable centering 可知正常輸入不大於 0。[arXiv v2 Eq. (5)–(6), §3.2](https://arxiv.org/html/2607.01798v2#S3.SS2)

### Fixed-point 與 table

原文明確寫出的資訊是：

- $x$ 使用 Q8.8。
- segment index 與 fractional offset 從 fixed-point bit fields 取出。
- 17 個 endpoint，每個 16 bits，共 $17\times16=272$ bits。
- endpoint 存在 distributed LUTRAM，因此不使用 BRAM。
- endpoint 是 fixed-point fractional format；若採 Q0.16，$x=0$ 的 $e^0=1$ 需以 saturation 表示。
- interpolation product 宣稱以 LUT logic 實作。

來源：[arXiv v2 §3.2](https://arxiv.org/html/2607.01798v2#S3.SS2)。

### 本專案公式的來源界線

對 Q8.8、$[-8,0]$、$\Delta=0.5$，令 $U$ 為 centered score 的 signed Q8.8 integer，則以下寫法是從上述規格直接推導出的自然硬體 mapping：

$$
Z=\operatorname{clip}(U,-2048,0)+2048,
$$

$$
i=Z\gg7,\qquad f=Z\ \&\ 127.
$$

對 $U<0$，$i\in\{0,\ldots,15\}$；$U=0$ 必須特判，直接輸出 $Y_{16}$。若 endpoint integer 為 $Y_i$，truncating interpolation 可寫為：

$$
Y=Y_i+\left((f(Y_{i+1}-Y_i))\gg7\right).
$$

這組 shift/mask 關係與論文的 Q8.8、uniform $0.5$ segments 和 bit-field extraction 一致，但**右移前是否加 rounding offset、endpoint 如何量化、差值與乘積的 signed width、overflow/saturation 次序均不是論文已指定內容**。因此應標示為本專案的 bit-true reference，而不是 faithful RTL reproduction。

## 2. 論文沒有提供、不能猜測的項目

截至 arXiv v2，以下細節未出現在正文或官方 arXiv source：

| 項目 | 原論文狀態 | 本專案應如何處理 |
|---|---|---|
| 17 個量化 endpoint 的完整 hex／integer 表 | 未提供 | 由明確公式與選定 quantizer 產生，連同 generator/version 提交 |
| endpoint 的唯一 Q format | 只說 fractional format，Q0.16 是條件式例子 | 專案自行固定，並處理 $1.0$ 的飽和值 |
| endpoint quantization rounding | 未提供 | 明定 round-to-nearest 或其他規則並測試 |
| interpolation rounding | 未提供 | truncation 與 rounding 至少做 bit-true error 對照後定案 |
| signed intermediate widths | 未提供 | 在 RTL spec 列出 $Y_{i+1}-Y_i$、乘積、輸出位寬 |
| saturation 次序 | 除 input clipping 與 Q0.16 的 $u=0$ endpoint 外未提供 | 明確列出 input、endpoint、product/output saturation |
| 作者 RTL／emulator code | arXiv 頁面未連結；本次 exact-title／作者關鍵字檢索未找到可確認的官方 repository | 不宣稱與作者 RTL bit-exact；若日後取得 code 再核對 |
| Binary-QK YOLO26 score 分布 | 不在論文範圍 | 必須在本專案 checkpoint、COCO2017 val 上量測 |

## 3. 原論文的 approximation 與 accuracy 證據

論文報告 $[-8,0]$ 的 16-segment PWL 最大 absolute exponential error 為 `0.0245`，出現在最靠近 0 的區段；32 segments 時約為 `0.006`，table 從 272 bits 增至 528 bits。作者據此把 16 segments 視為 approximation accuracy 與 LUTRAM footprint 的折衷。[arXiv v2 Figure 1–2, §3.2](https://arxiv.org/html/2607.01798v2#S3.SS2)

但其模型與資料不是本專案：

- INT8 實驗使用 Imagenette 3,550-image validation split、ViT-B/16 與 DeiT-S/16；Q/K/V 做 per-tensor symmetric fake quantization。
- INT16 matched protocol 使用 ViT-S/16、ViT-B/16、ViT-L/16。
- 作者報告 approximate INT8 與 exact INT8 的 top-1 差在 `0.2%` 內；INT16 相對 FP32 exact 的最大 top-1 degradation 為 `0.20%`。
- 這些實驗沒有涵蓋 object detection、COCO2017、Binary QK、PoT scale 或 decomposed 2D bias。

來源：[arXiv v2 Table 2–3, §5.1](https://arxiv.org/html/2607.01798v2#S5.SS1)。所以「論文上近乎無損」只能作為 starting-point evidence，不能替代目前 YOLO26 的 Exact／Float PWL／Bit-True PWL 同 checkpoint 評估。

## 4. 硬體成本主張與真正剩餘成本

論文的 target 是 Xilinx Zynq-7020 `xc7z020clg484-1`、100 MHz；row core 固定 $N=197$、$d_k=d_v=64$，採 two-pass schedule，最後用 serialized 16-bit restoring divider 正規化輸出。[arXiv v2 §4.1–4.2](https://arxiv.org/html/2607.01798v2#S4)

| 模組 | LUT | DSP | BRAM | Dynamic power |
|---|---:|---:|---:|---:|
| Score MAC | 198 | 8 | 0 | 11 mW |
| Numerator accumulator（weighted $V$） | 544 | 64 | 0 | 未個別列出 |
| Natural PWL weight | 71 | 2 | 0 | 3 mW |
| Denominator accumulator | 69 | 0 | 0 | 1 mW |
| Restoring divider | 487 | 0 | 0 | 6 mW |
| Post-route full row core | 1,444 | 77 | 0 | 21 mW dynamic／124 mW total |

來源：[arXiv v2 Table 1, §4.2](https://arxiv.org/html/2607.01798v2#S4.SS2)。完整 row latency 是 7,920 cycles，即 79.2 $\mu$s；這些結果不含 sensor input、external memory、DMA 與 Transformer 其他部分。[arXiv v2 §4.1、§5.2](https://arxiv.org/html/2607.01798v2#S5.SS2)

因此論文能支持的硬體優勢是：以很小的 distributed table 與線性插值取代 CORDIC／大型 exponential LUT，不改成 base-2，且 weight generation 不使用 BRAM。它**不能**支持「整個 Softmax 沒有除法」或「PWL 消除了 PV 成本」：原設計仍累加 denominator，並以 restoring divider 做最後 normalization；64-DSP numerator/value accumulator 是最大 DSP consumer，divider 則占 487 LUT。[arXiv v2 §4.1–4.2](https://arxiv.org/html/2607.01798v2#S4)

原文另有一個應在 RTL 前澄清的內部不一致：§3.2 說 interpolation product 由 LUT logic 實作並讓 weight unit free of DSP，但 Table 1 與 §4.2 又把 Natural PWL Weight 列為 2 DSP。沒有作者 RTL 時，不能自行判定哪一項才是最終 mapping；本專案成本報告應把 integer interpolation multiplier 明列，待 synthesis 後以實測 utilization 取代論文數字。[arXiv v2 §3.2](https://arxiv.org/html/2607.01798v2#S3.SS2)、[§4.2](https://arxiv.org/html/2607.01798v2#S4.SS2)

## 5. 對本專案驗證的直接結論

1. `[-8,0] / 16 segments / Δ=0.5 / Q8.8 / 17×16-bit endpoint` 是忠於論文的 starting point。
2. `[-8,0]` 是否適合本 Binary-QK YOLO26，論文沒有證據；必須量 clipping ratio 與 exact-Softmax tail probability mass。若 tail mass 顯著，維持 $\Delta=0.5$ 比較 `[-10,0] / 20 segments` 是合理且乾淨的 project ablation。
3. Float PWL 與 Q8.8 bit-true PWL 必須分開：前者驗證函數／range，後者驗證 endpoint quantization、rounding、width 與 saturation。
4. Bit-true class/docstring 應標為 `project bit-true reference derived from the paper's public numerical setting`，不可標為作者 RTL faithful reproduction。
5. 硬體成本至少分成 max/centering、segment/index、endpoint storage、integer interpolation、denominator accumulation、reciprocal/divider、$PV$；只報 272-bit endpoint table 會嚴重低估 normalization 與 Attention 的完整成本。

## 搜尋範圍與限制

本次以 2026-08-16 為截止日，只把以下 primary source 當作技術證據：

- [arXiv abstract and version record, arXiv:2607.01798](https://arxiv.org/abs/2607.01798)
- [作者提交的 arXiv v2 HTML full text](https://arxiv.org/html/2607.01798v2)
- [arXiv v2 官方 TeX source archive](https://arxiv.org/e-print/2607.01798v2)（用於確認正文之外沒有額外 endpoint／rounding 規格）

另外以完整論文標題、arXiv ID、`Natural PWL Weight`、作者／Q8.8 關鍵字搜尋 GitHub 與作者機構頁；截至本次搜尋，arXiv 沒有附作者官方 repository，本次也沒有找到可確認為作者發布的 RTL 或 emulator。搜尋結果缺失不能證明程式碼不存在，只表示目前不能以官方 code 驗證 bit-exactness。

本稿目前是 arXiv preprint，arXiv 頁面標示 v2 於 2026-07-06 上傳，正文標示 under review；尚未找到可取代 arXiv v2 的正式 IEEE publisher version。因此未引用 alphaXiv、ResearchGate 自動頁面、新聞稿或其他二手摘要作為證據。[arXiv version record](https://arxiv.org/abs/2607.01798)
