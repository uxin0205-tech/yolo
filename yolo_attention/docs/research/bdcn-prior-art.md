# Binary Distance-Codebook Normalization prior-art 初查

日期：2026-08-11

## 結論先行

目前**不能宣稱 Binary Distance-Codebook Normalization（BDCN）的核心流程是我們先提出**。

最接近的直接先例是 2025 年 11 月公開、2026 年 5 月修訂的 **IntAttention / IndexSoftmax**。它已明確執行：

$$
\Delta_{ij}=\max_k S_{ik}-S_{ij},
$$

再將非負距離 clipping、映射為有限 LUT index、查詢近似 $\exp(-\Delta)$ 的表格、做 row sum，最後輸出正規化的 UINT8 probability。這與 BDCN 的「max-relative distance → bucket/index → codebook/LUT → row normalization」核心骨架高度重疊。[IntAttention 論文 §3.1–3.2](https://arxiv.org/html/2511.21513v2)、[arXiv 版本紀錄](https://arxiv.org/abs/2511.21513)、[作者程式庫](https://github.com/WanliZhong/IntAttention)

目前較合理的研究定位不是「首次提出 distance-bucket LUT Softmax」，而是以下**特定組合與限制的 proposed design**：

> 針對 Binary QK / Hamming score，學習 per-Attention、per-head 的單調 distance codebook，再研究 two-term power-of-two 投影，以及等價的 bucket-count histogram dataflow。

這個完整組合在本次檢索的 primary sources 中尚未找到完全相同的實作；但「尚未找到」不等於可證明首創。在完成更廣的論文、專利與非英文檢索以前，只能稱為 contribution hypothesis。

## BDCN 與最接近方法的差異

| 元件 | IntAttention / IndexSoftmax | 目前 BDCN 構想 | 判讀 |
|---|---|---|---|
| QK precision | INT8 Q/K、INT32 logits | 1-bit Binary QK / popcount score | 可能形成應用場景差異 |
| 相對距離 | row max 減 score | row max 減 score | 幾乎相同 |
| 離散化 | clipping 後線性映射至 32-entry LUT | 例如 16 個固定 distance buckets | 高度相同 |
| LUT 內容 | offline 固定的 exponential-shaped UINT8 LUT | 可訓練、每個 head 獨立、受單調約束 | 主要可研究差異 |
| normalization | gather 後逐元素 row sum，再做 integer scaling | 精確 row normalization | 相同目的 |
| denominator dataflow | 逐元素累加 | 可用 bucket count：$D_i=\sum_l n_i[l]C_h[l]$ | 等價 dataflow，不宜獨立宣稱演算法 novelty |
| 硬體限制 | UINT8 LUT | floating upper bound → one-PoT / two-PoT | two-PoT 結合是差異，但 PoT 加法本身已有先例 |
| adaptation | training-free，固定超參數 | attention-only training / PMP recovery | 訓練策略不同 |

IntAttention 本身目前是 arXiv preprint；其頁面標示為 preliminary / under review。即使如此，它仍是早於本研究公開且公式高度重疊的 prior disclosure，不能因尚未正式發表便忽略。

## 已知元件先例

### 1. Binary QK 與 Hamming-distance 解讀

**BinaryAttention** 將 Q/K 保留為 sign，使用 bitwise 計算 similarity，並把 binary attention 解讀為 Hamming space 中的 distance-based metric；它後續仍使用 exponential/Softmax，沒有改成 learned distance codebook。因此 Binary-QK-specific normalization 仍有研究空間，但「Binary QK 是距離」並非本研究首創。[CVPR 2026 論文](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_BinaryAttention_One-Bit_QK-Attention_for_Vision_and_Diffusion_Transformers_CVPR_2026_paper.html)、[作者程式庫](https://github.com/EdwardChasel/BinaryAttention)

### 2. Max-relative distance、分桶與 LUT Softmax

- **IntAttention / IndexSoftmax** 已直接採用非負 max-relative distance、clipping、distance-to-index、固定 exp LUT 與 row normalization，是最接近的先例。[論文](https://arxiv.org/html/2511.21513v2)
- **EXAQ** 對 max-subtracted Softmax inputs 做低位 clipping/quantization，再以 LUT 近似 exponential，顯示「先限制 score range，再用很少 bit 表示 Softmax input」已有先例。[論文](https://arxiv.org/abs/2410.03185)
- **Efficient Softmax Approximation** 使用 max normalization 後的小型 numerator/denominator LUT，並在 DETR/COCO2017、Transformer、BERT 等模型與任務驗證 LUT Softmax。[論文](https://arxiv.org/abs/2111.10770)
- 2020 年公開的 **Softmax circuit** 專利使用 input differences 查詢 exponential LUT，再以 denominator/quotient LUT 取代除法。它不是 BDCN 的 row-max bucket 形式，但證明「difference-indexed exponential LUT + denominator lookup」已有更早硬體揭露。[US10949498B1](https://patents.google.com/patent/US10949498B1/en)

### 3. 可訓練 Softmax/LUT 參數

- **NN-LUT** 先訓練一層 ReLU approximator，再轉換成 LUT；對 Softmax 分別近似 EXP 與 DIV，並提出用 calibration 更新 approximation parameters。其 16-entry 實驗說明「可訓練的小型 LUT nonlinear approximator」已有先例，但它不是 task-trained per-head monotonic distance table。[論文](https://arxiv.org/abs/2112.02191)
- **ConSmax** 對每個 self-attention head 學習 normalization parameters $\beta,\gamma$，並使用 LUT 硬體。它不是學習每個 distance bucket 的 table，且不保證 probability row sum 為一；但「per-head learnable normalization」本身已有先例。[論文](https://arxiv.org/html/2402.10930v3)、[作者程式庫](https://github.com/ReaLLMASIC/ConSmax)

因此 BDCN 的 claim 必須精確到「per-head、monotonic、distance-indexed codebook」，不可只寫 learnable LUT 或 learnable normalization。

### 4. Base-2、shift 與 additive power-of-two

- **Softermax** 已用 base replacement、low-precision computation 與 online normalization 簡化 Softmax。[論文](https://arxiv.org/abs/2103.09301)
- **I-ViT / Shiftmax** 以 integer shifts/additions 實作 integer-only Softmax。[論文](https://arxiv.org/abs/2207.01405)、[作者程式庫](https://github.com/zkkli/I-ViT)
- **ITA** 以 max subtraction、base-2 exponent、bit shifting、integer denominator accumulation/inversion 建立量化 Softmax datapath。[論文](https://arxiv.org/abs/2307.03493)
- **APoT** 明確把 quantization levels 約束成數個 power-of-two terms 的和，以 shift/add 取代一般乘法。因此 two-term PoT 表示本身不是 novelty；可研究的是把它用作 learned BDCN codebook 的硬體 projection，以及其 accuracy/cost trade-off。[論文](https://arxiv.org/abs/1909.13144)、[作者程式庫](https://github.com/yhhhli/APoT_Quantization)

### 5. Histogram 聚合

- **LISA** 先以 codebook 表示 sequence items，再用 codeword occurrence histogram，把 attention numerator 與 denominator 改寫為 frequency-weighted sums。其 codebook 對象是 embedding/codeword，不是 BDCN 的 score-distance bucket；但「用重複 code 的 counts 合併 attention normalization」已是明確先例。[論文 §3.2–3.3](https://arxiv.org/abs/2105.14068)
- **AdaSplash-2** 對 attention scores 建立 coarse histogram，以加速 $\alpha$-entmax normalizer 的求解。其 histogram 用於 normalizer initialization，不是 BDCN 的 exact bucket-count denominator，但也表示 attention-score histogram 已被使用。[論文](https://arxiv.org/abs/2604.15180)

在 BDCN 中，一旦 $w_{ij}=C_h[b_{ij}]$，則

$$
\sum_j w_{ij}=\sum_l n_i[l]C_h[l]
$$

只是依 bucket 重排同一個總和。因此 direct row sum 與 histogram denominator 應做 bit-equivalence test，histogram 只比較 operation count、buffer、latency/dataflow，不應作為獨立 accuracy 方法或單獨 novelty claim。

## 建議的論文敘述與實驗

### 可使用的保守敘述

> We propose a Binary-QK-specific normalization design that learns a monotonic distance-indexed codebook per attention head, and study its projection to a two-term power-of-two representation and an equivalent histogram dataflow.

中文可寫成：

> 本研究提出一個針對 Binary QK 的 per-head learnable monotonic distance-codebook normalization，並研究 two-term PoT 投影與等價 histogram dataflow 所形成的精度—硬體成本折衷。

### 不應使用的敘述

- 「首次提出以距離分桶和 LUT 取代 Softmax。」
- 「首次利用 Binary score 的離散性進行 Softmax LUT。」
- 「首次使用 histogram 計算 attention denominator。」
- 「two-term PoT 是本研究提出的新量化表示。」

### 必要 ablation

1. IntAttention-style fixed exponential LUT。
2. learned shared monotonic LUT。
3. learned per-Attention LUT。
4. learned per-head monotonic LUT。
5. floating codebook、one-PoT、two-PoT。
6. direct row sum 與 histogram dataflow：必須輸出相同 P，只比較硬體 proxy。
7. INT8 QK + IndexSoftmax 與 Binary QK + BDCN 分開，避免把 QK binarization 的效果誤算成 normalization 貢獻。

上述順序才能回答 BDCN 的改善來自 Binary-QK specialization、per-head learning、monotonic constraint，或只是一般 LUT Softmax。

## 搜尋範圍與限制

本次以 2026-08-11 為截止日，查閱 arXiv、CVF Open Access、OpenReview、作者 GitHub repository，以及直接相關的 Google Patents 全文；檢索主題包含 binary QK attention、Hamming-distance attention、max-relative/index Softmax、learnable/monotonic LUT、histogram normalizer、integer/shift Softmax、additive power-of-two。只用原始論文、作者 repository 與直接相關專利支持技術判讀，未用部落格或二手 paper summary 作為證據。

限制如下：

- 這不是正式的 systematic review，也不是法律上的 patentability / freedom-to-operate 意見。
- 未完整覆蓋付費資料庫、所有專利家族、碩博士論文、非英文文獻、尚未公開稿件及可能使用不同名稱描述同一公式的工作。
- IntAttention、AdaSplash-2 等 2025–2026 工作很新，發表狀態與版本仍可能改變。
- 文獻中沒有找到完全相同組合，只能支持「尚未發現」，不能支持「世界首次」。

在投稿前，應以最終公式與 claim language 再做一次 citation graph、Google Scholar/IEEE/ACM 及專利分類檢索，並把 IntAttention / IndexSoftmax 放入最接近比較方法。
