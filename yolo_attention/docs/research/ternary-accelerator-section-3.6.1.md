# Multimax（論文第 3.6.1 節）研究筆記

## 來源與範圍

- 論文：陳法諭，〈三元權重變形器神經網路加速電路之設計與晶片實現〉，國立臺灣大學碩士論文，2025 年 7 月。
- DOI：[`10.6342/NTU202504546`](https://doi.org/10.6342/NTU202504546)。
- 本機原文：[三元權重變形器神經網路加速電路之設計與晶片實現.pdf](../../三元權重變形器神經網路加速電路之設計與晶片實現.pdf)。
- 本筆記只分析第 3.6.1 節「Multimax」，即印刷頁 39–42（PDF 第 58–61 頁），包括 3.6.1.1 與 3.6.1.2。

Multimax 是該論文的 Softmax 近似方法，不是三元權重量化本身。論文中的三元權重／INT4 activation（TW4A）是另一部分；兩者可以共同訓練，但不可把 Multimax 的效果歸因於三元權重。

## 3.6.1 在做什麼

傳統 Softmax 對每一列所有元素計算 exponential、加總與除法。Multimax 根據「注意力機率通常由少數大值主導」的觀察，只保留 score 最大的 Top‑K；論文硬體支援：

$$
K\in\{1,3,5\}.
$$

其餘位置直接設成 0。對排序後 score：

$$
s_1\ge s_2\ge\cdots\ge s_K,
$$

方法以相鄰 score difference 查 sigmoid LUT，再用類似 stick-breaking 的遞迴方式分配剩餘機率。以論文式 3.10 的 $K=3$ 為例：

$$
a_1
=\frac{e^{s_1-s_2}}{e^{s_1-s_2}+1}
=\sigma(s_1-s_2),
$$

$$
p_1=a_1,
$$

$$
a_2
=\frac{e^{s_2-s_3}}{e^{s_2-s_3}+1}
=\sigma(s_2-s_3),
$$

$$
p_2=(1-p_1)a_2,
$$

$$
p_3=1-p_1-p_2.
$$

因此：

$$
p_1+p_2+p_3=1.
$$

論文原式使用量化整數 $x_{int},y_{int},z_{int}$ 與共同 scale $S$；zero point $Z$ 在相減時消去：

$$
(Sx_{int}+Z)-(Sy_{int}+Z)=S(x_{int}-y_{int}).
$$

這讓 LUT 只需接收量化 score difference。論文式 3.10 使用 `diff_yz`，但正文沒有另外正式定義；依上下文可解讀為 $y_{int}-z_{int}$。

## 硬體意義

Multimax 改變的不是 exponential 的數值近似而已，而是整條 normalization 與後續矩陣乘法：

~~~text
一列 attention score
        ↓
Top-K selection + index
        ↓
相鄰 score difference
        ↓
sigmoid LUT
        ↓
遞迴 remainder allocation
        ↓
只有 K 個非零 probability
        ↓
sparse P × V
~~~

直接收益：

1. 不需要對整列做 exponential、完整 row sum 與除法。
2. probability sum 由遞迴 remainder 保持為 1；fixed-point 時可把最後一項設為剩餘整數，因此也能做 exact U8 row sum。
3. $P\times V$ 只需讀取 Top‑K 對應的 V。當 $K=1$ 時，輸出退化成直接選一列 V，不需 probability multiplication。

代價與限制：

1. 必須加入 Top‑K comparator、index tracking 與 sparse gather/control。
2. 它不是原 Softmax 的精確 Top‑K renormalization，而是以相鄰兩兩 sigmoid 做遞迴近似。
3. 被丟棄的 attention tail 永遠為 0；若 YOLO26 C2PSA 需要分散式空間資訊，精度可能明顯下降。
4. 論文在 Transformer 任務上的結果不能直接證明 YOLO26 COCO detection 也成立，必須重新量測。

## 與 Binary QK 的融合方式

本研究的 Binary QK 產生離散 score，因此比一般 floating score 更適合實作 difference-indexed sigmoid LUT：

~~~text
Binary Q/K
   ↓
XNOR + Popcount
   ↓
scale + relative bias
   ↓
integer score grid
   ↓
Top-K score/index
   ↓
difference-indexed sigmoid LUT
   ↓
normalized sparse U8 P
   ↓
sparse P × INT8 V → S32 accumulate
~~~

若 score scale 使用 fixed-head 或 power-of-two，LUT address 與硬體更單純；若加入 Dense／Decomposed relative bias，則必須先把 bias 對齊同一 integer score grid，再做 Top‑K。

## 對本研究的建議定位

不建議直接用 Multimax 取代目前的 L3-A LUT Softmax 主線，因為兩者回答不同問題：

- L3-A：保留 dense attention 語意，只近似數值運算。
- Multimax：把 normalization 改成 sparse Top‑K attention，連 $P\times V$ 一起簡化，演算法改動較大。

建議把 Multimax 設為 formal V1 與 scale/bias algorithm parent 之後的條件式 software ablation：先做 $K=1,3,5$ 無訓練診斷；只有結果可接受時，選單一 K 做短期 recovery。論文在 3.6 節開頭使用 Progressive Mixed Precision，亦即在訓練時混合原始函數與近似函數；這和本計畫目前混合 FP score／binary score 的 Progressive Blend 是兩個不同位置的方法，實驗名稱與歸因必須分開。

## 需回報的比較

- COCO mAP50–95 與 APs/APm/APl。
- Top‑K recall／與 dense Softmax top‑K 的一致率。
- attention output cosine similarity。
- Top‑K comparator、index buffer、sigmoid LUT 與 sparse gather 成本。
- dense 與 sparse $P\times V$ 的 MAC、memory read 與 CPU reference latency。
- U8 row-sum error；若最後一項使用 remainder，需驗證每列恰為 255。

以上「與 Binary QK 融合」及「建議定位」是依論文方法對本專案所做的工程推論，不是論文作者對 YOLO26 的直接結論。
