# YOLO11 BinaryAttention 相對位置 Bias 報告

## 1. 摘要

本報告說明 T7-D 與 T7-R 兩種 relative-position bias 的實際作法、參數量、額外運算量與硬體友善性。

兩者都不是改變 Q/K 二值化的方式，而是在 attention 的 QK score 計算後、softmax 前，加入一個依照 query/key 相對位置查出的 bias：

```text
score[h, i, j] = QK_score[h, i, j] + position_bias[h, i, j]
P = softmax(score, dim=-1)
```

目前結果顯示，T7-D（dense 2D）mAP50-95 為 50.616%，T7-R（decomposed 2D）為 50.577%。在本次 10-epoch attention-only adaptation 中，T7-D 高 0.039 個百分點，因此後續流程選擇 dense 2D。

這個結果不能解讀成實際推論速度提升：目前程式使用 fake quantization/QAT 與一般 PyTorch tensor 操作，沒有實際 1-bit 或專用硬體 benchmark。

## 2. Relative position 的意義

假設 feature map 上有兩個 token：

```text
query 位置 = (y_q, x_q)
key    位置 = (y_k, x_k)

dy = y_q - y_k
dx = x_q - x_k
```

attention 不是只看內容相似度，也可以學到某些位置關係，例如鄰近位置、上方位置、右下方位置是否比較重要。程式目前支援的相對位移範圍是 `[-31, +31]`；超過 `max_bias_size=32` 會拒絕執行，而不是默默截斷。

例如 query 在 `(3, 4)`、key 在 `(1, 5)`：

```text
dy = +2
dx = -1
```

程式會用 `(dy, dx)` 查表，將查到的數值加到該 head 的 QK score 上。

## 3. T7-D：Dense 2D relative-position bias

### 3.1 參數化

Dense 版本為每個 head 建立完整的二維表：

```text
B_dense[h, dy, dx]
```

程式對應的核心操作是：

```python
return self.relative_bias[:, dy, dx]
```

因此每一個 `(dy, dx)` 組合都有自己的可學習參數。以目前位移範圍計算：

```text
位移數量 = 63 × 63 = 3,969
```

由正式 artifact 的 trainable-parameter 差異可反推出目前 attention module 使用 4 個 heads，因此 T7-D 的 bias 參數量是：

```text
4 × 63 × 63 = 15,876 個參數
```

bias 以 truncated normal、標準差 0.02 初始化，訓練時由 detector loss/KD loss 透過梯度更新。

### 3.2 表達能力

Dense 2D 能分別學習：

- 左、右、上、下的偏好；
- 對角線方向的偏好；
- 同樣的垂直距離，在不同水平距離下有不同效果；
- 非可分解的二維位置模式。

因此它的表達能力最高，但參數表較大。

## 4. T7-R：Decomposed 2D relative-position bias

### 4.1 參數化

Decomposed 版本把二維 bias 拆成垂直與水平兩個一維 bias：

```text
B_decomp[h, dy, dx]
    = B_h[h, dy] + B_w[h, dx]
```

程式核心操作是：

```python
return self.relative_bias_h[:, dy] + self.relative_bias_w[:, dx]
```

每個 head 只需要：

```text
63 個垂直參數 + 63 個水平參數 = 126 個參數
```

目前 4 個 heads 的總參數量是：

```text
4 × (63 + 63) = 504 個參數
```

相較於 Dense 2D 的 15,876 個參數，Decomposed 版本少 15,372 個參數，約少 31.5 倍的 bias 參數。

### 4.2 表達限制

Decomposed 版本假設位置影響可以拆成：

```text
垂直方向影響 + 水平方向影響
```

因此它不能獨立表示所有 `(dy, dx)` 組合。例如「只有右上角特別重要，但右下角不重要」這類真正的二維交互關係，Decomposed 只能用水平和垂直兩個項目近似。

## 5. 額外運算量估算

令：

```text
N = H × W              # token 數
G = num_heads          # attention heads
```

每個 head 的 attention score 矩陣大小是 `N × N`，所有 heads 合計有 `G × N²` 個 score。

### 5.1 相對於沒有 bias

Dense 2D 的理想化額外工作是：

```text
每個 score：一次 bias lookup + 一次加法
總量：約 G × N² 次 lookup/add
```

Decomposed 2D 的理想化額外工作是：

```text
每個 score：一次垂直 lookup + 一次水平 lookup
          + 一次兩者相加 + 一次加到 score
總量：約 2G × N² 次 lookup + 2G × N² 次加法
```

實際實作若由 fused attention kernel 合併，部分中間 tensor 與加法可能被消除；目前 repository 的程式是先建立 bias tensor，再與 score 相加，不能直接宣稱已經使用 fused kernel。

### 5.2 相對於 QK 矩陣乘法

若每個 head 的 Q/K 維度是 `d`，QK score 的主要乘加量約為：

```text
G × N² × d
```

所以以純算術操作粗估：

```text
Dense bias：額外約 1/d 的 score-level 加法成本
Decomposed：額外約 2/d 的 score-level 加法成本
```

若把 PV 矩陣乘法也算入 attention 主體，額外比例大約再減半；但 lookup、記憶體讀取、kernel launch 和 tensor materialization 可能比單純加法更重要，因此這只是量級估算，不是實測 latency。

### 5.3 MACs 與 FLOPs 的嚴格區分

對標準 relative-position bias 而言，bias 是查表後直接加到 score：

```text
score = QK_score + bias
```

這個操作沒有乘法，因此：

```text
Dense 2D 額外 MACs       = 0
Decomposed 2D 額外 MACs  = 0
```

兩者的差異在於 elementwise add，而不是 multiply-accumulate。若採用常見的 FLOPs 計數方式「一次乘法 = 1 FLOP、一次加法 = 1 FLOP」，則：

```text
Dense：      約 G × N² 額外 FLOPs
Decomposed： 約 2G × N² 額外 FLOPs
```

Decomposed 的兩次加法分別是：

1. 將垂直 bias 與水平 bias 相加；
2. 將合成後的 bias 加到 QK score。

Dense 則只有第二個 score add。lookup、index 計算與記憶體搬移通常不算 FLOPs，但在真實硬體 latency 中可能是主要成本。

若只看 QK dot product，令每個 head 的 Q/K 維度為 `d`：

```text
QK MACs       = G × N² × d
QK FLOPs      ≈ 2G × N² × d

Dense bias FLOPs / QK FLOPs
              ≈ 1 / (2d)

Decomposed bias FLOPs / QK FLOPs
              ≈ 1 / d
```

若把 QK 與 PV 兩個矩陣乘法都算進 attention，令 V 的 head 維度為 `dv`：

```text
Attention MACs  = G × N² × (d + dv)
Attention FLOPs ≈ 2G × N² × (d + dv)

Dense total FLOPs
                ≈ 2G × N² × (d + dv) + G × N²

Decomposed total FLOPs
                ≈ 2G × N² × (d + dv) + 2G × N²
```

這些是 logical operation counts。若使用 fused attention kernel，bias add 可能被融合進 score/softmax kernel，外部看不到獨立的 FLOP kernel；但數學上仍代表相同的加法工作。

### 5.4 例子：只計算 bias 的額外 MACs/FLOPs

假設 `G=4`：

| Feature map | Token 數 N | Score 數 G×N² | Dense 額外 MACs | Dense 額外 FLOPs | Decomposed 額外 MACs | Decomposed 額外 FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| 7×7 | 49 | 9,604 | 0 | 9,604 | 0 | 19,208 |
| 14×14 | 196 | 153,664 | 0 | 153,664 | 0 | 307,328 |
| 28×28 | 784 | 2,458,624 | 0 | 2,458,624 | 0 | 4,917,248 |

上表重點是 `N²` 成長；feature map 變大時，runtime bias tensor 的成本會快速增加。參數表本身則固定是 15,876 或 504，不隨 `H×W` 增長。

若以「一次 MAC = 2 FLOPs」的硬體報表慣例呈現，bias 仍然是 0 MAC；上表的 FLOPs 只是加法 FLOPs，不能把 lookup 當成 FLOP。QK/PV 主體的 MACs 則必須另外依照實際的 `d` 與 `dv` 計算。

## 6. 硬體友善性分析

### 6.1 Dense 2D 的優點

- 查表邏輯直接，概念簡單。
- 參數表很小，容易放入 cache 或 on-chip memory。
- 對模型表達能力限制較少。
- 若硬體支援 relative-position lookup 或 fused score-plus-bias，容易整合。

### 6.2 Dense 2D 的缺點

- 對每個 score 都要依照 `dy, dx` 做 indexed gather。
- 目前程式會建立完整的 `G×N×N` bias tensor，可能增加記憶體頻寬與暫存空間。
- 它本身不是矩陣乘法，無法自然地使用一般 GEMM 資源。
- 若沒有 fused kernel，可能多出一個 kernel 或中間 tensor。

### 6.3 Decomposed 2D 的優點

- 參數量約少 31.5 倍。
- 垂直與水平表較容易快取。
- 在硬體中可以利用 separable/axis-wise 的資料流設計。
- 適合記憶體非常受限的部署環境。

### 6.4 Decomposed 2D 的缺點

- 每個 score 需要兩次 lookup，而不是一次。
- 目前實作還需要把兩個 lookup 相加，再加到 score；若沒有融合，runtime 未必比 Dense 快。
- 表達能力較弱，可能損失少量精度。

### 6.5 結論

從參數與儲存角度，Decomposed 2D 明顯更硬體友善；從單次 score 的 runtime 操作數來看，Dense 反而可能較簡單，因為它只需要一次二維 lookup 加一次 score add。真正誰比較快，取決於硬體是否支援：

1. relative-position bias 的 fused gather；
2. score、bias、softmax 的融合；
3. `N×N` score/bias 是否需要落地到記憶體；
4. cache、SRAM、HBM 或 GPU global-memory 頻寬；
5. attention 的實際 feature-map 尺寸。

因此不能只從參數量直接推論 Dense 或 Decomposed 的實際 FPS。

## 7. 本次實驗結果

| Variant | Bias | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| T6 | 無 bias（前一階段基準） | 71.769% | 62.611% | 67.423% | 50.624% |
| T7-D | Dense 2D | 72.343% | 62.004% | 67.395% | 50.616% |
| T7-R | Decomposed 2D | 71.913% | 62.532% | 67.402% | 50.577% |

相對於 T6：

- T7-D 的 mAP50-95 下降 0.008 個百分點，幾乎持平。
- T7-R 的 mAP50-95 下降 0.047 個百分點。
- T7-D 比 T7-R 高 0.039 個百分點。

這代表在本次 10-epoch adaptation 中，Dense 2D 的完整表達能力帶來非常小但可觀察的優勢；然而差異小於可能的訓練波動，不能只用這一輪就宣稱在所有資料集都更好。

## 8. 研究與部署上的重要限制

1. 目前是 YOLO11m 的 10-epoch attention-only adaptation，不是完整模型長時間訓練。
2. P/V 與其他量化設定主要是 fake quantization，尚未證明真實 1-bit/8-bit 硬體速度提升。
3. 尚未有 T7-D 與 T7-R 的同機、同 batch、同 kernel、warm-up 後 latency/FPS/功耗 benchmark。
4. 現有程式的 bias lookup 與 score add 尚未證明是 fused implementation。
5. 因此本報告可以支持參數量與理論運算量比較，但不能把理論比例當成實際速度倍率。

## 9. 建議的硬體 benchmark

若要回答「到底快多少」，應固定同一 checkpoint 與輸入，分別測試：

```text
無 bias / Dense 2D / Decomposed 2D
```

並記錄：

- warm-up 後平均與 P50/P95 latency；
- throughput（FPS）；
- GPU/加速器記憶體使用量；
- power 或 energy per image；
- 是否使用 fused attention kernel；
- 不同 `H×W` feature map 的單獨 latency。

只有完成這組 benchmark，才能把「參數少 31.5 倍」與「實際推論快幾倍」分開正確報告。
