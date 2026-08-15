# Attention 運算量節省、全模型占比與模型大小（整合版）

日期：2026-08-15

## 1. 最重要的結論

本研究只改 YOLO26m 的兩個 Attention sites 的 Q/K score 與 normalization，
不改 QKV/output projection、$PV+PE(V)$、FFN、Detect 或外層網路。固定 reference
shape 下，原始被分析的 $QK+normalization+PV$ 路徑是 129.286M proxy units；
A-FINAL 是 86.790M，局部減少 42.496M（32.87%）。若保守加入 decomposed
2D bias 的 2.560M streaming additions，則是 89.350M，局部仍減少 30.89%。

但局部 32.87% 不能寫成整個模型加速 32.87%。本機 Ultralytics 對 baseline
顯示 75.4 GFLOPs；其 FLOPs 規則以一個 MAC 約計 2 FLOPs。以此只做量級對照：

| 範圍 | 約當量 | 占 75.4 GFLOPs | 能否直接宣稱加速 |
|---|---:|---:|---|
| 兩處 $QK+PV$ | 245.760 MFLOPs | 0.326% | 否，尚未含/對齊異質運算 |
| 加上 exact normalization proxy 的整個分析區域 | 252.166M | **約 0.334%** | 否，只是量級 proxy |
| 本研究可修改的 $QK+normalization$ 上限 | 88.326M | **約 0.117%** | 否，是可節省區域上限 |
| 保留的 $PV$ | 163.840 MFLOPs | 0.217% | 本研究沒有節省 |

因此，這次工作的價值是把特定 Attention kernel 換成 packed binary、add/sub、
shift/LUT 友善的資料流，而不是讓整個 convolution-dominated YOLO26m 的理論
GFLOPs 大幅下降。沒有 GPU custom kernel、HLS 或 FPGA measurement 前，不能把
proxy reduction 換成 latency、power 或 energy reduction。

本地精度代價是 mAP50-95 從 0.517998 降至 0.506357，即下降 0.011641
（1.164 AP）；參數只增加 1,012（0.00462%）。

## 2. 我做了什麼，為何會省

| 原始操作 | 修改 | 節省來源 | 保留/新增代價 |
|---|---|---|---|
| FP QK dot product | Identity + normalized Hadamard MDB，XNOR + popcount | 40.960M FP MAC 改為 2.560M packed word ops + 1.024M add/sub | 兩個 basis 與 Hadamard butterfly |
| exact Softmax proxy | PoT/SHIFT normalization | 6.406M 降至 1.286M proxy，局部 normalization 減 79.92% | 仍保持逐列 normalization |
| 無相對位置補償 | decomposed 2D relative bias | 補回 binary score 的空間資訊 | 1,008 bias parameters；保守計 2.560M additions |
| Dense $PV$ | 完整保留 | 無節省，確保 value path 與官方結構不變 | 81.920M MAC，是 A-FINAL 局部 proxy 的 94.4% |
| $PE(V)$、projection、FFN、residual | 完整保留 | 無節省 | 維持 YOLO26m 功能邊界 |

數學上的局部差值為：

$$
C_{original}=40{,}960{,}000+6{,}406{,}400+81{,}920{,}000
=129{,}286{,}400,
$$

$$
C_{A\text{-}FINAL}=2{,}560{,}000+1{,}024{,}000+1{,}286{,}400
+81{,}920{,}000=86{,}790{,}400,
$$

$$
\frac{C_{original}-C_{A\text{-}FINAL}}{C_{original}}=32.87\%.
$$

加入 bias streaming additions 後：

$$
C_{A\text{-}FINAL+bias}=89{,}350{,}400,
\qquad reduction=30.89\%.
$$

FP MAC、packed word op、add/sub 與 shift 不是同一成本單位，所以這是固定假設下
的 analytical event-count proxy，不是硬體 cycle model。

## 3. 分析範圍與 reference shape

本章只計算兩個被替換的 YOLO26m Attention sites，不把 backbone、neck、Detect、QKV/output/PE convolution 或 FFN 混入 Attention score proxy。固定 batch 1 reference shape：

| 符號 | 意義 | 數值 |
|---|---|---:|
| $A$ | Attention sites | 2 |
| $T$ | 每處 tokens | 400 |
| $H$ | heads | 4 |
| $d_k$ | key dimension | 32 |
| $d_v$ | value dimension | 64 |
| $W$ | packed word bits | 32 |
| $L$ | BDCN levels | 16 |

這些 shape 已由兩個 checkpoint module 核對。所有數字是演算法 proxy，不是 GPU FLOPs、latency、FPGA cycles、BRAM/DSP 或 energy。

## 4. 基本公式

Score entries 與 rows：

$$E=AHT^2=2\times4\times400^2=1{,}280{,}000$$

$$R=AHT=2\times4\times400=3{,}200$$

### FP 與 Binary QK

FP QK MAC reference：

$$C_{QK}^{FP}=E d_k=40{,}960{,}000$$

單一 binary basis 的 packed XNOR+popcount word operations：

$$C_{QK}^{bin,1}=E\left\lceil\frac{d_k}{W}\right\rceil=1{,}280{,}000$$

Hadamard MDB 計算 Identity 與 Hadamard 兩個 basis：

$$C_{QK}^{MDB}=2C_{QK}^{bin,1}=2{,}560{,}000$$

兩處 Q/K Walsh-Hadamard transform 的 butterfly add/sub：

$$C_H=2AHTd_k\log_2(d_k)=1{,}024{,}000$$

Packed word operation 與 FP MAC 不是等價能耗單位；必須在指定硬體後才能換算 cycle/energy。

### 保留的 dense $PV$

$$C_{PV}=E d_v=81{,}920{,}000$$

它占 SHIFT 總 proxy：

$$\frac{81{,}920{,}000}{86{,}790{,}400}=94.4\%$$

因此目前簡化的是 QK 與 normalization，不是整個 YOLO26m 或 dense $PV$。

### Dense normalization

每個 score entry 使用固定比較權重，再加每列 reduction/normalization 的 $2R$：

$$C_{norm}=cE+2R$$

Exact、PWL、LUT/HardSigmoid/Multimax、Shift/ReLU 的 $c$ 分別為 5、3、2、1。這是同假設下的排序 proxy，不是 operator-level cycle model。

### BDCN

BDCN fused bucket-$PV$ 不 materialize dense $P$，但計入：

$$C_{bucket}=E d_v=81{,}920{,}000$$

$$C_{table}=AHTLd_v=3{,}276{,}800$$

$$C_{den}=c_{den}R$$

R0 exact、R1 reciprocal LUT、R2 PoT shift 的 $c_{den}$ 分別為 10、3、1。BDCN normalization proxy 為 $E+C_{table}+C_{den}$。

## 5. 各方法修正後計算量

| 方法 | Binary word ops | Hadamard add/sub | Normalization | Value path | 總 arithmetic proxy | Memory proxy |
|---|---:|---:|---:|---:|---:|---:|
| Exact | 2,560,000 | 1,024,000 | 6,406,400 | 81,920,000 | 91,910,400 | 2,764,800 |
| LUT | 2,560,000 | 1,024,000 | 2,566,400 | 81,920,000 | 88,070,400 | 2,764,800 |
| PWL | 2,560,000 | 1,024,000 | 3,846,400 | 81,920,000 | 89,350,400 | 2,764,800 |
| SHIFT / A-FINAL | 2,560,000 | 1,024,000 | 1,286,400 | 81,920,000 | **86,790,400** | 2,764,800 |
| BDCN R0 | 2,560,000 | 1,024,000 | 4,588,800 | 81,920,000 | 90,092,800 | 4,556,800 |
| BDCN R1 | 2,560,000 | 1,024,000 | 4,566,400 | 81,920,000 | **90,070,400** | 4,556,800 |
| BDCN R2 | 2,560,000 | 1,024,000 | 4,560,000 | 81,920,000 | 90,064,000 | 4,556,800 |
| BDCN v2 learn（$L=64$） | 2,560,000 | 1,024,000 | 27,526,400 | 81,920,000 | **113,030,400** | 14,387,200 |
| BDCN v2 R1（$L=64$） | 2,560,000 | 1,024,000 | 27,504,000 | 81,920,000 | **113,008,000** | 14,387,200 |

SHIFT 相對 Hadamard + exact normalization：

$$1-\frac{86{,}790{,}400}{91{,}910{,}400}=5.57\%$$

舊版 profile 只把 MDB 算成一個 binary basis，少算 1,280,000 word ops；本次已在 profiler 與 regression test 修正。此常數在同階段候選間相同，不改變既有 winner 排名，但舊總量與百分比不得再引用。

Dense memory proxy：

$$M_{dense}=2E+AHTd_v=2{,}764{,}800$$

BDCN memory proxy：

$$M_{BDCN}=E+AHTLd_v=4{,}556{,}800$$

這是元素存取 proxy，不含 datatype bytes、cache reuse、tiling 或 off-chip burst。BDCN 雖不存 dense $P$，16-level bucket buffer 使此 proxy 增加 64.8%。

BDCN v2 直接使用 $L=64,d_{max}=8$，令 $\Delta\approx0.127$；相對舊版
$L=16,d_{max}=1.875$，它保留近似 bucket 解析度但擴大覆蓋範圍。$C_{table}$
與 memory proxy 會隨 $L$ 線性增加，因此報告必須同時呈現精度恢復與成本代價。
新結果完成前標為 `not_run`，詳見 [BCND 工作區](../BCND/README.md)。

因此 defect fix 並非免費：BDCN v2 R1 相對舊 $L=16$ R1 增加 22.938M
arithmetic proxy，memory proxy 從 4.557M 增至 14.387M。另一方面，它相對
原始 129.286M Attention 分析區域仍減少 16.278M，約 12.59%。是否值得採用
必須看本次 10-epoch recovery 的 mAP 與 overflow 是否實際改善；尚未完成前
不能宣稱它優於 A-FINAL。

上述 BDCN v2 profiler 數字和其他候選一樣未含 decomposed bias。因 v2 直接
繼承 V1-BR，保守加入 $2E=2.560$M bias additions 後，learn/R1 分別為
115.590M／115.568M；R1 相對原始分析區域的 reduction 由 12.59% 改為
10.61%。兩組數字都必須連同假設一起報告。

![Accuracy versus corrected cost](figures/accuracy-cost.svg)

完整數據見 [compute_and_size.csv](compute_and_size.csv)。

## 6. 參數量與 checkpoint 大小

| 模型 | Parameters | Checkpoint | Parameter payload | Buffer payload |
|---|---:|---:|---:|---:|
| B26-FP | 21,896,248 | 42.21 MiB | 41.76 MiB | 0.108 MiB |
| H-SCR | 21,896,252 | 61.74 MiB | 41.76 MiB | 0.108 MiB |
| W-DIR | 21,896,252 | 61.75 MiB | 41.76 MiB | 0.108 MiB |
| V1-BR / A0 | 21,897,260 | 68.26 MiB | 41.77 MiB | 0.108 MiB |
| N1-SHIFT / A-FINAL | 21,897,260 | 68.26 MiB | 41.77 MiB | 0.108 MiB |
| D1-SHARED-10 | 21,897,275 | 55.29 MiB | 41.77 MiB | 0.109 MiB |

A-FINAL 比 B26-FP 多 1,012 parameters。checkpoint 檔案另含 class/config/metadata，不能當參數 RAM。A-FINAL parameters 為 FP16：

$$21{,}897{,}260\times2=43{,}794{,}520\ \text{bytes}=41.77\ \text{MiB}$$

![Retained checkpoint sizes](figures/checkpoint-size.svg)

原始 bytes 見 [checkpoint_sizes.csv](checkpoint_sizes.csv)。

## 7. 未執行量化時的理論 weight payload

若假設 A-FINAL weights 全部均勻 packed，忽略 scale、zero-point、alignment、bias、buffer 與 mixed precision：

| Weight bits | 理論 packed payload |
|---:|---:|
| 32 | 83.53 MiB |
| 16 | 41.77 MiB |
| 8 | 20.88 MiB |
| 6 | 15.66 MiB |
| 5 | 13.05 MiB |
| 4 | 10.44 MiB |

$$S_{weight}=\left\lceil\frac{N_{param}b}{8}\right\rceil$$

這些只是容量分析，不是已完成的量化 checkpoint，也不含 activation、accumulator 或 LUT。Optional Phase Q 仍未執行。

## 8. 可安全引用的結論

可以寫：

> 在兩個 YOLO26m Attention sites、固定 400-token reference shape 下，Hadamard
> MDB Binary QK 與 PoT-SHIFT normalization 將被修改的 QK-normalization-PV
> path analytical proxy 從 129.286M 降至 86.790M，減少 32.87%；保守計入
> decomposed bias additions 後仍減少 30.89%。該分析區域約為全模型 75.4
> GFLOPs 背景的 0.334%，因此局部比例不代表整體 latency 或 energy reduction。

不能寫「YOLO26m 整體 GFLOPs、速度或功耗降低 32.87%」。目前尚未量測 kernel
latency、吞吐、功率或 FPGA cycles。

機器可讀數據見 [operation_savings.csv](operation_savings.csv)、
[compute_and_size.csv](compute_and_size.csv) 與
[checkpoint_sizes.csv](checkpoint_sizes.csv)。CPU reference 可用：

```bash
python -m yolo_attention.cli profile
```
