# 原始 Attention 與 A-FINAL 運算量節省報告

日期：2026-08-15

## 1. 結論摘要

本研究沒有替換整個 YOLO26m，只修改兩個 Attention 的 Q/K score 與 normalization path。以兩處各 400 tokens、4 heads、$d_k=32$、$d_v=64$、32-bit packed word 的固定 reference shape 計算：

| 比較項目 | 原始 YOLO26m Attention | A-FINAL | 差異 |
|---|---:|---:|---:|
| QK | 40,960,000 FP MAC | 2,560,000 packed XNOR-popcount word ops + 1,024,000 Hadamard add/sub | 移除 FP QK MAC；同一 proxy 下減少 91.25% |
| Normalization | 6,406,400 exact-Softmax proxy ops | 1,286,400 PoT-SHIFT proxy ops | -5,120,000，-79.92% |
| $PV$ | 81,920,000 MAC | 81,920,000 MAC | 不變 |
| 修改區域 arithmetic proxy | 129,286,400 | 86,790,400 | **-42,496,000，-32.87%** |
| 加入 decomposed bias 的保守 proxy | 129,286,400 | 89,350,400 | **-39,936,000，-30.89%** |
| Parameters | 21,896,248 | 21,897,260 | +1,012，+0.00462% |
| 本地 COCO mAP50-95 | 0.517998 | 0.506357 | -0.011641，即 -1.164 AP |

最重要的解讀是：A-FINAL 在被修改的 Attention reference path 上，把 aggregate arithmetic proxy 降低 32.87%；若把 decomposed 2D bias 的兩次 streaming addition 也保守計入，仍降低 30.89%。這不是整個 YOLO26m 的實測加速比例。

## 2. 為什麼不能直接說整個模型省 32.87%

本地 Ultralytics 對原始 `yolo26m.pt` 顯示：

```text
YOLO26m: 280 layers, 21,896,248 parameters, 75.4 GFLOPs
```

對 A-FINAL checkpoint 顯示：

```text
YOLO26m: 300 layers, 21,897,260 parameters, 75.4 GFLOPs
```

通用 profiler 仍顯示 75.4 GFLOPs，因為它不會把 packed XNOR、popcount、Hadamard butterfly 或 shift normalization 正確換算成指定硬體的 cycle/energy；保留的 convolution、FFN、Detect、$PV+PE(V)$ 等大部分網路也沒有改變。因此：

- 75.4 GFLOPs 是全模型的通用浮點背景值。
- 32.87% 是兩個 Attention 被分析區域的 heterogeneous arithmetic proxy reduction。
- 在沒有 GPU kernel、HLS 或 FPGA measurement 前，不得把 32.87% 寫成整體 latency、energy 或 GFLOPs reduction。

## 3. Reference shape

| 符號 | 意義 | 數值 |
|---|---|---:|
| $A$ | 被修改的 Attention sites | 2 |
| $T$ | 每處 tokens | 400 |
| $H$ | heads | 4 |
| $d_k$ | key dimension | 32 |
| $d_v$ | value dimension | 64 |
| $W$ | packed word bits | 32 |

Score entries 與 query rows：

$$
E=AHT^2=2\times4\times400^2=1{,}280{,}000,
$$

$$
R=AHT=2\times4\times400=3{,}200.
$$

## 4. 原始 YOLO26m Attention 成本

原始 QK 使用 FP dot product：

$$
C_{QK}^{FP}=Ed_k
=1{,}280{,}000\times32
=40{,}960{,}000\ \text{FP MAC}.
$$

Exact Softmax 使用既有固定比較 proxy：每個 score entry 為 5 units，每列另計 reduction 與 normalization：

$$
C_{norm}^{exact}=5E+2R
=6{,}406{,}400.
$$

保留的 dense value path：

$$
C_{PV}=Ed_v
=1{,}280{,}000\times64
=81{,}920{,}000\ \text{MAC}.
$$

因此原始被分析區域的 aggregate proxy 是：

$$
C_{original}
=C_{QK}^{FP}+C_{norm}^{exact}+C_{PV}
=129{,}286{,}400.
$$

這裡不包含兩邊共同保留的 QKV projection、$PE(V)$、output projection、FFN、residual 與 C2PSA/C3k2 外層，因為它們不影響這次方法差值。

## 5. A-FINAL 做了什麼

### 5.1 Hadamard matched-dual-basis Binary QK

A-FINAL 同時計算 Identity 與 Hadamard 兩個 binary bases。每個 basis 使用 32-bit packed XNOR-popcount：

$$
C_{QK}^{bin,1}
=E\left\lceil\frac{d_k}{W}\right\rceil
=1{,}280{,}000.
$$

兩個 bases：

$$
C_{QK}^{MDB}=2C_{QK}^{bin,1}=2{,}560{,}000
\ \text{packed word ops}.
$$

Q/K Hadamard butterfly 額外需要：

$$
C_H=2AHTd_k\log_2d_k=1{,}024{,}000
\ \text{add/sub}.
$$

所以 Binary QK 並不是零成本；它以 2.56M packed word ops 和 1.024M add/sub，取代 40.96M FP MAC。只比較 packed QK word ops，dual basis 相對 FP MAC count 是 16 倍縮減；把 Hadamard add/sub 一起放進現行 aggregate proxy，QK 區域減少：

$$
40{,}960{,}000-(2{,}560{,}000+1{,}024{,}000)
=37{,}376{,}000,
$$

$$
\frac{37{,}376{,}000}{40{,}960{,}000}=91.25\%.
$$

FP MAC、packed word op 與 add/sub 的 cycle/energy 不等價；91.25% 只能稱為同一 analytical proxy 下的事件數減少。

### 5.2 Power-of-two SHIFT normalization

A-FINAL 使用 PoT/SHIFT normalization proxy：

$$
C_{norm}^{shift}=E+2R=1{,}286{,}400.
$$

相對 exact Softmax：

$$
\Delta C_{norm}
=6{,}406{,}400-1{,}286{,}400
=5{,}120{,}000,
$$

$$
\frac{5{,}120{,}000}{6{,}406{,}400}=79.92\%.
$$

### 5.3 保留 $PV+PE(V)$ 與外層結構

$PV$ 完全保留，仍是 81,920,000 MAC，占 A-FINAL core proxy：

$$
\frac{81{,}920{,}000}{86{,}790{,}400}=94.4\%.
$$

因此下一個真正的大型硬體瓶頸是 value path，而不是 normalization。這次研究不能宣稱已把整個 Attention 二值化。

### 5.4 Decomposed 2D bias 的新增成本

兩個 Attention 共新增 1,008 個 decomposed relative-bias parameters；另外 4 個參數來自其他 Attention scale/fusion state，因此 A-FINAL 總增量是 1,012 parameters。

目前正式 profiler 為了和所有 normalization variants 保持同一範圍，沒有把 relative-bias lookup/add 算進 86,790,400。若採 streaming 實作，每個 score entry 保守計入：

1. $B_y+B_x$ 一次 addition。
2. $S+B$ 一次 addition。

則額外成本為：

$$
C_{bias}^{stream}=2E=2{,}560{,}000\ \text{additions}.
$$

保守總 proxy：

$$
C_{A\text{-}FINAL+bias}
=86{,}790{,}400+2{,}560{,}000
=89{,}350{,}400.
$$

相對原始路徑仍減少：

$$
129{,}286{,}400-89{,}350{,}400
=39{,}936{,}000,
$$

即 30.89%。若硬體預先形成或重用 bias，實際 addition/traffic 會不同，必須在硬體 mapping 後重算。

## 6. 總節省與代價

不含 bias streaming additions 的既有正式 proxy：

$$
C_{A\text{-}FINAL}
=2{,}560{,}000+1{,}024{,}000+1{,}286{,}400+81{,}920{,}000
=86{,}790{,}400.
$$

$$
\Delta C
=129{,}286{,}400-86{,}790{,}400
=42{,}496{,}000.
$$

$$
\text{reduction}=32.87\%.
$$

精度與容量代價：

- mAP50-95：0.517998 → 0.506357，下降 1.164 AP，保留約 97.75%。
- Parameters：增加 1,012，約 +0.00462%。
- FP16 parameter payload 理論增量只有 2,024 bytes，約 1.98 KiB。
- A-FINAL checkpoint 比 baseline 檔案大，主要是 training/config/metadata，不能把 checkpoint file size 當部署 weight RAM。
- 尚未執行 INT quantization、kernel benchmark、HLS 或上板，因此沒有 measured latency、throughput、power 或 energy saving。

## 7. 可安全使用的論文表述

可以寫：

> 在兩個 YOLO26m Attention sites、固定 400-token reference shape 下，Hadamard MDB Binary QK 與 PoT-SHIFT normalization 將被修改 QK-normalization-PV path 的 analytical arithmetic proxy 從 129.286M 降至 86.790M，減少 32.87%；保守計入 decomposed bias additions 後仍減少 30.89%。本地 mAP50-95 下降 1.164 AP。

不能寫：

> YOLO26m 整體 GFLOPs、速度或功耗降低 32.87%。

因為目前沒有把 heterogeneous binary operations 映射成實際硬體成本，也尚未量測完整模型 latency/energy。

## 8. 證據與重算入口

- 公式與所有 normalization/BDCN 候選：[COMPUTE_AND_SIZE.md](COMPUTE_AND_SIZE.md)
- 機器可讀差值：[operation_savings.csv](operation_savings.csv)
- 各候選原始 profile：[compute_and_size.csv](compute_and_size.csv)
- 精度與方法結果：[REPORT.md](REPORT.md)
- profiler 實作：`src/yolo_attention/profiling.py`

重算 reference profile：

```bash
python -m yolo_attention.cli profile
```

該命令只做 CPU analytical calculation，不會啟動 GPU training。
CLI 顯示的是單一 Attention site 的 `estimate_operations` reference counts；本報告依正式研究邊界乘以兩處 Attention。variant 的兩處合計與 selected path 則以各 run 的 `profiles/analytical.json` 為準。
