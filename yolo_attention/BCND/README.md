# BCND：BDCN 改良重跑工作區

此資料夾依使用者指定命名為 `BCND/`；專案程式中的正式方法名稱仍是
BDCN（Binary Distance-Codebook Normalization）。這裡只保存改良實驗的
設定、操作方式與結果索引，演算法唯一實作仍在
`src/yolo_attention/bdcn.py`，queue 唯一入口仍是
`python -m yolo_attention.cli`。

## 為什麼要改：實際發現的缺陷

BDCN 先把每列 score 轉成相對 row maximum 的 distance：

$$
d_{ij}=\max_k S_{ik}-S_{ij},
$$

再做 bucket indexing：

$$
b_{ij}=\operatorname{clip}\left(
\operatorname{round}\left(\frac{d_{ij}}{\Delta}\right),0,L-1
\right),
\qquad w_{ij}=T[b_{ij}].
$$

舊 D0 使用 16 levels、step 0.125，只能表示

$$
d_{\max}=(16-1)\times0.125=1.875.
$$

超過此範圍的 score distance 全部落入最後一格；最後權重仍約為
$e^{-1.875}=0.153$，遠大於 PWL 在 $d=8$ 的 $e^{-8}$。因此這次直接固定
$d_{\max}=8$、64 levels（$\Delta\approx0.127$），並記錄 bucket histogram、
overflow rate、last-bucket rate 與實際最大 distance。

問題不只是「範圍有點小」，而是所有超出範圍的 distance 都被 clamp 到同一個
index 15。固定 exponential table 的最後權重是

$$
T[15]=e^{-1.875}\approx0.153355,
$$

所以 $d=1.875$、$d=8$ 甚至更大的 token 都得到相同未正規化權重。原本應該
接近 $e^{-8}\approx0.000335$ 的遠距 token 被高估約 457 倍，會讓 attention
tail 過度平均，改變 denominator 與 $PV$。

Scale 與 relative-position bias 位於 bucketization 之前，確實能調整 score；
但一旦不同 distance 已經落入同一個最後 bucket，learned codebook 也只能替
它們提供同一個 weight，無法在 bucket 之後重新分開。因此只繼續訓練舊
$L=16,\Delta=0.125$ codebook，不能從結構上消除這個資訊損失。

## 這次怎麼修

本次不再增加 16/32/64 screening，而是從已完成補償的 A0（V1-BR）直接建立
一條 defect-fix branch：

$$
L=64,\qquad d_{\max}=8,\qquad
\Delta=\frac{8}{64-1}=0.126984.
$$

這樣保留接近舊版 0.125 的 bucket 解析度，同時把可表示範圍從 1.875 擴到 8。
$d=8$ 的最後 table weight 約為 $e^{-8}$；只有 $d>8$ 才列為 true overflow 並
clamp 到最後 bucket。程式同時記錄完整 validation 的 bucket histogram、true
overflow rate、last-bucket occupancy 與 observed maximum distance。

除 normalization/codebook range 外，以下內容都沿用 A0：

- Hadamard MDB Binary Q/K 與 power-of-two scale。
- Decomposed 2D relative-position bias。
- Q/K/V projection 與完整 V path。
- $PV+PE(V)$、output projection、兩層 residual、FFN 與外層 YOLO26m 結構。
- 兩個 Attention sites：`model.10.m.0.attn` 與 `model.22.m.0.1.attn`。

## 是否真的處理好：目前可驗證的證據

### 1. 舊缺陷可重現，修正版不再 tail collapse

固定同一列 scores，其 distance 為 $[0,1.875,8]$。CPU bit-true reference 得到：

| 方法 | $P(d=0)$ | $P(d=1.875)$ | $P(d=8)$ | tail 判讀 |
|---|---:|---:|---:|---|
| 舊版 $L=16,\Delta=0.125$ | 0.765281 | 0.117360 | 0.117360 | 1.875 與 8 完全 collapse |
| 修正版 $L=64,d_{max}=8$ | 0.870175 | 0.129533 | 0.000292 | $P(8)/P(1.875)\approx0.225\%$ |

這證明此次修改確實消除了造成問題的「遠距 score 全部使用同一個高權重」機制。

### 2. 真實 A0 checkpoint 的補償狀態有完整繼承

以實際 `v1-br/best.pt` 在 CPU 建立 BDCN-V2 training graph，檢查結果為：

| 檢查 | 結果 |
|---|---:|
| Converted Attention paths | 2 / 2 |
| Custom-parent state coverage | 99.259% |
| 已比對 scale/bias compensation tensors | 10 |
| Compensation maximum absolute error | 0.0 |
| 實際 codebook levels | 64 |
| 實際 resolved step | 0.126984126984 |

新增的 63 個 learned monotonic ratios 原本不可能存在於 A0，因此是預期的新參數；
其餘相容的 projection、scale、bias 與 model state 由 parent 載入。coverage 高於
repository 的 95% fail-closed 門檻。

### 3. 測試能抓到邊界錯誤

回歸測試另外發現舊 diagnostics 用 round 後 bucket index 判斷 overflow，會把
$d=8.001$ 誤報為未 overflow。現在改成直接判斷

$$
d>d_{\max}=\Delta(L-1),
$$

因此 $d=8$ 是合法最後一格，$d>8$ 才是 overflow。range-collapse 與 overflow
boundary tests 均已通過。

### 目前能與不能下的結論

- **可以確認**：range/weight collapse、設定接線、A0 補償繼承、雙 Attention
  replacement 與 overflow diagnostics 已修正；v2/R1 正式 COCO mAP 分別為
  0.483281／0.483500。
- **可以確認**：V3-FIXED zero-train 為 0.506566，比 V2-LEARN 高 0.023285；
  因此 64-level fixed discretization 本身接近 A-FINAL，主要退化來自
  unconstrained codebook learning drift。
- **可以確認**：bounded codebook learning 為 0.506469，比 fixed table 低
  0.000097，沒有量測到學習收益；V3-R1 為 0.506562，reciprocal LUT 幾乎無損。

## v3：把 fixed table 與 codebook learning 分開

v2 已排除 range-collapse bug，但 10-epoch 訓練中 mAP 長期約為 $0.483$，且
learned table 逐步比 exponential initializer 更平。這不能直接證明 BDCN
離散化本身失敗，因為 v2 沒有保存 epoch-0 fixed-table COCO 結果。

v3 因此只處理這個可識別缺口，不增加其他 normalization 候選：

```text
A0 / V1-BR
   ↓
BDCN-V3-FIXED：64-level fixed exp，0 ep
   ↓
BDCN-V3-LEARN：同一初始化，只學 bounded codebook，5 ep
   ↓
BDCN-V3-R1：沿用 learned checkpoint，reciprocal LUT evaluation
```

v3 learned table 使用

$$
\log(C_{l+1}/C_l)=-\Delta+0.01\tanh(r_l),
\qquad \Delta=8/63.
$$

這項限制不是把 table 凍結：`raw_deltas` 仍有非零、有限梯度，且
`bdcn_codebook` scope 仍只解凍這一張跨兩處 Attention 共用的 table。
它只是防止 detection loss 把 tail 再推回過度平均。初始化時
$C_{63}=e^{-8}\approx0.000335$；理論最平極端仍小於 $0.001$，遠低於舊版
$e^{-1.875}\approx0.153$。

V3-FIXED 回答「不訓練的 64-level BDCN 本身多準」；V3-LEARN 與它的差值
才回答「學 codebook 是否有益」。若 learned 不高於 fixed，就保留 fixed
table 或放棄 BDCN，不能再把改善歸因於 codebook learning。


## 完整成本結論

固定兩個 Attention sites、$T=400,H=4,d_k=32,d_v=64$ 的 reference shape：

| 方法 | Profiler arithmetic | 含 bias 保守值 | Memory proxy | 相對原始區域 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| 原始 FP QK + exact Softmax + $PV$ | 129,286,400 | 129,286,400 | 2,764,800 | reference | 0.517998 |
| A-FINAL / SHIFT | 86,790,400 | 89,350,400 | 2,764,800 | -32.87%／-30.89% | 0.506357 |
| 舊 BDCN R1，$L=16$ | 90,070,400 | 92,630,400 | 4,556,800 | -30.33%／-28.35% | 0.499812 |
| BDCN-V2 learn，$L=64$ | 113,030,400 | 115,590,400 | 14,387,200 | -12.57%／-10.59% | 0.483281 |
| BDCN-V2 R1，$L=64$ | 113,008,000 | 115,568,000 | 14,387,200 | -12.59%／-10.61% | 0.483500 |
| BDCN-V3 fixed，$L=64$ | 113,030,400 | 115,590,400 | 14,387,200 | -12.57%／-10.59% | 0.506566 |
| BDCN-V3 learn，$L=64$ | 113,030,400 | 115,590,400 | 14,387,200 | -12.57%／-10.59% | 0.506469 |
| BDCN-V3 R1，$L=64$ | 113,008,000 | 115,568,000 | 14,387,200 | -12.59%／-10.61% | 0.506562 |

Profiler arithmetic 為所有 normalization candidates 共用的正式比較範圍；它沒有
計入 decomposed 2D bias。因 BDCN-V2 直接繼承 V1-BR bias，保守值另加

$$
C_{bias}^{stream}=2E=2{,}560{,}000
$$

次 additions。表中的兩個 reduction 因此分別代表「正式 profiler」與「保守
計入 bias」；不可只挑較大的數字而省略假設。

成本增加的原因不是 Binary QK 或 $PV$ 改回浮點，而是 fused bucket-$PV$ 的
codebook weighting term 與 $L$ 成正比：

$$
C_{table}=AHTLd_v.
$$

由 $L=16$ 增至 64，使 BDCN R1 arithmetic proxy 增加 22.938M，memory proxy
由 4.557M 增至 14.387M。相較 A-FINAL，BDCN-V2 R1 的 arithmetic proxy 高約
30.21%；相較原始 Attention 分析區域，未計 bias 時低約 12.59%，保守計入
bias 時低約 10.61%。因此這是一個用較高
codebook/memory 成本換取正確 score range 與可能精度恢復的設計，不是無代價
最佳化。

YOLO26m baseline 為 75.4 GFLOPs；兩個 Attention 的原始
$QK+normalization+PV$ 分析區域只約占其 0.334%，真正可修改的
$QK+normalization$ 上限約占 0.117%。不同 proxy units、FP MAC、packed
XNOR-popcount、add/sub 與 LUT 不能直接相加成實測 GFLOPs，所以 12.59% 或
32.87% 都是局部 analytical reduction，不能宣稱為整個 YOLO26m 的 latency、
power 或 energy reduction。

參數方面，BDCN-V2/V3 相對 A0 都只新增一個 global 64-level monotonic
codebook，即 63 個 trainable values；若只以 FP16 parameter payload 估算是
126 bytes，不含 buffer、alignment、optimizer state 與 checkpoint metadata。
V3 learned 的 best checkpoint 已保留在標準 artifact 位置；checkpoint file 約
55.30 MiB。它包含模型 metadata，不能直接當成純 parameter payload。

跨方法公式、全模型占比與模型大小的 canonical 整合版見
[`reports/COMPUTE_AND_SIZE.md`](../reports/COMPUTE_AND_SIZE.md)。
V2/V3 的完整缺陷、實作、訓練、結果與判定見
[`reports/BDCN_V3_REPORT.md`](../reports/BDCN_V3_REPORT.md)。

## 資料夾內容

```text
BCND/
├── README.md
├── RESULTS.md
└── configs/
    ├── README.md
    ├── bdcn-v2.yaml
    ├── bdcn-v2-r1.yaml
    ├── bdcn-v3-fixed.yaml
    ├── bdcn-v3-learn.yaml
    └── bdcn-v3-r1.yaml
```

## 執行流程

```bash
../.venv/bin/python -m yolo_attention.cli queue append-bdcn-v2 --json
../.venv/bin/python -m yolo_attention.cli queue validate --json
../.venv/bin/python -m yolo_attention.cli queue run --execute --json
```

v2 全部完成後，追加不覆寫歷史的 v3：

```bash
../.venv/bin/python -m yolo_attention.cli queue append-bdcn-v3 --json
../.venv/bin/python -m yolo_attention.cli queue validate --json
../.venv/bin/python -m yolo_attention.cli queue run --execute --json
```


流程直接從已完成 scale/bias 補償的 A0（V1-BR）做 10 epoch global
learned-codebook recovery，最後評估 reciprocal-LUT denominator；不增加
16/32/64 screening 或 winner selection。兩個 Attention site 都由
既有 fail-closed integration 同時替換。舊 runs 不會被覆寫。

大型 checkpoint 與 metrics 仍寫到 `artifacts/runs/<run_id>/`，不放在本
資料夾。量化仍鎖定，不會由此流程啟動。

## 外部接線

`BCND/` 擁有本次 branch 的 configs、方法說明與結果；共用實作不複製：

- `src/yolo_attention/bdcn.py`：distance/codebook/fused bucket-PV 唯一 source。
- `src/yolo_attention/evaluation.py`：將全 validation diagnostics 寫入結果。
- `src/yolo_attention/queue_workflow.py`：從 A0 追加 v2/v3 immutable jobs。
- `reports/COMPUTE_AND_SIZE.md`：跨方法與全模型占比的整合報告。
