# BinaryQK 少量 PoT scale／codebook 的精度與硬體取捨

日期：2026-09-03

## 結論先行

若部署端不能接受昂貴的 per-image dynamic scale，最合理的第一個正式候選不是「per-image global K-codebook」，而是 **`G=4` fixed channel-group PoT scale**：scale 與 group assignment 都在離線校正／訓練後固定，推論時沒有 magnitude reduction、nearest-code selector 或 activation index；代價是每個 QK pair 要保留 4 組 partial popcount，硬體 kernel 必須證明這個代價仍划算。

若先以精度保真為目的，**`K=8` per-token PoT codebook** 比 global scale 更可能恢復 token 間的 magnitude 差異，但它不是免費的「8 個常數」：每張影像仍要對 12,800 個 Q/K token-scale 項做 reduction 與 selector，並讓 2,560,000 個 basis-pair contribution 讀 index、合成 exponent、做 shift／alignment。它適合當 fidelity 上界或硬體可接受時的候選，不應直接宣稱是部署解。

本地單任務 YOLO26 中，dynamic scale 相對 PoT 只有 **`+0.000505537 mAP`**（`0.507457458 - 0.506951921`）；FP 到 PoT 的 gap 是 `0.011045914`，所以完整 global dynamic 也只關閉約 **4.58%** 的這段缺口。因此沒有證據顯示任何 4/8-scale codebook 可補回約 `0.011`，更不能把跨架構論文結果當作本專案的增益保證。

## 1. 證據邊界

### 1.1 本地正式 Full35

【本地證據】Full35 正式設定使用 `basis: hadamard`、`scale_mode: power_of_two`、bit-true PWL normalization，且 `use_ste: false`，見 [`bittrue-pwl-final.yaml` 第 1–11 行](<../../../yolo_combine/final/full35/source_bundle/configs/attention/bittrue-pwl-final.yaml#L1-L11>)。

【本地證據】正式 [`binary_basis.py` 第 35–43、87–113、126–200 行](<../../../yolo_combine/final/full35/source_bundle/code/yolo_attention/binary_basis.py#L35-L43>) 顯示：

- Hadamard/T5 都有兩個 basis；
- dynamic magnitude 是對 `[B,H,D,N]` 的 `D,N` 兩軸做 `mean(abs(.))`，語義是 per-image／per-head／per-basis global scale，而不是 per-token scale；
- PoT 固定係數用 `sign(x) * 2^round(log2(abs(x)))`；
- 正式 Full35 將觀察到的 combined coefficient 凍結為 per-head／per-basis fixed coefficient。

【本地證據】Full35 的 J3 best-joint 正式指標為 COCO box `0.49802233647`、person `0.62038148842`、BBAT box `0.63003552176`、pose `0.90371717411`，其餘 ball/bat 分項與 provenance 見 [`RELEASE_STATUS.json`](<../../../yolo_combine/final/full35/RELEASE_STATUS.json>)。這組是 Full35 多目標正式結果；下節的 `0.507...` 是先前單任務 YOLO26 scale ablation，兩者不可混成同一實驗 lineage。

【本地證據】正式 profiler 在 [`profiling.py` 第 13–26、47–81 行](<../../../yolo_combine/final/full35/source_bundle/code/yolo_attention/profiling.py#L13-L26>) 定義 packed-word 計數，預設代表形狀為 `N=400, H=4, D=32`，並以兩個 attention sites 計數。Full35 的 `bool_tiled` 是 reference/software backend；現有發布沒有 custom-kernel latency、FPGA 資源或 ASIC cycle 證據，故本文只報 operation／storage 上界，不把它寫成真實加速比。

### 1.2 本地 scale ablation

【本地證據】[`comparison.csv`](<../../../yolo_attention/reports/comparison.csv>) 記錄 `V1-DYN=0.507457458`、`V1-P2=0.506951921`，差值為：

```text
dynamic - PoT = 0.507457458 - 0.506951921
              = 0.000505537 mAP
```

同表 `B26-FP=0.517997835`，故 `FP - PoT = 0.011045914`、`FP - dynamic = 0.010540377`；dynamic 的 `0.000505537` 只關閉 `0.000505537/0.011045914 ≈ 4.58%` 的 FP–PoT 缺口。另有 `A-FINAL=0.506356605`，但它包含後續 normalization 選擇，不能當純 scale 分母。[`REPORT.md`](<../../../yolo_attention/reports/REPORT.md>) 選 PoT 的理由是落在 tie-band 內的複雜度勝者，不是它具有最高 mAP。這只能說 global dynamic 的額外收益很小；不能推出 per-token 或 channel-group 一定也只值 `0.0005`。

### 1.3 外部一手證據

【論文證據】[BinaryAttention 原論文](https://arxiv.org/html/2603.09582#S4.SS2)以 scaled binary Q/K 緩解 magnitude loss，並指出只保留 sign 會讓 attention 變平；其硬體加速依賴特製 binary/INT8 kernel，而不是把 `sign()` 接一般浮點 matmul。[作者官方實作](https://github.com/EdwardChasel/BinaryAttention/blob/main/models.py)的 Q/K 形狀為 `[B,H,N,D]`，對 `N,D` 平均；這和本地 `[B,H,D,N]` 對 `D,N` 平均語義相同，都是 per-sample/head global scale。它不是本文提出的 per-token codebook。

【論文證據】[LQ-Nets 原論文](https://openaccess.thecvf.com/content_ECCV_2018/html/Dongqing_Zhang_Optimized_Quantization_for_ECCV_2018_paper.html)及[作者官方 repository](https://github.com/microsoft/LQ-Nets)支持「量化值／basis 可由資料學得並以 bit operations 相容的形式表示」這個一般方向；研究對象是 CNN 權重／activation，沒有證明本文三種 QK scale 會提高 YOLO mAP。

【論文證據】[DeepShift 原論文](https://openaccess.thecvf.com/content/CVPR2021W/MAI/papers/Elhoushi_DeepShift_Towards_Multiplication-Less_Neural_Networks_CVPRW_2021_paper.pdf)以 signed power-of-two 讓乘法可改成 bit shift。這支持 PoT datapath 的形式，但不代表 variable shift、index fetch、reduction 與 group accumulation 沒有成本。

【官方規格】NVIDIA [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)定義了 `popc` 以及 b1 matrix operation 的 XOR/AND + population count 語義。實際 instruction shape、packing、padding 與 tile occupancy 仍受硬體限制；下列「一個 32-bit word popcount」是本地 profiler 的工作量單位，不等於一個 Tensor Core instruction 或一個 cycle。

## 2. 統一符號與目前基線

令：

```text
D = 32 channels/head
H = 4 heads
N = 400 tokens
S = 2 attention sites
B = 2 Hadamard bases
QK = 2 streams（Q 與 K）
batch = 1
```

因此：

```text
final token pairs P = S·H·N²       = 1,280,000
basis-pair terms E  = S·B·H·N²     = 2,560,000
token-scale slots T = S·B·H·2·N    = 12,800 / image
global-scale slots R= S·B·H·2      = 32 / image
Q/K values scanned  = T·D           = 409,600 / image
```

【本專案推論】目前 fixed PoT dual-basis 基線，每個 site/head/basis 一個 combined coefficient，共 `S·H·B=16` 個固定係數；每個 basis pair 做一個完整 D=32 binary dot，兩個 basis 再相加。以下各方案都保留 1-bit Q/K，但 scale 粒度與 datapath 不同。

### 2.1 `1/sqrt(32)` 的 PoT fold 條件

【本地證據】正式 score 在 `binary_basis.py` 第 181–200 行先令 `attention_scale=D^-0.5`，再乘入每個 basis coefficient。`D=32` 時 `1/sqrt(32)=2^-2.5`，不是整數 exponent 的 PoT。

【本專案推論】若要求純 integer shift，可行條件只有：（1）把 `1/sqrt(D)`、gamma 與 Q/K scale 一起離線 fold 成 combined coefficient，再把整體係數重新量化為 PoT；這不是精確 fold，會有 rounding error；或（2）保留共同的固定 `sqrt(2)` multiplier，因 `2^-2.5=sqrt(2)*2^-3`，此時不再是純 shift。若下一層 score quantizer 是線性、尺度固定，且 fold 不跨越 softmax/PWL 的 clipping、rounding 或飽和邊界，也可把這個共同常數吸收到其固定輸入 scale。Hadamard 的正 normalization 在 fixed-scale inference 可因不改 sign 而省略，但它的幅度效果必須已被 calibration/frozen coefficient 吸收；不能因 scale codebook 是 PoT 就宣稱整條 score path multiplication-free。

## 3. 三個方案的公式與成本

### A. K=4/8 per-token PoT codebook

對每個 site `s`、basis `b`、head `h`、token `i/j`：

```text
αq[s,b,h,i] = 2 ^ e[cq(s,b,h,i)]
αk[s,b,h,j] = 2 ^ e[ck(s,b,h,j)]
score[s,h,i,j]
  = Σ_b 2 ^ (e[cq] + e[ck]) · binary_dot(q̂[s,b,h,i], k̂[s,b,h,j])
```

其中 codebook exponent `e[0:K]` 可離線學得／校正；但只要 `c(.)` 依目前影像內容決定，runtime 就仍要：先對每個 token 的 32 channels 求 magnitude，再做 nearest-bin／threshold selector。

| 成本項目 | K=4 | K=8 |
|---|---:|---:|
| context-local exponent table 上界（site/basis/head/QK 各自一表）`32K`；若全 context 嚴格共用，僅 `K` entries | 128（共用時 4） | 256（共用時 8） |
| runtime index 數 `T` | 12,800 | 12,800 |
| 最小 index storage | 25,600 bit = 3,200 B（2-bit） | 38,400 bit = 4,800 B（3-bit） |
| 若介面統一固定 3-bit/index | 4,800 B | 4,800 B |
| magnitude abs | 409,600 | 409,600 |
| D=32 reduction adds `T(D-1)` | 396,800 | 396,800 |
| naïve selector comparisons `T(K-1)` | 38,400 | 89,600 |
| pair exponent adds，上界 `E` | 2,560,000 | 2,560,000 |
| variable shift/alignment，上界 `E` | 2,560,000 | 2,560,000 |
| 兩 basis 合併 adds `P` | 1,280,000 | 1,280,000 |

【本專案推論】`e_q + e_k` 可用 `K×K` LUT 取代顯式 exponent add，但每個 basis pair 仍要讀兩個 index／查表並執行 variable alignment；表中是直接算 exponent 的保守上界，不是 cycle 精確值。K 從 4 增到 8 主要增加 selector 與 index width，不增加 QK pair 數。

【本專案推論】若把每個 token 的 index 也離線固定（例如按絕對位置固定），reduction、selector、runtime index buffer 可以消失，但它已變成 input-independent positional assignment，不能再聲稱保留每張影像的 token magnitude。必須將「codebook 固定」與「assignment 固定」分開描述。

### B. G=4/8 fixed channel-group PoT scale

把 D=32 固定切成 G 組，`G=4` 時每組 8 channels，`G=8` 時每組 4 channels：

```text
score[s,h,i,j]
  = Σ_b Σ_g 2 ^ (eq[s,b,h,g] + ek[s,b,h,g])
      · binary_dot(q̂[s,b,h,i,g], k̂[s,b,h,j,g])
```

`eq/ek` 與 channel-to-group assignment 都在離線校正／訓練後固定，故推論時沒有 per-image reduction、selector 或 activation index。

| 成本項目 | G=4 | G=8 |
|---|---:|---:|
| separate Q/K exponent 上界 `32G` | 128 entries | 256 entries |
| 可預合併的 q+k exponent 常數 `S·B·H·G` | 64 | 128 |
| runtime activation indices | 0 | 0 |
| per-image magnitude reduction / selector | 0 / 0 | 0 / 0 |
| group partial popcount 上界 `E·G` | 10,240,000 | 20,480,000 |
| group shifts 上界 `E·G` | 10,240,000 | 20,480,000 |
| 合併 `2G` terms 的 adds `P(2G-1)` | 8,960,000 | 19,200,000 |

【本專案推論】目前一個 D=32 word 的 popcount 會把不同 group 的匹配數混在一起；各 group scale 不同時，必須取得 group partial counts。表中以「每 group 一次 partial popcount」估計直觀上界。硬體若有 lane-wise popcount、預遮罩或融合 shift-accumulate，可降低 cycle／instruction 數；反之，4/8-channel group 塞進 32-bit primitive 時的 masking 與 utilization 也可能更差。因此 B 的優點是控制面簡單、無 input-dependent selector，不是保證運算量比基線少。

### C. per-image global K-codebook

這是把本地 dynamic global magnitude 量化成 K 個 PoT exponent：

```text
αq[s,b,h] = 2 ^ e[cq(s,b,h; image)]
αk[s,b,h] = 2 ^ e[ck(s,b,h; image)]
```

【本專案推論】即使 codebook exponent 已離線固定，`c(...; image)` 仍取決於目前影像，所以仍需掃描 Q/K、做 reduction 與 selector。它只把 per-token 的 12,800 次選擇降成 global 的 32 次，沒有移除 dynamic data path。

| 成本項目 | K=4 | K=8 |
|---|---:|---:|
| context-local exponent table 上界 `32K`；若全 context 嚴格共用，僅 `K` entries | 128（共用時 4） | 256（共用時 8） |
| runtime global indices `R` | 32 | 32 |
| 最小 index storage | 64 bit = 8 B | 96 bit = 12 B |
| magnitude abs | 409,600 | 409,600 |
| global reduction adds `R(DN-1)` | 409,568 | 409,568 |
| naïve selector comparisons `R(K-1)` | 96 | 224 |
| q+k exponent combines，可每 image 預算 | 16 | 16 |
| basis-term shift/application `E` | 2,560,000 | 2,560,000 |

【本地證據＋本專案推論】因為完整 dynamic global 相對 fixed PoT 的本地收益僅 `+0.000505537 mAP`，global K-codebook 最多只能被視為逼近 dynamic 的低精度折衷；現有資料沒有理由期待它補回約 `0.0110` 的 FP–BinaryQK gap。若部署要求完全沒有 per-image reduction，它直接不符合需求。

## 4. 橫向比較

| 方案 | 能表達的 magnitude | input-dependent selector | per-image reduction | QK pair 端主要新增成本 | 適合角色 |
|---|---|---|---|---|---|
| A4/A8 per-token codebook | 每 token、每 head、每 basis | 有；12,800 indices | 有；按 D=32 | 每 basis pair index/LUT/variable shift | fidelity 上界；硬體成本待證 |
| B4/B8 fixed group | 固定 channel-group 統計 | 無 | 無 | G 倍 partial popcount/shift/add | selector 禁止時的部署候選 |
| C4/C8 global codebook | 每 image/head/basis global | 有；32 indices | 有；仍掃 409,600 values | 每 basis term 套 global shift | 不優先；本地增益上限訊號很小 |

關鍵取捨不是「scale 數量只有 8 個所以一定便宜」，而是 scale 的 **assignment 在何時決定**：

```text
offline learned values + input-dependent assignment
    → exponent table 固定，但 reduction / selector / index 仍存在

offline learned values + fixed assignment
    → 沒有 selector，但只表示固定的 head/channel/position 統計
```

## 5. 最小且必要的實驗

本節 `C0/A8/B4` 只是在 **scale 子問題已值得展開時** 的條件式矩陣，不能取代主線的兩個 fixed-PoT site isolation（`site10=fixed-PoT Binary, site22=FP`、`site10=FP, site22=fixed-PoT Binary`），以及 `兩 site=fixed-PoT Binary control`，也不能取代 matched low-LR QAT 與 FP-teacher attention-ranking KD。由於 global dynamic 只關閉 FP–PoT 缺口約 4.58%，剩餘大缺口應先由 site policy 隔離敏感節點，再靠 sign/ranking adaptation 回補；只有診斷顯示 magnitude/scale 仍是主要殘差時，才展開下列 codebook 比較。

不先擴成多條訓練 lineage；用固定 validation subset 做一次離線 score-fidelity／硬體計數篩選即可：

1. `C0`：目前 Full35 two-basis fixed PoT，作唯一 control。
2. `A8`：per-token、8-entry PoT codebook，保留 input-dependent selector；量 attention-score KL、top-k overlap、index entropy，以及上表的 reduction/index/shift 成本。
3. `B4`：4 個 fixed channel groups、每組 8 channels；量相同 fidelity 指標，並以目標 backend prototype 驗證 partial-popcount kernel，而不是只看 BitOPs。

若需要先做無 kernel 的便宜 screening，可額外離線模擬 `A4/A8/B4/B8`，但正式 validation 仍只帶 `A8` 與 `B4` 兩個極端取捨。`C4/C8` 暫不進正式實驗，因為它保留 deployment 最想移除的 reduction，且本地 dynamic–PoT 差只有 `0.000505537`。

通過條件應是相對 `C0` 同時滿足：attention fidelity 明顯改善、實際 backend latency／throughput 未失去 BinaryQK 的收益。只有 score fidelity 改善不等於 detection mAP；只有 operation count 下降也不等於實測加速。若候選通過，下一階段才做同 seed、同 checkpoint、同 schedule 的短 QAT/KD paired validation。

## 6. 最終建議

- **部署優先：先測 B4。** 它唯一明確消除 per-image selector/reduction，但要驗證 4-way partial popcount 是否吃掉收益。
- **精度診斷：保留 A8 作上界。** 若 A8 的 attention ranking 明顯勝 B4，代表主要缺口確實來自 token-wise magnitude；此時才值得評估 selector 融合或較粗的 token grouping。
- **不優先 C。** codebook 固定不代表 assignment 固定；C 仍需掃 activation，而本地完整 dynamic 也只比 PoT 高 `0.000505537 mAP`。
- **不承諾補回約 0.011。** 目前所有計算只建立可實驗的硬體／表示力假設，沒有 codebook mAP 證據；Full35 正式指標與先前 YOLO26 ablation 也必須分 lineage 解讀。
