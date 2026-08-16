# YOLO26 Binary Attention 程式架構

> 日期：2026-08-11
> 狀態：accepted
> 範圍：CPU reference、YOLO26 integration 與未來 training seam。

## 1. 外部 interface

### VariantConfig

演算法選擇集中在 immutable dataclass。YAML parser 會拒絕不合理組合，例如 FP basis 搭 integer LUT。

### HardwareFriendlyAttention

caller 提供官方 `Attention` 與 `VariantConfig`：

~~~text
official fused QKV
  → per-head channel gather
  → Q / K / V Conv+BN
  → configured score
  → configured bias
  → configured normalization
  → official PV + PE(V) + projection
~~~

### YOLO adapter

`convert_c2psa()` 是單一 C2PSA 的局部測試 seam；正式模型一律使用 `convert_yolo26_model()`。YOLO26m 有兩個 Attention：`model.10.m.0.attn` 與 `model.22.m.0.1.attn`。模型 seam 掃描官方 `Attention` 並驗證路徑集合完全一致，少一處或多一處都報錯；FFN、residual、C2PSA/C3k2 與 Detect 不變。

## 2. QKV layout

官方 reshape：

$$
[B,h,2d_k+d_v,N]
$$

channel order：

$$
[Q_0,K_0,V_0,Q_1,K_1,V_1,\ldots]
$$

`qkv_channel_indices()` 依 head 建立 gather；`interleave_qkv_maps()` 作反向重組與 P0 對照。不能改成三段 contiguous slice。

## 3. Score 與 normalization

`BinaryScore.forward(q,k)` 統一輸出 `[B,H,N,N]`：

- FP：scaled dot product。
- I：一組 sign + XNOR-popcount。
- H：Identity 與 normalized Hadamard matched bases。
- T5：兩組 matched residual bases，不含 cross terms。

`build_normalizer(VariantConfig)` 是唯一 construction seam。N0 使用 Exact、score-indexed exp LUT（浮點 P）、16-segment PWL exp、project PoT exp、normalized HardSigmoid、shifted ReLU 與 Multimax；它們都接受 `[B,H,N,N]` score 並輸出非負且 row-normalized 的 P。`ProgressiveNormalizer` 只負責 N1 的 probability-level PMP。

PoT 是本專案的軟體 reference approximation，不宣稱是特定 Shiftmax 論文的 bit-true reproduction。Optional Phase Q 才使用 `IntegerLUTSoftmax` 量化 U8 P，並保存 `last_u8` 與 `last_row_sums` diagnostics。

BDCN 由 `bdcn.py` 實作。`convert_yolo26_model()` 先依 global／per-attention／per-head 建立 table assignment，再把同一個 `BDCNCodebookBank` 注入兩個 Attention。`BDCNNormalizer.aggregate()` 直接把相同 bucket 的 value 相加，不 materialize dense $P$：

$$
G_{il}=\sum_{j:b_{ij}=l}V_j,\qquad
O_i=\left(\sum_l C_lG_{il}\right)\widehat{D_i^{-1}}.
$$

R0 使用 exact reciprocal 作數值 reference；R1 使用 leading-one/frexp normalization 加 reciprocal LUT，是主要硬體候選；R2 把 denominator 投影到最近的 $2^k$ 後用 shift，是 division-free diagnostic。nearest-PoT 使每列機率總和落在 $[2^{-1/2}, 2^{1/2}]$；它不是 exact normalization，也可能形成有意義的 accuracy 下界。Queue 必須保存其有限結果，即使 mAP 低於一般 child retention gate；R2 不參與 R1 的通過判定。BDCN 本身不能宣稱 division-free，只有 R2 可以這樣描述。fused path 仍保留官方 `PE(V)` 與 output projection。

從 learned BDCN checkpoint 轉成 D2/R variant 時，新的 shared `BDCNCodebookBank` 必須先載入 parent bank state，再套用 1-PoT/2-PoT projection 或 denominator。不得以預設 exponential initializer 取代訓練後的 codebook。

軟體 AMP reference 的 fused bucket-$PV$ 必須明確停用 autocast，使用 FP32 做 bucket accumulation、codebook weighting 與 reciprocal，再於 Attention 邊界轉回 V dtype。原因是未正規化的 bucket partial sum 可能超過 FP16 的 65504，即使最後除以 denominator 的結果完全有限。未來 fixed-point/FPGA 實作需依最壞情況配置 accumulator guard bits，不能把 FP32 reference 誤算成演算法額外 MAC。

## 4. Training construction

直接 conversion 後呼叫預設 `YOLO.train()` 不可靠，因 trainer 會由 YAML 重建模型。`HardwareAttentionTrainer.get_model()` 固定：

~~~text
DetectionModel(cfg)
  → load official weights
  → convert_yolo26_model()
  → optimizer build 前重新套用 trainable scope
~~~

screening／normalization recovery／bias／scale 只訓練兩個 converted Attention；`bdcn_codebook` 只解凍跨兩個 Attention 共用的 codebook；architecture recovery 使用 full model。Optional Phase Q 若另行核准，Q2 才使用 full-model fake quant。Progressive epoch callback 同時更新 score-level schedule 與 normalization-level PMP wrapper。

## 5. 修改指南

新增 basis：

1. 在 `BasisKind` 加 enum。
2. 在 `BinaryScore.forward()` 加實作。
3. 先加 shape、gradient、reference tests。
4. 加 variant YAML 與 registry entry。

新增 Softmax：

1. 在 `NormalizationKind` 或 `RowCorrection` 加 enum。
2. 在 `normalization.py` 實作共同 `scores -> P` 介面。
3. 測非負、row sum、NaN/Inf；integer 方法再測 U8 與 bit-true reference。
4. AMP fused bucket-$PV$ 必須以會使 FP16 partial sum overflow、但 normalized output 有限的案例測試；任一 batch loss 為 NaN/Inf 時 trainer 立即 fail closed。
5. 在 `build_normalizer()` 加 construction mapping，不改 Attention forward。

新增 experiment 應增加 `VariantConfig` 與 `ExperimentRun`，不要複製 Attention class。

## 6. Persistent queue seam

~~~text
QueueState / QueueJob
       ↓ atomic JSON
QueueStore + non-blocking worker lock
       ↓
QueueExecutor (ready → queued → running → succeeded/failed；selection rewind 可稽核封存)
       ↓ only with --execute
ResearchQueueBackend
 ├─ baseline/variant COCO evaluation；fixed scale 先 calibration，再把係數寫回未融合 Conv+BN training model 後 export
 ├─ parent-checkpoint training → best.pt → common evaluator
 ├─ P0 CPU full-model equivalence
 └─ pure selection policies → dynamic graph expansion
~~~

若 train job 已跑滿 recipe epochs 且 immutable run 同時存在 `best.pt`、
`last.pt` 與最後 epoch 完整的 `results.csv`，retry 只重做 evaluation/profile
postprocess，不重新啟動 trainer；任一證據缺失就不能套用此恢復路徑。

I/H/T5 的 scheduling dependency 依序串接，避免同時跑；但三者的 `model_parent_job_id` 都是 P0，所以模型權重不會錯接前一個 screening run。D1 三個 sharing 候選也依序排程，但各自從共同 D0 parent 進行一次完整 10-epoch codebook-only training；沒有 extension job 或 checkpoint-based 5+5 resume。`d1-select` 直接比較 `d1-shared`、`d1-pattn`、`d1-phead`；只有 top-two 差距小於 0.001 時，才從 D0 parent 建立 winner 的 seed-1 10-epoch confirmation，D2 仍接 seed-0 winner。每次 selection 後，`queue_workflow.py` 從實際 winner YAML 產生下一階段設定。Optional quantization 不在這張 graph 中。

COCO result 統一寫 `map50_95/map50/map75/maps`、checkpoint、profile 與 row-sum error。Custom training parent 至少需有 95% target state coverage，避免 fused inference checkpoint 靜默只載入少量權重；child mAP 低於 model parent 的 50% 時 executor fail closed。Final tie-break 只接受存在的 profile JSON；分析 profile 是固定兩個 Attention、reference shape 的比較 proxy，不是 hardware measurement。

BDCN v2 透過同一 QueueExecutor 的 `append-bdcn-v2` 在已完成 queue 後追加，不能
另開 worker script。它從已完成 scale/bias 補償的 A0 checkpoint 直接建立
$d_{max}=8,L=64$ 的 10-epoch learned-codebook 與 R1 reciprocal-LUT jobs，
不做額外 levels screening。BDCN validation 額外聚合兩處 Attention、所有 batches
的 bucket histogram、true overflow、last-bucket rate 與最大 distance 到
`metrics/queue-result.json`。`BCND/` 只保存設定與說明，唯一演算法 source 仍是
`src/yolo_attention/bdcn.py`。

BDCN v3 仍使用相同 source 與 executor，但由 `append-bdcn-v3` 追加三個
immutable jobs：fixed-exp zero-train control、anchored-codebook recovery、R1。
anchored bank 不直接自由學習每步 ratio，而是

$$
\log(C_{l+1}/C_l)=-\Delta+\epsilon\tanh(r_l),
$$

其中 $\Delta=8/63$、$\epsilon=0.01$。$r_l=0$ 精確重現 $e^{-l\Delta}$；
即使參數飽和，最後一格仍小於 $0.001$，因此不可能重現舊
$L=16,\Delta=0.125$ 將 $d\ge1.875$ 映射到約 $0.153$ 的 flat tail。v3
training scope 仍只有 codebook，使用獨立低 LR recipe；fixed control 不訓練。


## 7. 尚未執行

- canonical COCO JSON baseline 與最終候選重測。
- fixed-head/PoT calibration 已實作；2026-08-14 修正 validation 原地 BN fusion 後誤存 training parent 的問題，正式 scale screening checkpoint 需重建。
- PTQ calibration dataset pass。
- checkpoint export 後重新 BN fold report。
- GPU latency 與 FPGA/HLS。

以上是實驗執行狀態，不是缺少的程式 module。
