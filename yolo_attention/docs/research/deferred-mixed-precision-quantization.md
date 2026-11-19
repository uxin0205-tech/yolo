# 延後執行的混合精度量化研究筆記

## 結論先行

量化應保留為 **Binary-QK attention 架構、normalization 與 Multimax 實驗完成之後，且需另行核准的 Optional Phase Q**。A-FINAL 報告完成後主研究即可停止；不執行量化也算完成。目前不應把三元權重、SD4 或 3/4/5/6/8/16-bit 全部加入主線訓練，否則無法分辨精度變化來自 attention 架構還是數值格式。

現行計畫其實已經安排兩種不同訓練：

- Binary Q/K 的 `sign` 使用 STE，屬於架構本身必要的 binary-aware training。
- projection、$P$、$V$ 與 LUT path 的整數量化先做 PTQ；只有 PTQ 超過容許掉點才進 QAT。這個 QAT 是部署量化 recovery，不是目前 architecture screening 的一部分。

老師提出的 5-bit、6-bit 是合理候選。均勻整數量化公式本來就可接受任意整數 bit-width；「常見的是 4/8-bit」不表示只能使用 4/8-bit。不過軟體能模擬 5/6-bit，不代表既有 GPU、DPU 或 FPGA datapath 一定能得到等比例速度提升。最終成本必須使用目標硬體的 packing、DSP mapping、BRAM 與 accumulator 規格重新計算。

## 1. 目前有沒有安排量化訓練？

有保留方案，但它是可跳過、需另行核准的最後延伸：

```text
formal V1
   ↓
scale / bias / normalization 定案
   ↓
明確 quantization parent
   ↓
Q0：W8A8 projection + U8 P + S8 V，PTQ
   ↓
Q1：integer score grid + LUT normalization，PTQ
   ├─ 達標：不做 INT QAT
   └─ 未達標：先修 calibration / clipping / overflow
                    ↓
             仍未達標才做 Q2 QAT
```

因此目前不需要先跑「全模型 INT8 QAT」。先做 FP/binary attention 的架構研究，最後才量化，和老師的研究順序一致。唯一不能延後的是 Binary Q/K 的 STE，因為它不是部署 bit-width sweep，而是 Binary-QK 模型能否恢復精度的核心訓練。

PyTorch 對 QAT 的正式定義也是在訓練時插入 fake quantization，forward 模擬 round/clamp/dequantize，訓練後才 convert；PTQ 則直接由既有模型決定量化參數。[PyTorch `FakeQuantize`](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)；[torchao QAT workflow](https://docs.pytorch.org/ao/stable/workflows/qat.html)。

## 2. 為什麼 5-bit、6-bit 可以做

對 $b$-bit signed symmetric uniform quantization，可使用：

$$
q_{min}=-(2^{b-1}-1),\qquad
q_{max}=2^{b-1}-1,
$$

$$
s=\frac{\max |x|}{q_{max}},
$$

$$
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right),q_{min},q_{max}\right),
\qquad
\widehat{x}=s q.
$$

對 unsigned probability：

$$
q_{min}=0,\qquad q_{max}=2^b-1.
$$

公式中的 $b$ 可以是 3、4、5、6、8 或其他正整數。PyTorch 的 fake-quant API 也是用可設定的 `quant_min`、`quant_max` 模擬量化，而不是把演算法限制在 4/8-bit。[PyTorch `FakeQuantize`](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)。

AdaBits 原論文證明 weight/activation bit-width 可成為同一模型的可調設定，並比較 direct、progressive 與 joint training；PDF 第 5 頁第 4.2 節／Table 3 明確共同訓練與評估 8/6/5/4-bit。但它的目標是 runtime-adaptive network，不表示本研究需要照搬 super-network；該論文也顯示簡單依序降低 bit-width 會遺忘先前精度，若未來要「單一 checkpoint 支援多種 bit-width」，需另行研究 switchable normalization/clipping 與 joint training。[AdaBits, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_AdaBits_Neural_Network_Quantization_With_Adaptive_Bit-Widths_CVPR_2020_paper.pdf)。本研究只借用一個觀念：**bit-width 是實驗變數，不必只測二次方常見值**。

目前專案的 `fake_quant_probability`、`fake_quant_symmetric` 與 `VariantConfig` 已接受所有 `bits >= 2`，所以日後加入 5/6/7-bit 不必重做 quantizer interface；現階段只需保留設計，不需啟動訓練。

## 3. 不把所有 tensor 綁成同一 bit-width

Binary-QK attention 的資料型態不同，必須分開定義：

| Tensor／運算 | 第一輪狀態 | 最後量化候選 | 原因 |
|---|---:|---:|---|
| Q/K projection weight | FP → W8 PTQ | 8/6/5/4；三元與 SD4 另案 | 量化誤差會造成 `sign` 翻轉 |
| Q/K projection input/output | FP → A8 PTQ | 8/6/5/4 | 靠近 sign boundary 的值最敏感 |
| Binary Q/K | 1-bit sign | 固定 1-bit | 已是研究方法，不參加一般 bit sweep |
| Popcount | exact integer | 依 $d_k$ 決定 accumulator width | 它是計數值，不應硬套 W4/W8 名稱 |
| fusion scale、relative bias、score grid | FP → fixed-point | 16/12/10/8 等範圍分析 | integer bits 與 fractional bits需分開 |
| exp/sigmoid LUT address | integer index | 由有效輸入範圍決定 | address bits 不等於 LUT output bits |
| $P$ | U8 | 8/6/5/4 unsigned | 需同時量 row-sum 與 attention output error |
| V | S8 | 8/6/5/4 signed | 直接影響 $P\times V$ |
| $P\times V$ accumulator | S32 reference | 依 $b_P,b_V,N$ 推導 | 必須先證明不 overflow，再縮位 |
| PE(V)、output projection | FP → W8A8 | 8/6/5/4；特殊格式另案 | 先做 layer sensitivity，不和 QK 一次降低 |
| FFN、其餘 YOLO26 | 保留原精度 | 最後才做 sensitivity | 不屬於 Binary-QK 核心貢獻 |

對 dense $P\times V$，保守的 accumulator 位寬檢查可先用：

$$
b_{acc}\geq b_P+b_V+\left\lceil\log_2 N\right\rceil.
$$

若使用 Multimax Top-K，將 $N$ 換成 $K$，但仍需考慮實際 signed range、zero point、scale 與飽和規則。研究期間保留 S32 golden reference，縮 accumulator 應是最後的 bit-true ablation。

## 4. 避免組合爆炸的精簡漏斗

若同時讓六個 tensor group 各選 3/4/5/6/8-bit，完整笛卡兒積會有 $5^6=15{,}625$ 組，沒有研究必要。採固定順序的 coordinate sensitivity：

### Stage A：單一 anchor

以選定的 quantization parent 建立一個 W8A8/U8/S8 anchor，其他非研究模組保留 FP16/FP32。先校正並確認 bit-true 模型與浮點 parent 的差距。

### Stage B：逐群單變量 sweep，不訓練

固定其他群組為 8-bit，每次只降低一群：

```text
projection W：8 → 6 → 5 → 4
projection A：8 → 6 → 5 → 4
P：            8 → 6 → 5 → 4
V：            8 → 6 → 5 → 4
PE/proj：      8 → 6 → 5 → 4
```

這是節省測試量的 coarse sweep，不代表排除 7-bit：若某群 6-bit 首次失敗，就在 8 與 6 之間補 7-bit；3-bit 只在 4-bit 仍達標時追加；2-bit 不列第一輪。16-bit 只作 numerical ceiling，並需明確區分 INT16 與 FP16。

每個候選先用小型 calibration set 與 COCO2017 val 的代表性子集計算：

- tensor reconstruction error／SQNR；
- Q/K sign-flip rate；
- attention output cosine similarity；
- probability row-sum error；
- saturation／clipping ratio；
- detection mAP50–95 掉點；
- 理論 BitOps、權重/activation bytes 與 accumulator 需求。

### Stage C：只組合 Pareto 候選

每群只保留「相對 8-bit 最低且仍達標」的一個 bit-width，再組合成最多三個候選：

1. Accuracy：全 8-bit anchor。
2. Balanced：由單群 sensitivity 得到的 5/6/8-bit 混合方案。
3. Aggressive：只對不敏感群降至 4-bit，敏感群維持 6/8-bit。

這比 layer-wise 全搜索更適合目前尚未上板的階段。HAQ 原論文第 4 頁第 3.2 節把 action round 成 2–8 的整數 bit-width，因此包括 3/5/6/7-bit；同時指出 mixed-precision 搜尋空間很大，而且不同硬體的最佳 policy 不同，應以硬體 latency/energy feedback 而非只看 FLOPs。[HAQ, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf)。目前沒有目標板，先用上述可解釋的 sensitivity 漏斗，不宣稱 5-bit 一定比 6/8-bit 更快。NVIDIA TensorRT 官方列出的原生量化型態也包含 INT8/INT4 等而沒有 INT5/INT6，因此中間位寬的軟體結果不能直接當成標準 GPU 加速結果。[TensorRT quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)。

### Stage D：PTQ → QAT gate

每個入選 configuration 遵循：

```text
MinMax PTQ
   ↓ 不合格
Percentile / MSE calibration
   ↓ 不合格
找出 sign flip、clipping、LUT range 或 overflow 原因
   ↓ 規格合理但 mAP 仍超過允許掉點
只對最多 1–2 個 Pareto configuration 做 QAT
```

沿用現行門檻：相對 parent 的 mAP50–95 掉點不超過 0.002 就不因「通常會做 QAT」而額外訓練。若 5/6-bit PTQ 已通過，也不需要 QAT；若失敗才進 fake-quant recovery。4-bit 以下通常風險較高，但仍由量測決定，不預先把「bit-width」和「一定需要 QAT」畫上等號。

## 5. 三元權重放在哪裡

Ternary Weight Networks 將權重限制在：

$$
W_i^t\in\{-1,0,+1\},
$$

並以 scale $\alpha$ 近似原權重。原論文把它寫成最小化：

$$
\min_{\alpha,W^t}\left\|W-\alpha W^t\right\|_2^2,
$$

再用 threshold $\Delta$ 決定三個狀態。[Li et al., *Ternary Weight Networks*, 第 2.1–2.2 節](https://arxiv.org/abs/1605.04711)。

本機論文《三元權重變形器神經網路加速電路之設計與晶片實現》第 3.5.1 節採 TW4A：三元 weight 搭配 INT4 activation；印刷頁 33–37（PDF 第 52–56 頁）使用 threshold、scale、LSQ+、STE，並明確說直接 PTQ 掉點明顯，所以採 QAT，訓練順序為 FP32 → TW32A → TW4A，再加入 progressive quantization。[本機論文](../../三元權重變形器神經網路加速電路之設計與晶片實現.pdf)；[LSQ+ 原論文](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Bhalgat_LSQ_Improving_Low-Bit_Quantization_Through_Learnable_Offsets_and_Better_Initialization_CVPRW_2020_paper.html)。

因此三元權重不應被當成「大約 2-bit uniform」混在 3/4/5/6/8-bit 曲線中。它是另一種 codebook family，需獨立比較：

```text
Uniform branch：W8/W6/W5/W4 + Ax
Ternary branch：W∈{-α,0,+α} + A8/A6/A5/A4
```

第一個三元實驗只量化 Attention 的 Q/K/V projection 與 output projection；PE(V)、FFN 和其他 YOLO26 convolution 先保留原精度。若這一層級可恢復，再考慮擴大範圍。三元 branch 原則上直接規劃 QAT，但仍可保留 zero-shot/PTQ 結果當 diagnostic，不把它當最終模型。

## 6. 「SD4」目前最可能是什麼

老師同時提到「三元權重、SD4 等」，在這個硬體量化語境中，最可能指 2021 年 Hsieh、Liu、Chiueh 的 **FloatSD4／Signed Digit 4（SD4）權重格式**。原始論文為 *A Multiplier-Less Convolutional Neural Network Inference Accelerator for Intelligent Edge Devices*，DOI `10.1109/JETCAS.2021.3116044`；第一作者的臺大學位論文資料頁也把它描述為基於 Floating-point Signed Digit 的簡化 4-bit FloatSD4 encoding。[NTU thesis repository](https://tdr.lib.ntu.edu.tw/handle/123456789/68072)。但全文受限，因此臺大官方可取得的摘要目前只足以確認：

- SD4 是 neural-network weight 的 compact 4-bit number format；
- 目標是把 convolution 從乘加改成只做加法；
- 論文同時提供 software training 與 multiplier-less FPGA accelerator；
- 摘要報告 ImageNet ResNet 的 4-bit 與 FP32 top-1 差距小於 0.5%。

來源：[NTU Scholars 原始論文紀錄與摘要](https://scholars.lib.ntu.edu.tw/entities/publication/c1fea85e-53e2-44dc-8045-346f6b81ecb0)，期刊頁 739–750。

但「SD4」不是唯一可能的展開方式；例如 2026 年另有 SemanticDialect 也把其 4-bit mixed-format 稱為 SD4，而 `SDP4Bit` 則是分散式訓練通訊量化。這些與老師的 CNN/accelerator 脈絡不同。**在拿到老師指的論文標題、PDF 或 codebook 前，不應實作或宣稱 SD4 的精確數值集合。**

需要向老師確認一句：

> 您說的 SD4，是否是 Hsieh、Liu、Chiueh 2021 年 JETCAS 論文中的 Signed Digit 4？

若確認是該方法，後續應取得全文，先固定其 codebook、scale、rounding、gradient 與 accumulator 規格，再建立獨立 branch：

```text
Uniform INT branch
Ternary branch
Signed-Digit-4 branch
```

SD4 不能僅以「4-bit」和 W4 uniform 比 accuracy；還要比較 multiplier 是否真的可被 shift/add 或 LUT 取代、codebook/scale 儲存、稀疏度、weight decode 及目標硬體成本。

## 7. Optional Phase Q 的最小實驗

架構主線完成後，建議只保留：

| ID | 內容 | 是否訓練 |
|---|---|---:|
| MP0 | FP/binary formal parent | 否 |
| MP1 | uniform W8A8、U8 P、S8 V | PTQ |
| MP2 | 單群 6/5/4-bit sensitivity | 否 |
| MP3 | 最佳 balanced mixed precision | PTQ |
| MP4 | MP3 未達標時的 recovery | 條件式 QAT |
| TW0 | Attention projection ternary weight，activation 先 A8 | QAT |
| TW1 | TW0 通過後才降低 activation 到 6/5/4-bit | QAT，最多選一組 |
| SD4-0 | 老師確認格式且取得完整規格後才建立 | 待定 |

不需要把每個 normalization、basis、bias、Multimax K 都重新乘上所有量化組合。量化只接在已選定的 formal parent 後面，才能回答「精度掉點是量化造成的多少」。

## 8. 決策樹

```text
Binary-QK / bias / normalization / Multimax 研究完成？
   ├─ 否 → 不啟動部署量化
   └─ 是
       ↓
固定 formal parent、BN fold、bit-true FP reference
       ↓
W8A8/U8/S8 PTQ 是否達標？
   ├─ 否 → 先修 calibration / clipping / score grid / overflow
   │        └─ 仍未達標 → INT8 QAT
   └─ 是
       ↓
6/5/4-bit 單群 sensitivity
       ↓
選最多三個 Pareto candidates
       ↓
最佳 mixed-precision PTQ 是否達標？
   ├─ 是 → 完成 uniform branch
   └─ 否 → 只對最佳 1–2 組做 QAT
       ↓
另開 ternary branch（預設 QAT）
       ↓
老師確認 SD4 原始規格？
   ├─ 否 → 保留 backlog，不猜 codebook
   └─ 是 → 另開 Signed-Digit-4 branch
```

## 9. 來源

1. Qing Jin, Linjie Yang, Zhenyu Liao, [*AdaBits: Neural Network Quantization with Adaptive Bit-Widths*](https://openaccess.thecvf.com/content_CVPR_2020/html/Jin_AdaBits_Neural_Network_Quantization_With_Adaptive_Bit-Widths_CVPR_2020_paper.html), CVPR 2020, pp. 2146–2156。
2. Kuan Wang et al., [*HAQ: Hardware-Aware Automated Quantization with Mixed Precision*](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf), CVPR 2019, pp. 8612–8620。
3. Fengfu Li et al., [*Ternary Weight Networks*](https://arxiv.org/abs/1605.04711), arXiv:1605.04711, 2016，第 2 節。
4. Yash Bhalgat et al., [*LSQ+: Improving Low-Bit Quantization Through Learnable Offsets and Better Initialization*](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Bhalgat_LSQ_Improving_Low-Bit_Quantization_Through_Learnable_Offsets_and_Better_Initialization_CVPRW_2020_paper.html), CVPR Workshops 2020, pp. 696–697。
5. Ming-Hang Hsieh, Yu-Tung Liu, Tzi-Dar Chiueh, [*A Multiplier-Less Convolutional Neural Network Inference Accelerator for Intelligent Edge Devices*](https://scholars.lib.ntu.edu.tw/entities/publication/c1fea85e-53e2-44dc-8045-346f6b81ecb0), IEEE JETCAS 11(4), 2021, pp. 739–750, DOI `10.1109/JETCAS.2021.3116044`。
6. 陳法諭，[《三元權重變形器神經網路加速電路之設計與晶片實現》](../../三元權重變形器神經網路加速電路之設計與晶片實現.pdf)，國立臺灣大學碩士論文，2025，第 3.5.1 節，印刷頁 33–37（PDF 第 52–56 頁）。
7. [PyTorch `FakeQuantize` 官方文件](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)；[torchao QAT 官方 workflow](https://docs.pytorch.org/ao/stable/workflows/qat.html)。
8. Ming-Hang Hsieh，[《適用於智慧型邊緣裝置之無乘法器卷積神經網路推論加速器》臺大學位論文資料頁](https://tdr.lib.ntu.edu.tw/handle/123456789/68072)，DOI `10.6342/NTU202003812`。
9. [NVIDIA TensorRT quantized types and schemes 官方文件](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)。

## 10. 來源限制與工程推論

- 5/6-bit 對本專案的候選順序、PTQ/QAT gate、tensor grouping 與三個 Pareto configuration，是依原始方法對 YOLO26 Binary-QK attention 所作的工程設計，不是上述論文直接報告的 YOLO26 結果。
- 三元論文在 Transformer/ViT 或較早期 detection 任務的結果，不能直接推論 COCO2017 YOLO26m 精度；必須實測。
- SD4 的完整 codebook 尚未由可取得的 primary full text 驗證，因此本文只引用原作者機構頁可確認的摘要主張，不填入推測的數值集合。
- 尚未選定 FPGA/ASIC/DPU 時，BitOps、model bytes 和 accumulator width 只能當 algorithmic proxy，不能宣稱實際 latency、area 或 power 改善。
