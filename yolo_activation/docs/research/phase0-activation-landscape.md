# Phase 0：YOLO activation 演算法與硬體部署研究版圖

> 狀態：可供決策的初版（2026-08-24）
> 範圍：目前尚未收到實際模型／資料交付資料夾；已完成文獻、數學與純函數 prototype，但沒有宣稱任何候選已在 detection 勝過 SiLU。
> 證據規則：重要主張只引用原始論文、官方規格或官方原始碼。論文中的 FLOPs、單一 activation microbenchmark 或 FPGA activation core 結果，均不直接外推成完整 YOLO 在未指定硬體上的加速。

## 1. 先給結論

**2026-08-25 執行決策更新：**昂貴的既有函數 baseline 已縮成 SiLU 與 Hardswish；ReLU 只作便宜
primitive floor。Proposed 只跑 `poly_shift` 與 `poly_quality`，完整介面與驗證見
[硬體友善原型報告](phase0-hardware-activation-prototype.md)。下列項目保留為研究 frontier，不再
等同第一輪完整 sweep。

文獻盤點形成 **SiLU baseline + 5 個 frontier 位置**：

1. **ReLU**：最便宜的部署下界與 accuracy-loss control。
2. **Hardswish**：已有行動端設計與標準圖算子支援的 SiLU 鄰近控制。
3. **PReLU-shared**：用極少可學參數測試負半軸斜率是否有價值；先共享／逐區域，不先做逐通道大範圍搜尋。
4. **PWLU-lite**：受限、可學的 PWL 演算法上界；由原始 PWLU 縮成少段、少量共享 profile，必須明列為本專案變體而非原論文設定。
5. **FReLU-region**：只放在少數空間特徵區域的架構型上界，用來判斷「activation 是否應包含鄰域上下文」，不視為純 scalar drop-in。

Mish、GELU 可各做小規模 accuracy-oriented control，但不建議與上述五項同等投入完整 sweep；ReLU6 可在零樣本或短恢復測試中併入，若無優勢即淘汰。ACON／Meta-ACON 與 Dynamic ReLU 適合作為浮點動態上界或教師，不是第一批硬體部署候選。

另一條研究支線是本文第 8 節的 **even-residual、積分約束多項式 activation**。它已通過純函數、固定點與 ONNX gate，並以 `poly_shift`／`poly_quality` 進入目前短名單；但尚未通過 detection pilot，且已有多個相鄰先前技術，**不得宣稱 novel**。

是否需要重訓的短答是：

- 固定函數替換可先做 zero-shot 篩選，但正式精度比較至少要做相同配方的短期 recovery fine-tune。
- PReLU、PWLU、FReLU、ACON、Dynamic ReLU 或任何新參數化 activation 都要訓練其新增參數；若要作公平論文結論，決賽者應同配方從頭訓練或做等成本完整 fine-tune。
- 走 INT8／整數近似時，通常還要 calibration，且很可能需要 QAT／quantization recovery；只在 FP32 checkpoint 上換函數不能回答量化後精度。

## 2. 兩個資料集必須分開回答兩個問題

| 資料集 | 本專案用途 | 固定規則 | 主要報告 |
|---|---|---|---|
| BBAT5 v1 | 棒球領域、球棒／球等小物件與領域內行為 | 只使用不可變 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`；detect 用 `configs/detect.yaml`，不得自行重切、抽樣或改標註 | canonical evaluator 的 AP、AP50、AP75、逐類結果；另記錄合法且固定定義的小物件／尺寸分箱結果 |
| COCO2017 | 通用物件偵測泛化與外部可比性 | 標準 `train2017` 訓練、`val2017` 驗證；官方下載入口見 [COCO](https://cocodataset.org/#download)，資料集定義見 [原始論文](https://arxiv.org/abs/1405.0312) | AP@[.50:.95]、AP50、AP75、AP_S、AP_M、AP_L |

兩者應各自訓練／微調獨立 checkpoint、各自列表，**不得平均成單一分數，也不得在未授權下混合訓練**。BBAT5 的小物件定義與 COCO `AP_S` 不必然相同，數值不可直接橫向相減。相同資料集內才比較 activation，且模型、解析度、增強、optimizer、schedule、batch、量化與匯出設定均固定。

早期 ACON／FReLU detection 實驗使用 COCO `trainval35k/minival`，不是本專案要採用的現代 `train2017/val2017`；其數字只作方向證據，不作本專案 baseline。

## 3. 固定 scalar activation：公式、訓練與部署

### 3.1 SiLU／Swish baseline

\[
f(x)=x\sigma(x),\qquad
f'(x)=\sigma(x)+x\sigma(x)(1-\sigma(x)).
\]

Swish 原論文把一般式寫為 \(x\sigma(\beta x)\)，\(\beta=1\) 即 SiLU；它平滑、非單調，負半軸保留小幅輸出與梯度，但需 sigmoid 與乘法。[原始 NAS／Swish 論文](https://arxiv.org/html/1710.05941) 的主要證據是分類而非 YOLO detection，因此本專案仍須自行建立 baseline。SiLU 是比較中心，不是預設硬體最快者。

### 3.2 ReLU 與 ReLU6

- ReLU：\(f(x)=\max(0,x)\)，除 \(x=0\) 外導數為 0 或 1；無參數，只需 compare/max，負半軸為真零。原始研究顯示 rectifier 可有效訓練深網路，[Glorot 等人](https://proceedings.mlr.press/v15/glorot11a.html)；[ONNX Relu](https://onnx.ai/onnx/operators/onnx__Relu.html) 是標準算子。
- ReLU6：\(f(x)=\min(\max(0,x),6)\)，在負半軸及 \(x>6\) 飽和，官方公式見 [PyTorch 文件](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.relu6.html)。上界可限制動態範圍，但也可能剪掉有用的大正值；YOLOv4 論文將 ReLU6／hard-Swish描述為為量化網路設計的 activation，但沒有給出適用所有 detector 的保證。[YOLOv4 原始論文](https://arxiv.org/html/2004.10934)

兩者都可直接替換 scalar activation；然而函數與 SiLU 不等價，zero-shot 精度下降不等於重訓後精度。ReLU 是正式候選；ReLU6 與 ReLU／Hardswish 角色重疊，先作便宜 control。

### 3.3 Hardswish

\[
f(x)=x\frac{\operatorname{ReLU6}(x+3)}{6}
=\begin{cases}
0,&x\le -3,\\
x(x+3)/6,&-3<x<3,\\
x,&x\ge3.
\end{cases}
\]

中段導數為 \(x/3+1/2\)，邊界導數有跳躍。要注意：hard-sigmoid 是 PWL，但乘上 \(x\) 後 Hardswish 中段是**二次式，不是整體 PWL**。MobileNetV3 的原始工作以它近似 swish，並只在網路較深部分使用，顯示「不同區域不同 activation」可有工程價值；同篇也以實機量測說明相同理論運算量不代表相同 latency。[MobileNetV3 原始論文](https://arxiv.org/html/1905.02244)

原語為 add、clip/ReLU6、multiply、常數 scale；[ONNX HardSwish](https://onnx.ai/onnx/operators/onnx__HardSwish.html) 已標準化，但標準算子存在不代表每個 NPU 都有量化 fused kernel。它是正式候選，需同時測 export graph 是否保持單算子、是否插入 dequant/requant。

### 3.4 GELU 與 Mish：accuracy controls

- GELU：\(f(x)=x\Phi(x)\)，\(f'(x)=\Phi(x)+x\phi(x)\)。原始工作以輸入值乘上其隨機正規 gate 機率來解釋它，[GELU 原始論文](https://arxiv.org/abs/1606.08415)。exact 版含 `erf`，常見近似版含 tanh 與三次項；[ONNX Gelu](https://onnx.ai/onnx/operators/onnx__Gelu.html) 同時定義 exact／tanh approximation。它能測試更平滑的 accuracy 上界，但本次檢索沒有找到足以支持其成為 YOLO 硬體優先候選的直接證據。
- Mish：\(f(x)=x\tanh(\operatorname{softplus}(x))\)，若 \(s=\operatorname{softplus}(x)\)，則 \(f'(x)=\tanh(s)+x\sigma(x)(1-\tanh^2(s))\)。它平滑、非單調，但 forward 含 softplus、tanh、multiply。[Mish 原始論文](https://arxiv.org/abs/1908.08681) 與 [官方原始碼](https://github.com/digantamisra98/mish) 提供實作；[ONNX Mish](https://onnx.ai/onnx/operators/onnx__Mish.html) 可分解為 Softplus→Tanh→Mul。

Mish 有直接 YOLO 相鄰證據，但不能過度解讀：YOLOv4 的 COCO 實驗顯示 CSPDarknet53 使用帶 Mish 的預訓練設定由 42.4 AP 到 43.0 AP，CSPResNeXt50 則沒有同樣收益，說明結果依架構與預訓練而變。[YOLOv4 表 6 與討論](https://arxiv.org/html/2004.10934) 因此 Mish 適合小規模 accuracy control，不適合作為第一批硬體候選。

## 4. 可學、動態與架構型 activation

| 方法 | 核心機制與優化特性 | 新增參數／原語 | drop-in 與部署判斷 | detection 證據與限制 |
|---|---|---|---|---|
| PReLU | \(f(x)=x\) if \(x\ge0\)，else \(ax\)；導數為 1 或 \(a\)，避免 ReLU 全負半軸零梯度 | 每層共享或每通道 slope；compare + multiply | scalar drop-in；[ONNX PRelu](https://onnx.ai/onnx/operators/onnx__PRelu.html) 標準化，但 per-channel broadcast／量化 multiplier 的融合需實測 | [原始 PReLU 論文](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) 主要是 ImageNet；本次未找到現代 YOLO/COCO 的充分受控證據 |
| ACON-C | \((p_1-p_2)x\sigma(\beta(p_1-p_2)x)+p_2x\)，從 Maxout/Swish 形式學習「activate or not」 | 通常 channel-wise \(p_1,p_2,\beta\)；sigmoid、多次 mul/add | scalar 形狀但不是便宜固定函數；要訓練，量化 sigmoid 與多參數 scale 有風險 | RetinaNet-R50 舊 COCO split：ReLU 35.2、Swish 35.8、Meta-ACON 36.5 AP；各自使用對應 ImageNet 預訓練 backbone，非 YOLO 控制實驗。[原始 ACON 表 9](https://arxiv.org/html/2009.04759) |
| Meta-ACON | 令 \(\beta=G(x)\)，由輸入自適應控制 activation | GAP + bottleneck FC + sigmoid + ACON；含資料相依路由 | 非純 scalar drop-in，圖、記憶體存取及動態量化更複雜 | 同上；可作浮點動態上界／教師，不列首批部署候選 |
| Dynamic ReLU | \(y_c=\max_{k\le K}(a_c^k(x)x_c+b_c^k(x))\)，常用 \(K=2\)，係數由超網路產生 | GAP、FC、動態係數、max；論文 MobileNetV2 x1.0 約由 3.5M/300M 增至 7.5M/315.5M params/MAdds | 架構型、資料相依，不是靜態 activation swap；硬體延遲高度 backend-specific | 原始 COCO2017 結果是**單人 keypoint detection**而非 bbox detection，不能當作本專案 bbox AP 證據。[Dynamic ReLU 原始論文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640341.pdf) |
| FReLU | \(f(x_{c,i,j})=\max(x_{c,i,j},T(x_{c,i,j}))\)，\(T\) 通常為逐通道 2-D spatial condition（depthwise conv + norm） | 約 \(Ck^2\) 額外參數及逐位置 DWConv/MAC、norm、max | 非 scalar drop-in；ONNX 可由常見原語組圖，但融合、額外 feature-map traffic 與量化需實測 | 舊 COCO RetinaNet-R50：ReLU 35.2、Swish 35.8、FReLU 36.6 AP，AP_S 18.8/18.6/19.2；是有價值的 context upper bound，但含 activation-specific ImageNet 預訓練。[FReLU 原始論文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560341.pdf) |
| PWLU | 在可調區間內用 \(N\) 個連續線段與可學節點值，區間外另學 slope；以 activation 統計重置區間 | 原論文常用逐通道、16 段；segment select、係數讀取、mul/add | scalar drop-in，但原設定的係數量與索引不必然硬體便宜；`PWLU-lite` 必須限制段數／共享 profile | COCO train2017/val2017、from scratch：R50-FPN Mask R-CNN bbox AP 為 ReLU 39.06、Swish 40.05、FReLU 40.25、PWLU 40.48；RetinaNet 為 37.06/37.82/38.35/38.31。[PWLU 原始論文](https://arxiv.org/pdf/2104.03693) |

梯度觀點上，PReLU／PWLU 直接學靜態斜率；ACON／Dynamic ReLU 學輸入相依斜率；FReLU 改變的是 activation 的感受野。這三類不應混成「都是 drop-in activation」，否則參數、MAC、記憶體與訓練公平性會失真。

## 5. PWL、PoT／APoT、LUT 與 polynomial 近似

### 5.1 計算原語與真正風險

| 表示 | 典型原語 | 優點 | 主要風險 |
|---|---|---|---|
| PWL | breakpoint compare／segment select + \(a_ix+b_i\) | 可表示非單調函數；誤差、段數與成本直觀 | 控制分支、係數讀取、一個乘法；沒有 fused kernel 時可能比內建 activation 慢 |
| PoT-PWL | slope \(a_i=\pm2^{k_i}\) | 整數硬體可把乘法改為 shift | 係數格點粗，功能誤差可能在深網路放大；負指數與 rounding 還要定義 |
| APoT-PWL | slope 是少數 powers-of-two 的和 | 多個 shift+add 換取比 PoT 更細的 slope | add 數與 routing 增加；原始 APoT 論文是**量化 level codebook**，不是 activation slope 近似的 detector 證據。[APoT 原始論文](https://openreview.net/pdf?id=BkgXT24tDS)／[官方程式](https://github.com/yhhhli/APoT_Quantization) |
| LUT | address/index + memory read，可搭 interpolation | 常數時間、可容納任意形狀 | 直接表大小隨 address bits 指數成長；BRAM／cache／bandwidth、插值與 range clipping 皆影響結果 |
| polynomial | Horner form 的連續 mul/add | 無 segment branch，適合固定低階式 | 每升一階多一組乘加；fixed-point overflow、係數量化、輸入 range 與 DSP 成本 |

I-BERT 以二次 polynomial 做整數 GELU 近似，並在 Transformer/T4 設定報告相對 FP32 的 2.4–4.0× 加速；它同時指出 LUT 的記憶體問題及高階多項式的計算／overflow 代價。[I-BERT 原始論文](https://proceedings.mlr.press/v139/kim21d.html) 這證明 polynomial 可以是有效的**特定軟硬體共同設計**，不證明 YOLO 或未指定 NPU 會有同樣倍率。

### 5.2 與 SiLU 近似最接近的現有工作

- **GRAU** 已把非單調 SiLU 做 PWLF，再將 slope 近似成 PoT/APoT，並把 BN 與 output requantization 納入整數 activation unit；還支援 runtime reconfiguration。其 FPGA 單元相對 multi-threshold baseline 報告逾 90% LUT 降幅，但不是完整 YOLO accelerator；而其 VGG16/CIFAR-10 表也顯示低段數／低精度 SiLU 可能有明顯 accuracy loss。[GRAU 原始論文](https://arxiv.org/html/2602.22352)
- **DAPA** 已依實際 pre-activation PDF 的 quantiles 配置非均勻 knots，以 distribution-weighted MSE 同時近似 activation 與 derivative，並做 Fix16/HLS。它在自身 GELU HLS core 報告 16× latency／DSP 改善，但實驗集中在 ViT、Swin、GPT-2、BERT，沒有 YOLO detection。[DAPA 原始論文](https://arxiv.org/html/2603.19338)
- **PWLU** 已做可學、資料／架構專用的 activation shape，並有標準 COCO detection 證據；本專案若只是「讓 PWL 自己學節點」，不構成清楚缺口。[PWLU 原始論文](https://arxiv.org/abs/2104.03693)
- **Curl** 已利用 SiLU 的偶對稱殘差，把 SiLU 寫成 ReLU 減去只需 \(|x|\) 的 residual 並以 LUT 計算；因此「利用 SiLU 正負半軸恆等式節省表格」本身已有先前技術。[Curl 原始論文](https://ceur-ws.org/Vol-3920/paper02.pdf)

因此，`SiLU symmetry + PWL + PoT/APoT + fused requantization` 的泛稱已被高度覆蓋。後續若要形成研究貢獻，必須靠 detector-specific objective、受約束函數族、少量共享 profile、interaction-aware placement 與真實硬體驗證的**可區辨組合**，且仍需更廣的論文與專利檢索。

## 6. YOLO 逐層混合 activation 的直接碰撞：ActNAS

ActNAS 已在 YOLO5n／YOLO8 系列中逐層搜尋 ReLU、SiLU、Hardswish 等 activation，使用 COCO 並在 CPU、GPU、NPU 邊緣裝置量 latency／memory；最終論文報告最高 1.67× inference speedup 和／或 64.15% memory reduction，且發現針對一台硬體選出的結果不必然轉移到另一台。[CVPRW 2025 原始論文頁](https://openaccess.thecvf.com/content/CVPR2025W/MAI/html/Sah_ActNAS__Generating_Efficient_YOLO_Models_using_Activation_NAS_CVPRW_2025_paper.html)／[論文 PDF](https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Sah_ActNAS__Generating_Efficient_YOLO_Models_using_Activation_NAS_CVPRW_2025_paper.pdf)

這表示：

- 「YOLO 每層可以不同 activation」不能作為本專案新穎性主張。
- region-level、少量 profile 仍是合理工程約束，因為它比逐層任意搜尋更容易解釋、匯出與共享硬體，但必須與 ActNAS 比較搜尋成本、profile 數、精度與裝置轉移。
- 演算法階段可以先用 latency proxy 排序，最終硬體結論仍要等指定板卡／compiler／precision 上的端到端量測。

## 7. 研究 frontier 與淘汰邏輯

| 角色 | 候選 | 為何保留 | 何時提早淘汰 |
|---|---|---|---|
| Baseline | SiLU | 現有模型的 accuracy／收斂中心 | 不淘汰；所有結果相對它報告 |
| 部署下界 | ReLU | 最簡單、標準化、可測最大可能 activation 簡化收益 | recovery 後精度仍超過預設容忍區間 |
| mobile surrogate | Hardswish | 接近 swish 行為、有 YOLO/edge 直接先例與標準算子 | backend 未融合、端到端 latency 不升反降，或 AP 明顯落後 |
| 最小可學控制 | PReLU-shared | 只增加少量 slope，隔離「保留負梯度」的價值 | 學到接近 ReLU、無精度收益，或量化乘法成本不合目標 |
| 靜態形狀上界 | PWLU-lite | 有 COCO 證據，可測少段受限 PWL 是否能兼顧精度 | 需大量逐通道段數才維持 AP、匯出展開、或 profile 存取成本過高 |
| 架構／上下文上界 | FReLU-region | 有 detection 證據，回答 scalar activation 無法回答的空間條件問題 | 少數區域仍無收益，或 DWConv／feature traffic 抵銷精度效益 |

次順位：Mish、GELU 各做一個 seed 的短控制；ReLU6 做 zero-shot／短恢復。ACON／Meta-ACON、Dynamic ReLU 只在需要動態教師／浮點上界時啟用，避免把研究資源消耗在難部署的 routing 上。

## 8. Working hypothesis：even-residual、積分約束多項式

### 8.1 必須先修正的完整定義

給定

\[
q_a(z)=az+(9-4.5a)z^2+(6a-16)z^3+(7.5-2.5a)z^4,
\qquad z=\operatorname{clip}(u/T,0,1),
\]

直接令 \(H(u)=T\int_0^zq_a(s)ds\) **不會**在 \(u>T\) 接到 ReLU：此時 \(H=T/2\) 為常數，而不是 \(u/2\)。要讓

\[
A(x)=x/2+H(|x|)
\]

在兩端精確等於 ReLU，完整式應寫成

\[
H_{T,a}(u)=T\int_0^{z}q_a(s)\,ds+\frac{(u-T)_+}{2},
\quad u\ge0,
\]

或等價地分段定義 \(H(u)=u/2\) for \(u\ge T\)。

符號檢查顯示，對任意 \(a\)，給定係數確實滿足

\[
q(0)=0,\qquad q(1)=1/2,\qquad q'(1)=0,\qquad \int_0^1q(z)dz=1/2.
\]

所以修正後：積分條件保證 \(A(\pm T)\) 接到 ReLU 值；\(q(1)=1/2\) 保證一階導數接合；\(q'(1)=0\) 保證二階導數接合。內區間的導數為

\[
A'(x)=\begin{cases}
1/2+q_a(x/T),&0<x<T,\\
1/2-q_a(-x/T),&-T<x<0,
\end{cases}
\]

而 \(q(0)=0\) 令原點左右斜率皆為 1/2。參數 \(a=q'_a(0)\) 控制原點附近曲率，\(T\) 控制非線性區寬度；這提供「少量共享 \((T,a)\) profiles、訓練後逐層靜態凍結」的簡潔模型。

### 8.2 先前技術碰撞

| 相鄰工作 | 已涵蓋內容 | 此假說仍不同的部分 |
|---|---|---|
| Curl | SiLU 的 ReLU／偶對稱 residual 分解與 \(|x|\)-LUT | 不是本文的一參數 quartic derivative、積分與 exact-ReLU finite tail construction |
| xIELU | 明確從想要的 gradient 出發，積分導出 trainable piecewise activation，並用常數約束函數／梯度連續 | 原始 xIELU 是 ELU-derived、LLM-oriented，沒有此 even residual、有限 \(T\) 接 ReLU或 quartic 約束族。[xIELU 原始論文](https://arxiv.org/abs/2411.13010) |
| Smooth-step／Tree Ensemble Layer | 在有限區間用 polynomial，解 boundary value 與 derivative constraints，使區間外為精確常數 | 用途是 differentiable routing gate，不是積分成 SiLU-like detector activation；但「用端點條件解 polynomial」不是新概念。[原始論文](https://proceedings.mlr.press/v119/hazimeh20a/hazimeh20a.pdf) |
| DAPA | activation 與 derivative 的 distribution-aware approximation、fixed-point/HLS | DAPA 是非均勻 PWL 擬合，不是解析受約束一參數 polynomial；但 derivative-aware 與 distribution-aware 不能再單獨主張新穎 |
| GRAU | SiLU 的 PWLF、PoT/APoT slope、integer range、BN/requant fusion | 本假說是低階 polynomial 及 exact ReLU tails；若再做 dyadic coefficient／fusion，仍須正面比較 GRAU |
| ActNAS | YOLO 逐層靜態 mixed-activation placement | 少量共享 profile 與解析 shape 可能縮小搜尋空間，但「逐層凍結」本身不新 |

**檢索結論只能寫成：本次 primary-source 檢索未發現完全相同的「SiLU-even 形式 + 上述一參數 quartic derivative constraints + 積分 + 有限 \(T\) exact-ReLU tails + 少量共享 \((T,a)\) profiles + detector 訓練」組合。這不是 novelty 證明。** 在投稿或專利判斷前，仍需擴大關鍵字、引用鏈與專利檢索。

### 8.3 必做的前置驗證

1. 推導允許的 \(a,T\) 範圍：限制 derivative overshoot、輸出幅度與 fixed-point overflow；確認是否要保證單峰／非單調區數量。
2. 同時 fit SiLU function、derivative 與實際 activation distribution；不可只報均勻區間 MSE，並須與 DAPA、PWLU、等段 PWL 同參數／同硬體預算比較。
3. 用 Horner form 計算積分後的五次 polynomial，逐項量化係數；比較 DSP 次數、critical path、LUT/BRAM 與 PWL 的 comparator/MAC 成本。
4. profile 數先限制為 1、2、3；逐層只選靜態 profile ID。若每層都學獨立 \((T,a)\)，就失去共享硬體與統計效率的主張。
5. 先做 FP32 shape/recovery，再做 fake-quant/QAT，最後才下硬體結論。

## 9. 實驗與重訓決策

### 9.1 三段式漏斗

1. **Zero-shot／BN 控制篩選**：同一 SiLU checkpoint 替換固定候選，先測數值穩定、ONNX export、短 validation、activation distribution；BN 是否重算必須統一記錄。
2. **等成本 recovery**：所有存活候選使用完全相同 epoch、optimizer、LR、augmentation 與 seed 1。學習型 activation 此時才有公平機會。
3. **決賽完整訓練／QAT**：每個資料集獨立從相同初始化與 recipe 訓練；seed 1 先跑，資源允許時對決賽者與臨界結果補 seed 2。單一 seed 只作篩選，不作穩健優越性結論。

### 9.2 每個結果列必須同時有

- 精度：AP、AP50、AP75、AP_S/M/L（BBAT5 再列逐類／固定尺寸分箱）。
- 優化：收斂曲線、gradient norm／異常、activation/derivative 分布、新增參數。
- 軟體部署：匯出成功與否、ONNX graph 中的實際 op、fusion、precision conversion、完整模型 latency／memory、warm-up 與量測統計。
- 硬體準備：compare、mul、shift、add、LUT/BRAM、DSP 的**估計**；在上板前只能標為 proxy，不能標為實測 speedup。
- 公平性：資料版本、checkpoint 起點、seed、訓練成本、BN calibration、PTQ/QAT、compiler 版本與輸入 shape。

## 10. 尚未確認的風險與 Phase 0 出口

- 尚未收到交付模型資料夾，無法確認實際 YOLO 版本、activation 數量／位置、checkpoint、attention/FFN 結構與 exporter。
- 尚未指定最終板卡、NPU/FPGA/CPU、compiler、位寬與 latency／power 約束；因此目前的「硬體友善」只代表原語與文獻風險，不是排序終局。
- ACON、FReLU、PWLU 的 detection 證據來自不同 detector、pretraining 與 COCO protocol，不能把論文 AP 差直接搬到本專案。
- GRAU、DAPA 與 xIELU 是 2024–2026 的近年／預印本工作；本報告確認其原始稿內容，但未完成全面引用鏈、同族論文與專利檢索。
- 新多項式假說的 exact-tail 公式已修正並完成基本符號條件檢查，但尚未驗證可行 \(a,T\) 範圍、逼近誤差、訓練穩定性、量化誤差或 detector AP。

Phase 0 的合理出口不是宣佈勝者。純函數 registry 現已實作 SiLU、Hardswish、ReLU diagnostic、`poly_shift` 與 `poly_quality`；收到資料夾後先建立不可變 SiLU baseline，再依第 9 節漏斗於 BBAT5 v1 與 COCO2017 **分開**收斂候選。只有現行短名單被淘汰時才從本文件的研究 frontier 補位，最終仍以指定硬體的端到端實測決定 profile 與架構。
