# SIPA–BCSP 學術新穎性 Prior-art 稽核與 Claim Chart

日期：2026-08-25
狀態：學術主張稽核初版；不是法律意義的專利檢索，也不是新穎性證明
證據範圍：只採原始論文、arXiv／OpenReview、官方 proceedings 或作者程式碼；文獻敘述均附原始來源。

## 0. 結論先行

目前**不能**把 `SIPA` 或 `BCSP` 這兩個工作名稱本身當作論文貢獻，也不能主張下列任一單獨元件
是新的：SiLU 差分對稱、只算半邊 residual、平滑 polynomial ReLU、有限區間外 exact-ReLU tails、
\(C^2\) 接合、整數 polynomial activation、PoT／APoT、YOLO mixed activation、Pareto 搜尋或
progressive beam search。

本次稽核後，仍可能形成論文貢獻的最窄組合是：

> **零點錨定、SiLU 式非單調的 compact even-residual activation family**：以
> \(A(x)-A(-x)=x\) 為硬限制，從互補導數積分出低自由度 polynomial，在有限 \(T\) 外精確回到
> ReLU 且達 \(C^2\) 接合；再給出 dyadic profile 與在離散 Q-format 中仍保持差分恆等式及兩側
> tail 的 integer realization。

這只能稱為**候選貢獻**，原因是它位於三組已知方法的交集：

- [Curl](https://eprint.iacr.org/2024/1127.pdf) 已利用 `ReLU − SiLU` 是偶函數；
- [SmeLU](https://arxiv.org/abs/2010.09931) 已有 symmetric transition、polynomial middle 與
  exact-ReLU tails；
- [S-ReLU](https://www.mdpi.com/2227-7390/14/1/72) 更已是 \(C^2\) piecewise polynomial、
  compact transition 與 exact-ReLU tails。

SIPA 與它們目前可辨識的差異，是同時要求 \(A(0)=0\)、保留 SiLU 類負谷／非單調導數、使用
特定的一參數 quartic derivative constraint family，以及提供 bit-exact invariant-preserving
integerization。是否足以構成學術新穎性，必須由第 8 節實驗與更完整引用鏈／專利檢索決定。

`BCSP` 現階段應定位為**節省實驗成本、維持公平性的支援方法**，不宜列為獨立核心貢獻。

## 1. 本次被稽核的實際系統

### 1.1 SIPA 數學與實作

現行實作定義：

\[
A_{T,a}(x)=\frac{x}{2}+H_{T,a}(|x|),
\qquad A_{T,a}(x)-A_{T,a}(-x)=x.
\]

其中 \(H\) 由 quartic derivative \(q_a\) 積分而來，並受以下條件限制：

\[
q_a(0)=0,\quad q_a(1)=\frac12,\quad q'_a(1)=0,
\quad \int_0^1 q_a(s)\,ds=\frac12.
\]

這些條件分別支撐原點斜率、ReLU-tail 一階／二階接合與邊界函數值。`poly_shift` 固定
\(a=9/2,T=8\)，內區間係數為
\(9/32,-15/256,11/2048,-3/16384\)。程式實作見
[activations.py](../../src/activation_lab/activations.py)；Q-format 路徑見
[hardware.py](../../src/activation_lab/hardware.py)。

### 1.2 BCSP 實際做的事

現行 [search.py](../../src/activation_lab/training/search.py) 會：

1. 先完成各資料集自己的 SiLU reproduction；
2. 依序執行 uniform zero-shot、同成本 recovery、region one-at-a-time；
3. 對可行 observation 建立 non-dominated frontier；
4. 只從 frontier 擴張一個 region，受 beam width、最多三個 changed regions、最多三種 kernel
   限制；
5. 最多留下 accuracy extreme 與 hardware extreme 兩個 finalists；
6. COCO2017 與 BBAT5 v1 各自建 observation 與 frontier，不平均 raw AP。

這是合理且可稽核的實驗 scheduler，但目前沒有新的 Pareto 理論、搜尋收斂保證、joint
cross-dataset robustness objective 或 learned surrogate。

## 2. Claim chart 總表

| ID | 擬議主張 | 最接近的已知先例 | 碰撞程度 | 目前可安全主張 | 目前不可主張 |
| --- | --- | --- | --- | --- | --- |
| C1 | \(A=x/2+H(|x|)\) 保留 SiLU 差分對稱 | Curl、SmeLU、S-ReLU | 高 | SIPA **精確實作**已知差分不變量 | 首次發現／首次使用 SiLU 對稱 |
| C2 | exact-ReLU-tail、\(C^2\)、integral polynomial、dyadic/APoT | S-ReLU、xIELU、Compact、AFC、GRAU、FQA | 高，但特定組合尚可區辨 | 零點錨定、非單調、受約束一參數族與 dyadic instance 是候選差異 | 首個 polynomial SiLU、首個 \(C^2\) exact-tail activation、首個 APoT activation |
| C3 | complementary rounding 保持 fixed-point symmetry/tails | I-BERT、fixed-point AFC、FQA、GRAU；未找到相同 invariant contract | 中 | 指定 emulator 的 bit-exact 性質與可形式化 lemma | 首個 integer activation、已證明所有硬體／位寬都 exact |
| C4 | YOLO region-level static activation placement | MobileNetV3、YOLOv6 SimSPPF、ActNAS | 直接碰撞 | region abstraction 是受限、易部署的 search-space design | 首次在 YOLO 混用或逐區選 activation |
| C5 | hardware-cost Pareto／beam staged search、COCO/BBAT 分離 frontier | NSGA-Net、PNAS、PNAG、ActNAS、NAS-Bench-201 | 高 | BCSP 是本研究的成本控制與公平 protocol | 新 Pareto／beam 演算法；跨資料集 robust optimization 已成立 |

## 3. C1：差分對稱不是新穎性

SiLU \(S(x)=x\sigma(x)\) 滿足 \(S(x)-S(-x)=x\)。本專案以
\(A(x)=x/2+H(|x|)\) 強制保留這個恆等式，數學上正確，但不是新發現。

### 已知先例

- [Curl](https://eprint.iacr.org/2024/1127.pdf) 明確觀察
  \(D_{\mathrm{SiLU}}(x)=\operatorname{ReLU}(x)-\operatorname{SiLU}(x)\) 是偶函數，並只對
  \(|x|\) 建 LUT。這已直接覆蓋「利用 SiLU 正負半軸關係只算半邊 residual」。
- [SmeLU 原始論文](https://arxiv.org/abs/2010.09931) 的中段是
  \((x+\beta)^2/(4\beta)\)，區間外為 0 或 \(x\)。**由其公開公式可直接推得**
  \(\operatorname{SmeLU}(x)-\operatorname{SmeLU}(-x)=x\)；這是本稽核的數學推論，不是說
  原作者以該名稱宣告此性質。
- [S-ReLU 原始論文](https://www.mdpi.com/2227-7390/14/1/72) 在轉換區內可寫成
  \(x/2\) 加偶次 polynomial，區間外為 0 或 \(x\)。因此**由其公式同樣可推得**差分恆等式。

### SIPA 的差異

Curl 保留 SiLU 本身的零點與非單調形狀，但 residual 只漸近衰減；SmeLU／S-ReLU 有有限
exact-ReLU tails，卻是為 smooth ReLU 設計，且分別有
\(A(0)=\beta/4\) 與 \(A(0)=3\delta/16\)。SIPA 則固定 \(A(0)=0\)，並以 derivative overshoot
保留 SiLU 類負輸出與非單調性。

### 判定

- **安全主張：**「SIPA 將已知 SiLU 差分恆等式提升為設計與量化時不可破壞的 invariant。」
- **不可主張：**「我們首次發現 SiLU 對稱性」或「首次利用偶 residual 降低硬體成本」。

## 4. C2：exact tails、C2、積分與 dyadic 的真正邊界

### 4.1 各元件都有先例

- [SmeLU](https://arxiv.org/abs/2010.09931) 已用 quadratic transition 接上兩側 exact-ReLU
  tails，並使一階導數連續。
- [S-ReLU](https://www.mdpi.com/2227-7390/14/1/72) 已透過 compact-support kernel 得到
  piecewise polynomial 的 \(C^2\) smooth ReLU，且在有限區間外精確為 ReLU。故
  「polynomial + \(C^2\) + exact tails」三者合併仍不能單獨宣稱新穎。
- [xIELU](https://arxiv.org/abs/2411.13010) 已從想要的 gradient 出發，再以積分導出 activation；
  因此「先設計導數再積分」不是新的總體方法。
- [Compact](https://petsymposium.org/popets/2024/popets-2024-0065.php) 已針對 SiLU、GELU、Mish
  做 input-density-aware piecewise polynomial approximation；
  [Hardware-Based Activation Function-Core](https://www.mdpi.com/2079-9292/11/1/14) 已把 SiLU
  納入 fixed-point piecewise-polynomial FPGA activation core；
  [PolySwish](https://www.mdpi.com/2504-3110/8/8/444) 也已公開 polynomial Swish 形式。
- [APoT](https://openreview.net/pdf?id=BkgXT24tDS) 已用 powers-of-two sums 表示量化 level；
  [GRAU](https://arxiv.org/abs/2602.22352) 更已把 PoT／APoT slope 用於含 SiLU 的硬體
  activation approximation。故 dyadic／shift-add 係數本身不是貢獻。
- [FQA](https://arxiv.org/abs/2606.05627) 已把 coefficient quantization、fractional word length
  與 piecewise polynomial hardware search 放入同一設計流程，表示「量化空間內找 polynomial
  係數」也已有直接先例。

### 4.2 尚可能區辨的組合

SIPA 目前最值得保留的數學差異不是「smooth ReLU」，而是以下條件的交集：

1. `zero anchored`：\(A(0)=0\)，而非 SmeLU／S-ReLU 在原點的正偏移；
2. `SiLU-like non-monotonicity`：允許負半軸負谷與導數小於 0，正半軸對應導數可大於 1；
3. `finite exact tails`：\(|x|\ge T\) 後精確為 ReLU，不只是漸近接近；
4. `C2 gluing`：函數、一階與二階導數都接到 tail；
5. `constrained integral family`：所有滿足四個線性限制的 quartic \(q_a\) 形成顯式一參數族，
   而不是先自由擬合大量 coefficients；
6. `hardware instance`：從此族選出 \(a=9/2,T=8\)，使 threshold 與 direct coefficients
   可用二進位 shift-add 表示。

本次 primary-source 檢索**沒有找到完全相同的六項組合**，但這不是新穎性證明。尤其 S-ReLU
已消除其中第 3、4 項作為獨立 contribution 的可能，Curl 已消除「SiLU 偶 residual」的獨立
主張。

### 4.3 建議的安全論文措辭

在完成第 8 節前，只能寫：

> 「我們研究一個 zero-anchored、non-monotone、symmetry-constrained integral-polynomial
> family；它在有限 transition 外精確回到 ReLU，並提供一個 dyadic deployment profile。」

不能寫「首個 polynomial SiLU」、「首個 smooth exact-ReLU activation」或「首個 hardware-friendly
SiLU approximation」。

## 5. C3：Q-format invariant 是較有機會的窄貢獻

### 5.1 已知先例與尚未找到的部分

- [I-BERT](https://proceedings.mlr.press/v139/kim21d.html) 已用 integer-only polynomial 計算
  GELU 等非線性，證明整數 polynomial activation 不是新概念。
- [Hardware-Based Activation Function-Core](https://www.mdpi.com/2079-9292/11/1/14) 已給出
  SiLU piecewise polynomial 的 fixed-point coefficients 與 FPGA core。
- [GRAU](https://arxiv.org/abs/2602.22352) 已處理 SiLU、PoT／APoT、mixed precision、BN folding
  與 output requantization。
- [FQA](https://arxiv.org/abs/2606.05627) 已把截位與係數量化誤差直接納入 full-space search；
  [AQuant](https://arxiv.org/abs/2208.11945) 則顯示 activation rounding policy 本身早已是可優化
  對象。

本次檢索未找到上述方法明確要求並證明「量化後仍對所有成對可表示整數輸入滿足
\(\widehat A(n)-\widehat A(-n)=n\)，且 tail 為 0 LSB 誤差」。這是目前相對乾淨的窄缺口，但仍
需要擴大檢索。

### 5.2 現行 rounding 可形式化成一般 lemma

現行 emulator 使用 signed arithmetic right shift：

\[
\widehat A(n)=\left\lfloor\frac n2\right\rfloor+\widehat H(|n|).
\]

只要正負輸入共用完全相同的 \(\widehat H(|n|)\)，便有：

\[
\widehat A(n)-\widehat A(-n)
=\left\lfloor\frac n2\right\rfloor-\left\lfloor\frac{-n}{2}\right\rfloor=n.
\]

tail 再令 \(\widehat H(u)=\lceil u/2\rceil\)，即可得到正尾 \(\widehat A(n)=n\) 與負尾
\(\widehat A(-n)=0\)。這比單純報告某個 Q16.10 grid 的 0 LSB 更有學術價值，因為它可以寫成
與 polynomial 係數無關的一般 invariant-preserving integerization lemma。

### 5.3 目前可主張與不可主張

- **可主張：**目前 reference emulator 在指定 two's-complement／rounding contract 下保留差分與
  tail；上述式子可進一步形式化證明。
- **不可主張：**已證明任意位寬、任意 saturation、任意 compiler lowering 或實體 RTL 都會保持
  性質；也不能把 0 LSB 直接等同 AP、latency、area 或 power 優勢。

## 6. C4：YOLO region-level static placement 是工程約束，不是首創

### 已知先例

- [MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html)
  已按網路深度混用 ReLU 與 h-swish，顯示不同 stage 使用不同 activation 是既有設計模式。
- [YOLOv6 作者程式碼](https://github.com/meituan/YOLOv6/blob/main/yolov6/layers/common.py) 的
  `SimCSPSPPF` 明確以 ReLU 簡化特定 detector block；對應原始模型報告見
  [YOLOv6 v3.0](https://arxiv.org/abs/2301.05586)。
- [ActNAS](https://openaccess.thecvf.com/content/CVPR2025W/MAI/html/Sah_ActNAS__Generating_Efficient_YOLO_Models_using_Activation_NAS_CVPRW_2025_paper.html)
  已對 YOLO 逐層選 ReLU、SiLU、Hardswish、ReLU6、LeakyReLU，並把 target-device latency、
  memory 與 accuracy constraint 放入 random／ILP search。

### BCSP 的合理差異

BCSP 不先假設各層成本／精度 delta 可線性相加，而是以完整 policy recovery observation 逐步擴張；
region abstraction、最多三個 kernels 與 manifest 靜態凍結，也比任意逐層組合更容易部署與稽核。
這些是有價值的設計差異，但尚未由實驗證明搜尋成本或最終 frontier 優於 ActNAS。

- **安全主張：**「我們採 semantic-region constrained placement，以減少候選並限制 deployment
  kernel diversity。」
- **不可主張：**「首次在 YOLO 混用 activation」、「首次做 hardware-aware activation NAS」。

## 7. C5：Pareto、beam 與雙資料集 frontier 的邊界

### 7.1 搜尋元件已有先例

- [NSGA-Net](https://arxiv.org/abs/1810.03522) 已用 non-dominated multi-objective search 同時處理
  error 與 computation complexity。
- [Progressive NAS](https://openaccess.thecvf.com/content_ECCV_2018/html/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.html)
  已按複雜度逐步擴張候選，並以 top-\(K\) beam 保留下一輪架構。
- [PNAG](https://openaccess.thecvf.com/content/CVPR2023W/NAS/html/Guo_Pareto-Aware_Neural_Architecture_Generation_for_Diverse_Computational_Budgets_CVPRW_2023_paper.html)
  已直接學習跨不同 computational budgets 的 Pareto architectures。
- [ActNAS](https://arxiv.org/abs/2410.10887) 已將 YOLO activation placement 與 latency／memory／
  accuracy budgets 結合。

因此，「從 non-dominated candidates 擴張」、「限制 beam」與「用 accuracy/hardware 向量而非
單一 weighted sum」都不能單獨當作新演算法。

### 7.2 COCO／BBAT5 分開是正確 protocol，但不是 joint robustness algorithm

[NAS-Bench-201](https://arxiv.org/abs/2001.00326) 已在多個資料集分別提供同一架構空間的結果；
[NASTransfer](https://ojs.aaai.org/index.php/AAAI/article/view/17121) 也顯示 source/proxy dataset
會改變 NAS 方法排名與轉移表現。這支持本專案不應平均 COCO 與 BBAT5 raw AP，但「分開報告」
本身是實驗衛生，不是新優化法。

更重要的是，現行 BCSP **沒有**跨兩資料集求一個共同 non-dominated policy；它只是讓兩個 planner
各自前進。故目前只能主張：

- observations、frontiers、checkpoints 與 raw metrics 不混合；
- 同一 activation family 可在兩個獨立 detector 任務上驗證。

不能主張已完成「cross-dataset robust optimization」。若要形成此主張，必須實作 shared-profile／
shared-policy constraint，對 \((\Delta\mathrm{COCO},\Delta\mathrm{BBAT5},\mathrm{cost})\) 做共同
frontier 或 frozen worst-case regret，並與 independent oracle 比較。

## 8. 驗證學術新穎性所需的完整實驗

### 8.1 M0：形式化數學與最小度數

1. 證明所有滿足四個 boundary／integral constraints 的 degree-4 \(q\) 恰形成現行一參數族。
2. 證明在指定 \(a\) 範圍內的負谷數量、導數 extrema、輸出界與 non-monotonic interval。
3. 回答「要同時滿足 zero anchor、可調 SiLU-like overshoot、\(C^2\) exact tails 時，degree 4 的
   derivative／degree 5 的 integral 是否為最小自由度」；若不是，就縮成更低成本族。
4. 對 SmeLU、S-ReLU 與 SIPA 列出 \(A(0)\)、tail、smoothness、導數範圍、負谷、degree、乘法數的
   解析比較。

**出口條件：**若無法提出比「任選 polynomial 解線性條件」更強的結構定理，數學貢獻只能降為
hardware co-design component。

### 8.2 M1：同預算函數與導數比較

最低必要 prior-art controls：

- exact SiLU；
- Hardswish；
- SmeLU（\(C^1\) exact-tail symmetry control）；
- S-ReLU（\(C^2\) exact-tail symmetry control，最重要）；
- SIPA `poly_shift`／`poly_quality`；
- ReLU 只作 primitive floor。

另以 [Compact](https://arxiv.org/abs/2309.04664) 類 piecewise polynomial 或等段 PWL 做
equal-cost approximation control；硬體比較需加入 GRAU-style PoT/APoT PWL。所有候選都報：

- uniform-grid 與 empirical pre-activation distribution 的 function／derivative MSE、MAE、max error；
- negative-valley location/value、derivative extrema、tail error；
- coefficient bytes、比較器、一般乘法、常數乘法、shift、add、critical dependency depth；
- 相同 degree、相同 multiplier budget、相同 coefficient-bit budget 三種公平比較。

### 8.3 M2：整數不變量與 rounding ablation

1. 對 Q8、Q12、Q16 等候選格式，枚舉所有可成對表示的輸入 code，而非只測均勻 float grid。
2. 形式化並 property-test：difference identity、正／負 tail、單調區、overflow、saturation 與
   intermediate range。
3. 比較 nearest-even、away-from-zero、floor、現行 paired floor/ceil，以及對正負半軸各自量化的
   naive implementation。
4. 消融「保持 invariant」是否真的降低 layer-output bias、累積 feature drift、PTQ/QAT AP loss。
5. 將 reference emulator、compiler IR、HLS C model 與 RTL 做逐 code bit-match。

**出口條件：**只有在 bit-exact 性質能轉成較低誤差、較少修正邏輯或更佳硬體／模型結果時，
integerization 才足以列為論文貢獻；純代數技巧本身可能過窄。

### 8.4 M3：真實 detector 的 uniform activation 實驗

COCO2017 與 Canonical BBAT5 v1 必須使用各自不可變 baseline、checkpoint 與 evaluator，分欄報告：

1. all-SiLU reproduction；
2. 同 checkpoint zero-shot；
3. 同 optimizer、iterations、LR、augmentation 的 short recovery；
4. 只對 finalists 做 full recovery／必要 QAT。

每列至少包含 AP、AP50、AP75；COCO 加 AP_S/M/L，BBAT5 加 ball/bat AP。另報收斂曲線、gradient
norm、NaN/Inf、activation／derivative distribution、negative ratio 與 saturation ratio。

使用者暫定 seed 1、資源允許 seed 2，可用於篩選；但**1–2 seeds 只能支持 exploratory claim**。
若要主張穩定優越，finalists 至少需要更多獨立 runs 或可辯護的 confidence interval／paired
bootstrap。

### 8.5 M4：placement／搜尋演算法比較

在相同候選、相同 recovery budget、相同 target profile 下比較：

1. uniform activation；
2. naive early-region replacement；
3. region greedy one-at-a-time；
4. ActNAS-style additive table + ILP／random search；
5. BCSP Pareto beam。

必報 candidate evaluations、實際 training GPU-hours、search wall time、frontier hypervolume、
final AP–latency／area trade-off、kernel count，以及多次 search run 的選擇穩定度。另做以下消融：

- Pareto vs. weighted scalar；
- region vs. layer search；
- 只用結構 proxy vs. 加 target latency；
- beam width、changed-region budget、kernel budget；
- one-at-a-time observation vs. interaction-aware recovered policies。

**出口條件：**若 BCSP 只是在較少實驗下得到相近解，將它報成 efficient protocol；只有在同搜尋
預算下穩定改善 frontier，才考慮演算法 contribution。

### 8.6 M5：跨資料集 robustness，而不是 AP 平均

凍結三種 protocol：

- `D0 independent oracle`：兩資料集各自 profile 與 assignment；
- `D1 shared profile`：共享 \((a,T)\)／量化係數，assignment 可不同；
- `D2 shared policy`：只有 manifests 的 semantic regions 可對齊時，才共享 profile 與 assignment。

報告 \((\Delta\mathrm{COCO},\Delta\mathrm{BBAT5},\mathrm{cost})\) 向量、各自 frontier、共同可行率
與事先凍結的 worst-case normalized regret。不得把 raw AP 相加，也不得看完正式 validation 再
調權重。

### 8.7 M6：真正硬體證據

現有 ONNX operator count 只算 proxy。要宣稱 hardware-friendly，至少需要：

1. 固定板卡／製程、runtime/compiler、precision、batch、input shape；
2. custom kernel 或 HLS/RTL，證明 dyadic constants 確實 lowering 為 shift/add；
3. synthesis／implementation 的 DSP、LUT、FF、BRAM、Fmax、critical path、area、power；
4. activation microbenchmark 與完整 detector latency／memory／energy，含 warm-up、同步與重複統計；
5. 對 fused SiLU、fused Hardswish、S-ReLU/SmeLU，以及 GRAU-style PWL 的同條件比較。

四次 data-power multiplication 必須照實計入。若目標 backend 已有高效 fused SiLU/Hardswish，
`poly_shift` 可能反而更慢；只有實測能決定。

## 9. 建議的論文貢獻層級

### 最穩健的三層敘事

1. **主要候選貢獻：**zero-anchored、non-monotone、symmetry-constrained integral-polynomial
   family，以及它相對 Curl／SmeLU／S-ReLU 的解析差異。
2. **次要候選貢獻：**一般化的 invariant-preserving fixed-point realization，若能證明並在 RTL／
   detector 中顯示實際效益。
3. **支援方法：**BCSP 作為低預算、region-constrained 的 co-design evaluation protocol；除非
   M4 顯示明確優勢，不列獨立演算法創新。

### 現階段絕對不可寫的句子

- 「首次利用 SiLU 對稱性。」
- 「首次提出 polynomial／hardware-friendly SiLU。」
- 「首次提出具有 exact ReLU tails 或 \(C^2\) 的 polynomial activation。」
- 「首次讓 YOLO 不同層／區域使用不同 activation。」
- 「首次以 Pareto 或 beam search 做硬體感知 activation placement。」
- 「已證明比 SiLU 更準、更快、更省面積／功耗。」
- 「COCO 與 BBAT5 已證明跨資料集泛化。」

## 10. 最終稽核判定

| 項目 | 現況判定 | 可否當核心 contribution |
| --- | --- | --- |
| SiLU 差分對稱／even residual | 明確有 Curl、SmeLU、S-ReLU 先例 | 否 |
| \(C^2\) exact-ReLU-tail polynomial | S-ReLU 已直接覆蓋 | 否 |
| zero-anchor + non-monotone + constrained integral family | 本次未找到完全相同組合 | **候選，可以 M0–M3 驗證** |
| dyadic／APoT coefficients | APoT、GRAU 等已覆蓋一般概念 | 否，僅 co-design 組件 |
| fixed-point exact difference/tails | 未找到相同明示 contract | **窄候選，可以 M2/M6 驗證** |
| YOLO region placement | ActNAS 等直接碰撞 | 否，工程約束 |
| Pareto／beam staged search | NAS 文獻已有大量先例 | 否，除非 M4 證明新 objective／效率優勢 |
| COCO/BBAT 分離 frontier | 正確實驗 protocol | 否 |

因此，學術路線不應包裝成「一個新 activation 名稱加一個新搜尋縮寫」，而應集中證明一件窄而
清楚的事：**能否在不破壞 SiLU-like zero crossing／negative valley 的前提下，以 compact、\(C^2\)、
exact-tail、可 bit-exact integerize 的低自由度函數族，得到 detector accuracy 與實體硬體成本的
可重現 Pareto 改善。**在這個問題被 M0–M6 回答之前，所有新穎性結論都維持「候選」狀態。
