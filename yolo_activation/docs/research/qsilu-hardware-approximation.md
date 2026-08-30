# qSiLU／SiLU 硬體友善近似研究與候選規格

日期：2026-08-27
狀態：研究結論與候選預規格；尚未代表 Full35 模型精度或實機硬體結果
證據邊界：外部主張只採原始論文、作者／學校典藏、官方程式碼與 ONNX 規格；本文提出的
`qsilu_pq` 公式是本專案的可驗證設計，不冒充既有論文公式。

## 1. 結論

建議新增一個候選，registry key 固定為 `qsilu_pq`，報告名稱寫成
「QSILU-PQ（本專案 C1 dyadic piecewise-quadratic SiLU）」：

- 保留 SiLU 的核心恆等式 \(A(x)-A(-x)=x\)、原點值與斜率；
- 在 $|x|\ge 8$ 精確接到 ReLU tails；
- 在 $0,1,2,4,8$ 全部保持函數值與一階導數連續（$C^1$）；
- 每筆輸入只需一次 \(u^2\)；區段係數全為 denominator \(2^{10}\) 的 dyadic/APoT，
  每個常數最多四個 signed power-of-two 項；
- float reference 可以用標準 ONNX `Abs/Less/Where/Mul/Add` 表達；真正的 signed fixed-point
  shift-add 仍應由 bit-accurate emulator 與目標 backend lowering 保證。

這是目前最值得加進 repo 的「品質／成本平衡」候選。若最終 FPGA／ASIC 的 DSP 比 LUT/BRAM
更稀缺，第二順位不是更高次多項式，而是以實際 activation histogram 最佳化的 non-uniform PWL
或小型 LUT；若部署平台已有原生 fused SiLU，則原生 SiLU 可能仍是最快的實際方案，必須以板上
量測而不是 operator 數量判定。

## 2. 名稱稽核：qSiLU／QGELU 並沒有唯一公認含義

SiLU 的原始定義是

\[
\operatorname{SiLU}(x)=x\sigma(x),
\]

而且具有非單調負谷與大幅值時接近 ReLU 的性質；這些都是近似時應保留的 target，不應先換成
另一個 activation 再宣稱是「SiLU 的硬體實作」。來源見
[Elfwing、Uchibe、Doya 的 SiLU 原論文](https://arxiv.org/abs/1702.03118)。

`QGELU` 已至少有三種互不相同的第一手用法：

1. 陳法諭 2025 年臺大碩論的 QGELU 是 $[-3,3]$ 內七個二次區段、區外為 0／identity，
   且係數設計成 shift-add；論文用實際 GELU 輸入分布產生 100,000 筆樣本，報告
   $1.87\times10^{-5}$ MSE，訓練時 forward 用 QGELU、backward 改用原 GELU 梯度，並在
   輸入／輸出模擬 truncate。這是一個具體且有價值的 QGELU 設計，但 target 是 GELU、模型是
   Transformer，不可直接把其係數或晶片節省比例移植到本專案的 SiLU/YOLO。
   [臺大典藏與摘要](https://tdr.lib.ntu.edu.tw/handle/123456789/100980)、
   [公開全文](https://tdr.lib.ntu.edu.tw/retrieve/9657822c-043f-48a0-982c-7b27ebf6b824/ntu-114-1.pdf)。
2. Q-S5 論文把基於 `ReLU4`、可用 bit shift 代替除法的另一個函數也稱為 qGELU；作者還指出
   低於 8-bit 時會系統性低估 GELU。因此「qGELU」本身不能唯一指向 piecewise quadratic。
   [Q-S5 Appendix A.3](https://arxiv.org/html/2406.09477v1#A3.SS3)。
3. NVIDIA Transformer Engine 的 `QGELU` 是 QuickGELU，公式為
   $x\sigma(1.702x)$，完全不是上述兩種量化／分段二次版本。
   [NVIDIA 官方 API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.ops.QGELU)、
   [官方數值測試程式碼](https://github.com/NVIDIA/TransformerEngine/blob/main/tests/pytorch/test_numerics.py)。

`QSiLU` 也不是安全的通用名稱：2025 年已有原始論文用 QSiLU 表示逐一作用在 quaternion
虛部通道上的 Quaternion SiLU，而不是 quantized/quadratic SiLU。
[QResNet 原論文公式 (18)](https://www.csroc.org.tw/journal/JOC36-1/JOC3601-06.pdf)。

因此本專案不應只登錄 `qsilu`。穩定 key 建議用 `qsilu_pq`，所有表格同時寫出「本專案公式、
版本、係數 hash」，避免和 Quick、Quantized、Quadratic、Quaternion 四種 Q 前綴混淆。

## 3. 為何 piecewise quadratic 值得加入，但不能只看「沒有一般常數乘法器」

HardSwish 本身已是最簡單的 piecewise-quadratic Swish 近似：ONNX 定義為

\[
y=x\max(0,\min(1,x/6+1/2)),
\]

其標準 graph 是 `HardSigmoid + Mul`；這使它具有成熟的 framework/backend 支援，但係數 $1/6$
不是 power-of-two，且區段接點只有 $C^0$ 而非 $C^1$。
[ONNX HardSwish 規格與 function body](https://onnx.ai/onnx/operators/onnx__HardSwish.html)、
[MobileNetV3 原論文](https://arxiv.org/abs/1905.02244)。

已發表的 SiLU 硬體核心也支持「piecewise quadratic 是合理路線」：González-Díaz-Conti 等人
用八個區段、二次多項式與 fixed-point 係數在 Artix-7 上實作 SiLU；該設計的 SiLU input/output
使用 16-bit、11 fractional bits，係數使用 16-bit、13 fractional bits。它證明方法可落地，
但其係數是一般 fixed-point 常數，並非本專案要求的 APoT-only。
[原始論文與 SiLU coefficient table](https://doi.org/10.3390/electronics11010014)。

此外，Choi 等人的 peer-reviewed Swish 專用硬體近似報告其 delay、area、power 優於 Hardswish；
在取得可核對全文公式與相同製程／位寬的 synthesis 條件後，應列為硬體先前技術 baseline，不能只
根據摘要重建公式。
[KAIST 作者典藏紀錄與 DOI](https://pure.kaist.ac.kr/en/publications/hardware-friendly-approximation-for-swish-activation-and-its-impl/)。

本次可核對的一手來源中，沒有找到把「\(A(x)=x/2+h(|x|)\)、節點 1、2、4、8、
\(C^1\)、dyadic coefficients、exact-ReLU tails」這個精確組合命名為 qSiLU 的來源；但這只支持
使用專案限定名稱，不構成新穎性證明。廣義的「分段二次 SiLU」已有上述 2022 年實作，Swish 專用
硬體近似也已有先前工作。因此目前只能把特定係數組、限制條件與驗證流程寫成「本專案候選」；若要
主張方法創新，仍需完整論文與專利檢索，以及同位寬 synthesis 與模型品質證據。

關鍵成本事實是：二次函數 $ax^2+bx+c$ 就算 $a,b,c$ 全能用 shift-add，$x^2$ 仍是一個
data-by-data squaring。故 `qsilu_pq` 的正確主張應是「一個平方器、沒有一般常數乘法器」，而不是
「零乘法器」；是否佔用 FPGA DSP 取決於目標 RTL/HLS lowering 與 synthesis。

## 4. 首選候選 `qsilu_pq`

### 4.1 數學形式

利用精確恆等式

\[
\operatorname{SiLU}(x)=\frac{x}{2}+
\frac{|x|}{2}\tanh\left(\frac{|x|}{2}\right),
\]

定義 $u=|x|$ 與

\[
\operatorname{QSILU\mbox{-}PQ}(x)=\frac{x}{2}+h(u),
\]

其中初始凍結版本為：

| $u$ 區間 | $h(u)=Au^2+Bu+C$ |
| --- | --- |
| \(0\le u<1\) | \(\frac{57}{256}u^2\) |
| \(1\le u<2\) | \(\frac{23}{256}u^2+\frac{17}{64}u-\frac{17}{128}\) |
| \(2\le u<4\) | \(-\frac{11}{512}u^2+\frac{91}{128}u-\frac{37}{64}\) |
| \(4\le u<8\) | \(-\frac{5}{1024}u^2+\frac{37}{64}u-\frac{5}{16}\) |
| $u\ge8$ | \(\frac12u\) |

這組係數是本專案的 constrained dyadic 初始設計，不是外部論文的抄錄。其解析性質可直接驗證：

- \(A(0)=0\)、$A'(0)=1/2$；
- 在 $u=1,2,4,8$，左右函數值與一階導數完全相同；
- $u\ge8$ 時，正輸入輸出為 $x$，負輸入輸出為 0，因此是 exact ReLU tails；
- 因 $h$ 為偶函數，對所有 $x$ 都有 \(A(x)-A(-x)=x\)；
- 二階導數在 knots 可跳變，所以此候選是 $C^1$、不是 $C^2$。

等價的 $C^1$ truncated-power 表示為：

\[
h(u)=\frac{57}{256}u^2
-\frac{17}{128}[u-1]_+^2
-\frac{57}{512}[u-2]_+^2
+\frac{17}{1024}[u-4]_+^2
+\frac{5}{1024}[u-8]_+^2.
\]

此表示方便做連續性證明；實際 graph/RTL 應使用前一個 coefficient-select 形式，才能共享唯一的
$u^2$，不要在一般 static runtime 同時計算五個平方項。

### 4.2 純函數初算，不等同模型 AP

以下是本專案用 float64、`[-8,8]` 等距 400,001 點所得的可重現數值；只衡量函數本身，不支持
任何 COCO／BBAT5 accuracy 結論：

| 候選 | MAE vs SiLU | RMSE vs SiLU | max abs error vs SiLU |
| --- | ---: | ---: | ---: |
| `qsilu_pq` | 0.0051297 | 0.0060978 | 0.0108319 |
| repo `poly_quality` | 0.0094349 | 0.0113170 | 0.0210018 |
| repo `poly_shift` | 0.0139339 | 0.0165891 | 0.0269959 |
| Hardswish | 0.0462761 | 0.0606127 | 0.1422776 |

`qsilu_pq` 在同一網格的一階導數 MAE 為 0.0072994、max absolute derivative error 為
0.0342158，候選導數範圍為 \([-0.125,1.125]\)。另在 `[-12,12]` 等距 600,001 點上，
MSE 為 \(2.51284\times10^{-5}\)，max absolute error 仍為 0.0108319。這些數字顯示它值得進
zero-shot gate，但不能拿來預測 detector AP；正式排名必須改用完整 training split 的 site/region
histogram 加權誤差與完整 validation。

### 4.3 初始硬體契約

- breakpoints 固定為 `1, 2, 4, 8`，比較器與常數皆為 power-of-two 友善；
- 所有 \(A,B,C\) 都可寫成 \(n/1024\)，constant multiply 可分解為 shift/add/sub；
  \(91/128\) 的普通 binary 有五個非零 bits，但 signed-digit 可寫成
  \(1-1/4-1/32-1/128\)，所以各係數最多四個 signed shift 項；
- 每元素資料路徑：`abs -> range decode/coefficient mux -> one square -> two constant shift-add
  networks -> adds -> output saturation`；
- fixed-point 原型先沿用 Q16.10 input/output，平方與累加至少用 signed int32，bit-accurate
  reference 可先用 int64 防止 emulator 中間溢位；
- 必須另外報告 comparator、mux、register、rounder 與 saturation 成本，不能只計平方器；
- 直到 RTL/HLS synthesis 前，hardware proxy 只能寫「1 variable square、dyadic constants」，
  不得寫實際 latency、energy、area 或 DSP 數。

## 5. 其他方法何時可能更好

| 方法 | 精度／連續性 | 主要硬體代價 | 對本 repo 的判斷 |
| --- | --- | --- | --- |
| 標準 Hardswish | 中等誤差、$C^0$ | 一次 data multiply、clip、$1/6$ scale；常有 fused op | 必留作成熟 hardware-neighbor baseline，不是首選品質候選 |
| `qsilu_pq` | 四內部區段、$C^1$、exact tails | 一平方器、四 threshold、coeff mux、shift-add | 首選新增候選；比現有 degree-5 path 少 variable multiplies |
| non-uniform PWL | 可 $C^0$，斜率在 knots 跳變 | comparator/LUT/mux；dyadic slope 時可無平方器 | DSP 極少時的首選備案；需較多區段與係數儲存 |
| 小型 LUT／LUT+linear interpolation | 量化格點上可很準 | ROM/BRAM/LUT、index、interpolator | 若板上 BRAM/LUT 充裕，可能優於 polynomial；需板上量測 |
| 全域低階 polynomial | 區間內平滑 | Horner multiplies；區外可能爆掉 | 不適合直接覆蓋無界 SiLU；必須加 clip/exact tails |
| Padé／rational | 少量係數可有高精度與光滑性 | denominator、reciprocal/division、zero/overflow 防護 | 只有 target 已有可共享 reciprocal SFU 時才值得 |
| CORDIC／近似 tanh | 可共享多種 transcendental function | iterative shift-add、較多 cycles/control | 多函數共用 SFU 時合理；單一 SiLU 通常不如 PQ/PWL 直接 |
| 原生 fused SiLU | 數學精確 | backend 專用 SFU/kernel | GPU/NPU 已原生支援時可能實際最快，不能強迫展開自訂 graph |

PWL 不應因「只有一次項」就被排除。Flex-SFU 用 non-uniform piecewise interpolation 與 32 個
segments，在超過 700 個 vision/NLP 模型的評估中做到可忽略的模型精度影響，並明確量測其
accelerator area/power overhead；這支持以實機成本共同最佳化 breakpoints，而不是預設多項式一定
較好。[Flex-SFU 原始論文](https://arxiv.org/abs/2305.04546)。Auto-LUT 也把 approximation error、
segment 數與 bit width 放在同一個硬體感知搜尋中，並以真實 FPGA 實作驗證；其公開硬體節省數字
是 Sigmoid case，不能直接當作 SiLU 數字。
[Auto-LUT 原始論文](https://confcats-event-sessions.s3.amazonaws.com/iscas23/papers/1791.pdf)。

對純 INT8 datapath，小型 LUT 也可能勝出。FQ-PETR 的 DULUT 以兩級 32+32 entries 處理包括
SiLU 的非線性，作者報告 approximation error 低於 0.1%；但那是該量化系統與任務的結果，仍需
在本模型、相同位寬與相同 memory primitive 重做。
[FQ-PETR 原始論文](https://ojs.aaai.org/index.php/AAAI/article/download/38204/42166)。

若希望共用一個 tanh/Sigmoid/SiLU 特殊函數單元，K-TanH 提供以低精度 integer、shift-add 與
可放入 CPU register 的小表格做 tanh 的原始路線，論文明確指出可衍生 Swish/GELU；這是共享
SFU baseline，不是 `qsilu_pq` 的直接替代公式。
[K-TanH 原始論文](https://arxiv.org/abs/1909.07729)。

## 6. 係數與量化策略

### 6.1 第一輪先凍結，不用 validation 調公式

先以本文件的 `[1,2,4,8] + Q10 coefficients` 跑完整 E0、profile-weighted function error、uniform
zero-shot 與 region zero-shot sensitivity。第一輪固定公式的理由是隔離「activation family 效果」
與「對 Full35 feature distribution 調參效果」，避免同時改函數與 placement。

### 6.2 若要重擬合，使用 joint、受限制且可合成的目標

只使用完整 COCO2017 train 與 Canonical BBAT5 v1 formal train 的 activation histogram；兩任務
分別正規化後，最小化 worst-task objective，而不是讓 COCO 的元素數量淹沒 BBAT5：

\[
\min_\theta\max_{d\in\{\mathrm{COCO},\mathrm{BBAT5}\}}
\left(\operatorname{RMSE}_{p_d}(A_\theta,\operatorname{SiLU})
+\lambda\operatorname{RMSE}_{p_d}(A'_\theta,\operatorname{SiLU}')\right),
\]

並加上硬限制：

- $A(0)=0,A'(0)=1/2$；
- $C^1$ knots 與 $|x|\ge T$ exact-ReLU tails；
- shared coefficients，禁止每資料集／每 layer 產生另一套硬體函數；
- breakpoints 優先取 integer/power-of-two；coefficients 投影到 denominator $2^{10}$；
- 每個係數最多四個 signed power-of-two 項，並把 comparator、coefficient storage、adder depth
  加進 cost；
- validation 只作一次凍結後評估，不回饋 coefficient search。

外部 QGELU 與 Auto-LUT 都顯示「依實際輸入分布設計近似」是合理做法，但本專案必須以 training
profile 避免 validation leakage，且同時保護 COCO 與 BBAT5，不能照抄 Transformer/GELU 分布。

### 6.3 INT8／INT16 注意事項

dyadic coefficient 不自動等於整條 quantized graph 都只有 shifts。若 activation scale $s_x$ 是
任意實數，$x^2$ 的 scale 是 $s_x^2$，轉回 output scale 仍可能需要一般 requant multiplier；
只有 power-of-two scale、或 backend 能把 requant 合併／共享時，shift-add 優勢才會完整落地。

因此需要兩種獨立驗證：

1. Q16.10 bit-accurate emulator：rounding mode、negative right shift、intermediate width、saturation、
   symmetry LSB 與 tail LSB 全部凍結；
2. 真實 INT8 PTQ/QAT graph：使用部署 backend 的 scale/zero-point 與 requant 規則量測；若 PTQ
   超出 accuracy budget，才對 `round/clamp` 使用 STE 做短 QAT，不要因 piecewise 公式本身是
   $C^1$ 而套用 QGELU 的 GELU-backward surrogate。

## 7. ONNX 靜態圖與 backend 邊界

float/QDQ reference 建議匯出 coefficient-select 形式：

1. `u = Abs(x)`、`u2 = Mul(u,u)`；
2. 用 `Less + Where` 選擇 $A,B,C$；
3. `y = 0.5*x + A*u2 + B*u + C`。

`Abs`、`Mul`、`Clip`、`Where` 都是 ONNX standard operators；`Where` 的語意是依 condition 從兩個
tensor 選值，`Clip` 在 opset 13 亦支援 signed integer tensors。
[ONNX operator inventory](https://onnx.ai/onnx/operators/)、
[Where 規格](https://onnx.ai/onnx/operators/onnx__Where.html)、
[Clip 規格](https://onnx.ai/onnx/operators/onnx__Clip.html)。

但「ONNX graph 中常數是 dyadic」不保證 runtime 會自動產生 shift-add。標準 ONNX `BitShift`
只接受 unsigned integer tensor，不能直接表達本專案需要的 signed arithmetic right-shift 與
round-to-nearest 規則；因此 portable ONNX 應保持數學 reference，signed fixed-point shift/add 由
自訂 lowering/RTL 與 emulator 驗證。
[ONNX BitShift type constraint](https://onnx.ai/onnx/operators/onnx__BitShift.html)。

驗收時要同時檢查：

- graph 不含 `Exp/Sigmoid/Tanh/Erf/Div`；
- 全 graph 只有一個 data-by-data square path；
- ONNX Runtime 與 PyTorch float reference 的 max error；
- 目標 compiler 是否把 `Where`、signed `Abs` 或 coefficient mux fallback 到 CPU；
- custom integer backend 的逐碼 exhaustive／boundary test，而非只跑隨機 float tensor。

## 8. 不應把 GELU 近似直接拿來替 SiLU 的情況

GELU 的 target 是 \(x\Phi(x)\)，SiLU 的 target 是 \(x\sigma(x)\)。即使兩者都有 zero/identity tails，
中央曲率、負谷與導數不同；QuickGELU 的 $x\sigma(1.702x)$ 也是為了近似 GELU，不是原始
\(x\sigma(x)\) SiLU。因此：

- 不可把臺大 QGELU 的分段係數直接登錄成 SiLU approximation；
- 不可把 NVIDIA `QGELU`／QuickGELU 當成「較快的 SiLU」而不改實驗名稱；
- 只有當它被明確視為另一個 activation candidate，從同一 accepted checkpoint 做完整 zero-shot、
  等成本 recovery、COCO/BBAT5 各自 gate 與硬體量測時，才可以比較；
- 本輪已有 `qsilu_pq`、Hardswish、`poly_shift`、`poly_quality`，不建議再加入 QGELU，避免在昂貴
  Full35 recovery 前擴張候選空間。

## 9. 本 repo 的執行順序

1. 新增 `qsilu_pq` float module、hardware proxy、Q16.10 emulator 與 stable registry name；不動原
   `poly_shift/poly_quality`。
2. E0：float64 function/derivative、所有 knots 左右極限、symmetry、exact tails、fixed-point
   boundary/exhaustive、ONNX graph gate。
3. 用完整 COCO2017 train 與 BBAT5 formal train profile 計算每 region empirical function/derivative
   error；不使用 30% 資料或另建 split。
4. 在 recovery 前先跑 uniform zero-shot 與 one-region-at-a-time zero-shot sensitivity；
   training-only branch 不算部署節省。
5. 若通過 graph/數值 gate，才把 `qsilu_pq` 加入與 SiLU sham、Hardswish、`poly_shift` 相同步數、
   相同 LR 的 short recovery；不因純函數誤差較低就給更多 epoch。
6. 只有 PTQ 的真實 INT graph 超出凍結 budget 才做 QAT；最後以 FPGA/ASIC synthesis 或目標板
   latency/energy 決定 PQ、PWL、LUT 或 native fused SiLU，operator proxy 不作最終結論。

最終晉級標準必須同時包含：COCO mAP、BBAT5 box mAP、BBAT5 pose mAP、跨 seed 穩定性、
ONNX/backend 可執行性，以及同位寬的 LUT/FF/DSP/BRAM、latency、energy；沒有這組證據前，
`qsilu_pq` 只能稱為「首選候選」，不能稱為較佳硬體方法。
