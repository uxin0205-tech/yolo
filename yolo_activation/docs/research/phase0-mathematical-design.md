# Phase 0：受約束 Activation 數學設計草案

日期：2026-08-24
狀態：純函數／固定點／ONNX prototype 已實作；等待 baseline model integration 與真實 detection 實驗
主張邊界：本文不得被引用為 accuracy、泛化、硬體加速、新穎性或 SOTA 證明

## 0. 2026-08-25 實作決策覆寫

使用者要求減少驗證 baseline、允許乘法但優先硬體友善後，第一輪實作已收斂為：SiLU accuracy
reference、Hardswish hardware-neighbor baseline、只作便宜 primitive floor 的 ReLU，以及
`poly_shift`／`poly_quality` 兩個 proposed profiles。PReLU、PWLU-lite、FReLU 與其他文獻候選
保留為 frontier，不進第一輪昂貴 recovery。

`poly_shift` 採 `a=9/2,T=8`，direct polynomial 常數為 `9/32,-15/256,11/2048,-3/16384`，
皆可用 dyadic/APoT shift/add 表示；`poly_quality` 採原 Shape 設計 `a=4,T=109/16`。若後方
早期 shortlist、Lite／Shape 命名或搜尋順序與本節衝突，以本節及
[硬體友善原型報告](phase0-hardware-activation-prototype.md)為準。

模型層級的特殊優化已實作為 **BCSP**：只從 non-dominated policy 做受限 beam expansion，限制
changed regions、kernel 數與 finalists，並將 profile ID 靜態凍結。完整訓練 module 與資料契約見
[training/README](../../training/README.md)；尚未收到 baseline，故沒有真實 model adapter 或 AP。

## 1. 本輪決策與範圍

本研究以演算法貢獻為主要目標，同時保留工程可部署性。使用者已確認：

- 起點是未來交付的不可變 baseline 資料夾，不預設一定是 YOLO26m。
- 僅替換模型中原有且經 manifest 確認的 SiLU；不額外串接 activation，不替換 softmax、
  detection output sigmoid、`act=False` 或 raw output convolution。
- 先做 detection；Pose 與融合需在 detection 方法成立後另行授權與驗證。
- 候選不限定 PWL，先縮減成少量強候選，再研究一個數學／架構型 proposed candidate。
- 允許逐層不同 activation，但採全網、region、逐層的階層式搜尋；最終部署模型最多三種
  activation kernel。
- 廣泛篩選先用 1 seed；有餘裕時只替 finalists 增加第 2 seed。此證據只能標為探索性，
  論文等級穩定性主張仍需日後增加 seeds 或合理的變異估計。
- 已完成文獻、數學、純函數、固定點與 ONNX protocol；checkpoint 整合、訓練與正式選型等待資料夾到位。

## 2. 雙資料集邊界

兩個資料集是兩個獨立 Detect 任務，不得混用 head、label space、split 或 raw metric：

| 資料集 | 正式入口 | 任務與角色 |
| --- | --- | --- |
| Canonical BBAT5 v1 | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml` | `nc=2` ball/bat domain test；固定 formal train/val assignment |
| COCO2017 | `/home/uxin/yolo/coco2017.yaml` | COCO80 通用 detection 與泛化 test；使用既有 train2017/val2017 manifest |

規則如下：

1. BBAT5 v1 不重切、不抽樣、不改影像或標註，也不建立假 test split。
2. COCO2017 不得把 80 類 head 指向 BBAT5 二類 YAML。
3. 每個資料集用自己的 all-SiLU baseline 計算 `delta`；不直接比較兩者 raw mAP，也不把兩個
   mAP 平均成單一分數。
4. 第一個跨資料集目標是「同一數學 activation family 可否在兩個任務都恢復」，不是把資料
   混合訓練。
5. 若兩個 baseline 的模型 graph 相容，才追加共享 profile／region policy 實驗；不相容時只
   共享公式，分別搜尋 policy。

建議的跨資料集 ablation：

- `D0 independent`：各資料集獨立擬合 profile 與 assignment，作 oracle 上界。
- `D1 shared-profile`：共享 activation profile，assignment 可依資料集不同。
- `D2 shared-policy`：profile 與 semantic-region assignment 都共享；只在 manifests 可對齊時執行。

## 3. Prior-art 防線

本輪檢索已足以否定幾個過強敘事：

- [ActNAS](https://openaccess.thecvf.com/content/CVPR2025W/MAI/html/Sah_ActNAS__Generating_Efficient_YOLO_Models_using_Activation_NAS_CVPRW_2025_paper.html)
  已針對 YOLO 做逐層混合 activation 與裝置感知搜尋，因此「YOLO 每層 activation 不同」不是
  本研究的創新點。
- [Curl](https://eprint.iacr.org/2024/1127.pdf) 已明確利用
  `ReLU(x) - SiLU(x)` 的偶對稱性；2026 年公開的
  [hardware-embedded SiLU patent application](https://patents.justia.com/patent/20260037791)
  也描述 SiLU 線性部分與偶對稱 residual 的硬體分解。因此 SiLU 正負對稱或只算半邊本身不足以
  主張新穎；這裡也不提供專利法律意見。
- [PWLU](https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_Learning_Specialized_Activation_Functions_With_the_Piecewise_Linear_Unit_ICCV_2021_paper.html)
  已研究可學習 PWL activation；[DAPA](https://arxiv.org/abs/2603.19338) 已研究 distribution-aware
  piecewise activation；[FQA](https://arxiv.org/abs/2606.05627) 已研究量化空間中的 piecewise
  polynomial hardware approximation。因此 PWL、distribution weighting、低階多項式或係數量化
  各自都只能當組件或 baseline。
- [PolySwish](https://www.mdpi.com/2504-3110/8/8/444) 已提出 polynomial Swish 形式，故「把
  SiLU 改寫成多項式」也不是充分 contribution。

因此本文只提出一個待驗證組合：以 **導數互補＋積分與尾端接合約束** 定義低自由度函數族，
再搭配少量共享 profiles、逐層靜態凍結與跨資料集 robustness protocol。完整檢索與實驗前只能
稱為 `proposed integral-polynomial family`。

## 4. 少量候選的角色

文獻證據先形成 SiLU baseline 加五個位置，避免無限制 activation zoo：

| 角色 | 暫定候選 | 原因 |
| --- | --- | --- |
| Baseline | SiLUExact | 原 checkpoint 的函數與恢復基準 |
| 極簡控制 | ReLU | 比較最低非線性與最簡 operator path |
| 成熟 hard control | Hardswish | 比較固定 hard gate；源自 [MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) |
| 低成本可學控制 | PReLU | 只有負斜率參數；源自 [Delving Deep into Rectifiers](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) |
| 可學分段控制 | PWLU-lite 或受限 shared PWL | 檢查「增加函數自由度」本身能帶來多少增益 |
| 架構控制 | FReLU-region | 檢查少數 region 加入 spatial context 的上界；不與 scalar 候選混算成本 |

`proposed integral-polynomial family` 先走獨立的 `P0 math gate`，不是不經驗證就加成第六個昂貴
候選。由於使用者把演算法發展列為優先方向，若它通過公式、autograd、export 與 toy recovery，
正式 model sweep 時就由 proposed 取代 FReLU 的主候選席位；FReLU 降為小規模 architecture
control。若 proposed 未通過，維持文獻短名單，不為了創新敘事硬留。

以下不進入第一批完整重訓：

- GELU 或 Mish：最多選一個作 accuracy-oriented diagnostic，不作硬體候選。
- ACON／Dynamic ReLU：屬於額外 context/gating 架構，另列 architecture control；不與純
  elementwise drop-in 候選混成同一排行榜。
- ReLU6／LeakyReLU：保留為便宜 smoke controls；若 pilot 未顯示額外資訊，不升級 recovery。

最終身份仍須根據完整文獻報告與交付 baseline 的 operator/runtime 支援再凍結。

## 5. Proposed integral-polynomial family

### 5.1 從導數互補條件開始

SiLU 滿足：

\[
S(x)-S(-x)=x,
\]

因此：

\[
S'(x)+S'(-x)=1.
\]

所有滿足第一個式子的函數都可寫成：

\[
A(x)=\frac{x}{2}+H(|x|),
\]

其中 \(H\) 定義於非負半軸。這個分解並非新發現；本候選的研究點是直接限制 \(H'\) 的
邊界值、積分與尾端光滑接合，而非先自由 fit 一個 polynomial。

令 \(u=|x|\)、\(T>0\)、\(z=\min(u/T,1)\)，並定義：

\[
q_a(z)=a z+\left(9-\frac{9a}{2}\right)z^2
       +(6a-16)z^3
       +\left(\frac{15}{2}-\frac{5a}{2}\right)z^4.
\]

這個係數族由下列四個限制解出：

\[
q_a(0)=0,\qquad q_a(1)=\frac12,\qquad q_a'(1)=0,
\qquad \int_0^1q_a(s)\,ds=\frac12.
\]

其積分為：

\[
R_a(z)=\frac{a}{2}z^2
       +\left(3-\frac{3a}{2}\right)z^3
       +\left(\frac{3a}{2}-4\right)z^4
       +\left(\frac32-\frac{a}{2}\right)z^5.
\]

必須把尾端線性項明寫出來：

\[
H_{T,a}(u)=T R_a(z)+\frac{(u-T)_+}{2},
\qquad
A_{T,a}(x)=\frac{x}{2}+H_{T,a}(|x|).
\]

若漏掉 \((u-T)_+/2\)，\(H\) 在 \(u>T\) 會錯誤地固定為 \(T/2\)，無法接到 ReLU tail。

### 5.2 由構造直接得到的性質

- `difference identity`：\(A(x)-A(-x)=x\)。
- 零點：\(A(0)=0\)、\(A'(0)=1/2\)。
- 尾端：\(x\ge T\) 時 \(A(x)=x\)；\(x\le -T\) 時 \(A(x)=0\)。
- 內部導數：

  \[
  A'(x)=\frac12+\operatorname{sgn}(x)q_a(|x|/T),\qquad 0<|x|<T.
  \]

  因此正負導數精確互補。
- `q(1)=1/2` 讓一階導數接到 ReLU tail；`q'(1)=0` 讓二階導數也在 tail 接合。
- 部署 forward 只需 `Abs`、`Clip/Min`、常數乘法、加減與 polynomial Horner path；不需要
  `Exp`、`Sigmoid` 或 runtime `Div`。實際 constant reciprocal 是否能被 backend 有效降低，仍
  必須上板或在目標 runtime 實測。

### 5.3 分析過的低自由度 profiles

#### Lite profile：\(a=3\)（保留數學控制，未進第一輪 registry）

\[
R_3(z)=\frac32z^2-\frac32z^3+\frac12z^4.
\]

它比下一個版本少一個 polynomial degree，可作低成本 profile；初始 \(T\) 暫設 \(11/2\)，
之後只能由 training/search 資料調整，不可用正式 validation 偷調。

#### Quality profile：\(a=4\)

\[
R_4(z)=2z^2-3z^3+2z^4-\frac12z^5,
\]

\[
q_4(z)=4z-9z^2+8z^3-\frac52z^4.
\]

此時有容易稽核的解析式：

\[
q_4'(z)=2(1-z)^2(2-5z),
\]

所以 \(q_4\) 在 \(z=2/5\) 取得最大值 \(76/125\)，activation derivative 被限制在：

\[
-\frac{27}{250}\le A'(x)\le\frac{277}{250}.
\]

此外：

\[
q_4(z)-\frac12=\frac12(5z-1)(1-z)^3,
\]

負半軸的內部極小值出現在 \(|x|=T/5\)。以函數形狀 heuristic 取
\(T=109/16\) 時，該最小輸出約為 `-0.27904`，接近 exact SiLU 的負谷；這只是初始化理由，
不是最佳參數或 task evidence。

#### Shift profile：`a=9/2,T=8`

將 threshold 摺入後，內區間 `H(u)` 的四個係數是
`9/32,-15/256,11/2048,-3/16384`；常數可降成 shift/add，但仍需四次資料相乘建立 powers。
同一均勻網格的 MSE 為 `0.000183797`、max error 為 `0.0269959`。完整定義、固定點 rounding
與 ONNX 結果見硬體友善原型報告；它是目前第一順位 proposed candidate。

### 5.4 共享 profile 架構

1. 先各自測 `all-shift`、`all-quality`，不得一開始就把 profile learning 與 layer search 混在一起。
2. 每層只持有靜態 profile ID；若搜尋階段使用 continuous relaxation，匯出前必須 `argmax`／
   projection 並移除 gating graph。
3. 最多保留三個 deployment kernels：例如 SiLU/Hardswish、Shift、Quality；若 proposed profiles 沒有互補性，只保留一個。
4. `T` 與可選的 `a` 只能在少量共享 profiles 中學習，不能每 channel 或每 sample 各自一套。
5. float profile 凍結後才做 fixed-point／APoT coefficient projection；projection 前後結果分欄。

## 6. 純函數 sanity check

為了確認公式沒有明顯錯誤，本輪使用 `float64`、均勻網格 `[-12,12]`、24,001 點，比較 exact
SiLU。這不是 activation distribution，也不是模型或硬體 benchmark。

| 函數 | Uniform MSE | Uniform MAE | Max abs error | 最小輸出 | 導數範圍（約） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hardswish | 0.00244951 | 0.0310944 | 0.142278 | -0.375000 | [-0.500, 1.500] |
| Lite：`a=3,T=5.5` | 0.000220584 | 0.00991823 | 0.0359192 | -0.290039 | [-0.125, 1.125] |
| Shape：`a=4,T=109/16` | 0.000085719 | 0.00653458 | 0.0210018 | -0.279040 | [-0.108, 1.108] |

這張表只支持三件事：公式與 tail 可運作、增加一個 degree 能降低此網格上的 curve error、參數量
值得進入 toy implementation。它不支持：

- detection accuracy 比 Hardswish 或 SiLU 好；
- 真實 activation distribution 上仍有相同比例的誤差；
- polynomial 比 fused SiLU／Hardswish kernel 快；
- fixed-point projection 後仍維持此誤差。

## 7. 實驗與評估設計

### 7.1 Fidelity 階段

1. `math/toy`：函數、導數、tail、連續性、量化與 export graph。
2. `zero-shot`：各資料集從自己的乾淨 baseline checkpoint 替換，不微調。
3. `short recovery`：相同 optimizer、iteration、LR policy 與 augmentation budget。
4. `full validation`：pilot 後先凍結 accuracy gate，再跑完整 validation。
5. `final recovery`：只替 2–3 個 Pareto finalists 執行；1 seed 為必要、2 seed 為資源允許項。

### 7.2 搜尋順序

1. all-SiLU baseline reproduction。
2. 全網統一 Hardswish、Shift、Quality；ReLU 只作便宜 smoke control，不預設進 recovery。
3. region-wise one-at-a-time sensitivity。
4. survivors 的逐層 one-at-a-time sensitivity。
5. 受限制 beam/evolutionary search；不得暴力枚舉 `K^L`。
6. 最佳 mixed policy recovery 與 component ablation。

ActNAS 已涵蓋一般 YOLO layer-wise activation NAS，因此本研究報告必須把搜尋方法當 implementation
工具，而非單獨 contribution；比較時至少要包含 uniform、naive mixed 與 ActNAS-style baseline。

### 7.3 必報指標

數學／數值：

- uniform-grid、empirical-distribution-weighted 與 tail 的 function MSE/MAE/max error；
- derivative MSE/max error、導數上下界、零點斜率與負谷；
- float、quantized coefficient、integer emulator 三種結果；
- coefficient/profile 數量、一般乘法、constant multiply、add、compare、branch 與 LUT bytes。

模型：

- 各資料集自己的 `mAP50-95`、`AP50`、`AP75`、recall 與 loss；
- COCO 的 `AP_S/AP_M/AP_L` 與必要 class AP；
- BBAT5 的 ball AP、bat AP、formal-val aggregate，且明記沒有 test split；
- 收斂 epoch、gradient norm、NaN/Inf、activation saturation/zero/negative ratio；
- zero-shot、short-recovery、full-validation 不得混欄。

跨資料集選擇：

- 保留 `ΔCOCO` 與 `ΔBBAT5` 向量，另報 worst-case normalized regret；
- pilot 看清量級後凍結 gate，不得看完正式結果再更換權重；
- 任一資料集被支配的 candidate 不得只靠另一資料集平均分數掩蓋。

部署 proxy：

- PyTorch/ONNX export 是否成功與 graph operator counts；
- 不在未指定 target 時把 FLOPs、operator counts 或 curve MSE 稱為 latency／energy gain；
- 實際板卡、runtime、precision 與 toolchain 到位後才建立 hardware ranking。

## 8. 未來交付資料夾最低契約

每一個 dataset/model baseline 都至少需要：

- model config/source identity、checkpoint 與 SHA-256；
- framework commit、Python/PyTorch/Ultralytics 版本；
- dataset YAML、dataset fingerprint、task、split；
- baseline metrics、imgsz、batch、seed、augmentation 與 evaluation recipe；
- activation manifest，包含唯一 module ID、shape、region 與 eligibility；
- profiling/calibration sample list hash，以及統計是來自 train/search-train 還是 val；
- 若有 hardware claim，另需 target profile。

缺少必要欄位時 validator 應列出缺件並停止，不得猜 checkpoint、改 split 或產生假統計。

## 9. 下一次與使用者討論的決策

在實作 proposed activation 前，請使用者評估：

1. 是否接受 `Lite + Shape + static profile ID` 作第一個數學 prototype。
2. 第一批主候選是否採 SiLU/ReLU/Hardswish/PReLU/shared-PWL/proposed；FReLU 只作額外 architecture
   control。
3. 兩資料集是否各自交付 baseline checkpoint；若只先交付一個，執行順序仍是 detection 方法先
   成立，再驗證另一資料集。
4. 是否先只建立 math/reference tests，通過後再擴到模型 replacement framework。

## 10. 目前風險

- 本次 primary-source 檢索不是法律意義的專利／新穎性檢索；新名稱與 contribution 尚未成立。
- baseline 資料夾未到，無法判斷模型結構、eligible activation 數、checkpoint 相容性或真實成本。
- polynomial path 乘法較多，可能在已有 fused SiLU/Hardswish 的 CPU/GPU/NPU 上更慢。
- 只有 1–2 seeds 時只能形成探索性證據。
- COCO2017 與 BBAT5 v1 的資料量與 label space 差異很大；跨資料集 robust 結果必須用相對 delta
  與分開報告解讀。
