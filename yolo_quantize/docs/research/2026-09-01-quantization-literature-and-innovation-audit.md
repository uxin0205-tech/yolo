# Full35 量化文獻、架構與研究新意稽核

日期：2026-09-01  
性質：第一手文獻查核＋現行計畫唯讀稽核；未使用 GPU、未執行訓練、PTQ calibration 或 mAP validation  
稽核對象：`IMPLEMENTATION_PLAN.md`、`full35-quantization-plan-v3.yaml`、`v4-plus-prepared-plan-v3.yaml`、`2026-08-31-hardswish-policy-revision.md`，以及 Fixed SD4／uniform scale fitting 的現行實作

## 一、結論先行

現行方向的核心是合理的：activation 與 weight 綁成完整 policy、使用 matched control、保護 Binary Q/K 與 decode、保留 unfused／BN-folded 雙視圖、逐項檢查八個任務指標，而且 W5／W6／W7 在沒有 kernel 時不宣稱 GPU 加速。這些應保留。

但**目前不宜直接照既有 V4→V5→V6 全矩陣開跑**。正式使用 GPU 前，至少要處理下列四項：

1. **Fixed SD4 與 uniform W4 的 `mse` baseline 目前不是全域 MSE 最優解。**現行程式只搜尋十個 `absmax × fraction` 候選。CVPR 2021 已對任意固定 codebook 給出 $O(NK\log K)$ 的全域 MSE 最優 scale／assignment 演算法，因此現有方法應明確改名為 `mse_grid_v1`，另加 `optimal_scaled_codebook`，再發新版 routing；否則 LS-SD4 可能只是在擊敗偏弱初始化，而不是證明 learned SD4 的價值。[Optimal Quantization Using Scaled Codebook, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Idelbayev_Optimal_Quantization_Using_Scaled_Codebook_CVPR_2021_paper.html)
2. **QAT 的 unfused-master／folded-deployment 契約還缺 fold-aware fake quant。**計畫寫的是「BN running statistics frozen、affine trainable」，但 deployment weight 是 fold 後的 $W_{eff}=\gamma W/\sqrt{\sigma^2+\epsilon}$。若 QAT 只量化 fold 前的 $W$，affine 更新後的 deployment grid 可能失效。這是計畫風險，不表示目前已有 QAT bug；在實作前須選擇 fold-aware shadow weight，或凍結 affine 並在 folded graph 上訓練。
3. **目前 graph parity 尚不等於完整 integer graph 可部署。**Add／Concat／residual、attention、MASF、bias、requant、accumulator、rounding 與 saturation 的 scale contract 應提早納入 probe；不應等 finalist export 才第一次檢查。AQD 的 detector 整數化工作也顯示 normalization、skip connection 與完整資料流的固定點處理是量化偵測器的重要邊界。[AQD, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.html)
4. **V5 的 150 格應保留為可產生的 sensitivity universe，不應全部反覆用 full validation 選 winner。**建議分成「static 全做、diagnostic probe 全做或分層做、full validation 只做預先鎖定的 promoted cells＋sentinel」。否則 full validation 同時充當搜尋集與最終證據，容易產生 selection leakage。不得為解決此問題另切 BBAT5；應維持不可變 split，改以預註冊 routing、paired uncertainty 與 finalist 多 seed 降低過度選擇。

因此，本稽核的總評是：**架構骨架可保留，但 Fixed SD4 baseline、QAT fold contract、integer-boundary contract 與 evaluation tier 必須先修正；Hardswish／qSiLU／poly_shift、W4–W8、SD4 與 ternary 目前都仍是候選，不是 winner。**

## 二、證據層級與範圍限制

下文嚴格區分：

- **已發表事實**：論文、作者／會議正式頁、官方程式碼或官方 runtime 文件直接支持的內容。
- **專案推論**：把文獻方法映射到 Full35 shared Detect＋Pose、BBAT5、COCO person、Q3 regional activation 與目前程式架構後提出的設計；不代表原論文已在本模型證明。
- **待實驗假說**：可以公平否證的研究候選；不得先寫成已證明的 contribution。

本次是範圍明確的 primary-source search，不是完整專利與引用網路檢索。尤其 2025–2026 的 arXiv 工作、尚未被索引的技術報告、中文／非英文資料與專利可能仍有遺漏。「未找到完全相同組合」只能叫**目前檢索到的研究缺口**，不能等同正式 novelty 證明。

## 三、現行架構做對的地方

| 現行設計 | 稽核判斷 | 原因 |
|---|---|---|
| activation function、A-bit、checkpoint hash、weight map、scale recipe 綁成一個 cell | 保留 | [QDrop](https://openreview.net/pdf?id=ySQH0oDyp7) 的 reconstruction 設計直接處理 activation quantization 與 weight reconstruction 的耦合；獨立挑 activation winner 再拼 weight winner沒有充分因果依據。 |
| qSiLU、uniform Hardswish、poly_shift 三個 A8 parent 並行 | 保留，但跨 parent 只作 system-policy 比較 | 三者 checkpoint 與 recovery recipe 不完全相同；同 parent 的 FP-weight→quantized-weight delta 才是乾淨的 weight 因果比較。Hardswish checkpoint 還有 FP32 CIoU recipe 差異，現行報告已正確揭露。 |
| Q3 regional Hardswish 與 uniform Hardswish 分開 | 保留 | Q3 證據只支持 qSiLU root 上的單區 placement；不能外推成 uniform Hardswish 或量化 winner。混合 activation 的 device-aware 搜尋已有 [ActNAS, CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/MAI/html/Sah_ActNAS__Generating_Efficient_YOLO_Models_using_Activation_NAS_CVPRW_2025_paper.html)，因此「單區 Hardswish」本身也不宜宣稱新穎。 |
| W8／W7／W6／W5／W4 都保留 | 保留 | 逐層 1–8 bit 搜尋不是不合理；[HAQ, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html) 就搜尋 1–8 bit。非 2 次冪位寬可做精度、容量、BOP 與 custom hardware 研究，但不自動代表現成 GPU 更快。 |
| W5／W6／W7 無 kernel 時不宣稱 speedup | 必須保留 | 目前 [TensorRT quantized types 官方文件](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html) 的原生量化類型不提供通用 W5／W6／W7 Conv kernel；真實效能仍取決於 target、packing 與 kernel。 |
| 八項 metric，含 COCO person 與 BBAT ball／bat／pose | 強烈保留 | shared backbone 的平均分數可能掩蓋最差任務；逐項 gate 比只看 headline 更適合本專案。 |
| Binary Q/K、TopK、gather、decode 等 protected boundary | 保留並提早驗證 requant 邊界 | 這能避免改變模型語意；但保護 operator 本身不等於其輸入／輸出的 scale、accumulator 與 Q/DQ 已正確。 |
| PTQ 通過即可不做 QAT；低 bit 才做短 recovery | 保留 | NVIDIA 的官方 YOLOv7 QAT 範例同樣把 PTQ 與 QAT 分層，並強調 Q/DQ placement；不是所有通過 PTQ 的 cell 都需要訓練。[NVIDIA YOLOv7 QAT official code](https://github.com/NVIDIA-AI-IOT/yolo_deepstream/blob/main/yolov7_qat/README.md) |
| Paper-TWN、Channel-TWN、TTQ 分開命名 | 保留 | [TWN](https://arxiv.org/abs/1605.04711) 是 threshold＋共享 $\alpha$；[TTQ, ICLR 2017](https://openreview.net/pdf?id=S1_pAu9xl) 是正負兩個可學 scaling coefficients 與 QAT，訓練成本和方法不同。 |
| A-SD4 延後且不從 W-SD4 外推 | 保留 | 原始 [SD4, IEEE JETCAS 2021](https://scholars.lib.ntu.edu.tw/entities/publication/c1fea85e-53e2-44dc-8045-346f6b81ecb0) 的核心證據是 4-bit weight 格式與 multiplierless accelerator，不是 activation SD4。 |

## 四、需先修正的架構與實驗問題

### P0-1：`mse` 命名與 Fixed SD4 baseline 不夠強

**專案事實。**現行 `src/yolo_quantize/weight_formats.py` 的 uniform 與 SD4 `mse` 都只嘗試：

```text
[1.0, 0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50] × absmax scale
```

它是離散 grid heuristic，不是數學意義上的 MSE optimum。

**已發表事實。**[Optimal Quantization Using Scaled Codebook](https://openaccess.thecvf.com/content/CVPR2021/html/Idelbayev_Optimal_Quantization_Using_Scaled_Codebook_CVPR_2021_paper.html) 對任何固定 codebook 的 scaled quantization 給出全域最優演算法，且論文明確討論 scaled powers-of-two 與 ternary codebook。

**必改項目。**

1. 既有 artifact 保持不可變；把既有語意標成 `mse_grid_v1`，不得回寫歷史數字。
2. 新增 `optimal_scaled_codebook`，至少支援 per-tensor、per-output-channel；group32／64 可在相同演算法上處理每 group。
3. uniform W4 與 Fixed SD4 必須使用同 granularity、同 exact solver、公平比較。
4. 用新 solver 重新產生 Fixed SD4 routing 新版 manifest；舊 v2 只保留歷史地位。
5. 可另加 exact-scaled ternary baseline，和 Paper-TWN、Channel-TWN、TTQ 分開。
6. LS-SD4 的研究敘事只能是 task-loss／activation-conditioned／multi-task constrained learning；「替固定 codebook 學一個 scale」本身不可宣稱新穎。

**另有一個 SD4 encoding contract 需要補齊。**計畫宣告 `unique_values: 15`、`encoded_codes: 16`、`duplicate_zero_code: true`；目前 value-level projector則使用15個數值與邏輯 code ID `-7..7`，沒有保存兩個不同 zero bit-pattern。這不會改變浮點重建 MSE，卻不能代表 bit-true 4-bit encoding／occupancy。正式 export前應提供明確的 `0..15 → signed-digit bits → value` mapping、canonical zero規則、pack／unpack round-trip與16種pattern測試；在此之前，`code_bytes`只能解讀成4-bit packed容量模型，不能視為已驗證硬體編碼。

### P0-2：QAT 必須量到 deployment 真正使用的 weight

**專案推論。**計畫的 unfused master 保留 BN 供訓練，而 deployment graph 折疊 BN。當 BN affine 可訓練時，fold 後 weight 與 fold 前 weight 的有效範圍會改變。建議二選一並寫成 machine-readable contract：

- `fold_aware_shadow_qat`：forward 時由 FP32 Conv＋BN 參數計算 (W_{eff}, b_{eff})，fake quant 套在 effective deployment tensors；訓練 supervision 仍可使用 master graph。
- `folded_graph_qat`：先 fold，凍結／移除 BN，再在部署等價 graph 短恢復；代價是和原本 training graph 的一致性較難維持。

第一方案更符合目前 dual-view 設計，但每個 QAT checkpoint 必須重新跑 master↔deployment parity。若暫時不能實作，則至少凍結 BN affine；不能維持「affine trainable」又假設原 scale 永遠適用。

### P0-3：提早加入 full-integer boundary audit

**已發表事實。**[Jacob et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html) 建立整數推論的 scale、zero-point、bias 與 requant 基礎；[AQD](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.html) 則把 detector 的 convolution、normalization、skip 與資料流一併納入固定點設計。

**必改項目。**在 V4 W8 probe 前建立 `integer_boundary_contract`：

- 每個 Add 的兩支輸入是否共 scale，若否，指定哪一支 requant；
- Concat 後是否需要統一 scale；
- MASF、attention PWL、Binary Q/K 前後的 dtype／scale／zero-point；
- INT32 bias 與 accumulator 的 worst-case bound；
- RNE 或其他 rounding、saturation 與 overflow 規則；
- protected FP island 的 Q→DQ→FP→Q 成本與實際 latency；
- bit-true emulator 和浮點 fake-quant 的逐 boundary 誤差。

最終 packed export 仍可留在 finalist，但 graph feasibility 與 boundary tax 應在 V4 W8 就知道。

### P0-4：把「搜尋」和「正式證據」分層

現行 V5 設定的 `report_after_all_no_training_cells: true` 容易被解讀為 150 格都需完整 validation。建議重新定義 `all`：

| Tier | 範圍 | 用途 | 可否選 winner |
|---|---|---|---|
| T0 static | 150 格全部建立 weight artifact／reconstruction／成本 | 排除 structural、non-finite 與明顯 dominated strata | 不可 |
| T1 diagnostic | W8／W6／W4 全做；W7／W5 作邊界補點與 sentinel，或在資源足夠時全做 | 任務 probe、output reconstruction、routing 排序 | 只可 promotion |
| T2 canonical full validation | 預先鎖定的 promoted cells／mixed routes＋每 stratum sentinel | 八指標 gate | 可決定 finalist |
| T3 formal | top policies，完整 train view、多 seed、實機 | 最終結論 | 可報正式結果 |

V4 的 15 個 quantized cells仍全部保留；只改執行順序為 `W8 → W6 → W4 → boundary W7/W5`。這不是排除 W5／W7，而是先用較少點找出曲線轉折，再補非 2 次冪位寬。若使用者願意付出時間，T1 可跑齊五個 bits；仍不必讓 150 格全部進 T2。

### P1-1：calibration 32/task 適合 smoke，不適合微小差異的 winner 宣稱

[Q-YOLO](https://arxiv.org/abs/2307.04816) 指出 YOLO activation distribution 的不平衡會影響 calibration；[DetPTQ](https://arxiv.org/abs/2304.09785) 也顯示 detector 的局部 MSE／cosine 不一定對應終端 detection loss。

本專案應保留 cal32／probe64 作快速檢查，但 finalists 增加：

- calibration-size ablation：固定 hash manifests 的 32／128／512（若 512 超出資料限制則在開跑前明訂可用上限）；
- COCO person、object size 與 BBAT ball／bat／pose／`.rf.` group coverage；
- 相同 method 一律共用同一 manifest，不得為某方法重抽 calibration；
- 每項八指標報 paired bootstrap 或以 BBAT source group 為單位的 group bootstrap CI；
- 明寫 `0.04` 是 0–1 AP 座標的 **0.04 absolute AP（4 AP points）**，不是相對下降 4%。

### P1-2：cross-parent 公平性需要兩種結論語言

- **因果較乾淨的 weight 結論**：同一 activation parent、同一 checkpoint、同一 calibration，FP-weight matched control 對 quantized weight 的 incremental delta。
- **可部署 system-policy 結論**：不同 parent 的最終八指標／成本 Pareto；可說哪個完整系統較好，不能只把差異歸因於 activation function。

若論文要主張「Hardswish 本身勝過 qSiLU」，必須額外建立完全相同 data、loss precision、optimizer、epoch selector、seed 的 matched activation recovery。現有 Hardswish 與 qSiLU checkpoint 不能提供此因果主張。

### P1-3：single-region sensitivity 不可直接相加

現行 Q3 已正確禁止把 `neck_attention` 與 `masf` 的單區 delta 相加。建議在 top regions 加一個小型 $2\times2$ interaction audit：

```text
region X off / on × region Y off / on
```

並計算：


\[
\Delta_{int}(X,Y)=M_{11}-M_{10}-M_{01}+M_{00}.
\]

Activation×weight 亦用相同觀念：

\[
\Delta_{int}(A,W)=M(A,W)-M(A,FPW)-M(A_0,W)+M(A_0,FPW).
\]

這能直接回答先前核心疑問：activation 與 weight 是否強耦合，以及獨立最佳是否真的能拼接。

### P1-4：硬體角色尚缺固定 target

[HAQ](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html) 使用實際 hardware simulator；[HAWQ-V3](https://proceedings.mlr.press/v139/yao21a.html) 以 ILP 納入真實硬體限制。這些文獻支持的是**裝置條件化**策略，不是跨裝置通用 winner。

在宣稱 `standard_hardware`、`dyadic_hardware` 或 latency Pareto 前，凍結：device、runtime、compiler／版本、kernel source、input shape、batch、warmup、repetition、power mode、clock policy、host-device transfer 是否計入。Hardswish 的標準 operator 身分與 poly_shift 的 dyadic 身分目前是合理假說，不是效能結果。

### P1-5：低 bit QAT 加入 oscillation telemetry

[WACV 2024 的 quantized YOLO oscillation 研究](https://openaccess.thecvf.com/content/WACV2024/html/Gupta_Reducing_the_Side-Effects_of_Oscillations_in_Training_of_Quantized_YOLO_WACV_2024_paper.html) 顯示 3／4-bit YOLO QAT 的 weight oscillation 可造成副作用，並評估 model EMA 與短 quantization correction。

W4／SD4／ternary QAT 應先記錄：code flip rate、per-layer scale variance、clipping、codeword occupancy、master-weight 到 threshold 的距離。只有觀察到 oscillation 時，再做 matched EMA／correction ablation；不要未診斷就把額外技巧變成所有 policy 的預設，避免改變公平成本。

## 五、各類文獻方法的適用性

### 5.1 可直接採用其核心原則或作正式 baseline

| 方法 | 已發表事實 | 本專案採用方式 |
|---|---|---|
| LSQ | [LSQ, ICLR 2020](https://arxiv.org/abs/1902.08153) 以 QAT 學習 uniform quantizer step size，涵蓋 2／3／4-bit。 | 可作 W4／A4 QAT 的標準均勻 baseline；需用 matched training budget。 |
| LSQ+ | [LSQ+, CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Bhalgat_LSQ_Improving_Low-Bit_Quantization_Through_Learnable_Offsets_and_Better_Initialization_CVPRW_2020_paper.html) 為 Swish、H-swish、Mish 類偏斜且含負值 activation 引入可學 scale／offset 與 MSE initialization。 | qSiLU／Hardswish A8／A4 的重要 baseline。若只做初始化而沒有 gradient update，名稱必須是 `LSQ+-initialized asymmetric PTQ`，不能寫成完成 LSQ+ QAT。 |
| Exact scaled codebook | [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Idelbayev_Optimal_Quantization_Using_Scaled_Codebook_CVPR_2021_paper.html) 對任意固定 codebook 全域最小化 MSE。 | uniform、Fixed SD4、exact ternary 的必要強 baseline。 |
| APoT | [APoT, ICLR 2020](https://openreview.net/pdf?id=BkgXT24tDS) 以 additive powers-of-two 建立非均勻量化，官方實作亦提供 5-bit 配置。[官方程式碼](https://github.com/yhhhli/APoT_Quantization) | 作 SD4 的 matched 非均勻／POT comparator，特別能回答 W4／W5 的 codebook 形狀效果；需報每值 shift-add 成本，不能把 APoT 等同單一 shift SD4。 |
| TWN／TTQ | [TWN](https://arxiv.org/abs/1605.04711) 含 VOC detection 實驗；[TTQ](https://openreview.net/pdf?id=S1_pAu9xl) 學正負 scale。 | 現有 Paper-TWN static 與未來 TTQ QAT 分開，方向正確。可新增 exact-scaled ternary，使 heuristic、exact PTQ、QAT 三層完整。 |
| Hardswish | [MobileNetV3, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf) 以 ReLU6 形成 piecewise hard-swish，目標含部署效率。 | 值得列為 activation parent／regional candidate；是否更快仍需在指定 target 實測。 |

### 5.2 需修改後才適用 Full35

| 方法 | 已發表事實 | 為 Full35 所需修改 |
|---|---|---|
| HAWQ family | [HAWQ, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Dong_HAWQ_Hessian_AWare_Quantization_of_Neural_Networks_With_Mixed-Precision_ICCV_2019_paper.html) 以 Hessian aware sensitivity 作 mixed precision；[HAWQ-V2, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d77c703536718b95308130ff2e5cf9ee-Abstract.html) 使用平均 Hessian trace 並涵蓋 activation mixed precision與 COCO detection；[HAWQ-V3](https://proceedings.mlr.press/v139/yao21a.html) 加 ILP 和 dyadic integer-only inference。 | 不以單一平均 task loss 建 score；要分別算 COCO Detect、BBAT Detect、BBAT Pose，採 worst-task constraint／Pareto，並對 non-power bits 使用真實 packed bytes 或 custom-hardware cost。 |
| AdaRound／BRECQ／QDrop | [AdaRound, ICML 2020](https://proceedings.mlr.press/v119/nagel20a.html) 學習 rounding；[BRECQ, ICLR 2021](https://arxiv.org/abs/2102.05426) 做 block reconstruction與二階近似；[QDrop, ICLR 2022](https://openreview.net/pdf?id=ySQH0oDyp7) 在 reconstruction 中隨機略過 activation quantization。 | 先定義 C2f／CSP、attention、MASF、Detect、Pose 的合理 block seam；只套 shortlisted W4／SD4 cells，不把 reconstruction optimization 加到 150 格。 |
| DetPTQ | [DetPTQ](https://arxiv.org/abs/2304.09785) 使用 object detection output loss 選擇 layer-wise重建準則，指出 local MSE／cosine 未必對應 detection outcome。 | 擴成 Detect＋Pose output loss：raw one-to-one class／box、keypoint／visibility 與 OKS-like surrogate；decode 仍保持 protected。 |
| Reg-PTQ | [Reg-PTQ, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html) 針對 detector regression head 使用 filtered global loss 與 logarithmic-affine quantizer。 | 初期保持 box／pose predictor W8 或 FP；shortlist 後才分別設計 box 與 keypoint-validity／OKS loss。原論文未涵蓋 shared keypoint head，不能直接照搬 filtering。 |
| YOLO-Pose mixed precision | [YOLO-Pose, CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/ECV/html/Maji_YOLO-Pose_Enhancing_YOLO_for_Multi_Person_Pose_Estimation_Using_Object_CVPRW_2022_paper.html) 報告其模型純 8-bit PTQ 有明顯 AP 下降，而約 30% layers 保留 16-bit 可縮小差距。 | 它支持 joint detection＋pose 必須做 layer sensitivity；但模型、資料、任務與 8／16-bit 都和 Full35 不同，不能外推哪個 region 必須 W8／FP。 |
| MPQ-YOLO | [MPQ-YOLO](https://doi.org/10.1016/j.neucom.2023.127210) 研究 YOLO backbone／head 的 mixed-precision與 progressive QAT。 | 可作 backbone→neck→head staged order 的參考；Full35 仍需分開 Detect／Pose predictor 和 shared regions。 |
| MQAT | [MQAT, TMLR 2024](https://openreview.net/forum?id=ArWQ9ZyA6J) 顯示 6D pose 模組量化順序與模組位寬會影響結果。 | PTQ sensitivity 完成後，可對 top policy 做 neck-first、backbone-first、head-first 的短 QAT 順序消融；不是證明哪個順序適合本模型。 |
| RAPQ | [RAPQ, IJCAI 2022](https://www.ijcai.org/proceedings/2022/219) 研究 PoT-constrained scale 與 reconstruction 中 clipping／rounding 的共同影響。 | 可作 POT-scale PTQ comparator或 scale constraint ablation；其 quantizer 不等同 SD4 codebook。 |

### 5.3 只適合借用概念，不宜直接移植或宣稱適用

| 方法 | 邊界 |
|---|---|
| AWQ | [AWQ, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) 是以 activation 統計保護 salient channels 的 LLM weight-only 方法，核心設定是 Transformer／W4A16。可測 `activation-energy weighted channel score`，但不能稱為「把 AWQ 套到 YOLO」後就具有相同結論。 |
| OmniQuant | [OmniQuant, ICLR 2024](https://openreview.net/pdf?id=8Wuvhh0LYW) 以 learnable weight clipping與等價變換做 LLM block-wise calibration。其 LayerNorm／Transformer 結構假設不直接對應 Conv-BN、C2f、MASF；只適合借用 learnable clipping／equivalent transform 的 matched ablation。 |
| Task-specific zero-shot QAT | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Task-Specific_Zero-shot_Quantization-Aware_Training_for_Object_Detection_ICCV_2025_paper.html) 為沒有真實資料的 detection 建 synthetic task data與 distillation。[官方程式碼](https://github.com/DFQ-Dojo/dfq-toolkit) | 本專案有 canonical training data，沒有理由用 synthetic data取代真實資料；只能借用 task-specific teacher target 概念。 |
| GABFusion | [GABFusion 2025 預印本](https://arxiv.org/abs/2511.05898) 探討 multi-task QAT 的 gradient conflict／fusion。 | 這提高「fusion-aware quantization」泛稱的新穎性風險；它不是 Full35 Detect＋Pose＋SD4 證據，且只是預印本。 |
| FQA | [FQA 2026 預印本](https://arxiv.org/abs/2606.05627) 聯合搜尋 piecewise polynomial activation 的係數、fractional word length與硬體架構。 | poly_shift 的硬體主張應提供 bit-exact coefficients、word length、overflow／rounding與實測；僅有函數 NRMSE 不足。該來源是預印本，不應當成已被同行審查的定論。 |
| A-SD4 | 原始 SD4 證據是 weight與 custom accelerator；把 signed SD4 直接用於偏斜、含負值的 SiLU／Hardswish output 需要另定 offset、range、observer和實際 activation datapath。必須與 LSQ+ A4、APoT A4 matched 比較，維持 deferred branch。 |

## 六、建議的修正版實驗順序

### G0：GPU 前 CPU／規格修正

1. 新增 exact scaled-codebook solver；保留 `max`、`mse_grid_v1` 作歷史 ablation。
2. 重新建立 uniform W4／Fixed SD4／exact ternary static results與 Fixed SD4 routing v3。
3. 補齊 SD4 16-code bit-pattern mapping、duplicate-zero canonicalization與 pack／unpack測試。
4. 在 YAML 明定 `evaluation_tiers`、`full_validation_budget`、`promotion_rule`、`sentinel_rule`。
5. 明定 `fold_aware_qat` 與每 checkpoint deployment parity。
6. 明定 integer boundary、target hardware與 benchmark protocol；尚未選 target時成本只報 bytes／metadata／BOP proxy。
7. 建立 calibration 32／128／512 的版本化候選 manifests；不改 canonical split、不物理合併資料、不從 validation calibration。

### G1：W8 graph bridge與 activation policy 基準

初始固定 12 格：

| 類別 | 格數 | 說明 |
|---|---:|---|
| Uniform FP-weight matched controls | 3 | qSiLU／Hardswish／poly_shift，各自 checkpoint |
| Uniform W8 | 3 | 同 parent incremental delta |
| Q3 regional FP-weight control／W8 | 6 | 3 single regions × 2；base仍是 qSiLU checkpoint |

只有 `neck_attention` 與 `masf` 單區 W8 都通過才新增組合 control／W8 兩格。G1 同時檢查 Q/DQ boundary、observer、bit-true feasibility與八指標，不只看 mAP。

### G2：uniform bit curve

1. 三 parent 先跑 W6、W4，連同 G1 W8 得到 coarse curve：`3 × {W8,W6,W4}=9` quantized cells。
2. W7、W5 全部保留；優先補在 gate crossing、Pareto knee與至少一格 dominated sentinel。若資源允許，T1 probe 可跑齊 15 格。
3. 只讓通過 matched incremental／total gate的 policy進 V5；不得因 weight NRMSE好看直接晉級。
4. 報 activation×bit interaction；不要只報每個 parent 的獨立排序。

### G3：region sensitivity與 mixed map

1. 原 150 格保留為 machine-readable universe與 T0 static coverage。
2. T1 先做存活 parent × 10 regions × W8／W6／W4；W7／W5補邊界。
3. 每格同時產生 weight reconstruction、block output reconstruction、八項 diagnostic metrics、bytes／metadata／BOP；若 target已凍結再加 latency。
4. 先保護 box／pose predictor為 FP／W8建立 safe route；低 bit predictor作 sensitivity cell，不自動放入 mixed policy。
5. 以 top regions建 mixed map，再對 top pair做 $2\times2$ interaction audit。
6. T2 full validation只跑預註冊 mixed routes、promoted isolated cells與 sentinel，不把150格 full validation當搜尋資料庫。

### G4：特殊格式的公平比較

每個被 routing 選中的 region至少比較：

```text
exact uniform W4
vs exact Fixed SD4
vs APoT4
vs Paper-TWN / exact-scaled ternary（不同容量層級，分開報）
```

所有方法固定 parent、layer set、granularity、calibration data與 evaluation tier，另報 scale metadata、code occupancy與實際算術成本。Fixed SD4 routing 不再使用舊 `mse_grid_v1` 當唯一選擇依據。

### G5：有條件 QAT，不從零訓練

- W8 PTQ 通過所有 gate：不做 QAT。
- W7–W4／SD4／ternary進 recovery band：才做低 LR、matched QAT。
- S15 可保留現有最多6個主 policy，另允許最多2個預註冊 sentinel，避免只對最看好的方法投入訓練而高估效果。
- D20 最多3個；D60建議最多2個 base finalists。正式 top2 再做 seeds 0／1／2。
- 每個 optimizer／格式都有 matched sham；AdamW與 MuSGD 不共享 optimizer state。
- 30% QAT view必須固定 hash、BBAT group-safe、COCO person／size coverage；search只用同一個 view。正式 finalist仍依現行計畫回到100% train。
- 加入 fold-aware weight、code flip／scale variance／clipping telemetry。

### G6：lower A-bit與 A-SD4

weight map穩定後才測 A7／A6。A-SD4只作探索研究，至少比較：

```text
uniform LSQ+ A4
vs APoT A4
vs Fixed A-SD4
vs learned A-SD4
```

四者必須明定 signed／asymmetric range、offset、per-tensor／channel policy、activation memory traffic與 target operator。若沒有對應 activation kernel，只能報 accuracy／格式代理，不宣稱硬體收益。

## 七、每階段應報的指標

| 階段 | 必報指標 | 決策用途 |
|---|---|---|
| Static | MSE、NRMSE、SQNR、cosine、max error、clipping、code occupancy、scale count、code／metadata bytes | 健全性與粗篩，不能代表 mAP |
| Output reconstruction | block output NRMSE／cosine、Detect／Pose raw-output loss、activation energy | 比單看 weight MSE更接近任務，但仍不是正式結果 |
| Diagnostic | 八項 AP、每項 delta、worst-task delta、interaction term、paired CI | promotion／racing |
| QAT | 八項 AP、sham drift、loss curve、code flip、scale variance、clipping、checkpoint parity、實際 epochs／samples | 判斷 recovery是否真由量化學習造成 |
| Cost | packed weight bytes、scale／offset metadata、BOP／traffic、accumulator bound | 無 kernel時的可比較代理 |
| Hardware | latency median／p90、throughput、peak memory、power／energy，附完整 target protocol | 只在固定 device/runtime/kernel後作 Pareto |
| Formal | 每 seed八項 AP、mean／std／worst seed、calibration-size sensitivity、100% train結果 | 最終可報結論 |

`0.04` gate應保持逐項 worst-task hard constraint；另增加連續 Pareto score供排序，但不得用平均 score覆蓋任一任務的 fail。

## 八、可落地、可公平否證的研究候選

以下都只能先稱「候選貢獻」。

### 候選 A：activation-conditioned、worst-task-constrained mixed precision／mixed format routing

**研究問題。**同一 region 的最佳位寬或 SD4 suitability 是否隨 qSiLU／Hardswish／poly_shift 改變？共享 backbone若只最小化平均 sensitivity，是否會犧牲 COCO person、ball、bat或 pose 的最差任務？

**可實作算法。**對 activation parent $a$、region $r$、task $t$、format／bit $b$ 建立：

```text
task Fisher/Hessian trace
× activation-weighted block reconstruction
× measured quantization distortion
```

再用 ILP／Pareto在 bytes、BOP或真實 latency budget下最小化 worst-task normalized risk。W5／W6／W7可自然留在離散候選集合；若 target不支援則只用 bytes/BOP constraint。

**公平對照。**uniform W4/W6/W8、weight-NRMSE router、AWQ-like activation-energy router、單一加權 HAWQ router、實測 single-region oracle。固定 parent、calibration、budget與 protected regions。

**驗證。**報預測 sensitivity與實測八項 delta的 Spearman、top-k recall、routing regret，以及跨 activation parent的 interaction。這比單純「用了 Hessian」更有辨識力。

**新穎性邊界。**HAQ／HAWQ已有 mixed precision與硬體限制，AWQ已有 activation-aware channel saliency；可能的研究缺口是 shared Detect＋Pose下的 activation-conditioned、worst-task constraint與 SD4／非標準位寬共同 routing，不是任一元件本身。

### 候選 B：shared Detect＋Pose 的 task-output-loss block PTQ

**研究問題。**weight MSE與一般 feature reconstruction是否會錯排 regression／keypoint-sensitive blocks？

**可實作算法。**在 AdaRound／BRECQ／QDrop 式 block reconstruction中加入 joint output loss：

- one-to-one class logits與box regression；
- pose keypoint coordinates、visibility與valid-keypoint mask；
- OKS-like normalized distance；
- worst-task或constraint-based loss balance，不把 COCO與BBAT raw mAP混合成一個數字。

Decode仍為 protected FP；loss可作用在 decode前的 raw output與教師 FP output。

**公平對照。**RTN／exact-scale PTQ、BRECQ-MSE、detect-only output loss、pose-only output loss、joint output loss、joint＋QDrop。所有方法固定 calibration samples、block順序、iterations、optimizer與 wall-clock budget。

**驗證。**八項 AP、block reconstruction、最差任務、同成本下的 recovery；另做 box predictor與 pose predictor protected/unprotected ablation。

**新穎性邊界。**DetPTQ／Reg-PTQ已有 detector task loss，BRECQ／QDrop已有 advanced PTQ；候選價值在 shared YOLO Detect＋Pose、keypoint-aware objective與最差任務公平性，不是「把 BRECQ 用在 YOLO」本身。

### 候選 C：Exact-to-learned SD4 continuum

**研究問題。**SD4改善究竟來自 codebook形狀、精確 scale fitting、activation-conditioned calibration，還是 task-loss QAT？

**方法階梯。**

```text
歷史 max / mse_grid_v1
→ exact scaled-codebook PTQ
→ activation-weighted scale-only PTQ
→ fold-aware task-loss LS-SD4 QAT
```

每層記錄 scale、clipping、每個 codeword occupancy、zero使用率與 routing穩定性。

**公平對照。**exact uniform W4、APoT4、exact Fixed SD4、Paper-TWN、exact-scaled ternary、TTQ；同 layer set、granularity、scale metadata budget與 train/calibration budget。TTQ是 QAT，結果表必須另列訓練成本。

**驗證。**每一階梯相對上一階梯的增量八指標與成本；三 seeds只用於真正 learned finalists。若 learned SD4未勝 exact baseline，應如實結論為初始化不足被修正，而非新方法失敗。

**新穎性邊界。**scaled codebook與learnable quantization都有先例。較可辯護的候選是 activation/task-conditioned routing、shared multi-task constraints、fold-aware deployment與bit-exact硬體驗證的整合。

### 候選 D：Q3-seeded regional activation×weight co-design

**研究問題。**Q3找出的 `neck_attention`／`masf`／`backbone_attention` Hardswish placement，是否改變相鄰 weight region 的bit sensitivity與真實 latency？

**可實作設計。**Q3單區先驗只用來決定測試順序，不直接指定 winner；在每個優先 region測 FP／W8／W6／W4，做 activation-placement×weight-bit interaction，再只對 top pair做二因子組合。

**公平對照。**uniform qSiLU、uniform Hardswish、uniform poly_shift、regional matched FP-weight control、獨立選 activation＋weight的 naive composition。

**驗證。**交互作用、八項 AP、packed cost與固定 target latency；若 joint policy沒有顯著勝過 naive composition，就不能宣稱 co-design帶來價值。

**新穎性邊界。**ActNAS已研究 YOLO 的 mixed activation與 device-specific search；本候選必須以 shared Detect＋Pose、weight quantization coupling與實測硬體共同形成差異，regional Hardswish本身不是新方法。

## 九、建議加入 machine-readable plan 的欄位

以下是交給實作階段的建議鍵名，不代表本次已修改 YAML：

```yaml
scale_methods:
  historical: [max, mse_grid_v1]
  strong_baseline: optimal_scaled_codebook
  do_not_relabel_historical_artifacts: true

evaluation_tiers:
  static: all_cells
  diagnostic: promoted_plus_boundary_bits_plus_sentinels
  full_validation: preregistered_promoted_plus_sentinels
  formal: locked_finalists_only

calibration_ablation:
  sizes_per_task: [32, 128, 512]
  same_manifest_across_methods: true
  validation_as_calibration: forbidden

qat_graph_contract:
  quantized_tensor: bn_folded_effective_weight
  bn_running_stats: frozen
  bn_affine: explicit_policy_required
  deployment_parity_each_checkpoint: true

integer_boundary_contract:
  add_scale_alignment: required
  concat_requant: explicit
  accumulator_bound: required
  rounding: round_to_nearest_even
  saturation: explicit
  protected_fp_island_cost: reported

uncertainty:
  coco: paired_bootstrap
  bbat: source_group_bootstrap
  report_all_eight_intervals: true

hardware_target:
  device: unresolved
  runtime: unresolved
  compiler_version: unresolved
  native_kernel_required_for_speed_claim: true
```

## 十、推薦決策

### 可以直接保留

- 三個 active A8 parents與 Q3 regional branch分開。
- W4–W8完整候選集合；W8→W6→W4先找 coarse curve，W7／W5再補點。
- 八項 gate、matched controls、immutable BBAT5、COCO person coverage。
- Binary Q/K、training-only branches與 decode protection。
- Paper-TWN／Channel-TWN／TTQ分離；A-SD4 deferred。
- PTQ通過即可停止，不自動 QAT；final optimization仍延後到 base finalists。

### 必須修改後才執行

- `mse`改成可稽核的 `mse_grid_v1`，新增 exact scaled-codebook並重新 routing。
- 補齊 SD4 的15個 unique values／16個 encoded codes／duplicate zero bit-true mapping。
- QAT明定 fold-aware effective weight與checkpoint parity。
- V5分為 static／diagnostic／full／formal；不要150格都反覆拿 full validation選 winner。
- calibration-size與uncertainty稽核。
- Add／Concat／MASF／attention／accumulator的integer boundary contract前移到 W8。
- 硬體 target凍結後才使用 latency或「硬體 winner」語言。

### 目前不應做或不應宣稱

- 不把 A8 activation winner與獨立 weight winner直接拼接。
- 不把現有 ten-point `mse`當全域最優。
- 不把 LS-SD4的 scale learning本身宣稱 novelty。
- 不把 AWQ／OmniQuant原封不動套入 Conv YOLO後沿用其 LLM 結論。
- 不因 W5／W6／W7容量較低就宣稱 TensorRT／GPU更快。
- 不以原始 SD4 weight論文支持 A-SD4。
- 不以 Q3單區結果推算多區組合。
- 不以 weight NRMSE取代八項 AP。

## 十一、第一手來源清單與檢索限制

本報告使用 **33 組第一手來源**（原始論文／會議正式頁／官方程式碼／官方 runtime 文件）；其中 GABFusion與 FQA明確標為預印本，不能與已同行審查來源同等解讀。主要來源已在相應主張旁直接連結，依主題整理如下：

1. 整數 QAT／部署：[Jacob et al.](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html)、[AQD](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.html)、[TensorRT quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html)。
2. YOLO／detector PTQ-QAT：[Q-YOLO](https://arxiv.org/abs/2307.04816)、[DetPTQ](https://arxiv.org/abs/2304.09785)、[Reg-PTQ](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html)、[YOLO oscillation](https://openaccess.thecvf.com/content/WACV2024/html/Gupta_Reducing_the_Side-Effects_of_Oscillations_in_Training_of_Quantized_YOLO_WACV_2024_paper.html)、[NVIDIA YOLOv7 QAT](https://github.com/NVIDIA-AI-IOT/yolo_deepstream/blob/main/yolov7_qat/README.md)。
3. Joint detection／pose與模組策略：[YOLO-Pose](https://openaccess.thecvf.com/content/CVPR2022W/ECV/html/Maji_YOLO-Pose_Enhancing_YOLO_for_Multi_Person_Pose_Estimation_Using_Object_CVPRW_2022_paper.html)、[MPQ-YOLO](https://doi.org/10.1016/j.neucom.2023.127210)、[MQAT](https://openreview.net/forum?id=ArWQ9ZyA6J)。
4. Learned quantizer：[LSQ](https://arxiv.org/abs/1902.08153)、[LSQ+](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Bhalgat_LSQ_Improving_Low-Bit_Quantization_Through_Learnable_Offsets_and_Better_Initialization_CVPRW_2020_paper.html)。
5. Mixed precision／Hessian：[HAQ](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html)、[HAWQ](https://openaccess.thecvf.com/content_ICCV_2019/html/Dong_HAWQ_Hessian_AWare_Quantization_of_Neural_Networks_With_Mixed-Precision_ICCV_2019_paper.html)、[HAWQ-V2](https://proceedings.neurips.cc/paper/2020/hash/d77c703536718b95308130ff2e5cf9ee-Abstract.html)、[HAWQ-V3](https://proceedings.mlr.press/v139/yao21a.html)。
6. Advanced PTQ：[AdaRound](https://proceedings.mlr.press/v119/nagel20a.html)、[BRECQ](https://arxiv.org/abs/2102.05426)、[QDrop](https://openreview.net/pdf?id=ySQH0oDyp7)、[RAPQ](https://www.ijcai.org/proceedings/2022/219)。
7. Activation-aware／LLM-derived ideas：[AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)、[OmniQuant](https://openreview.net/pdf?id=8Wuvhh0LYW)。
8. Codebook／POT／ternary：[SD4](https://scholars.lib.ntu.edu.tw/entities/publication/c1fea85e-53e2-44dc-8045-346f6b81ecb0)、[Optimal Scaled Codebook](https://openaccess.thecvf.com/content/CVPR2021/html/Idelbayev_Optimal_Quantization_Using_Scaled_Codebook_CVPR_2021_paper.html)、[APoT](https://openreview.net/pdf?id=BkgXT24tDS)、[TWN](https://arxiv.org/abs/1605.04711)、[TTQ](https://openreview.net/pdf?id=S1_pAu9xl)。
9. Activation與搜尋：[MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)、[ActNAS](https://openaccess.thecvf.com/content/CVPR2025W/MAI/html/Sah_ActNAS__Generating_Efficient_YOLO_Models_using_Activation_NAS_CVPRW_2025_paper.html)。
10. 邊界與近期相關工作：[Task-specific zero-shot QAT](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Task-Specific_Zero-shot_Quantization-Aware_Training_for_Object_Detection_ICCV_2025_paper.html)、[GABFusion預印本](https://arxiv.org/abs/2511.05898)、[FQA預印本](https://arxiv.org/abs/2606.05627)。

正式對外寫 novelty 前，仍應補做：各候選論文的 forward／backward citation graph、IEEE Xplore／ACM DL／CVF／OpenReview／Scopus／Web of Science／Google Scholar 交叉查核、相關 FPGA／ASIC patents、2025–2026 最新版本與老師／共同作者的領域審查。方法組合沒有被逐字搜尋到，也不會自動構成新穎性。

## 十二、研究執行紀錄

- 變更內容與原因：建立本稽核報告，將現有計畫逐項對照第一手量化、偵測、姿態、mixed-precision、codebook與硬體文獻；特別揭露 Fixed SD4 baseline、QAT fold與 evaluation tier風險。
- 驗證方式與結果：唯讀比對四份指定計畫／報告與現行 uniform／SD4 scale fitting程式；查核33組第一手來源；未執行GPU、訓練或資料變更。
- 困難與解法：部分近期工作只有預印本，已明確標示；尚未固定 target hardware，因此只提出 protocol，不做速度結論。
- 未解事項或風險：本稽核初始時exact solver與SD4 encoding尚未實作；後續工程處理狀態見下節。integer boundary CPU reference與新版routing已補齊，但GPU boundary calibration、fold-aware QAT、hardware target與正式novelty search仍未完成。

## 十三、本稽核後續工程處理狀態

截至2026-09-01本輪收尾：

- `optimal_scaled_codebook_scales`已實作並以獨立assignment搜尋驗證全域objective；uniform與Fixed SD4共用同一Module。這不宣稱與論文實作具有相同複雜度。
- `SD4Encoding`已補齊16個nibble patterns、雙zero、canonical padding與pack／unpack round-trip。
- active PTQ runner已改用BN-folded deployment View、由plan獨立釘住SHA的固定diagnostic manifest與fail-closed執行授權。
- 完整Full35 exact W4／SD4 profile已在qSiLU、Hardswish、poly_shift三parent完成，共3,552筆；routing v3有36層cross-parent、cross-view static候選。舊十點grid產物未覆寫。
- integer boundary CPU reference已補Add／Concat、RNE、saturation、protected islands與148層INT32 MAC bound；下一步仍需GPU凍結LSQ+ scale／offset並驗證bias／padding lowering與實際boundary error。fold-aware contract只阻擋QAT；未固定target hardware前不作速度或能耗winner宣稱。
- 完整結果見[`../reports/2026-09-01-cpu-p0-integer-exact-routing.md`](../reports/2026-09-01-cpu-p0-integer-exact-routing.md)。
