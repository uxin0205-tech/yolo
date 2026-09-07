# 2026-08-29 Full35 weight region敏感度重新規劃

> 歷史狀態：本文保存2026-08-29當時的activation shortlist。2026-08-31後active parent已改為qSiLU／Hardswish／poly_shift，現行矩陣見[Hardswish policy修訂報告](2026-08-31-hardswish-policy-revision.md)。

## 建議方向

最好的第一步不是立刻全網INT4，也不是直接開始QAT，而是固定三條A8 activation policy，從backbone開始做INT8／INT4隔離敏感度，再逐區域累積到Detect與Pose heads。

優先壓縮backbone_deep與neck。兩區合計16,269,312個部署候選weights，占72.075%；若這兩區可用W4，就能取得大部分weight節省。MASF與兩個最終predictor合計只有0.623%，不值得為了很小容量收益先承擔精度風險，建議先W8。

不再新增SiLU run。既有accepted Full35／SiLU結果只作歷史絕對基準；新weight矩陣只使用qSiLU、poly_quality與poly_shift。

## A3至A8結果

本次指標是worst Detect／Pose one-to-one raw NRMSE與minimum TopK selected-pair overlap，不是mAP。

| A-bit | 最佳raw | 最佳overlap | 結論 |
|---|---|---|---|
| A3 | poly_shift 0.629569 | 0.0000 | 不可作主線 |
| A4 | poly_quality 0.601968 | poly_shift 0.0233 | 不可作主線 |
| A5 | poly_quality 0.379727 | 0.0967 | 不晉級 |
| A6 | qSiLU 0.204463 | poly_shift 0.2467 | 只作後續研究 |
| A7 | poly_quality 0.138725 | poly_shift 0.4433 | 可探索 |
| A8 | poly_quality 0.081644 | poly_quality 0.6933 | weight分析主線 |

排除SiLU後，Hardswish在A3至A8每個bit都被至少一個其他候選支配，因此不再投入。qSiLU雖然A8 proxy不是最佳，但上游short-recovery的三任務mAP headroom最大，仍保留accuracy recovery角色。三條主線各自解決不同風險：

- qSiLU＋A8：上游mAP headroom。
- poly_quality＋A8：activation量化數值。
- poly_shift＋A8：硬體取向。
- qSiLU＋A6、poly_quality＋A7、poly_shift＋A7延後到weight policy選定後。

## Weight graph盤點

| Region | Modules | Weights | 占部署候選 |
|---|---:|---:|---:|
| backbone_early | 21 | 1,218,240 | 5.397% |
| backbone_deep | 22 | 8,126,464 | 36.001% |
| backbone_attention_safe | 7 | 919,808 | 4.075% |
| neck | 33 | 8,142,848 | 36.074% |
| MASF | 3 | 74,240 | 0.329% |
| neck_attention_safe | 5 | 395,520 | 1.752% |
| Detect O2O tower | 18 | 1,390,592 | 6.161% |
| Detect predictor | 6 | 62,208 | 0.276% |
| Pose O2O tower | 24 | 2,238,464 | 9.917% |
| Pose predictor | 12 | 4,224 | 0.019% |
| 合計 | 151 | 22,572,608 | 100% |

另有96個training-only Conv／Linear modules、3,747,712個weights保持FP32／原訓練精度；兩處attention的4個Q／K projections共131,072個weights受Binary Q/K契約保護。三者相加可完整對回251 modules與26,451,392 weights。

## 實驗流程與使用者選擇點

### W0：分布與重建

對151個部署候選modules做BN-folded weight分析，先算W8、W4重建，也同時計算Fixed SD4、LS-SD4與ternary適配proxy。輸出layer與region ranking，不跑訓練。

使用者會看到每layer的MSE、NRMSE、SQNR、clipping、code occupancy、zero ratio、scale metadata與格式推薦，再決定哪些layer進W1。

### W1：隔離region probe

三條A8 policy乘10 regions乘W8／W4，共60格無訓練probe。每次只量化一個region，其他weights保持FP32，藉此分辨真正敏感位置。

完成後交付每格的forward NRMSE、TopK overlap、容量收益與Pareto判定，停下來讓使用者選。

### W2：Backbone到Head累積validation

依序加入backbone_early、backbone_deep、backbone attention安全子集、neck、MASF、neck attention安全子集、Detect tower、Detect predictor、Pose tower、Pose predictor。每一步從上一階段父policy分叉W8與W4，跑完整COCO／BBAT5 validation。

每個region完成後都停下來，提供accuracy、hardware、balanced三個角色的推薦。只有使用者確認後才進下一個region。

### W3：QAT recovery

W8若三任務相對matched activation FP-weight都下降不超過0.01，可直接保留PTQ，不必QAT。W4通常需要QAT；最多三個完整policy進20 epochs diagnostic recovery，且每個都有fake-quant-disabled matched sham。

現在沒有啟動W3。

### W4／W5

Uniform W8／W4穩定後，才在分布適合的高收益layers測Fixed SD4、LS-SD4、Paper-TWN與Channel-TWN／TTQ。最後才將winning weight policy分別搭配A7或A6重驗。

## 是否要重新訓練

不需要從零訓練。流程是：

1. 先PTQ／no-training敏感度。
2. W8達到0.01 incremental gate就不QAT。
3. W4從各activation自己的completed short-recovery checkpoint做低learning-rate QAT fine-tune。
4. 保留FP32 master weights，forward使用fake quant，backward用STE／LSQ。
5. activation A8 observer先校正並freeze；weight scale用per-output-channel MSE初始化。
6. 20 epochs只用固定30% diagnostic train，validation仍完整；通過才續60 epochs。
7. 100／120 epochs、3 seeds是正式訓練，目前不執行。

QAT仍保留Detect logical batch 128、Pose logical batch 16、BN frozen、fresh optimizer／scheduler／EMA／RNG、相同資料順序與trainable mask。O2M與pose_flow維持FP32但繼續提供訓練supervision。

## 每階段指標

每個scorecard固定包含：

- 三任務mAP50-95及相對matched parent、accepted Full35兩套delta。
- worst-task drop與0.01／0.04／0.06 gate狀態。
- weight與forward NRMSE、SQNR、cosine、clipping、saturation及TopK overlap。
- quantized modules／elements、各region bit與protected exclusions。
- weight code、scale、bias、requant metadata bytes及compression ratio。
- 若有QAT：loss、gradient norm、scale trajectory、best epoch與最近5 epochs趨勢。
- green／recover／reject、Pareto角色及我的推薦理由。

這讓使用者能在每個region選擇W8、W4、條件式W6、回退FP或停止，而不是等所有長訓練完成才看到結果。

## 尚未執行

本次只重新規劃與唯讀盤點，沒有建立weight quantizer、沒有跑W0／W1、沒有修改資料、沒有完整validation、沒有QAT或正式訓練。下一個實作節點是以TDD建立WeightRegionCatalog與W8／W4 WeightQuantizationAdapter。
