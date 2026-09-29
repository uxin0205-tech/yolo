# V29七組來源與qSiLU接手實驗設計

日期：2026-09-05

## 結論

Poly_shift V29已依使用者新決定，在第二組matched-sham完成epoch 1 checkpoint後安全暫停。原Queue、checkpoint與metrics完全保留，可從last.pt續跑；封存清單是artifacts/archives/v29-poly-shift-paused-for-qsilu-v1/archive-handoff.json，SHA-256為ce73d390f2c62421d7c98a26e5b46e980d89bc5443641bb7956e87031cdffb17。

接手主線使用uniform qSiLU+A8，不直接混入Hardswish，也不移植poly_shift的weight winner。理由是activation與weight量化耦合，qSiLU必須建立自己的all-W8 parent。

## 七組V29是如何選出來的

七組不是隨機選擇，也不是七個epoch。來源是V29完成backbone到head十階段PTQ後，將原先因門檻符號問題標為reject的候選，用修正後的雙指標re-gate重播。十二個候選進入可恢復區，再受最多六個main及兩個sentinel的預算限制；最後有六個main與一個具資訊價值的sentinel，共七組。

進入短QAT的條件是PTQ尚未達最終部署門檻，但仍位於recovery floor：worst total mAP50不低於−0.04，worst total mAP50-95不低於−0.08。選擇同時保留不同region與format，避免所有預算集中在同一區。

| 順序 | 候選 | 角色 | PTQ worst total mAP50 | PTQ worst total mAP50-95 | 要回答的問題 |
|---:|---|---|---:|---:|---|
| 1 | Pose tower exact ternary | main | −0.013842 | −0.071933 | 三元容量損失能否由QAT恢復 |
| 2 | Detect predictor Fixed-SD4 | main/control | −0.015784 | −0.016058 | 小predictor是否適合固定SD4 |
| 3 | Neck attention TWN-v3 | main | −0.016555 | −0.017826 | filter-wise三元是否適合attention FFN |
| 4 | Neck attention exact ternary | main | −0.016849 | −0.017625 | 同一path比較TWN heuristic與exact scale |
| 5 | Neck W4 | main | −0.018357 | −0.015020 | 一般uniform W4能否在neck恢復 |
| 6 | Backbone deep Fixed-SD4 | main | −0.018521 | −0.016347 | 深層backbone是否適合SD4分布 |
| 7 | Pose predictor TWN-v3 | sentinel | −0.028368 | −0.045725 | 高風險predictor三元化的否證界線 |

每組都有matched-sham與QAT兩個arm，最多15 epochs、patience 5。第一組已完成：QAT在epoch 6後early stop，共7 epochs；所有epoch的mAP50最差下降仍在−0.015內，但mAP50-95最差下降介於−0.0701與−0.0804，沒有best_joint通過總門檻。因此它是負結果，不是winner。

## 為何現在以qSiLU為主線

qSiLU checkpoint固定為SHA-256 7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e。既有Q3 matched CPU證據顯示：

| Activation policy | 最差global delta | 相對qSiLU |
|---|---:|---:|
| uniform qSiLU | −0.008810 | 基準 |
| qSiLU + Hardswish neck-attention | −0.008581 | +0.000229 |
| qSiLU + Hardswish MASF | −0.010301 | −0.001491 |
| qSiLU + Hardswish backbone-attention | −0.013518 | −0.004708 |

只有neck-attention略優於qSiLU，差距0.000229小於合理的promotion效果，而且Q3沒有A8/W8或QAT證據。故主線先使用uniform qSiLU；Hardswish只保留一格neck-attention條件旁支，MASF與backbone-attention只有在該格顯示activation×weight正向交互時才執行。不得把三個單區delta相加，也不採uniform Hardswish。

## qSiLU實驗順序

1. Q0：驗證V29 archive handoff SHA、qSiLU checkpoint、資料與graph contract。
2. Q1：在qSiLU+A8根上，把全部十個deployment regions／148 paths套W8 PTQ，補齊歷史九區證據。
3. Q2：Q1為green或recover才執行matched-sham與all-W8 QAT；15 epochs、patience 5。

Q1現已實測為recover：worst total mAP50為−0.020298、worst total mAP50-95為−0.021367；其中W8相對qSiLU parent的worst incremental mAP50為−0.008722並通過−0.01 gate。也就是mAP50-95已合格，但總mAP50仍需QAT恢復約0.0053，故進Q2而非直接鎖parent。
4. Q3：只有best_joint同時通過所有mAP50及mAP50-95門檻，才鎖定qSiLU all-W8 parent。
5. Q4：對148 paths做八格式CPU profile，共1184格：W7、W6、W5、W4、optimal Fixed-SD4、Paper-TWN v2、TWN-v3 filter-wise、exact ternary。此結果只排序，不直接promotion。
6. Q5：依backbone early→backbone deep→backbone attention→neck→MASF→neck attention→Detect tower/predictor→Pose tower/predictor逐區PTQ。每個stage都保留八格式，以前一個green parent累積。
7. Q6：只有recovery-band候選進matched短QAT，最多六個main與兩個sentinel；Fixed-SD4 recovery對應LS-SD4。
8. Q7：比較qSiLU與已封存poly_shift的相同指標Pareto，停在finalist review，不進formal或長epoch。

每個GPU候選輸出16項指標：八個mAP50及八個mAP50-95，包括COCO box、COCO person、BBAT box/pose及ball/bat class-level。最終門檻是每項total mAP50下降不超過0.015、每項total mAP50-95下降不超過0.04；平均值不能掩蓋最差任務。

訓練固定AdamW、weight decay 0.00027、betas 0.948/0.999、Detect logical batch 128、physical microbatch 16、Pose batch 16、不加noise，沿用accepted Full35 augmentation與canonical BBAT5。matched-sham與量化arm使用相同資料順序、optimizer budget及trainable mask。

## Hardswish條件旁支

機器規劃是configs/experiments/v32-qsilu-regional-hardswish-conditional-v1.yaml。它依賴qSiLU主線完成，不會現在搶GPU，也不會自動擴展成完整weight matrix。

第一格只測qSiLU all-W8 parent上的neck-attention單區Hardswish。除了通過總門檻，還必須相對uniform qSiLU parent的每項下降不超過0.002，且worst-task至少改善0.001，或有目標backend實測收益，才會晉級。若不成立，整個Hardswish旁支停止。

## Queue與監測

V30 Queue會自動執行Q0到Q7，例行錯誤每個arm最多重試一次並從last.pt恢復；hash、graph或資料契約漂移會fail closed，交由代理分析，不盲目重跑。

qSiLU的事件檔不是V29的execution-status.json，而是artifacts/queues/v30-qsilu-complete-quantization-lane-v1/status.json。Blocking monitor每600秒只比較kind、stage、candidate_id、arm、attempt、error_type與message；完全沒有變化時不輸出，發生transition才回報一行compact JSON並進入診斷。
