# YOLO26m PWL 最終訓練領域

本流程使用可微分 Float-PWL surrogate 訓練，再將候選權重獨立轉成 Bit-True PWL checkpoint 評估。已完成的 staged-recovery queue 以分層遞減學習率，依序測試 Attention 所在 block、Neck/Detect、Backbone 最後 stage 與 full model；BatchNorm running statistics 全程鎖定。

## 專案詞彙

- **Parent**：保留且不可變的 `v1-br` 起始 checkpoint。
- **Float-PWL**：只在訓練反向傳播使用的可微分 PWL exponential surrogate。
- **Bit-True PWL**：用於 gate 與正式交付的 Q8.8/UQ1.15 整數分子 reference。
- **Phase gate**：child 相對直接 parent 的 mAP50-95 下降不超過 `0.001` 才接受。
- **Best-observed**：單一路徑中實測 Bit-True 指標最高的候選。
- **Formal winner**：只有三個 seeds 的平均值比 zero-train parent 至少高 `0.001`，訓練權重才能成為正式 winner。

## 已完成決策

LR sweep／recovery queue 已完成 19/19 個工作。block x1 是最高觀測候選，mAP50-95 為 `0.5069387`，但只比 audited parent 高 `0.0001944`。依使用者指示沒有續跑 seed 1/2，因此正式 winner 仍是 zero-train fallback `final/pwl-final-best.pt`。可攜交付位於 `final/`。
