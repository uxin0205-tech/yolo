# 新 P3 bridge：Activation → 雙任務 KD

最新狀態：四臂零樣本、兩臂各 10 epoch 及選定 qSiLU E2 匯出重驗均已完成，無執行中的 GPU job。見 [完整結果](<RESULTS.md>) 與 [教師候選研究](<../../kd/dual_task_v1/TEACHER_OPTIONS.md>)。以下配置及啟動敘述保留為實驗歷史；KD 尚未訓練。

依 2026-09-11 使用者最新指示，先 activation，再方向 2 KD；不再先插入共享層 BBAT 恢復。來源是 optimize 重新訓練的 P3 bridge + Pose head recovery E5，不是舊 combine，也不是未採用的 BN 校準候選。原融合 BBAT gate 尚未全過，必須持續揭露，不能藉 activation 基準重設宣稱問題消失。

Activation 優先 qSiLU-PQ，Hardswish／PolyShift 為評估對照；重用唯讀 `yolo_activation/src/activation_lab/activations.py`。先 CPU 契約／SiLU 重現／三個零樣本候選，再同起點、seed 1 的 SiLU／qSiLU 10 epoch 配對短恢復，warmup 1。依原 short recovery recipe 採 AdamW、J3 LR×0.1、Detect logical128／physical32、Pose physical16，完整 COCO 與 canonical BBAT5。其他候選是否短訓視零樣本結果決定，不照抄舊 queue 中其他模型已完成的狀態。

Activation gate 沿用原 recipe 相對 SiLU 最大單项 AP 下降 0.015，同時列出原融合 gate；通過淘汰線不等於勝過 SiLU。MASF P3 bridge、α 路徑、PWL [-10,0]／20 段與 BinaryQK 硬體契約不變。

目前四臂零樣本已完成；qSiLU通過0.015線，Hardswish／PolyShift未通過。兩臂配對短訓已由run_pair.py啟動。qSiLU physical32反向OOM，實際兩臂統一physical16、logical128不變；兩臂新smoke通過，qSiLU峰值allocated約19.2GB。原始32設定僅作來源說明，不是目前執行值。

## KD 雙資料集契約

沿用 `optimizations/round2-innovation/region-ranking-full-spec.md`，不是一般單資料集分類 KD。

```text
COCO batch  → 相同增強影像 → teacher Detect／student Detect → Detect native + KD_D
BBAT5 batch → 相同增強影像 → teacher Pose／student Pose     → Pose native + KD_P
                                                       ↓
                                  既有 macro task 權重與 batch 正規化
```

兩份資料沒有每張影像的雙任務完整標註，不混 class IDs、不把 COCO 未標 keypoints 當零真值、不把 BBAT2 標註送進 COCO80 head。KD 僅訓練期存在。
先查核真正 FP-QK joint teacher（Float PWL 的 BinaryQK 不算）、任務別優勢與 student live score 梯度，再 K0／K-UNIFORM；普通 KD 有訊號才 K-REGION。原規格限制單 joint teacher；若現有來源不足而需兩個 standalone teachers，必須明確另立方案，不靜默偷換。KD 尚未啟動，未承諾不存在的 teacher 或可用梯度。

GPU 使用 `combine/monitor.py` 的 600 秒 child wait，正常期間不讀長 log；完成／錯誤再介入。所有結果另存本目錄，舊專案與資料集唯讀。
