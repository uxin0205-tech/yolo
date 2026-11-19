# 第五階段：Detect 訓練用 HOG

2026-10-21 使用者授權進入本階段，並保留全部 MASF 正式結果、候選與必要 resume，待融合後再決定是否重試。HOG 從 [第四階段 Detect 交接](../yolo_masf/result/stage4/detect/handoff.json) 的原 parent 出發：第三階段 KD 第五輪，官方 AP50–95 0.514620791。本階段沒有擴展到 Pose；後續 Stage6～9 已完成，見 [統一交接](../result/README.md)。

**本階段已完成負結果覆核並停止。** H2 完整訓練一輪後，整體 AP 0.509686298、球棒 AP 0.462289209，球棒下降 2.5305 個百分點觸發事前停止條件。本輪不採用；H3 實際 E1 剝除／重載與全量 AP 一致。主線及未採用候選見 [result/stage5](result/stage5/README.md)，八項指標與限制見 [正式結果](artifacts/stage5/formal-20261021-v1/RESULTS.md)。下方是當時固定的研究配方，實際未跑滿十輪。

**後續資源清理：** 依使用者要求，HOG 的未採用候選、E0 初始副本、正式 resume、預覽圖與快取已直接刪除，沒有封存。只保留程式、小型數據、manifest、SHA、圖表與原主線連結；本資料夾不再含 HOG 實體 checkpoint，也無法直接續跑此候選。`handoff.json` 的 candidate／resume 已設為 null，歷史路徑另存為刪除證據。見 [清理紀錄](../docs/worklogs/2026-10-21-hog-resource-cleanup.md)。

先唯讀核對 optimize 的原始 HOG 算子、訓練／coverage 腳本、實際 summary 與負結果，來源 SHA、配方及差異已在實作前寫入 [reference-review](artifacts/stage5/reference-review.json)。本目錄原先不存在，沒有需要刪除的舊內容；不動其他階段資料夾。

## 方法與範圍

raw P3（layer 16、256 channels）接 1×1 Conv 預測 9-bin unsigned HOG。目標來自同一批增強後 RGB；cell 8、方向線性插值、cell 內 L1 正規化。有效 cell 中心必須落在 GT 框內且具有非零梯度能量。每張有效 cell 平均後 batch-sum，與原生 loss 一起乘 64／actual macro。

只訓練 Neck layer 13／16／17／19／20／22 的非 Attention 參數、Detect head 與 HOG head；backbone、全部 Attention 參數、scale／bias／定點常數及全部 BN running statistics 固定。head／neck／HOG LR 為 5e-6／1e-5／3e-4，AdamW betas .948／.999、weight decay .00027、clip 10。新 optimizer，EMA age 從 SHA 固定且逐位等於 parent 的 KD resume 繼承 2315。原 E2E loss 已到終點，維持 .1／.9。

單一 H2 候選：完整 COCO train118287／val5000，physical16／logical128，最多 10 輪，至少 5 輪後 patience 4。第一輪漸開 HOG，第 2～8 輪啟用，第 9～10 輪關閉輔助 loss 作同一候選末段適應；沒有另跑 control。μ 由固定 256 張 train prefix 的 raw P3 梯度校準至 5%，不使用 val 選 μ，不沿用 optimize 舊 μ。

正式選型要求整體 AP 提升，且八項監測 AP 均不下降超過 .001；候選與建議分開保留。任一監測 AP 下降超過 .02 時停止分析。單 seed、無額外 control，不能把所有變化歸因 HOG。

## 覆蓋與部署驗證

[CPU coverage](artifacts/stage5/coverage.json) 固定檢查 1,024 張增強後 train prefix：59 個 sports ball 框有 6 個沒有自身有效 cell（10.17%）；38 個 baseball bat 框沒有此情況。這是 prefix 診斷，不是全資料比例，也不代表無原生／重疊框梯度。H4 可提出「只補零 cell 框」的候選，尚未實作或啟動第二組訓練。

HOG 只有 2,313 個 train-only 參數；推論完全移除 head，新增 HOG 參數與運算為零。HOG target 的 sqrt／atan2／除法及訓練 loss 的 log-softmax 均不進部署圖。B2／S1 Q10／完整 integer PWL exp＋D1 reciprocal、原 activation 類別與係數完整繼承，不回退浮點 normalization；全模型 qSiLU 主線仍依第七階段建立。

CPU 已驗證方向、邊界、切批、空框、常數圖、停梯度 target、μ=0 原生 loss／梯度等價、HOG 到 P3 producer 的梯度、剝除輸出逐位等價、training／deployment strict reload 及禁止 normalization fallback。GPU 校準與 smoke 通過；訓練後已獨立重載實際 E1 剝除模型，完整 val5000 的全部 official 指標與原 E1 完全一致。

## 操作與交付

```bash
bash yolo_hog/run_stage5.sh coverage
bash yolo_hog/run_stage5.sh init --output /home/uxin/yolo/yolo_hog/artifacts/stage5/formal-20261021-v1
bash yolo_hog/run_stage5.sh status --output /home/uxin/yolo/yolo_hog/artifacts/stage5/formal-20261021-v1
bash yolo_hog/run_stage5.sh run --output /home/uxin/yolo/yolo_hog/artifacts/stage5/formal-20261021-v1 --execute
```

run 沒有 `--execute` 只預覽。沿用 Attention QueueExecutor 和 binaryqk 訓練引擎；背景每 600 秒監測，STOP 檔只停止本 run，保留最後完整 epoch 的 resume。新 GPU 工作先核對其他 compute process，避免資源衝突。

`result/stage5/` 現保留 manifest、八項 AP、推薦 parent 連結及已刪候選／resume 的歷史 SHA；正式負結果小型數據保留，所有本輪 HOG 實體權重均已依使用者要求清理。本階段當時完成後停止，後續研究已完成並另有正式 result。Git 發布依根層政策排除 checkpoint、runtime dataset 與 cache；未要求發布時不 commit／push。

中文紀錄見 [第五階段工作紀錄](../docs/worklogs/2026-10-21-hog-stage5.md)。


原 queue／supervisor 的 `failed` 狀態保留 AP guard 當時的真實事件；停止後的獨立覆核狀態見 `artifacts/stage5/formal-20261021-v1/safety-stop-review/audit.json` 及 `working-handoff.json`。不要為清掉 failed 狀態而重啟原 queue。

本輪負結果圖不依賴 GPU 或 checkpoint，可執行：

```bash
.venv/bin/python yolo_hog/scripts/close_safety_stop.py plot --output yolo_hog/artifacts/stage5/formal-20261021-v1
```

歷史停止後驗證使用 `scripts/close_safety_stop.py verify`，已完成的結果保留；目前實體權重與 resume 已刪，不能直接重跑 verify 或續訓。`plot` 仍只讀 JSON，可獨立重畫。完整驗證過程見 [完成工作紀錄](../docs/worklogs/2026-10-21-hog-stage5-results.md)。
