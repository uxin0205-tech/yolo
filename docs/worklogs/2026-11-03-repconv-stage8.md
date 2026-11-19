# 2026-11-03：啟動第八階段 qSiLU RepConv

## 任務與範圍

使用者指定「進入下一步」，啟動 Stage8。三組 Rep17、Rep20、Rep17＋20 從同一 Stage7 qSiLU E8 獨立開始。保留 qSiLU、B2 sign＋Hadamard、S1 Q10／signed16 bias、D1 整數 PWL exp＋LUT256 reciprocal。Stage8 完成即交付停止；Stage9 未啟動。工作延續至 2026-11-04 凌晨。

## 事前參考與決策

已唯讀核對 optimize 的 `pose_only_rep_v1` 實作、preflight、config、三組實際 resolved-config／smoke／summary、RESULTS、final-audit，以及上游 pose_masf_training_v1/training_b.py 與共用 repconv.py。26 項 SHA、實際結果及採用／差異理由保存於 [reference-review.json](../../yolo_achitechure/stage8/reference-review.json)，在新實作前寫入。

沿用雙線性分支相加後 qSiLU、新 1×1 分支 BN gamma／beta 為零、只更新指定 Rep 與兩個 heads、BN running 固定、Attention 固定、AdamW、warmup／cosine、5 輪初始預算、fresh EMA、FP32 CIoU＋AMP、16×16 Detect 累積與 Pose16。Rep LR 1e-5、head LR 5e-6，weight decay .00027，beta1 .948，clip10。

差異：本輪 parent 為融合後的 qSiLU E8，沒有 MASF；pose weight 從既有 .04 出發，以 train macro 梯度重新校準，不照搬 optimize .25。optimize 舊 PWL 尚有浮點分母除法，本輪明確繼承 D1 完整 reciprocal。部署 Rep 真正折為單 Conv3×3+bias，不留下 identity BN。以當前 parent 的十項 AP、相對跌幅及平衡分數選最佳輪，原 optimize 八項工程條件（COCO overall/person＋BBAT 六項）另列；有改善時允許一次半 LR 延長 5 輪，最多共 30 輪。

## 實作與清理

新增 `yolo_achitechure/stage8/`，包含嚴格 manifest 載入、Rep／folded 模型、CPU 診斷、算子實際呼叫盤點、校準／短訓練／正式訓練、條件式 queue、600 秒監測、結果及可攜交接。部署和多分支 EMA／完整 resume 分開；qSiLU 共享 alias 與固定係數都保留。共用算子沿用原維護位置，交接 runtime 是固定版本的推論快照。

清理前查核 process 與外部依賴，刪除 achitechure_1/artifacts 下 6 個無引用舊 checkpoint，共 253.92 MiB，另清除 cache。156 個受保護來源／權重 SHA 保持相同；optimize 及正式資料不變。仍被 optimize／combine 使用的 achitechure_1/src、final、inputs 及 achitechure_2 正式資料不清空。詳見 [清理清單](../../yolo_achitechure/stage8/cleanup-20261103.json)。

## 驗證方式與結果

CPU 三組均通過：初始化輸出等價；刻意設定非零 1×1 分支後的 FP32／FP16 線性及 qSiLU 完整輸出 folding；160／640 全模型輸出；training／folded 重載；Detect／Pose routing；錯誤 SiLU 回退拒絕；兩處 D1 normalizer 與 qSiLU 確實呼叫。初始模型未使用 GPU。

全模型算子盤點保留原生 head confidence／visibility sigmoid 三次呼叫；Attention／activation 沒有回退 exp／softmax／SiLU。其他 ConvBN 尚未全網融合，Conv／QKV／P·V／projection 與 qSiLU 任務路徑仍浮點。部署版 Rep 本身無 BN。此為公式與軟體路徑驗證，不是 RTL／HLS 或實板效能。

獨立程序可攜交接整合測試已通過（pytest 1 passed），驗證非零 Rep folded 輸出逐位相同、qSiLU 回退拒絕與搬移後 SHA。Ruff 與 shell 語法檢查通過。

正式 AP／訓練尚待 queue 實際結果；不可把 CPU passed 當精度改善。完成後自動補結果於本文及 result/stage8。

## 困難與解法

Decoded top-k 在微小浮點誤差下可能改變列排序；比較器保留原 atol／rtol，以完整一對一配對驗證，raw anchor 輸出仍按原順序比較，並確認錯誤類別無法通過。清理保護清單起初誤納可重建 pyc cache，已修正為來源／唯一權重並核對全部 SHA，沒有刪除受保護模型。

## 未解事項或風險

GPU 校準、非零分支實際更新、完整 AP 與訓練品質需實測。所有候選只使用一個 seed，沒有未改方法 control 加訓；精度未達門檻時保留 parent，不硬選 Rep。尚有浮點運算及 head sigmoid；不宣稱全整數或已量測加速。其他 GPU 工作出現時 queue 會讓出本輪 worker。

## GPU 啟動

2026-11-04 00:07 左右啟動本階段 supervisor（PID 3131327），單 RTX 5090。啟動前沒有其他 compute process，空閒顯存約 31 GiB。初始任務為 Rep17 E0 完整 AP；之後校準、短訓練通過才啟動長訓練。來源與 protocol 已鎖定，每 600 秒監測，60 秒檢查其他 GPU 工作並避讓。

## 首組 GPU 前置結果

Rep17 E0 完整 AP 通過：COCO AP 0.513950617、Pose box AP 0.613904202、Pose AP 0.900761588；與 parent 最大單項差值為 COCO 球棒 AP -0.000011308，來自 FP32 folding／排序微差。其他 BBAT 六項完全相同。

梯度校準通過：初始 .04 權重時 Pose／Detect 中位梯度比 .20258，選定 Pose weight .195，確認比 .91238。2 macro smoke 通過：318 個參數 tensor 更新，新增 1×1 conv／BN gamma 及兩 heads 都確實更新；固定 state／EMA 不變，EMA updates=2，測試權重丟棄。peak allocated 6.165 GiB、reserved 6.281 GiB，初測約 5.70 秒／macro。Rep17 正式訓練已開始。可攜測試的 106.18 MiB 暫存包已刪除，保留程式與測試結果紀錄。

## 長訓練交接

2026-11-04 00:12 左右：Rep17 第 1／5 輪前 30 個 macro 已持續完成，EMA 同步更新，loss 有限，沒有 worker 異常。前段有 AMP scaler 從 1024 降至 512，既有重試機制處理後正常前進，並非未處理的 NaN。當下平均約 4 秒／macro，加完整驗證後每輪約 32–36 分鐘。基本三組共 15 輪預估 11 月 4 日 08:00–10:00 完成；若三組均觸發延長，暫估 17:00–20:00。其他 GPU 工作插入會順延；估計與來源鎖檢查保存在 startup-check.json。

依使用者要求退出互動 working，背景 supervisor 保持每 600 秒監測；Stage8 完成後輸出結果並停止，不啟動 Stage9。

## 自動完成紀錄

2026-11-04T07:54:04+08:00：三組完成，建議 `rep17_20`，最佳 Rep `rep17_20`。folding、重載完整 AP 與可攜 CPU 驗證均通過；詳見 [正式結果](../../yolo_achitechure/result/stage8/README.md)。Stage9 未啟動。
