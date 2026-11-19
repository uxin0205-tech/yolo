# 2026-10-19：第 3 階段完成、KD 判斷與舊產物清理

## 任務與結果

使用者要求查看結果、判斷是否需要恢復訓練與 KD，並移除舊資料夾內無用內容。第 3 階段於 11:57:24（Asia/Taipei）完成，耗時 540.08 秒；14 個 queue jobs 全部 succeeded，其中 9 個為完整 val5000、3 個為診斷排除，另有診斷與選型各一個。GPU 已釋放，第 4 階段未啟動。

D1 官方 AP50–95 0.514189349，S1 0.514151658；依事前規則選 D1。三種完整 PWL／reciprocal 都通過數值與 AP gate。N7～N9 僅診斷排除，不記假 AP；N6 的正式 AP 0.510462290，作為負結果保留。

## 恢復與 KD

沒有本階段掉點需要恢復，故不追加一般訓練。KD 可以處理相對原始模型 0.524685533 的累積差距，但原始教師候選的 bat AP 比 D1 低 0.001489；不能只看 overall 就直接啟動全圖模仿。已唯讀查核 optimize 的雙教師／Pose-head KD 實作、梯度校準、實際未通過 gate 的結果與原始文獻，形成 [恢復與 KD 判斷](../../yolo_attention/docs/stage3-recovery-kd.md)。本次沒有新增 GPU 訓練或 unchanged control。

## 整理與驗證

先保存逐檔清理清單、SHA 與歷史證據索引，再處理已由正式選型及本輪取代的舊 checkpoint、預覽圖、曲線圖片與快取。必要唯一權重集中到 `yolo_attention/weights/`，舊入口保留相容連結；正式資料、Pose、optimize、當前候選與上游 parent 均不刪。舊 metrics、配置、results.csv 仍在原路徑，沒有建立 archive／backup。實際刪除 37 個舊 checkpoint、217 張預覽／曲線圖、63 個快取檔案，另去除 5 份位元相同權重，合計 322 個檔案、87 個空資料夾，釋放 **2.601010738 GiB**。四份必要唯一權重實體搬移，SHA 不變。`yolo_attention_final` 實體用量由約 2.8 GiB 降至 7.4 MiB；小型唯一 state、指標、配置與相容入口保留。見 [逐檔清理清單](../../yolo_attention/artifacts/stage3/cleanup-final-delivery.json)。

## 交付與驗證結果

D1 已由原 run 實體搬至 [result/stage3](../../yolo_attention/result/stage3/README.md)，搬移時六個檔案 SHA 全部相同，權重 SHA `a691a600715b58bf7d1438e379022443fde784c95ae75bf13336da28cc728a11`。舊 run 是相對連結，沒有第二份權重。handoff 含 S1 parent、scale／bias 常數、exp／reciprocal、整數格式、activation call-sites 與完整 metrics。

- 整理後完整 CPU suite：**193 passed／1 skipped，22.06 秒**；唯一 skip 是既有 BDCN CUDA 測試，本次沒有再占 GPU。
- `ruff check --no-cache src tests scripts` 通過。
- 九個正式 candidate checkpoint 全部嚴格重載；同一 S1 SHA、COCO annotation SHA、每組 val5000 與實際 metrics 一致。D1 FP32 的 64×96／96×64 smoke 均為有限輸出。
- 僅複製 comparison.json 到臨時目錄，即可在無 GPU／checkpoint 下重畫 SVG。
- 由真正舊 `yolo_attention_final/src` import 入口載入相容連結的歷史正式權重，CPU 輸出正常。
- 清理清單逐項核對通過，保留證據 SHA 不變、相容連結無斷鏈；13,320 個 optimize 實體檔案的 size／mtime 與清理前一致，另由 Stage3 source guard 核對關鍵 optimize 原始碼 SHA 不變。

驗證詳細資料見 [handoff-verification.json](../../yolo_attention/result/stage3/handoff-verification.json)。本次未 commit／push，也未啟動下一階段。

## 困難與解法

舊目錄不能直接整個刪除：optimize 的 preflight、發布文件及既有模型仍引用其來源、配置與唯一權重；已將必要依賴與被取代產物分開處理。初次清理分類遇到小於盤點大小門檻的唯一 `pwl-attention-state.pt`，程式在刪除前停止，補入保留清單，沒有誤刪。其餘無。

## 未解事項與限制

尚無本輪 KD 收益、RTL／HLS 或板端速度量測。D1／D2／D3 的微小 AP 差異不代表統計顯著性；D2 仍是較小 ROM 的可用選項。qSiLU／RepConv 與 MASF 等後續階段尚未開始。

## 13:47 狀態核對

使用者詢問目前狀況；唯讀核對 supervisor、handoff、交付驗證、KD 與清理紀錄，均與完成狀態一致。D1 權重存在，官方 AP 0.514189349；KD 與第 4 階段均未啟動。GPU utilization 0%，無 compute process，顯示用量 611 MiB。本次未重跑測試、AP 或訓練；193 passed／1 skipped 為前次交付驗證結果。困難：無。下一階段仍待使用者指定。
