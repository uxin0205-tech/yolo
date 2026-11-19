# 第五階段 HOG：完成覆核，保留原 parent

> 後續依使用者要求，未採用 HOG 的全部實體權重與 resume 已刪除；本頁指標與驗證紀錄仍保留。原主線模型與下一階段入口不受影響。清理明細見 `resource-cleanup-20261021.json`。

2026-10-21：H2 第一輪完整訓練後因球棒 AP50–95 下降 2.5305 個百分點，超過事前 2 個百分點停止門檻，於 12:14:26 停止。本輪不採用 HOG 適應候選；主線仍為第三階段 KD 第五輪，官方 AP50–95 **0.514620791**。此結果不能推論所有 HOG 方法無效，也不能推論延長訓練必定無效。

| 指標 | Parent | HOG 第 1 輪 | 差值（百分點） |
|---|---:|---:|---:|
| 整體 AP50 | 0.688156664 | 0.683722683 | -0.4434 |
| 整體 AP50–95 | 0.514620791 | 0.509686298 | -0.4934 |
| Person AP50 | 0.865282951 | 0.863359850 | -0.1923 |
| Person AP50–95 | 0.643206570 | 0.641292874 | -0.1914 |
| 球 AP50 | 0.719326365 | 0.710547802 | -0.8779 |
| 球 AP50–95 | 0.513224007 | 0.512737434 | -0.0487 |
| 球棒 AP50 | 0.724143671 | 0.708419207 | -1.5724 |
| 球棒 AP50–95 | 0.487594089 | 0.462289209 | -2.5305 |

## 實際完成與停止範圍

- H1：CPU coverage 1,024 張；16 項 CPU 測試通過。GPU 校準使用 256 張 train，μ=0.149032491，raw P3 梯度比中位數 4.3884%，目標 5%；未用 val 選 μ，未更新 optimizer。
- H2：GPU smoke 通過，完整 train118287 一輪、925 次更新、val5000。訓練 1,630.89 秒，含該輪評估 1,695.44 秒；AMP 重試 0 次，訓練峰值 allocated 5.543 GiB。最多十輪、至少五輪的 patience 設定受精度停止條件優先約束，實際只有一輪。最佳仍是未適應的 E0；表中列的是實際訓練過的 E1，不能以 E0 冒充 HOG 改善。
- H3：從已保存 resume 的 E1 EMA 還原，驗證 Neck、head、aux 確有更新；live／EMA 的固定參數與全部 BN buffers 相對 parent 逐位一致。移除 HOG head 的 CPU FP32 160×160 輸出與原生權重逐位一致；部署權重 strict reload 一致，另跑完整 FP32 val5000，八項 AP 及全部 official 欄位與 E1 完全相等。
- H4：僅提出對零有效 cell 框補最近有效 cell 的研究方向。固定 prefix 的球 59 框有 6 框無自身有效 cell，球棒 38 框沒有此情況；不能用這個覆蓋盲區解釋此次球棒下降。須先診斷背景污染與梯度密度，尚未實作或訓練第二候選。

## 方法來源與原因判讀

沿用事前 reference-review 的 11 份 optimize 程式、設定與結果來源 SHA；本次再次核對 optimize 的 HOG 負結果與歸因紀錄。保留既定配方，不放寬停止門檻，不追加 control，不追加 KD 或新候選。HOG／Neck／head 都有變動，單 seed、無本輪 matched control，不能把全部 AP 下降單獨歸因 HOG。已排除 parent 載錯、來源漂移、固定權重漂移與匯出／重載造成這次 AP 差異；校準初始梯度範數合格不代表後續任務精度必然改善。

資料仍為完整 COCO2017 train118287／val5000；沒有替換、重切或改標註。BBAT5、Pose 與 optimize 均未更動。

## 硬體與交接

HOG head 的 2,313 個參數僅用於訓練，部署移除後新增參數與運算為零。B2 二值 QK、S1 Q10、分解式 bias、完整 integer PWL exp 與 D1 reciprocal 契約保留。QKV／P·V／projection／Conv 仍浮點，原 activation 保留；全模型 qSiLU 依第七階段建立。本輪沒有 RTL／HLS 或板端加速量測。

正式入口為 `yolo_hog/result/stage5/handoff.json`。`selected.pt` 是既有主線權重的連結，格式仍由 stage3 loader 載入；`rejected-epoch01.pt`、E0 副本與正式 E1 resume 已依使用者後續要求刪除，僅留 manifest、SHA 與驗證數據。handoff 明確標記候選及 resume 不可用；選定 parent 與其 source resume 保留。

原 queue／supervisor 保留 AP guard 的 failed 狀態與錯誤訊息，沒有改寫成十輪成功；新增 `safety-stop-review/audit.json` 記錄停止後已完成的獨立驗證與交付。完整數據在 `metrics.csv`、`audit.json`，圖為 `delta.svg`；`scripts/close_safety_stop.py plot` 只讀 JSON，可無 GPU、無 checkpoint 重畫。

第四階段 MASF 正式結果、候選與必要 resume 全數保留，等待融合後再決定是否重試。第五階段在此停止供使用者覆核，第六階段待指定。
