# 2026-10-19：第四階段延長與分組學習率比較

## 任務與範圍

使用者在詢問狀態後明確授權「訓練的時間可以拉長，也可以嘗試一些不同的方法」。本輪仍是第四階段 MASF，沒有啟動 HOG、Combine 或其他階段。

## 第一輪結果

第一輪於 20:53:22 完成，9 個 jobs 成功。Detect 五輪中最高 AP50–95 為第五輪 0.514576492，低於 parent 0.514620791；Pose 關鍵點最高為第五輪 0.919994127，低於 parent 0.920709389。Pose 第五輪 bbox AP 從 0.628401841 升至 0.630224466，但關鍵點下降，需分開呈現。Detect formal AMP retry 共 1 次，Pose 共 3 次，均重算原 macro，沒有跳過資料。

原自動選型保留 epoch 0，因此原 alpha-on/off 只證明初始等價，不能代表已訓 MASF 貢獻。本輪補匯出兩任務第五輪 EMA、strict reload，並正式評估實際已訓 checkpoint 的開關。原流程的實作與來源 SHA 已在修改前通過稽核，保存於第一輪 `final-audit.json`。

## 參照與變更

實作前先讀 optimize 的 `continue_masf_head.py`、`train_masf_task_bridge.py`、`train_masf_p3.py` 與 Pose 五輪配方，保存七個來源 SHA 到 `yolo_masf/artifacts/stage4/extension-reference.json`。參考其保存 optimizer／EMA／scaler 並調整角色 LR 的方式；本輪 L2 數值是針對新 context 的研究延伸，不宣稱重現歷史已訓 donor 結果。

兩任務分別從同一份第五輪 live、EMA、AdamW moments、loss scale 與 EMA 次數獨立出發。L1 接續原末端 LR：head 2.5e-6、context 5e-6、alpha 5e-5。L2 的 head 降至 5e-7、context 升至 2.5e-5、alpha 不變。兩組同資料順序與 cosine 衰減至 .2，不重新 warmup；原 E2E loss 已到終點，維持 one2many／one2one 的 .1／.9，不因延長 epochs 默默重啟 loss 排程。

每組增加 5～15 輪，最多累計 20 輪；至少五輪後，primary AP 連續五輪未改善即早停。保存最高 primary AP 候選，不因類別安全門檻丟失權重；正式推薦仍同時檢查 parent、其他 AP 與已訓 alpha-on/off，供使用者覆核。早停可能使兩組輪數不同，報告須明記。

硬體圖、qSiLU、Detect 的 B2／S1／PWL／D1、beta 1 與所有 BN running 保持原契約。資料仍是完整 COCO 與不可變 BBAT5 v1；沒有 control 加訓、額外 KD、split 或 label 變更。

## 驗證方式與結果

原五輪 resume 與匯出已確認逐位相等。Detect alpha EMA 0.003220698，Pose 0.000509449，context 也有實際更新。後續 CPU bootstrap、resume、loss clock、GPU smoke 與正式開關結果續記。

## 困難與解法

原選型停在 epoch 0，開關診斷不足；改用唯一正式 resume 的第五輪 EMA 補測。延長訓練時直接更改 epochs 會改變 E2E loss 時鐘，因此明確保存已完成五輪的 loss age 並固定末端比例。

## 未解事項或風險

延長不保證改善；單 seed、沒有額外 control，不能分離 MASF 與 head 適應的獨立貢獻。所有結果供使用者檢視，工作停止於第四階段。


## CPU 與啟動核對

17 項 bootstrap／resume／loss clock／LR／既有引擎測試通過，另補 1 項完整佇列初始化測試，共 18 項。Ruff 通過。初始化曾因 L1／L2 大寫 job ID 被共用 QueueExecutor 拒絕，GPU 尚未啟動；改為小寫 ID，保留報告方法名稱，直接刪除未啟動的錯誤初始化並保存小型原因紀錄。

正式延長 run：`yolo_masf/artifacts/stage4/extension-20261019-v1`。Supervisor PID 801908，GPU 啟動前無其他 compute process；先補第五輪 alpha-on/off，再四個 smoke，接續 Pose L1／L2 與 Detect L1／L2。每組 5～15 個新增 epochs，600 秒監測，完成後停在第四階段。

## 已訓開關、GPU 短測試與背景交接

兩任務的第五輪 EMA 已完成正式全量 alpha-on/off，重新載入後 alpha-on 各項 AP 與原第五輪相同。Detect alpha 為 0.003220697865，primary AP 開啟減關閉為 +0.000008448032；Pose alpha 為 0.000509449397，差為 +0.000001551426。差距非常小，不能據此宣稱有明確增益；alpha-on 的 primary AP 也仍低於各自原 parent。

Detect／Pose × L1／L2 四個 GPU smoke 全數通過，各完成 2 個 optimizer steps，均無 AMP retry。alpha、head 與 context 均有實際更新；兩個方法同任務的首批資料順序相同。短測試只檢查訓練行為，不作正式 AP 結論。

2026-10-19 22:04（Asia/Taipei）Pose L1 正式續訓已通過完整 683 張初始驗證，精確承接第五輪 AP，並完成首輪 20 個更新步驟，loss／梯度有限，無 AMP retry。GPU 為 RTX 5090，當時只有本流程 PID 801920 使用 GPU，約 6 GiB 程序顯存。Supervisor PID 801908 已脫離互動工作階段，持續每 600 秒監測；四組訓練依序執行。

依首輪實測 Detect 每輪約 29.5 分鐘、Pose 約 1.5 分鐘估算，包含評估及早停差異，整體預計在 **2026-10-20 03:30～14:00（台灣時間）** 完成；這是估計，GPU 資源競爭或異常會延後。完成時背景流程產出比較、開關診斷、最佳候選與交接，並停在第四階段。交接狀態保存為 run 內 `working-handoff.json`，不需要維持互動工作階段等待訓練。

本次同步修正根層 README 的舊「第四階段待指定」與專項 README 的舊「流程已完成」語句，明確區分首輪完成和目前延長研究。此輪 GPU 檢查無新困難；延長的最終 AP 與採用建議仍待完整結果。


## 背景流程完成

# 第四階段延長結果

| 任務／方法 | 總輪數 | 最佳 AP | 與 parent 差 | Alpha-on/off 差 | 建議採用 |
|---|---:|---:|---:|---:|---|
| detect/L1 | 13 | 0.514675024 | +0.000054233 | -0.000007983 | False |
| detect/L2 | 20 | 0.514728468 | +0.000107677 | -0.000005570 | False |
| pose/L1 | 11 | 0.920154750 | -0.000554639 | +0.000001261 | False |
| pose/L2 | 10 | 0.919994127 | -0.000715261 | +0.000001551 | False |

每個任務的 L1／L2 都從同一 E5 完整訓練狀態出發；早停導致預算可能不同，不能宣稱嚴格等輪數比較。單 seed、未新增 control、沒有硬體實測。最高 AP 與類別保護建議分開，完整結果和權重均保留供覆核。第 4 階段完成，未啟動第 5 階段。
