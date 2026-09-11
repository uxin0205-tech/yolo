# 現有 MASF 權重直接驗證與停止交付

## 變更與原因

使用者要求 P2／P3 都看 Pose 資料集，合併 COCO 後呈現並停止；隨後要求優先使用已訓練的權重。改為直接驗證，不重啟新 MASF 訓練。新增 `combine/evaluate_existing_masf.py`，使用原 Pose labels 的框，對 COCO Detect 輸出做 32→0、34→1 類別映射，不改資料或增加未訓練的 Pose head。

## 驗證與結果

GPU 驗證 exit 0，UTC 03:15:37–03:16:03。七個現成 Detect checkpoint 各完整 BBAT5 val 683 張，另原 Pose checkpoint 關閉 MASF 驗證 683 張；開啟組重用同 SHA256、同設定已完成 baseline。所有 PWL 實際範圍 [-10, 0]，類別映射與資料數量斷言通過。COCO 欄位對應同 epoch 完整既有 EMA 驗證。

原 Pose 開 MASF 相對關閉：ball box +0.005220、bat box +0.004674、ball keypoint +0.000253、bat keypoint -0.001746。P2 Detect 相對其 E5 control：COCO overall -0.000149、person +0.000099、BBAT ball box -0.000957、bat box +0.000606。不能一概判定 MASF 無效，也不能稱全部指標改善。完整 P3／P2 表格與限制見 [權威比較報告](<../../experiments/combine/pose-masf/RESULTS.md>)。

已執行無 MASF J0 完成 8 epoch；最佳 Pose 在 E8，整體 keypoint 0.807073，仍低於原獨立 Pose 0.912161，best_joint 未通過。這不是同 trunk 的 MASF 消融，不混為因果結論。

## 困難、處理與未解事項

交付前 CPU 檢查通過：17 個本地 Python 檔案語法、七組結果與來源路徑、全部 AP 有限且在 [0,1]、停用狀態與實際訓練入口閘門。GPU 訓練／smoke／queue 入口現在會在使用者尚未決定時直接拒絕執行。精確程序查詢未發現本次 combine 訓練或驗證仍在執行（pgrep 無匹配 exit 1，並非測試失敗）。未執行不相關全測試或再次 GPU 驗證。

新 Pose MASF smoke 的附加梯度斷言失敗：原 MacroStepEngine 更新後清空 `.grad`，而新檢查在清空後讀取。此為檢查時機問題，不能說 MASF 沒有梯度。正式 candidate 未開始；依使用者最新優先現成權重要求，不重跑 smoke 或啟動訓練。若未來恢復，須先修正觀測位置並完成驗證。

P2 Pose 新訓練準備碼未完成驗證，明確停用；目前 P2 結論限於已訓練 COCO Detect 的 BBAT box 泛化，沒有 P2 keypoint 結論。直接 Pose 開／關消融不等於成對重訓。未做多 seed 或顯著性驗證；既有資料上反覆研究有 validation 選擇偏差，不能當独立測試結論。

`finish-work` 用於彙整方法、原始來源、比較限制與同步停止狀態，不代表提交或清理授權。所有來源 checkpoint、run、失敗 smoke 與事件紀錄均保留；未刪任何檔案，未 commit／push。比較完成後停止，等待使用者決定是否使用 MASF；不接續 J1/J2、activation 或方向 2。
