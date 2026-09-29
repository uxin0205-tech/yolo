# 2026-09-13：目前回合結束後暫停 Attention 恢復

## 變更內容與原因

使用者要求「這個部份跑完一個回合可以先暫停」。接到要求時，原生 QK＋PWL 已完成 E1 並進入 E2，故安排完成目前 E2 的 Float／BitTrue 驗證、所有選定 checkpoint 與 inference/last.pt 後暫停。scale_bias 保留待辦但不啟動，額外 BinaryQK 對照仍取消。未更動超參數、20 epoch 上限、資料集或訓練中的模型程式。

新增 pause_at_boundary.py，等待 inference/last.pt 的原子 rename 事件，隨即暫停目前訓練程序，CPU 核對完整續訓檔與推論檔的 epoch、step、來源 SHA、optimizer／scheduler／scaler／RNG 等狀態，核對成功後結束訓練及其已核對的資料載入子程序，釋放資源。不在寫檔中途停止，不把暫停偽裝成全部訓練完成。

run_queue.py 新增 pause-request.json 入口防護；未獲使用者恢復指示，不再啟動任何後續工作。後續恢復應先確認 pause-outcome-v1.json，保留暫停紀錄，獲授權後解除該旗標，再從 checkpoints/last.pt 的原 optimizer／scheduler／scaler／RNG 狀態接續 E3。

## 困難與解法

目前 trainer 沒有已載入的回合結束暫停介面；修改檔案不會改變執行中的 Python 類別。因此採用系統檔案事件與經 PID／啟動時間／argv 核對的訊號，在原子存檔完成後停止。讀取原始儲存流程確認 last 為最後保存標籤，inference/last.pt 落盤前完整續訓快照已寫完並計算 SHA。沒有直接改動外部 trainer 或重啟当前回合。其他困難：無。

## 驗證方式與結果

Python AST 檢查通過。CPU 暫存 fixture 驗證 inotify 原子 rename 偵測；短暫 CPU 子程序驗證 SIGSTOP → SIGTERM／SIGCONT 可安全結束。暫存 fixture 已自動清除，不是訓練結果。

唯讀核對既有 E1 快照通過，來源 SHA 為 482be4ae327fcd5a8c8d893590c44edec225f29025d671af0e3b179496a54319；完整續訓與推論檔的 epoch／step／SHA 一致，具備 optimizer、scheduler、scaler、EMA、RNG。這是暫停驗證程式的實檔測試，並不代表 E2 已完成。

2026-09-13 16:33（Asia/Taipei）只停止 queue PID 2351392，保留原訓練 PID 2342759 及啟動身分不變。暫停監測器成功取得 queue lock，輸出 PAUSE_ARMED，目標為完成 2 個回合。正常等待只由核心等待存檔／程序退出事件，不輪詢 log 或 GPU。

## 未解事項與風險

截至此紀錄，狀態是「已安排 E2 結束後暫停」，尚非「已暫停」。最終以 artifacts/queue-v1/pause-outcome-v1.json 的 PAUSED 結果為準；若核對失敗則保留已停止計算的程序並輸出 ERROR，需診斷後處理，不自動丟棄 checkpoint。E2 精度與續訓檔完整性待存檔事件發生後核對。未新增 Git commit／push，原正式權重未改動。

## 16:49 進度查詢補記

依使用者查詢，queue 最新仍為 PAUSE_ARMED，尚無 pause-outcome-v1.json。最新訓練事件為 epoch 1（第 2 回合）、macro_in_epoch 444、global step 907；該筆 loss 有限值、AMP overflow_retries 為 0，三個訓練部分均有梯度。暫停監測 session 仍在等待、未輸出錯誤。此次未更動程序或配置；目前仍需完成本回合剩餘更新、驗證及存檔後，才會執行已安排的安全暫停。查詢未發現新困難，不把 PAUSE_ARMED 當作 PAUSED。

## 暫停完成補記

16:52:37（Asia/Taipei）監測器輸出 PAUSED：已完成 2 個回合，global_macro_step=926；E2 完整續訓 SHA 為 6e960fd77666ba334fa5d2ba26a31031e26a9e895277ac9ff0dd4bb8ebb928f7，推論來源 SHA 核對成功。未啟動後續 job。結果保存在 artifacts/queue-v1/pause-outcome-v1.json；這更新先前「尚未暫停」的時點狀態。
