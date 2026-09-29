# 2026-09-08：單一主代理、取消配額門檻與 400 秒監測

## 使用者最新要求

只由主代理執行，不再啟動、呼叫或委派子代理；先前 weekly remaining 到 73% 收尾的限制已取消。GPU 工作改為每 400 秒完整監測一次，取代本輪先前 600 秒設定。沒有修改全域或兄弟專案檔案；歷史工作紀錄中的 600 秒樣本保留，不改寫已發生事件。

## 變更與原因

對話中斷後，原 supervisor session 已失效，但訓練 PID 3617439 仍在原地執行。唯讀核對 `/proc` 與 command line，确认它是同一 workspace recovery train，沒有重新啟動或重跑。E4 已完成，E5 繼續。

新增 `scripts/monitor_existing_recovery.py`，只接受 workspace 內既有 run 與可核對的 recovery PID，使用 PID start-time 避免誤認重用 PID。獨立監測程序 PID 3639592 不因對話中斷而被一同關閉。每 400 秒讀 GPU、磁碟與最新進度；間隔內只每 10 秒檢查程序是否退出，提早完成／失敗就立即寫事件。它不啟動、停止或修改訓練。非其 child 的退出代碼不可取得，因此明記 `exit_code_observable=false`，不假造 exit code 0。

未來 `supervise_recovery.py` 的完整硬體採樣間隔也改為 400 秒。同步 README、機器可讀計畫與工作紀錄索引；原始模型、checkpoint、資料及 optimizer 設定不變。

## 驗證

新 CLI help 成功。實際接手檢查成功，stderr 為 0 bytes；首筆監測於 21:39:28（Asia/Taipei），GPU 59°C、使用率 30%、顯存 11633 MiB，訓練仍在 E5。日誌：`artifacts/direction1-20260908/logs/native-parent-ema-control-adopted.monitor-400.jsonl`。採用小範圍執行檢查，不擴充模型測試。

## 困難與解法

原 terminal session 隨對話中斷失效，訓練子程序仍活躍；以只讀身份核對後接手監測，不重複啟動 GPU 工作。其他困難：無。

## 未解事項

本紀錄建立時 E5 尚未完成最終評分；後續結果見原生對照工作紀錄。HOG 尚未正式啟動。監測不代表精度已回升。
