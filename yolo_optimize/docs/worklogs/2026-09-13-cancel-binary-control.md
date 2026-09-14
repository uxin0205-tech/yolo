# 2026-09-13：取消額外 BinaryQK 對照，不中斷目前訓練

## 變更內容與原因

使用者明確表示「BinaryQK 對照 not need」。取消尚未開始的 binary_control 正式 20 epoch 訓練與其獨立驗證，保留 native_qk、scale_bias 兩組及既有 qSiLU E2 比較基準。已完成的 CPU／E0／GPU smoke 與原始權重全部保留，不將取消工作偽装為訓練成功。

新增 queue-plan.json 作為正式待辦範圍；queue 每個 job 前檢查目前範圍。summarize.py 不再要求 binary_control 結果，也不產生 delta_vs_control，改為既有正式模型對兩組新成果的比較。同步更新根 README 與實驗 README。

## 困難與解法

舊 queue 已將三組工作讀入記憶體，只改原始碼無法取消它尚未執行的對照。新增可驗證 PID、啟動 tick、完整 argv 的接管方式，使用 Linux pidfd 靜默等待既有工作結束；身分不符即拒絕，避免 PID 重用或重複啟動訓練。

2026-09-13 15:47（Asia/Taipei）只對已核對的舊 queue PID 2342757 發送 SIGTERM，沒有對訓練子程序或 process group 發送訊號。native_qk 訓練 PID 2342759 與啟動身分保持不變，新 queue 記錄 JOB_ADOPTED。沒有覆寫 checkpoint，沒有重新跑正常 job。接管資訊保存在 artifacts/queue-v1/adopt-native-qk-v1.json。其他困難：無。

## 驗證方式與結果

三個新／改程式通過 Python AST 檢查。CPU 測試確認正式排程僅包含 train-native_qk、train-scale_bias，沒有 binary_control job；以短暫子程序驗證 pidfd 退出等待與錯誤身分拒絕。以獨立暫存假資料（不是實驗指標）驗證只有兩組檔案也能成功生成比較 JSON／Markdown，無 delta_vs_control 或額外對照欄位。暫存測試資料已由 TemporaryDirectory 自動移除，不是正式結果，不影響訓練。

實際接管核對同一訓練 PID／start_ticks／argv，並以 pidfd 確認訓練當時未退出；新 queue 輸出 JOB_ADOPTED。尚未完成正式精度驗收，不能聲稱恢復精度。

## 未解事項與風險

取消同預算對照後，能比較新模型和現有正式模型的精度，但不能完全區分 scale／bias 修改與額外訓練的獨立收益。兩組的既有超參數、PWL [-10,0]、MASF 固定與驗收門檻不變。正常等待不讀 log、不查 GPU；完成／錯誤事件仍需按實際結果處理。此變更未新增 Git commit／push。

## 16:11 進度查詢補記

依使用者查詢，只讀 queue 尾端、目前訓練 log 尾端與最新兩筆結構化事件，並作一次程序／GPU 快照。native_qk 仍為原 PID 2342759，已執行約 34 分鐘；epoch 0（對使用者為第 1 回合）最新 macro_in_epoch 為 421，尚未出現首輪完整驗證與選定 checkpoint。GPU 利用率當下 100%，使用顯存 23087 MiB。最新兩筆 loss 為有限值，AMP overflow_retries 均為 0，Detect head、Pose head 與共享部分均有梯度。這些只證明目前持續更新，不代表最終精度提升。scale_bias 尚待前組完成後自動接續；BinaryQK 額外正式對照保持取消。此次查詢無發現需要修復的問題，沒有改動訓練配置或程序。
