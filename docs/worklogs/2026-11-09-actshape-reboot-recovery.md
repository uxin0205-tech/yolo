# Stage9 電腦重啟後從中斷處恢復

日期：2026-11-09。使用者要求確認進度，並從電腦出問題的位置接續。工作範圍為原 Stage9 queue 的復原；不更換訓練配方、parent、正式資料或部署算子。

## 已完成與中斷位置

384×768 的 10 輪適應、10 輪降 LR 與最佳模型重載驗證均完成。重載 AP 最大差異為 0，qSiLU／D1 契約通過；綜合最佳仍為第一段 E4。此組 20 輪與所有已完成 E0／校準／smoke 保留，不重跑。

電腦於 12:20:20 開機；查核時原兩個 supervisor 及 GPU worker 均不存在，兩把 flock 都已釋放，GPU 沒有 compute 程序。原狀態最後更新於 00:55:37，仍寫 running，但只是過期狀態。原 512×1024/adapt10 最後訓練記錄是 E1 step340/463，沒有 resume.pt、history.json 或完成標記，因此無法從第 341 步還原 optimizer／EMA／scheduler；須從這個未完成 epoch 的開頭重算。

前次開機的 kernel journal 沒有可讀條目，沒有證據將問題判定為 OOM、GPU 錯誤或供電故障。確認的事實是系統重啟與原程序中斷。

## 啟動前核對

依規則先唯讀重查 optimize 的 `experiments/pose_only_rep_v1/run_arm.py`、config，以及 activation bridge 實際 qsilu_pq-short-e10 配置、訓練程式和已有 reference-review。optimize 的恢復會載入完整 epoch snapshot 與訓練狀態；本輪沿用相同原則，未完成 epoch 不假裝可由 progress.json 還原。

原來源鎖 436 項、擴充來源鎖 450 項均通過；14 個已完成 job 的 job SHA／source SHA、兩份已完成 resume SHA／config、共同 parent 與已選 best 權重 SHA 均一致。共同 parent 仍為 `2454548104ed64768eb9ecdb99d44804ee9d8e1e371e2a508cf052c0911fdd35`。完整核對與中斷前的小型狀態見 `yolo_actshape/artifacts/recovery-20261109.json`。

沿用 neck／雙 head、固定 backbone／Attention／BN running、AdamW、warmup／cosine、原 pose weight 與原 10E＋半 LR 10E 配方；部署保留 qSiLU／D1／B2／S1／folded17+20。剩餘 512×1024 20E 加擴充 60E，整階段上限仍為 100 個完整 epoch。使用 GPU 0；該組原 smoke 峰值 reserved 約 5.86 GiB。COCO／BBAT5 v1 membership、影像與標註不變。

## 恢復時發現的監測問題與處理

12:30 第一次重啟時，原 supervisor 立即報「兩小時無進度」。查明原 runner 在啟動 worker 後仍讀取昨晚 progress.json 的 mtime，將電腦停止期間誤判為新 worker 停滯。這不是本次 GPU 訓練已跑兩小時沒有更新。

在確認所有本次程序停止後，只移除中斷 job 的過期即時 `progress.json` 一份；其原內容已留在 recovery JSON，原 progress.jsonl 亦完整保留。未刪除 checkpoint、正式 history、done 或其他實驗結果，未更動來源鎖中的程式。新 worker 尚無進度時，原 runner 會以此次啟動時間計算逾時，第一筆新進度出現後再監測實際心跳。

12:32:15 再啟動初輪 supervisor PID 8381，worker PID 8387。已確認第一個實際 GPU macro 完成、EMA 更新為 1、共享／雙 head 均有梯度且 AMP overflow retries 為 0，才於後續啟動擴充 supervisor PID 8592。擴充繼續等待初輪完成，避免只見持鎖卻讀到過期 failed 狀態的競爭。

## 驗證與剩餘事項

復原前以實際 PID／cmdline 檢查得到 FAIL：running 狀態對應程序不存在。復原後已看到新 worker 與新時間戳的 GPU macro；會再確認持續前進、兩個 supervisor 的監測與已完成檔案未改，再交回背景執行。本次是程序／狀態復原，沒有修改訓練程式，無需重跑整套模型測試或已完成 AP。

每 600 秒監測、每 30 秒檢查資源的規則維持。由於中斷及未保存的 E1 需重算，原完成時間順延；以復原後速度更新 ETA，不沿用停機前的日期。新增尺寸尚未開始 GPU 工作，整個 Stage9 完成後停止。

其他困難：無。尚未查明電腦最初故障的系統原因；本次未調整驅動、系統服務或新增開機自動執行。

## 復原驗收通過

2026-11-09T12:34:18.934976+08:00：實際 GPU 訓練 E1 step20/463 已持續前進；初輪 supervisor 8381、worker 8387、擴充 supervisor 8592 均存活，原與擴充 queue flock 均有持有者。GPU compute 只有 worker 8387，約 6588 MiB；共享／雙 head 梯度正常。復原前 FAIL 的程序存活檢查已轉為 PASS，兩次資源檢查後仍正常。

再次核對所有啟動前記錄的正式檔案 SHA，14 個 done、兩份已完成 resume、parent、best、job/config 及來源鎖皆未改變。只重算未持久化的 512×1024 E1；原 starts 計數記錄本次監測失敗與成功重啟，不抹去失敗事實。

重啟後實測約 5.061 秒／macro。整體中央估計為 2026-11-10T22:27:14.441069+08:00，對外粗估 11 月 10 日深夜至 11 月 11 日上午；新增三尺寸仍使用成本比例估算，尚未實測。背景每 600 秒監測繼續，退出 working。
