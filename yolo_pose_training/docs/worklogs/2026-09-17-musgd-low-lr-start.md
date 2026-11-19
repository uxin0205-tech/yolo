# 2026-09-17 啟動 MuSGD 低 LR 候選

## 變更與原因

使用者批准啟動前一輪準備的候選。從 runs/stage3_adamw_960/weights/best.pt 初始化，lr0=0.00038；imgsz960、batch16、epochs100、patience30、MuSGD momentum0.9、lrf0.05、weight_decay0.0005 等參數皆與原 Stage 4 一致。沒有續訓 Stage 4，也沒有修改資料集 split、樣本或 labels。

新增 scripts/run_candidate.py，沿用既有每600秒GPU監測與每5秒程序檢查，獨立保存狀態至 artifacts/candidates/stage4_musgd_lr00038/status.json。共用 queue.lock 防止本專案兩個佇列同時啟動。背景 tmux 為 bbat5_pose_lr00038_20260917。

訓練完成自動以960／batch8評估候選 best.pt，對照快照中的既有 Stage 4 統一評估結果，保存 comparison.json、report.md。原 artifacts/final 不替換。候選 weights 包含 best.pt、last.pt、last-resumable.pt，由既有訓練腳本保存。

## 驗證方式與結果

新腳本語法檢查通過；候選經官方 get_cfg 解析通過，程式斷言與原 Stage 4 僅 name、lr0 不同；確認 Stage 3 初始化 checkpoint 存在、候選 run 原先不存在。啟動前 GPU 閒置，約489 MiB。啟動實測追加於後。

## 困難與解法

無。使用獨立狀態與比較檔案，避免覆寫已完成四階段的報告與權重。

## 未解事項與風險

收斂、精度與完成後評估尚待驗證；低LR不保證改善。只有seed0，formal val也用於選模，沒有獨立test。背景監測記錄GPU並在失敗時停止，不代表代理交付後仍持續在線。

## 啟動實測

00:08:36 背景啟動。實際 args.yaml 確認 lr0=0.00038、batch16、960、resume=False，模型來源為 Stage 3 best.pt。第1輪已完成超過90個batch，日誌 loss 為有限值；GPU 使用率92%，nvidia-smi顯存22578 MiB，訓練日誌約20.7 GB，目前無OOM。完整epoch與最終品質仍待完成。

## 2026-09-17T01:07:17+08:00 狀態查詢

候選已完成36輪，正在第37/100輪，程序存活；GPU使用率75%、22596 MiB、58°C。最佳綜合結果為第32輪：Box mAP50=0.89669、mAP50–95=0.61791；Pose mAP50=0.90160、mAP50–95=0.89986。最後完成第36輪的Box／Pose mAP50–95為0.60696／0.89211。目前仍低於原Stage4訓練中最佳的0.62416／0.90494，最終統一評估尚未執行。best、last、last-resumable均存在。此次僅查詢並記錄，沒有更改訓練；困難：無。

## 完成狀態查詢

2026-09-17 01:46:57（台北時間）訓練與統一評估完成。實跑62輪，最佳第32輪，30輪未改善正常提早停止；GPU使用率0%。統一960／batch8驗證，低LR候選Box／Pose mAP50–95=0.616998／0.899951，原Stage4=0.624332／0.905072，分別下降0.7334／0.5121個百分點，平均下降0.6227個百分點。候選Precision／Recall略高但AP較低，不視為整體提升。best、last、last-resumable均存在；既有最終權重未替換。此次查詢困難：無；限制仍為單seed、formal val選模，未新增GPU工作。
