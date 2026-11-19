# 2026-10-29：H1 進度與 head 停止規則修正

## 任務與狀態

使用者詢問目前狀況。14:41 查核發現 H0 已完成 8 輪，H1 在第 3 輪於 14:01 觸發停止，背景程序已結束、GPU 空閒。尚未進入完整 Pose 分支 P、混合比例比較或聯合訓練。第六階段仍在既有授權範圍內，第七階段未啟動。

| 指標 | 原獨立來源 | H0 E7 | H1 E1 最佳 |
|---|---:|---:|---:|
| COCO AP | 0.514621 | 0.514621 | 0.514621 |
| BBAT box AP | 0.628402 | 0.551269 | 0.548764 |
| BBAT pose AP | 0.920709 | 0.852077 | 0.858949 |

H1 第 3 輪 ball box AP 由起點 0.464057891 降至 0.433584769，下降 0.030473122，觸發本輪實作中的 0.03 規則。沒有數值錯誤、OOM 或固定 Detect 漂移。這些數字是本輪重建的新結果，未恢復先前已刪除的實驗資料。

## 原因與事前 optimize 核對

重新唯讀核對 `j0_runtime.py`、`bridge_v1/pose_first.py`、`full_pose.py`、`full_pose_gentle.py`、`merge_pose.py`、實際 H1 resolved config 與 E36 validation，以及正式 trainer 的 gate 使用方式。

確認代理實作有範圍錯誤：optimize 的 head-only 適應以固定 COCO 精確不變、40 輪上限及 patience 10 停滯規則控制；各項 BBAT 較起點下降超過 0.03 的停止規則來自 `full_pose.py`，只適用會更新 shared trunk 的完整 Pose 副本。本輪 `policy.safety()` 誤套到 H0／H1／M0，造成過早中止。optimize 的 AccuracyGate 記錄最終選型狀態，不等同這個 0.03 即刻停止條件。

實作前保存 [修正來源、SHA、實際觸發值與續訓核對](../../yolo_combine/artifacts/stage6/head-safety-review.json)。本次修正是對齊原方法的適用範圍，不以放寬最終門檻掩蓋退化。

## 變更

- head scope 仍要求四項 COCO 完全不變，使用既有預算與 plateau；不套用完整 Pose 的 BBAT 下降規則。
- P 保留 0.03 安全線，joint 保留既有 COCO／BBAT 安全線；最終 COCO 0.005／BBAT 0.02 gate 不變。
- 新 checkpoint lineage 加入執行來源鎖的路徑與 SHA，讓這次方法修正可追溯。
- 只修復 H1 resume 的錯誤停止標記；歷史 E3 的原停止紀錄保留。原結果已抽取至必要的小型修正紀錄，再移除失效的終止入口，沒有建立另一套權重或 archive。

## 驗證

以實際下降幅度建立最小 regression test：修正前三個 head cases 都失敗，完整 Pose／joint case 通過；修正後全部 13 項 Stage6 CPU 測試通過（8.47 秒），Ruff 通過。

逐項遞迴比對續訓檔，除 `reason` 由錯誤停止改為空值外，live、EMA、optimizer、scaler、scheduler、criterion、RNG、best、history、parent、config 完全相同。續跑 epoch 4，EMA updates 1119、scheduler step 1119；沒有重置 EMA、優化器、LR 或 loss horizon，也沒有重跑前三輪。

來源鎖只允許兩個修正程式及新增 review JSON 改變，其餘來源完全一致。H0 已完成的來源鎖不改寫；H1 的修正前後 SHA 保存在 review 中。正式資料、原始 Pose、Stage5 Detect 與 optimize 全部維持原樣。

## 後續與限制

14:49:30 第一次重啟因監測器誤讀舊 progress 時間而立即停止，尚未進入新訓練。修正後於 14:54:18 再啟動同一 queue，supervisor PID 2580256、worker PID 2580265；已實際確認從 H1 第 4 輪續跑，原上限仍為 40 輪／patience 10，每 600 秒監測。第六階段完成後停止。

困難與解法：head／full Pose 的停止規則混用，已用對應 optimize 原始程式確認並加上分範圍 regression test。修正停止邏輯不代表精度已恢復；仍須看完整適應與融合結果，且目前只有單 seed。

## 續跑監測修正與實際進度

第二個錯誤來自本輪 supervisor：重啟時以舊 progress.json 的時間判斷停滯，在 worker 尚未初始化前便誤殺。先唯讀核對 optimize `pose_only_masf_v1/run_queue.py` 每次 worker 建立新 log，再取 heartbeat／log 最新時間的做法；本輪修正為 `max(job_started_at, progress.time_unix)`，保留真正 1800 秒停滯判斷。

新增四個監測 regression cases，修正前舊 heartbeat case 失敗，修正後四項通過；先前 13 項 Stage6 測試亦已通過。另保存 [監測修正稽核](../../yolo_combine/artifacts/stage6/monitor-restart-review.json)，確認這次重啟未改動 resume SHA。沒有以移除停滯監測規避問題。

14:54:38 實際觀察 H1 E4 step100，EMA updates1219 = 原1119 + 100；loss updates3、LR約1.9917e-4，證明沿用原狀態接續。GPU使用約4.5GiB。14:57已完成E5並進入E6，後續仍正常；首段停止規則修正不影響已固定的 Detect AP。H1若跑滿40輪，粗估15:45～15:55結束，停滯則提早進入分析。
