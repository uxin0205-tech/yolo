# MuSGD 低 LR 候選比較

相同 Stage 3 best.pt 初始化；原 Stage 4 lr0=0.001，候選 lr0=0.00038，其餘訓練參數一致。
固定 bbat5-v1 formal val 683 張、960、batch8。控制組使用既有統一評估結果，候選為本次評估；沒有重跑控制組。

| 指標 | 原 Stage 4 | 低 LR 候選 | 差值 |
| --- | --- | --- | --- |
| metrics/precision(B) | 0.941612 | 0.949119 | +0.007507 |
| metrics/recall(B) | 0.855866 | 0.866627 | +0.010761 |
| metrics/mAP50(B) | 0.902422 | 0.896806 | -0.005617 |
| metrics/mAP50-95(B) | 0.624332 | 0.616998 | -0.007334 |
| metrics/precision(P) | 0.947119 | 0.952282 | +0.005164 |
| metrics/recall(P) | 0.859661 | 0.869410 | +0.009749 |
| metrics/mAP50(P) | 0.905415 | 0.901703 | -0.003712 |
| metrics/mAP50-95(P) | 0.905072 | 0.899951 | -0.005121 |

Box／Pose mAP50–95 平均差值：-0.006227。

候選權重：/home/uxin/yolo/yolo_pose_training/runs/candidate_stage4_musgd_lr00038/weights/best.pt；同目錄保留 last.pt 與 last-resumable.pt。

困難與解法：執行日誌見 logs/；本表不代表跨 seed 穩定性或獨立 test 成績。
未解事項：單 seed、相同 formal val 選模，差異不能視為統計顯著；既有 artifacts/final 權重未替換。
