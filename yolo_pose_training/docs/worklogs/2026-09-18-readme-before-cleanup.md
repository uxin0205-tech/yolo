# 清理前 README 歷史快照

以下是歷史狀態，權重與輸出路徑多數已清除；目前入口請見根層 README。

# BBAT5 v1 YOLO26m Pose 分階段訓練

從官方 `yolo26m-pose.pt` 開始，訓練 ball／bat 與 2 個 keypoint。四階段與統一评估已於 2026-09-16 20:24:28（台北時間）完成。Stage 4 在第 45 輪提早停止，最終 best.pt 選中其第 15 輪最佳權重。完整結果見[指標報告](docs/worklogs/2026-09-16-completed-metrics.md)。

| 階段 | Epochs | 解析度／batch | Optimizer／LR | Freeze | Mosaic |
| --- | --- | --- | --- | --- | --- |
| Stage 1 | 20 | 640／32 | AdamW／0.001 | 前 10 層 | 0.5，最後 5 輪關閉 |
| Stage 2 | 50 | 640／32 | AdamW／0.0003 | 全部解凍 | 0.5，最後 10 輪關閉 |
| Stage 3 | 40 | 960／16 | AdamW／0.0001 | 全部解凍 | 關閉 |
| Stage 4 實驗 | 最多 100，patience 30 | 960／16 | MuSGD／0.001 | 全部解凍 | 關閉 |

每階段從前一階段 `best.pt` 接續，重新建立 optimizer 與 scheduler；Stage 2 的 `data` 仍是同一資料 YAML，不是上一階段權重。前三階段遵循使用者提供參數，另明確設 `warmup_bias_lr=0.0`，避免 AdamW 使用預設 0.1 bias 暖身 LR。Stage 4 是候選接續實驗，不保證改善；不是平行搜尋，也尚未得到最佳參數結論。YOLO26 官方 head 的 `reg_max=1`，保留 `dfl=1.5` 不代表啟用 DFL。

## 資料集規範

唯一正式來源為 `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`，registry 為 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。沿用 formal train 5,964 張、formal val 683 張；沒有 test split，不切分、不抽樣、不改動標註。

`configs/data.runtime.yaml` 僅將 path 指向 `artifacts/datasets/bbat5-v1/pose/` 的逐檔符號連結，讓 cache 留在本專案。`artifacts/dataset-audit.json` 記錄數量、標註 SHA256 與 canonical 來源。重建：`.venv/bin/python scripts/prepare.py`（也重建初始訓練 YAML，執行期間不要重建）。完整規範見[共用資料規範](../docs/agents/bbat5-datasets.md)。

## 設定與產物

- `configs/stage*.yaml`：各階段完整設定；`configs/queue.yaml`：执行順序。
- `weights/yolo26m-pose.pt`：官方初始化；來源與 SHA256 在 `artifacts/environment.json`。
- `runs/<stage>/args.yaml`：實際訓練參數；`results.csv`：逐輪 loss 與 metrics。
- `runs/<stage>/weights/best.pt`、`last.pt`：最佳／最後權重。
- `runs/<stage>/weights/last-resumable.pt`：保留 optimizer 狀態，避免官方結束清理導致不能續訓；每 10 輪另存週期 checkpoint。
- 全部完成後產生 `artifacts/comparison.json`、`artifacts/final/report.md` 與 `artifacts/final/{best,last,last-resumable}.pt`。`best` 取統一 960 formal val 下 Box／Pose mAP50–95 平均最高者，`last` 是最後 MuSGD 階段最後權重。各類別分數也保存在 comparison.json。

上述 checkpoint 與最終報告已產生並驗證。formal val 同時用於選模，不能宣稱是独立 test 成績。

## 操作與監測

獨立 tmux session：`bbat5_pose_20260916_stage3_b16`。

```bash
cat artifacts/queue-status.json
tail -f logs/queue.log
tail -f logs/stage1_adamw_640.log
tmux attach -t bbat5_pose_20260916_stage3_b16
```

佇列每 600 秒記錄 GPU 使用率、記憶體、溫度與日誌更新時間到 `logs/monitor.jsonl`；每 5 秒檢查子程序退出，正常完成自動接下一階段，全部完成即比較。若訓練報錯，立即標記 failed 並停止後續階段，保留日誌與權重；日誌超過 30 分鐘沒有更新會記錄警告。此背景監測不是代理持續在線，也不會發送外部通知。各階段固定使用 YAML 的 batch：Stage 1～2 為 32，Stage 3～4 為 16；若 OOM 則保留 checkpoint 並停止，不自動修改 batch。實際 batch 記錄於各 run 的 batch-policy.json 與 args.yaml。

環境使用本專案官方 `ultralytics==8.4.90`，唯讀共用父環境的 CUDA 依賴（PyTorch `2.11.0+cu128`），不載入其他子專案修改版。主機重啟不會自動恢復 tmux；有未完成 run 且具有 last-resumable.pt 時，佇列會接續；缺少完整 checkpoint 則停止避免覆寫。

官方依據：[YOLO26](https://docs.ultralytics.com/models/yolo26/)、[訓練與 MuSGD 參數](https://docs.ultralytics.com/modes/train/)。

中文[工作紀錄索引](docs/worklogs/README.md)。

2026-09-16 最新使用者修正：從官方權重重新開始，不接續舊訓練。Stage 1～3 固定 batch 32、Stage 4 固定 batch 16。此前 runs 與 logs 封存於 `artifacts/archive/20260916-170408/`。正式新訓練仍輸出至 `runs/`，獨立比較評估 batch 保持 8。

2026-09-16 後續修正：Stage 1／2 已完成。Stage 3 因 960／batch 32 OOM，經使用者同意改為 batch 16，從 Stage 2 best.pt 接續；不重跑前兩階段。Stage 2 綜合最佳為第 24 輪，後段 Pose 驗證略退步，因此目前不延長 epoch 或調高 LR。Stage 3 維持 40 輪、LR 1e-4，Stage 4 維持最多 100 輪、LR 1e-3、patience 30。使用者已授權後續依收斂證據調整 epoch 與 LR；目前沒有自動改 LR 或延長 epoch 的機制。

改善方案：[參考 yolo_combine 的 Pose 方法比較](docs/research/2026-09-16-combine-pose-training-review.md)。低 LR 候選設定在 `configs/candidates/stage4_musgd_lr00038.yaml`，已於 2026-09-17 另行啟動。

低 LR 候選背景 session：`bbat5_pose_lr00038_20260917`；目前狀態：`artifacts/candidates/stage4_musgd_lr00038/status.json`，訓練日誌：`logs/candidate_stage4_musgd_lr00038.log`。原四階段 queue-status.json 仍表示該實驗已完成。候選完成後比較報告將存於 `artifacts/candidates/stage4_musgd_lr00038/report.md`，不自動覆蓋既有最終權重。

低 LR 候選已於 2026-09-17 01:46:57 完成，62 輪提早停止，最佳第 32 輪。統一評估 Box／Pose mAP50–95 為 61.70%／90.00%，未超越原 Stage 4，保留原最終權重。詳見 `artifacts/candidates/stage4_musgd_lr00038/report.md`。

2026-09-17 已啟動 **combine 式官方 Pose 新實驗**：P1 head-only → P2 neck/head → P3 full，640、batch32、nbs128、MuSGD LR0.00038／lrf0.5、無Mosaic，最多40／60／200輪（patience20／25／40）。詳見[工作紀錄](docs/worklogs/2026-09-17-combine-recipe-start.md)。設定在 `configs/combine_recipe/`；狀態 `artifacts/combine_recipe/status.json`；tmux `bbat5_combine_long_20260917`；完成後比較報告 `artifacts/combine_recipe/report.md`。

依最新要求增加epoch，P1從既有完整checkpoint接續；排程調整與驗證見[延長紀錄](docs/worklogs/2026-09-17-combine-extend-epochs.md)。

combine 式實驗已於 2026-09-17 14:34:58 完成：P3最佳第37輪、第77輪提早停止。640新P3為最佳，Box／Pose mAP50–95=63.45%／91.70%；960綜合仍由原Stage4最佳。結果見 `artifacts/combine_recipe/report.md`，原final權重未替換。

已啟動新 P3 的 **640／960 低 LR 精修**。兩組共同起點，batch32／16、nbs128、MuSGD LR1e-4，最多40輪、patience15。基線框／keypoint診斷已完成：[診斷報告](artifacts/refinement/baseline-errors/report.md)。新設定在 `configs/refinement/`，背景 `bbat5_refinement_20260917`，狀態 `artifacts/refinement/status.json`，最終比較將輸出 `artifacts/refinement/report.md`。詳見[工作紀錄](docs/worklogs/2026-09-17-error-analysis-refinement.md)。

640／960精修與誤差比較已於2026-09-17 18:28:36完成。640原P3仍最佳；新960精修平均AP提升，但bat Pose下降超過保護門檻，未替換舊模型。完整結果見 `artifacts/refinement/report.md`。

已新增並啟動逐階段降 LR 實驗：P1／P2／P3 lr0=`0.00038／0.00019／0.000095`，各stage cosine至0.5倍，從官方權重重新開始。配置 `configs/stagewise_lr/`，狀態 `artifacts/stagewise_lr/status.json`，tmux `bbat5_stagewise_lr_20260917`。[設計與工作紀錄](docs/worklogs/2026-09-17-stagewise-lr.md)。

逐階段降 LR 實驗於2026-09-17 22:38:23完成全部300輪與評估，P3最佳第163輪。640結果Box／Pose mAP50–95=62.88%／92.06%；Pose較原combine P3提高、Box較低，原combine P3仍是640綜合最高。原權重未替換，詳見 `artifacts/stagewise_lr/report.md`。

依要求保留目前Pose最佳模型：`artifacts/preserved/stagewise_lr_p3_640/`（含best、last、續訓checkpoint、實際配置、結果及SHA256 manifest）。已啟動兩組Box精修對照，僅box loss=7.5／10不同；共同freeze11、640／batch32、MuSGD LR5e-5、最多40輪。狀態 `artifacts/box_recovery/status.json`；背景 `bbat5_box_recovery_20260917`；[工作紀錄](docs/worklogs/2026-09-17-box-recovery.md)。
