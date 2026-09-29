# YOLO11m P2 雙版本全流程實驗與自動分析計畫

## 目標與實驗組

所有實作、訓練、日誌與產物限定在 `/home/uxin/yolo/yolo_p2/.worktrees/p2-study`，分支為
`feature/yolo11m-p2-study`。不修改主 checkout、不 push、不建 PR。

| ID | 模型 | Detect 尺度 | 差異 |
| --- | --- | --- | --- |
| A0 | YOLO11m | P3/P4/P5 | 官方 checkpoint 基線，不參與本次 fine-tune |
| A1 | YOLO11m-P2 Direct | P2/P3/P4/P5 | 保留原 neck，新增獨立 P2 concat 分支 |
| A2 | YOLO11m-P2 Direct Staged | P2/P3/P4/P5 | 與 A1 同架構，改用兩階段 fine-tuning |

A1/A2 使用 COCO2017、`imgsz=640`、`seed=0` 與同一固定 batch，各自 fine-tune 100
epochs。A0 使用官方 `yolo11m.pt` 及官方公布 mAP 51.5，另做本機 val/latency，不重新訓練。

A2 的推論架構與 A1 完全相同，不增加 Params/GFLOPs/latency。第一階段 20 epochs 凍結 layer
0–22 及已映射的 P3/P4/P5 Detect towers，僅訓練 layer 25 P2 branch 與 Detect index 0；第二階段由
stage-1 `best.pt` 開始，全部解凍後以明確指定的 MuSGD `lr0=0.001` 訓練 80 epochs。

## 模型變更

- `yolo11-p2.yaml`：保留原模型 layer 0–22，由 layer 16 的 P3 上採樣，與 backbone layer 2 的
  P2 concat，再以 `C3k2` 輸出 128 channels；Detect 順序為 `[P2, P3, P4, P5]`。
- A1/A2 均要有 4 個 Detect 輸出，640 輸入對應 `160², 80², 40², 20²`，stride 固定為
  `[4, 8, 16, 32]`。
- A1/A2 共用同一份初始權重：依名稱與 shape 載入共通 layer 0–22，並將原 Detect `cv2/cv3[0:3]` 顯式映射到新 Detect
  `cv2/cv3[1:4]`。逐 tensor 驗證 shape 與數值一致，寫出 `weight_transfer.json`。

## 可恢復的無人值守流程

主介面：

```bash
python -m p2_study.run --config p2_study/config.yaml
python -m p2_study.run --config p2_study/config.yaml --resume
```

控制器以 lock 防止重複啟動，在 `p2_study/artifacts/state.json` 記錄階段、開始/結束時間、command、
exit code、attempt、checkpoint、metrics 和 ETA。中斷後優先由當次 run 的 `last.pt` 使用
`resume=True` 續跑。

執行順序：

1. Preflight：驗證 editable package、Torch/CUDA/RTX 5090、COCO train/val 與 annotation、80 classes、影像數量、
   `faster-coco-eval>=1.6.7`、`yolo11m.pt` 與至少 80 GB 磁碟。
2. 靜態/單元測試：建模、params/GFLOPs、Detect 尺度、權重映射、FP32/AMP 與 P2 梯度、
   各版一個 batch forward/loss/backward。
3. Batch 校準：用 A1 以 70% VRAM 目標取整數 batch，寫入 manifest，後續所有版本共用。
4. 1% gate：A1 跑 1 epoch，要求 train/val、loss 有限、P2 梯度有限非零、
   並產生 checkpoint。
5. 10-epoch health gate：A1 使用完整資料集跑 10 epochs，僅檢查穩定性。
6. 100-epoch formal：順序為 A1 direct 100、A2 staged 20+80；共用資料、batch、seed、
   `deterministic=True, amp=True, workers=8, cache=False`。
7. 最終 COCO 驗證：各自 `best.pt`、batch 1、`save_json=True`，收集 AP/AP50/AP75/AP S/M/L/AR。
8. 效能：同一 RTX 5090，batch 1，FP16 為主、FP32 為輔；50 warmup、500 張、3 次重複、
   CUDA synchronize，記錄 inference/end-to-end median/p95、params、GFLOPs、模型大小與峰值 VRAM。
9. 分析：產生 `comparison.csv`、`summary.json`、中文 `REPORT.md` 和圖表。

## 關閉 Codex 後持續執行

```bash
# 啟動（優先建立 systemd user service，不可用時自動改用 nohup）
./p2_study/ctl.sh start

# 關閉 terminal/Codex 後仍可執行；日後查看
./p2_study/ctl.sh status
./p2_study/ctl.sh logs

# 停止（checkpoint/state 保留）與恢復
./p2_study/ctl.sh stop
./p2_study/ctl.sh resume
```

service 固定 `WorkingDirectory` 為本 worktree，使用當前 Python 虛擬環境的絕對路徑，stdout/stderr 落盤至
`p2_study/artifacts/logs/controller.log`。若要登出主機帳號後仍執行，需主機管理員啟用 user lingering；只關閉
Codex 或 terminal 不需額外設定。

## 錯誤處理與驗收

- 每階段最多兩次嘗試。OOM 首次將全域 batch 減半；若 formal 已開始，作廢同一 batch 的既有
  formal 結果並全部重跑。
- NaN/Inf 從最近有效 checkpoint 恢復一次，不改 optimizer/LR/augmentation；第二次停止。
- 第二次仍失敗時產生 `FAILURE_REPORT.md`，不產生假勝者。
- 準確度勝者以 AP_small 最高為準；差距小於 0.1 AP 時依總 mAP、較低 FP16 latency 決勝。
- 另列 AP_small、mAP、GFLOPs、latency、VRAM 的 Pareto frontier，並明確報告 S/M/L 任一退化。
