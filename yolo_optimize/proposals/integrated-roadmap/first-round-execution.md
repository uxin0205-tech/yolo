# 方向 1：第一輪執行報告

本報告讀取現有產物，不會啟動訓練。未完成實驗不作精度收益結論。

## BEST 選擇

共同 parent：原始 `J3 best_joint.pt` 的 EMA。原檔不變；兩臂重新建立 optimizer，延續原 E2E loss state。

| 指標（BitTrue internal AP） | BEST_joint | BEST_pose |
|---|---:|---:|
| COCO Box | 0.498022 | 0.498072 |
| COCO Person Box | 0.620381 | 0.620513 |
| BBAT Box | 0.630036 | 0.630445 |
| BBAT Pose | 0.903717 | 0.903339 |
| Ball Box | 0.507437 | 0.508312 |
| Bat Box | 0.752634 | 0.752579 |
| Ball Pose | 0.859909 | 0.859571 |
| Bat Pose | 0.947526 | 0.947107 |
| Joint score | 0.71117474 | 0.71114152 |

分數差距約 0.00003322，只是沿用 joint 選擇規則，不代表顯著優勢。外部 COCO API AP 不混入這張表。

## Batch 測速

| Arm | Physical batch | 結果 | Images/s | Peak reserved GiB |
|---|---:|---|---:|---:|
| native | 32 | ok | 171.04 | 10.47 |
| hog | 32 | ok | 167.29 | 10.52 |
| native | 64 | ok | 172.33 | 20.57 |
| hog | 64 | ok | 169.88 | 20.57 |
| native | 128 | oom | — | — |
| hog | 128 | oom | — | — |

短測為每 case 2 warmup + 5 timed macros，包含真正反向與 optimizer step；所有測速更新已丟棄。

自動吞吐量推薦 64；工程選用 32：64 短測僅快約 1–2%，顯存接近兩倍；32 延續歷史 J3 physical batch，減少 BN 行為變動。Detect logical batch 仍為 128（32×4），每次 optimizer 更新為 Detect 256 張 + Pose 16 張。

## 本次架構與訓練範圍

下圖省略既有 lateral concat，只標出 P3 與兩個任務的關係。此階段不移動 MASF、不替換 RepConv、不改 BinaryQK。

```text
原生對照：
layer16 內 raw_p3 → 既有 MASF → p3_shared ─┬→ layer17 → P4 → layer20 → P5
                                        └→ [P3, P4, P5] → Detect + Pose

HOG 候選（訓練專用支線）：
layer16 內 raw_p3 ─┬→ 既有 MASF → p3_shared ─┬→ layer17 → P4 → layer20 → P5
                  │                        └→ [P3, P4, P5] → Detect + Pose
                  └→ 1×1 HOG head → HOG loss ← 同一增強後 RGB + GT box mask

部署：移除 HOG head，保留原本 Detect + Pose 推論圖。
```

訓練 Neck 與 Detect/Pose heads；backbone、attention、既有 MASF 凍結。Shared BN running statistics 凍結；Neck BN affine 可訓練，head BN 維持原生 train 行為。

## 訓練設定與公平性

- 原生 5 epochs；HOG 最多 10 epochs、patience 4、min_delta 0.0001。
- AdamW：Neck LR 1e-5、兩個 head LR 2.5e-5、HOG head LR 3e-4；betas=(0.948, 0.999)、weight decay=0.00027、clip=10、AMP。
- 兩臂共用 10-epoch cosine LR trace，warmup=1 epoch，起始 factor=0.1、最終 factor=0.5；比較共同 epoch，不能把多訓 5 epochs 的收益全部歸因 HOG。
- 原 E2E criterion 延續：Detect horizon=120/updates=51，Pose horizon=128/updates=59；不是重新開始 progressive loss。
- HOG μ 尚須 train-only 梯度校準，目標為 raw P3 native gradient 的 5%，不是 μ=0.05；分任務記錄 coverage 與比值。
- HOG E1 關閉、E2 ramp、E3–8 啟用、E9–10 關閉；若提前停止，不宣稱已完成尾段。
- 每 epoch 保存完整快照與去除 HOG 的 inference 權重，並驗證 Float/BitTrue 八項 AP；任一 AP 比 parent 跌超過 0.005，先保存並暫停分析。

## 實際執行狀態

原生對照：`paused_for_analysis`，已完成 1 epochs。

| Epoch | BitTrue joint | 相對 parent 最低 AP 差 |
|---:|---:|---:|
| 1 | 0.70756101 | -0.008656 |

HOG 候選：尚未啟動。

## 原生第 1 epoch：下降診斷

原生對照只完成 1/5 epochs 就觸發安全暫停，HOG 正式訓練為 0/10。run 內產生的 `best_joint.pt` 只是該 run 最佳，不代表勝過原始 BEST，不能升格部署。

以下只用完整 BBAT5 validation 683 張做唯讀混合 state 對照；沒有追加訓練，也沒有把混合權重存成部署候選。數字為相對原 BEST 的 AP 差（非百分比相對變化）。

| 對照 | BBAT Box Δ | BBAT Pose Δ | Ball Box Δ | Ball Pose Δ | Bat Box Δ | Bat Pose Δ |
|---|---:|---:|---:|---:|---:|---:|
| E1 原樣重驗 | -0.006856 | -0.004754 | -0.008656 | -0.005332 | -0.005056 | -0.004176 |
| E1 參數 + 原 head BN 統計 | -0.007980 | -0.004394 | -0.008684 | -0.003606 | -0.007275 | -0.005183 |
| 原參數 + E1 head BN 統計 | -0.002014 | -0.001950 | -0.004785 | -0.003481 | +0.000756 | -0.000420 |
| 原 state + E1 Pose head 參數 | -0.004528 | -0.000790 | -0.007952 | +0.001480 | -0.001104 | -0.003060 |
| 原 state + E1 Neck 參數 | -0.006864 | -0.004923 | -0.005483 | -0.005112 | -0.008244 | -0.004733 |

推論：換回 BN 統計不能救回 Box AP；Neck 與 Pose head 參數各自更新，也能讓部分 AP 下降。這些混合 state 可能打破共同適應，不能將差值相加、宣稱唯一主因，或推斷後續 epochs 必定無法恢復。HOG 尚未正式加入，不能歸因 HOG。Float 與 BitTrue 同向下降，也不支持新 backend 分歧是這次主要原因。

### 已確認的程式問題與修補

| 原本 | 現在 | 驗證與界線 |
|---|---|---|
| AMP overflow 後直接重跑，失敗 forward 已累計 BN/RNG | 每個 macro 保存起點；retry 恢復 BN/RNG，再重試 | 真實 native engine CPU 回歸；沒有改 optimizer/scaler step 順序 |
| EMA 對凍結浮點 state 仍做加權，會有微小舍入漂移 | 固定參數、硬體 buffer、凍結 BN 統計 exact-copy；可訓練 state 照原 EMA 平滑 | 硬體固定值可 bit-exact；未改 EMA age/decay/tau |

HOG 日誌另修正 retry double-count：每次 attempt 開始只重設診斷統計，μ、criterion 與梯度不變。實際 router + engine 的 CPU 回歸先重現 1／4 次 retry 將 Detect 4 images 記成 8／20，再確認修後 images、有效 cells 與 physical batch 清單均與單次成功參考一致；這不是 optimizer 重複更新。

訓練契約 17 項、HOG 12 項、資料唯讀保護 8 項、安全回歸 8 項，共 45 項唯一 CPU 測試已通過。AMP／EMA 主修補整合時重跑相關 24 項通過；最後日誌修補由 8 項安全測試驗證，沒有重複全套或追加 GPU。這些測試不是 AP 改善證據。

原生 GPU 短驗證：`passed`，1 macro、4 次 AMP retry、1 次 optimizer/EMA 更新，480 個固定 state 完全不變。
HOG GPU 短驗證：`passed`，1 macro、4 次 AMP retry、1 次 optimizer/EMA 更新，480 個固定 state 完全不變。

短驗證沒有保存更新權重、沒有增加正式訓練 epochs、沒有重驗修補後 AP。首次 HOG 短驗證工具在 native engine 的 zero_grad 之後讀取 .grad 而失敗；改檢查實際參數更新及 AdamW 非零有限動量後，僅重跑 HOG 並通過，原失敗紀錄保留。

### 待確認的下一個最小對照

EMA age 重置仍是待驗假說：原 parent updates=26597；本次 fresh EMA 在 E1 結束 updates=463。依 d(u)=0.9999×(1−exp(−u/2000))，更新時舊 EMA 保留係數約為 0.999898 與 0.206637，兩者平滑強度差很多。這可解釋為何候選 EMA 更貼近 live，但尚不能證明它造成 AP 下降。

若追加診斷，建議從同一原 BEST 重建修補後的配對起點，只改 EMA 起始 updates；LR、BN scope、optimizer、criterion、資料 trace 都固定，並同時驗證 live/EMA，避免把平滑掩蓋退步誤判成學習改善。嚴格歸因需要修補後兩臂各 1 epoch，不把修補前 E1 當完全等價 control；這個追加成本尚未排程。先不疊加新的 LR、凍結策略或架構。

## 來源與後續

結果根目錄：`/home/uxin/yolo/yolo_optimize/experiments/artifacts/direction1-20260908`。實際超參數見各 run 的 `resolved-config.json`；逐步進度見 `progress.jsonl`；GPU 每 600 秒監測見 `logs/*.monitor.jsonl`。

原始 final、COCO 完整清單與 canonical BBAT5 未改動。後續 MASF、BinaryQK、單層 RepConv 依證據逐一決定，不與本輪 HOG 同時混入；第二輪及 person-only 不執行。
