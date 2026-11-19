# 第四階段：Detect／Pose task-specific P3 MASF

2026-10-21：使用者明確要求保留本階段結果、四份候選與必要 resume，待融合後再決定是否重試。主線保留原 parent，由 [yolo_hog](../yolo_hog/README.md) 接續第五階段。以下為第四階段完成紀錄。

**最新狀態（2026-10-20）：第四階段首輪與延長研究均已完成，09:50 停止，GPU 已釋放。** Detect L1／L2 最佳 AP 為 0.514675／0.514728，但球棒退步且 MASF 開關未顯示整體收益；Pose 兩組 keypoint AP 未超過 parent。兩任務建議保留原 parent，候選與必要 resume 已交付。[完整結果](artifacts/stage4/extension-20261019-v1/RESULTS.md)、[完成紀錄](../docs/worklogs/2026-10-20-masf-stage4-results.md)。L1／L2 各從同任務 E5 完整狀態出發，實際追加 Detect 8／15 輪、Pose 6／5 輪。第五階段待指定。以下為首輪設定與證據。

目前接續整合計畫。M1 使用使用者選定的第三階段 KD 第五輪 Detect EMA；M2 使用 `yolo_pose_training/artifacts/preserved/stagewise_lr_p3_640/weights/best.pt`。兩者分別建立 head P3 MASF，不做 shared MASF，不重跑原始 Pose，也沒有 unchanged control 加訓。

先依 optimize 核對方法、來源與配方，證據見 [事前參照](artifacts/stage4/reference-review.json)。[CPU 診斷](artifacts/stage4/cpu-diagnostics.json) 已驗證 alpha 零值完全等價、raw P3／P4／P5 不變、one2one 梯度能到達 MASF、beta 不縮放 head 梯度、qSiLU 重載、非平凡 BN folding。正式 GPU 結果以 run 內實際狀態為準。

```text
raw P3 ─────────────→ 後續 P4、P5
   └→ DW3(qSiLU) + DW5(qSiLU) → 1×1(qSiLU) → ×alpha + raw P3 → task head P3
```

新 context 沿用 optimize 的 `P3MASFFull35`，初始化 alpha 0。兩個獨立 parent 都沒有 MASF，不套用其他模型已訓 donor。訓練時 one2one 在 MASF 前 detach，beta 1 僅影響反向；所有 BN running statistics 固定，只訓各自 head 與 MASF。五輪、seed 1、AdamW，head／context／alpha LR 分別為 `5e-6／1e-5／1e-4`；詳見事前參照與 run 的 recipe。

Detect 保留 B2／S1 Q10／完整 integer PWL exp＋D1 LUT reciprocal。Pose 原生浮點 Attention 仍是其已訓契約，並未宣稱已變成 Detect 整數路徑；融合階段必須接續選定 Detect 契約。新增 context 採 canonical `qsilu_pq`，原模型其餘 activation 待第七階段統一建立 qSiLU parent。

每個 MASF 新增 75,777 個參數，640 × 640 時增加 475,136,000 個 Conv MAC，另有 alpha 乘法、加法及 qSiLU。BN 可線性折疊；此處仍是浮點 Conv／alpha 與 qSiLU reference，沒有硬體實測加速結論。

## 操作

```bash
bash yolo_masf/run_stage4.sh diagnostics
bash yolo_masf/run_stage4.sh init --output /home/uxin/yolo/yolo_masf/artifacts/stage4/formal-20261019-v1
bash yolo_masf/run_stage4.sh status --output /home/uxin/yolo/yolo_masf/artifacts/stage4/formal-20261019-v1
bash yolo_masf/run_stage4.sh run --output /home/uxin/yolo/yolo_masf/artifacts/stage4/formal-20261019-v1 --execute
```

未提供 `--execute` 只預覽。沿用 Attention QueueExecutor；背景 supervisor 每 600 秒檢查 GPU、進度與異常，完成後整理結果並停止。`STOP` 檔停止本 run；已完成 epoch 的 resume 保留。smoke 不作正式精度。選型同時列 parent、alpha-on 與 alpha-off；只在 primary AP 提升、其他監測 AP 未下降逾 .001，且優於同權重 alpha-off 時建議採用 MASF。否則交回 parent，候選與最後 resume 仍保留供覆核。無額外 control，不能把全部改善歸因 MASF。

完整 AP、候選與推薦交接寫入 `result/stage4/{detect,pose}/`，第五階段待使用者指定。`plot` 可只讀小型 JSON 重畫，不需要 GPU 或 checkpoint。

## 清理與保護

[清理清單](artifacts/stage4/cleanup.json) 記錄被取代的舊 YOLO11 matrix 權重、圖檔、predictions 與舊程式的 SHA，已刪除 479 個檔案，釋放約 2.858 GiB。必要歷史數據已抽取至 [evidence](clean_bbt5_study/results/extracted-evidence.json)，optimize 使用的兩個 CSV 保持原路徑與原內容。沒有保存完整 legacy／archive／backup。

`bbt5-detect-baseline/` 依既有明確要求保留；locked dataset lineage、field_check 唯一論文／分析證據與 [歷史研究報告](clean_bbt5_study/results/REPORT.md) 也保留。歷史 raw split 不再作新訓練入口；新 Pose 僅使用 BBAT5 v1 的 5,964／683 固定 assignment 與 labels，runtime View 只隔離 cache。

中文紀錄見 [根層工作紀錄](../docs/worklogs/2026-10-19-masf-stage4.md)。


2026-10-19 18:16 交接：[正式 run](artifacts/stage4/formal-20261019-v1/README.md) 已通過兩個 parent 重載與兩個 smoke，M1 正式五輪進行中。預估台灣時間 21:00～21:30 完成整個第四階段，實際以 supervisor／queue 狀態為準；不啟動第五階段。


第四階段首輪五輪流程已完成：[結果](artifacts/stage4/formal-20261019-v1/RESULTS.md)。延長研究也已完成，完整交付見頁首；第五階段待指定。
