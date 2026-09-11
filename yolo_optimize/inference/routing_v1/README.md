# qSiLU 同權重 one2many＋NMS 推論比較

2026-09-11 完成；不訓練、不修改權重。原 qSiLU E2 的 one2one 完整結果重用既有基準，新增 one2many 路徑的 Float／BitTrue 完整 BBAT5 683 張驗證。 COCO head 不改路徑，本次未重跑 COCO，也不把舊 COCO 數字冒充新測。

來源：`activation/bridge_v1/artifacts/runs/qsilu_pq-short-e10-seed1-v1/inference/best_joint.pt`（相對 optimize），SHA256 `1bbbbe0f87cb853827af1bb625e7066178e3fb2c7e14c410d997efb24e94834a`，前後 hash 不變。

## BitTrue 實測

| AP50–95 | 原 one2one | one2many＋NMS | 差值 |
| --- | ---: | ---: | ---: |
| BBAT 框 | 0.618008 | 0.618969 | +0.000962 |
| BBAT Pose | 0.891329 | 0.899451 | +0.008122 |
| ball 框 | 0.505192 | 0.494348 | -0.010844 |
| ball Pose | 0.859649 | 0.843451 | -0.016199 |
| bat 框 | 0.730823 | 0.743590 | +0.012768 |
| bat Pose | 0.923008 | 0.955451 | +0.032443 |

bat 明顯改善，但 ball 退化超過保護線，**不全面採用 one2many**。這支持推論候選路徑是影響 bat 效果的一個因素，不能證明所有框差距都由 NMS 造成：此處同時換到不同已訓練預測分支並使用 NMS，不是只加 NMS 的單變因。

下一個候選是固定 class routing：ball 用 one2one 、 bat 用 one2many，但尚未實作／驗證。不可以直接把本表每類最好 AP 拼接成新模型成績；候選篩選、 top-k 、分數與後處理仍須完整重驗。若同時算兩條 head，會增加 head 運算和記憶體，不增加 backbone 也不代表零成本；須與硬體預算一起判斷。預設仍保留 one2one 。

## 執行與驗證

`evaluate.py` 使用同一 SelectedSource 重建 qSiLU／PWL[-10,0]20 段／BinaryQK，在 fuse 前設 end2end=False；不從已移除 one2many 的模型硬切換。 CPU160×160 檢查輸出(1,12,525)、 raw 分支、有限值與 NMS 後 12 欄通過，Float／BitTrue 均完成。

正式參數：imgsz640 、 physical batch16 、 workers4 、 FP32 、 rect=True 、 conf0.001 、 iou0.7 、 class-aware NMS 、 max_det300 。未 sweep 閾值、未 TTA 、未改資料。 canonical registry `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`，Pose YAML `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`，固定 val683；runtime 只重現 assignment／labels 。

完整結果：[summary.json](<artifacts/one2many-v1/summary.json>)。每後端保存 metrics 與 validator 速度觀察，但沒有成對隔離 warmup 、 backend 、 I/O 的正式 benchmark，不用它宣稱部署加速。 GPU 工作由`combine/monitor.py`的 child.wait(timeout=600)管理，提前完成即返回 JOB_DONE；整個工作約 21 秒，不是訓練 epoch 。

沒有新增 checkpoint；部署候選是同權重加明確推論設定。程式、設定、 raw metrics 与事件保存在補充封存包；Git 只考慮小型程式／文件，未執行發布。困難：無。尚未驗證 class routing 、多 seed／獨立 test 、真實影片、 ONNX／板端。
