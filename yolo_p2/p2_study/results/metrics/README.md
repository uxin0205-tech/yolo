# 正式指標與訓練歷史

- `A0/A1/A2/coco_metrics.json`：完整 COCO val2017 的 COCO API 與 Ultralytics 指標。
- `A0/A1/A2/benchmark.json`：RTX 5090、imgsz 640、batch 1 的 FP16/FP32 benchmark。
- `A1/args.yaml`、`A1/results.csv`：A1 100 epochs 實際參數與歷史。
- `A2/stage1/`：P2-only 20 epochs 實際參數與歷史。
- `A2/stage2/`：全模型 73/80 epochs 實際參數與歷史。

所有 retained args 都保存 `seed: 0` 與 `deterministic: true`。

- COCO API mAP50-95：官方 annotation JSON 配合 faster-coco-eval。
- Ultralytics mAP50-95：Ultralytics validator 對同一 val2017 預測計算。
- benchmark：500 張固定影像、50 次 warmup、3 次重複；每種精度每個模型 1,500 個計時樣本。
