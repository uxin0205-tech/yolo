# 第 1 階段離線分析工具

`stage1_cost.py` 計算已完成 B2 的浮點 MAC、Hadamard、尺度與二值成本。分析來源為本輪已驗證的正式權重，以及唯讀 optimize accounting 方法；不修改正式訓練程式，不啟動下一階段。

從 repository 根層重算公式，只需 Python 標準庫與小型 JSON，不需 checkpoint 或 GPU：

```bash
python yolo_binaryqk/scripts/stage1_cost.py
```

重新執行完整 CPU 模型核對，需要本機依賴與 checkpoint：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=yolo_binaryqk .venv/bin/python yolo_binaryqk/scripts/stage1_cost.py --verify-models
```

輸出固定為 `artifacts/stage1/formal-20261018-v2/cost-model-check.json` 與 `operation-cost.json`；前者只在 CPU 核對時更新。數據與腳本可納入研究交付；checkpoint 仍僅本機保存。本工具不量測推論速度、能耗或硬體面積。[成本分析](../artifacts/stage1/formal-20261018-v2/COST.md) 說明口徑與限制。
