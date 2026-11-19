# Ultralytics 原始碼與 P2 修改

此目錄是 Ultralytics 8.4.90 套件原始碼。本研究保留完整上游結構，只加入 YOLO11m P2 所需的最小修改：

- `cfg/models/11/yolo11-p2.yaml`：P2/P3/P4/P5 四尺度模型。
- `nn/tasks.py`：解析 P2 模型所需設定。
- `nn/modules/head.py`：支援舊 Detect towers 對應到四尺度 head。

```bash
python -m pip install --no-deps -e .
```

上游使用方式與授權見根目錄 `LICENSE`、`docs/` 及 https://docs.ultralytics.com/。
