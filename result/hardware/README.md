# 硬體選擇 · 固定 512×768

使用第 9 階段 `rect512_768/adapt10` 第 2 輪的 folded 權重。高 512、寬 768，輸入為 RGB BCHW、FP32、0～1。Detect 與 Pose 共用 qSiLU Rep17＋20、B2、S1 與 D1。權重 SHA-256 為 `a7526f060612fdada757679b6b32d408b760d4c53f99e6adcb9923f16636b457`。[形狀契約](shape.json) · [檔案與算子契約](manifest.json)

在具備本機模型檔的工作區執行以下 CPU 驗證。載入器使用包內凍結的 runtime。

```bash
.venv/bin/python -B result/hardware/load_shape.py
```

從 `result/hardware/` 開啟獨立 Python 程序，將 OpenCV 讀入的 BGR 影像轉成固定尺寸，再執行兩個任務。

```python
import cv2
import torch
from load_shape import load, prepare

model = load(device="cpu")
image = cv2.imread("frame.jpg")
tensor, meta = prepare(image, model.shape_contract["shape"])
with torch.inference_mode():
    outputs = model(tensor, task="both")
```

`meta["ratio_pad"]` 記錄縮放與補邊資訊，供框和關鍵點換回原圖座標。交接至其他機器時，需取得 `rep17_20.pt`，並攜帶同包的架構、runtime、manifest 與 sidecar。完整 AP 與精度版本見[統一交接](../README.md)。
