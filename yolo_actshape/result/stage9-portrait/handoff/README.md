# Stage9 交接

選定 `rect768_512`，固定 H×W：`[768, 512]`（null 表示保留原 rect640 流程）。

在獨立 Python 程序 `from load_shape import load, prepare`，`model = load()`；固定尺寸使用 `tensor, meta = prepare(bgr_image, model.shape_contract['shape'])`，再 `model(tensor, task='both')`。輸入 RGB BCHW FP32、0–1；bbox/keypoint 依 meta.ratio_pad 反算。原 rect640 請沿用來源 evaluator 的 batch letterbox。

執行 `python -B load_shape.py` 可重驗矩形及選定尺寸的完整輸出、qSiLU/D1 契約。可攜包已獨立 CPU 重載驗證，完整 AP 見上層 README；未執行 ONNX/TensorRT/RTL/HLS/板端量測。
