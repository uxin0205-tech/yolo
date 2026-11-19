"""最終固定尺寸入口；發行時與 Stage8 已鎖定 runtime 一起複製。"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def load(device="cpu"):
    from load_model import load_model
    manifest = json.loads((HERE / "manifest.json").read_text())
    path = HERE / "shape.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"]["shape.json"]["sha256"]:
        raise ValueError("shape 契約 SHA 不符")
    contract = json.loads(path.read_text())
    model = load_model(device=device)
    selected = contract["selected"]
    entry = manifest["models"][manifest["default_model"]]
    if entry["sha256"] != selected["sha256"] or manifest["shape"] != contract["shape"]:
        raise ValueError("shape 與選定權重血緣不符")
    model.shape_contract = contract
    return model


def prepare(bgr, shape):
    """OpenCV BGR 影像 → RGB BCHW FP32 與可逆座標資料。"""
    from ultralytics.data.augment import LetterBox
    if shape is None:
        raise ValueError("原rect640依batch決定形狀，請使用原驗證資料層；本介面只適用固定canvas")
    h, w = shape
    h0, w0 = bgr.shape[:2]
    gain = min(h / h0, w / w0)
    resized = (round(h0 * gain), round(w0 * gain))
    padding = ((w - resized[1]) // 2, (h - resized[0]) // 2)
    image = LetterBox((h, w), auto=False, scaleup=True, stride=32)(image=bgr)
    tensor = torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
    return tensor, {"ori_shape": (h0, w0), "ratio_pad": ((gain, gain), padding)}


def records(value, name="output"):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("非有限可攜輸出")
        return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                       "sha256": hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()}}
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else ()
    return {k: v for key, child in children for k, v in records(child, f"{name}.{key}").items()}


def validate():
    from load_model import verify_contract
    torch.set_num_threads(2)
    reference = json.loads((HERE / "validation/reference.json").read_text())
    model = load()
    for entry in reference["outputs"]:
        torch.manual_seed(entry["seed"])
        with torch.inference_mode():
            actual = records(model(torch.rand(1, 3, *entry["shape"]), task="both"))
        if actual != entry["records"]:
            raise ValueError("可攜模型與來源輸出不等價")
    model.graph.model[17].act = torch.nn.SiLU()
    try:
        verify_contract(model, model.stage8_manifest)
    except ValueError:
        pass
    else:
        raise ValueError("錯誤 SiLU 未被拒絕")
    if torch.cuda.is_initialized():
        raise ValueError("CPU 交付驗證意外使用 GPU")
    result = {"status": "passed", "bitwise_equal": True, "silu_rejected": True, "GPU_used": False}
    (HERE / "validation/result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS Stage9 portable shape／qSiLU／D1")


if __name__ == "__main__":
    validate()
