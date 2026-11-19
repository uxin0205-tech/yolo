"""Stage8 可攜推論入口；僅載入可信交付包，核對 qSiLU 與完整 D1。"""

import argparse
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage7_contract import RUNTIME, canonical, digest, read
from stage7_contract import verify_contract as upstream_contract


def verify_contract(model, expected):
    import torch

    from stage8.rep import FoldedRepConv, RepConv, AsymmetricRepConv

    if (
        canonical(model.stage8_manifest) != expected
        or expected["schema"] != "repconv-stage8-v3"
    ):
        raise ValueError("Stage8 manifest 不符")
    if expected["mode"] != "folded" or expected["operators"] != "hardware":
        raise ValueError("部署入口只接受 folded checkpoint")
    result = upstream_contract(model, expected["stage7_manifest"])
    expected_paths = {f"graph.model.{i}" for i in expected["layers"]}
    actual = {
        n
        for n, m in model.named_modules()
        if isinstance(m, (FoldedRepConv, RepConv, AsymmetricRepConv))
    }
    if actual != expected_paths:
        raise ValueError("Rep 位置不符")
    for path in actual:
        block = model.get_submodule(path)
        if type(block) is not FoldedRepConv or block.conv.bias is None:
            raise ValueError("Rep 必須為單 Conv3x3+bias")
        if any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            for m in block.modules()
        ):
            raise ValueError("Rep 仍有 BN")
        desc = expected["layer_contract"][path.split(".")[-1]]
        for key in (
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "groups",
        ):
            value = getattr(block.conv, key)
            if (list(value) if isinstance(value, tuple) else value) != desc[key]:
                raise ValueError(f"Rep geometry 不符：{path} {key}")
    result.update(
        repconv=expected["candidate"], rep_mode="folded", rep_sites=sorted(actual)
    )
    return result


def load_model(device="cpu", candidate=None):
    import torch

    manifest = json.loads(read(HERE / "manifest.json"))
    candidate = candidate or manifest["default_model"]
    if candidate not in manifest["models"]:
        raise ValueError(f"未交付模型：{candidate}")
    entry = manifest["models"][candidate]
    for name in [*manifest["runtime_files"], *entry["files"]]:
        path = HERE / name
        if not path.is_file() or digest(path) != manifest["files"][name]["sha256"]:
            raise ValueError(f"交付檔案缺失或 SHA 不符：{name}")
    for package in (
        "ultralytics",
        "activation_lab",
        "binaryqk",
        "yolo_attention",
        "yolo_combine",
        "stage8",
    ):
        for name, module in tuple(sys.modules.items()):
            if name == package or name.startswith(package + "."):
                origin = getattr(module, "__file__", None)
                if origin and not Path(origin).resolve().is_relative_to(RUNTIME):
                    raise RuntimeError(f"已載入另一版 {name}；請在獨立程序載入交接包")
    sidecar = json.loads(read(HERE / entry["sidecar"]))
    payload = torch.load(
        HERE / entry["checkpoint"], map_location="cpu", weights_only=True
    )
    if (
        canonical(payload["manifest"]) != sidecar["manifest"]
        or sidecar["sha256"] != entry["sha256"]
    ):
        raise ValueError("checkpoint／manifest 不符")
    with gzip.open(HERE / entry["architecture"], "rb") as stream:
        model = torch.load(stream, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(
        v.is_floating_point()
        and (v.dtype != torch.float32 or not torch.isfinite(v).all())
        for v in model.state_dict().values()
    ):
        raise ValueError("權重必須為有限 FP32")
    model.eval()
    verify_contract(model, sidecar["manifest"])
    model.to(device).eval()
    for head in (model.detect_head, model.pose_head):
        for name in ("stride", "anchors", "strides"):
            setattr(head, name, getattr(head, name).to(device))
        head.shape = None
    model.graph.stride = model.graph.stride.to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    model = load_model(candidate=args.candidate)
    with torch.inference_mode():
        outputs = model(torch.zeros(1, 3, 640, 640), task="both")
    print(
        json.dumps(
            verify_contract(model, canonical(model.stage8_manifest)),
            ensure_ascii=False,
            indent=2,
        )
    )
