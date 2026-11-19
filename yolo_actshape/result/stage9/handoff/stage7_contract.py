"""Stage7 可攜 tensor 推論入口；保留已驗證的完整硬體友善契約。"""

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUNTIME))


def read(path):
    return Path(path).read_text(encoding="utf-8")


def canonical(value):
    return json.loads(json.dumps(value))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(candidate):
    manifest = json.loads(read(HERE / "manifest.json"))
    if candidate not in manifest["models"]:
        raise ValueError(f"沒有交付此候選：{candidate}")
    entry = manifest["models"][candidate]
    for name in [*manifest["runtime_files"], *entry["files"]]:
        expected = manifest["files"][name]
        path = HERE / name
        if not path.is_file() or digest(path) != expected["sha256"]:
            raise ValueError(f"交付檔案缺少或 SHA 不符：{name}")
    return manifest, entry


def normalizer_settings(module):
    fields = ("score_floor", "segments", "score_step", "score_min", "top_k",
              "score_fraction_bits", "endpoint_fraction_bits", "fraction_bits", "endpoint_bits")
    return canonical({
        name: {"type": type(child).__name__,
               **{key: getattr(child, key) for key in fields if hasattr(child, key)},
               **({"config": asdict(child.config)} if is_dataclass(getattr(child, "config", None)) else {})}
        for name, child in module.named_modules()
    })


def verify_contract(model, expected):
    """類別、係數、alias、B2、Q10 與整數 exp／reciprocal 均須吻合。"""
    import torch
    from activation_lab.activations import build_activation, get_profile
    from binaryqk.scale_checkpoint import AffineBranchScore
    from yolo_attention.attention import HardwareFriendlyAttention
    from yolo_attention.config import Stage3NormalizationConfig
    from yolo_attention.normalization import build_stage3_normalizer

    if canonical(model.stage7_manifest) != expected or expected["schema"] != "activation-stage7-v1":
        raise ValueError("Stage7 manifest 不符")
    policy = expected["activation"]
    base = expected["parent_manifest"]
    if policy["region"] is not None or policy["repconv"] is not None or base["repconv"] is not None:
        raise ValueError("交付僅允許完整 Activation 替換且尚未加入 RepConv")
    if canonical(model.stage6_manifest) != base or canonical(model.contract()) != base["graph"]:
        raise ValueError("雙任務模型架構不符")
    actual = {n for n, m in model.named_modules(remove_duplicate=False)
              if isinstance(m, (torch.nn.SiLU, torch.nn.Hardswish))
              or type(m).__module__ == "activation_lab.activations"}
    if actual != set(policy["sites"]):
        raise ValueError("Activation 位置不符")
    groups = defaultdict(list)
    for path, desc in policy["sites"].items():
        module = model.get_submodule(path)
        name = policy["candidate"]
        profile = get_profile(name)
        if (desc["name"] != name or type(module) is not type(build_activation(name))
                or desc["class"] != f"{type(module).__module__}.{type(module).__qualname__}"
                or getattr(module, "profile", None) != profile
                or desc["profile"] != canonical(asdict(profile) if profile else None)
                or getattr(module, "inplace", None) != desc["inplace"]):
            raise ValueError(f"Activation 類別／固定係數不符：{path}")
        groups[id(module)].append(path)
    if sorted(sorted(g) for g in groups.values()) != policy["alias_groups"]:
        raise ValueError("Activation 共享 alias 不符")
    upstream = base["upstream"]
    scale = upstream["parent_manifest"]
    if (upstream["candidate"] != "D1" or scale["scale_candidate"] != "S1"
            or scale["variant"] != "B2" or scale["scale_slots"] != 16):
        raise ValueError("不是已選定的 B2／S1／D1 主線")
    reference = build_stage3_normalizer(Stage3NormalizationConfig("D1")).eval()
    sites = {n for n, m in model.named_modules() if isinstance(m, HardwareFriendlyAttention)}
    if sites != {"graph." + n for n in scale["constants"]}:
        raise ValueError("Attention 位置不符")
    for path in sites:
        module = model.get_submodule(path)
        if (not getattr(module, "integrated_stage3", False)
                or type(module.score) is not AffineBranchScore
                or module.score.variant != "B2" or module.score.candidate != "S1"
                or tuple(module.score.bases) != ("sign", "hadamard")
                or (module.num_heads, module.key_dim, module.head_dim) != (4, 32, 64)
                or canonical(module.score.affine.export_constants()) != scale["constants"][path.removeprefix("graph.")]):
            raise ValueError("Binary QK／scale／bias 不符")
        normalizer = module.normalizer
        if (type(normalizer) is not type(reference)
                or normalizer_settings(normalizer) != normalizer_settings(reference)
                or canonical(normalizer.reciprocal.contract()) != upstream["normalization"]["reciprocal"]):
            raise ValueError("完整 PWL／reciprocal 契約不符")
        buffers, refs = normalizer.state_dict(), reference.state_dict()
        if buffers.keys() != refs.keys() or any(not torch.equal(buffers[n].cpu(), refs[n]) for n in refs):
            raise ValueError("PWL／reciprocal 固定常數不符")
    return {"activation": policy["candidate"], "registered_activation_sites": len(actual),
            "alias_groups": len(groups), "attention_sites": sorted(sites),
            "binary": "B2", "scale": "S1-Q10", "normalization": "D1-PWL-LUT256",
            "repconv": False, "parameters": sum(p.numel() for p in model.parameters())}


def load_model(device="cpu", candidate="qsilu_pq"):
    """載入可信交付包；預設 qSiLU，共用原生 Detect／Pose tensor 介面。"""
    _, entry = verify_files(candidate)
    import torch

    # 避免與同一程序內已載入的其他版本混用；請以獨立程序載入交付包。
    for package in ("ultralytics", "activation_lab", "binaryqk", "yolo_attention", "yolo_combine"):
        for name, module in tuple(sys.modules.items()):
            if name == package or name.startswith(package + "."):
                origin = getattr(module, "__file__", None)
                if origin and not Path(origin).resolve().is_relative_to(RUNTIME):
                    raise RuntimeError(f"已載入另一版 {name}；請在新 Python 程序載入交付包")
    sidecar = json.loads(read(HERE / entry["sidecar"]))
    payload = torch.load(HERE / entry["checkpoint"], map_location="cpu", weights_only=True)
    if (canonical(payload["manifest"]) != sidecar["manifest"]
            or sidecar["sha256"] != entry["sha256"]):
        raise ValueError("checkpoint／sidecar manifest 不符")
    # 同 optimize 交付模式：骨架是 Python 物件，只載入本包可信任且已核對 SHA 的骨架。
    with gzip.open(HERE / entry["architecture"], "rb") as stream:
        model = torch.load(stream, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
           for v in model.state_dict().values()):
        raise ValueError("交付權重必須為有限 FP32")
    model.eval()
    verify_contract(model, sidecar["manifest"])
    model.to(device).eval()
    # 原生 head 的非 buffer cache 也必須隨 device 移動。
    for head in (model.detect_head, model.pose_head):
        for name in ("stride", "anchors", "strides"):
            setattr(head, name, getattr(head, name).to(device))
        head.shape = None
    model.graph.stride = model.graph.stride.to(device)
    return model


def tensors(value, prefix="output"):
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensors(child, f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from tensors(child, f"{prefix}.{index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("qsilu_pq", "poly_quality"), default="qsilu_pq")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, choices=(160, 640), default=160)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    model = load_model(args.device, args.candidate)
    with torch.inference_mode():
        outputs = model(torch.zeros(1, 3, args.size, args.size, device=args.device), task="both")
    if not all(torch.isfinite(t).all() for _, t in tensors(outputs)):
        raise ValueError("輸出非有限")
    print({k: list(v[0].shape) for k, v in outputs.items()})
    print("PASS：strict 載入、契約核對與 CPU／指定 device smoke；這不是 AP 驗證。")


if __name__ == "__main__":
    main()
