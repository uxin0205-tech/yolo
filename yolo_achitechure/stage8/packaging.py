"""凍結可攜推論 package；folded 權重搬移而不複製，保留相容入口。"""

import copy
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .common import HERE, ROOT, read, sha256, write_json


def records(value, prefix="output"):
    import torch

    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("非有限輸出")
        return {
            prefix: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(
                    value.cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
            }
        }
    children = (
        value.items()
        if isinstance(value, dict)
        else enumerate(value)
        if isinstance(value, (tuple, list))
        else ()
    )
    return {
        k: v
        for name, child in children
        for k, v in records(child, f"{prefix}.{name}").items()
    }


def bundle(destination, models, recommendation, sizes=(160, 640)):
    import torch

    from .model import load, manifest_for

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    prior = ROOT / "yolo_activation/result/stage7/handoff"
    prior_manifest = read(prior / "manifest.json")
    for name in prior_manifest["runtime_files"]:
        if sha256(prior / name) != prior_manifest["files"][name]["sha256"]:
            raise ValueError(f"Stage7 可攜 runtime 改變：{name}")
    shutil.copytree(
        prior / "runtime",
        destination / "runtime",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(prior / "licenses", destination / "licenses", dirs_exist_ok=True)
    for name in ("environment.json", "requirements.txt"):
        shutil.copyfile(prior / name, destination / name)
    shutil.copyfile(prior / "load_model.py", destination / "stage7_contract.py")
    for source, target in (
        ("portable_loader.py", "load_model.py"),
        ("portable_validate.py", "validate.py"),
        ("rep.py", "runtime/stage8/rep.py"),
    ):
        path = destination / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / source, path)
    (destination / "runtime/stage8/__init__.py").write_text(
        '"""Stage8 frozen inference runtime."""\n'
    )
    entries, references = {}, {}
    for name, entry in models.items():
        source = Path(entry["checkpoint"])
        if sha256(source) != entry["sha256"]:
            raise ValueError("已選 folded SHA 不符")
        model = load(source)
        manifest_for(model)
        if model.stage8_manifest["mode"] != "folded" or model.stage8_manifest["operators"] != "hardware":
            raise ValueError("handoff 只放 folded；多分支續訓另外保留")
        references[name] = {}
        for size in sizes:
            torch.manual_seed(20261103 + size)
            with torch.inference_mode():
                outputs = model(torch.rand(1, 3, size, size), task="both")
            references[name][str(size)] = records(outputs)
        template = copy.deepcopy(model)
        with torch.no_grad():
            for tensor in template.state_dict().values():
                tensor.zero_()
        for head in (template.detect_head, template.pose_head):
            head.shape = None
            head.anchors = torch.empty(0)
            head.strides = torch.empty(0)
        architecture = f"{name}.architecture.pt.gz"
        with gzip.open(destination / architecture, "wb", compresslevel=6) as stream:
            torch.save(template, stream)
        target = destination / f"{name}.pt"
        if not target.exists():
            source.rename(target)
            source.with_suffix(".json").rename(target.with_suffix(".json"))
            source.symlink_to(os.path.relpath(target, source.parent))
            source.with_suffix(".json").symlink_to(
                os.path.relpath(target.with_suffix(".json"), source.parent)
            )
        if sha256(target) != entry["sha256"]:
            raise ValueError("搬移改變 SHA")
        entries[name] = {
            "checkpoint": target.name,
            "sidecar": target.with_suffix(".json").name,
            "architecture": architecture,
            "sha256": entry["sha256"],
            "files": [target.name, target.with_suffix(".json").name, architecture],
            "metrics": entry["metrics"],
        }
        del outputs, template, model
    default = (
        recommendation
        if recommendation in entries
        else max(entries, key=lambda n: models[n]["balanced_score"])
    )
    write_json(
        destination / "validation/reference.json",
        {"seed": 20261103, "device": "cpu", "dtype": "FP32", "models": references},
    )
    (destination / "README.md").write_text(
        "# Stage8 可攜 RepConv 交接\n\n"
        f"研究建議：`{recommendation}`；本包預設示範 Rep：`{default}`。若建議為 parent／previous_rep17_20，本包是比較候選；真正主線由上層研究 handoff 指向既有較佳模型。\n\n"
        "所有交付模型都是單 Conv3×3 → qSiLU，保留 B2／Q10／D1 PWL＋LUT256 reciprocal。\n"
        '在新 Python 程序執行 `from load_model import load_model; model = load_model(candidate="rep17")`，'
        '再呼叫 `model(tensor, task="both")`；tensor 為 RGB BCHW／0–1。Detect／Pose 可單獨 routing。\n\n'
        "先按 requirements 安裝依賴；此包自帶客製 Ultralytics。用 `python -B validate.py` 重做 CPU 驗證。"
        "骨架為 Python 物件，只可載入可信來源；入口先核對 SHA。模型未含影像前處理與外部評估資料。\n\n"
        "多分支 EMA 權重及含 optimizer 的 resume 由上層 result/stage8-retrain 保存，不能拿 folded checkpoint 當多分支 resume。"
        "Conv、QKV、P·V、projection、head sigmoid 等仍有浮點運算；未宣稱 RTL/HLS 或板端實測。\n"
    )
    write_json(
        destination / "provenance.json",
        {
            "stage7_runtime_manifest_sha256": sha256(prior / "manifest.json"),
            "sources": {
                n: sha256(HERE / n)
                for n in ("rep.py", "portable_loader.py", "portable_validate.py")
            },
            "weights_moved_not_copied": True,
        },
    )
    files = {
        str(p.relative_to(destination)): {
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(destination.rglob("*"))
        if p.is_file()
        and p.suffix != ".pyc"
        and p.name != "manifest.json"
        and "validation" not in p.relative_to(destination).parts
    }
    runtime_files = [
        n
        for n in files
        if n.startswith("runtime/") or n in ("load_model.py", "stage7_contract.py")
    ]
    write_json(
        destination / "manifest.json",
        {
            "schema": "stage8-portable-handoff-v3",
            "stage": 8,
            "next_stage_started": False,
            "recommendation": recommendation,
            "default_model": default,
            "models": entries,
            "files": files,
            "runtime_files": runtime_files,
        },
    )
    subprocess.run(
        [sys.executable, "-I", "-B", str(destination / "validate.py")],
        cwd=destination,
        check=True,
        timeout=600,
    )
    validation = read(destination / "validation/result.json")
    if validation["status"] != "passed":
        raise ValueError("交接驗證未通過")
    return {"models": entries, "validation": validation, "path": str(destination)}
