"""從已驗證模型匯出可攜交接包；無訓練、無 GPU、不複製已訓權重。"""

import ast
import copy
import gzip
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "yolo_activation"
RESULT = PROJECT / "result/stage7"
OUT = RESULT / "handoff"
for path in ("yolo_activation", "yolo_activation/src", "yolo_combine", "yolo_combine/src",
             "yolo_attention/src", "yolo_binaryqk", "yolo_masf", "yolo_achitechure/achitechure_1/src"):
    sys.path.insert(0, str(ROOT / path))
sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tensor_records(value, prefix="output"):
    import torch

    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("非有限參考輸出")
        return {prefix: {"shape": list(value.shape), "dtype": str(value.dtype),
                         "sha256": hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()}}
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (tuple, list)) else ()
    return {k: v for name, child in children for k, v in tensor_records(child, f"{prefix}.{name}").items()}


def main():
    import torch
    import ultralytics
    from stage7.model import load, manifest_for

    torch.set_num_threads(2)
    reference = json.loads((RESULT / "handoff-reference.json").read_text())
    assert reference["reviewed_before_implementation"]
    for path, digest in reference["sources"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"事前核對來源改變：{path}")
    selection = json.loads((RESULT / "handoff.json").read_text())
    if selection["status"] != "complete" or selection["next_stage_started"]:
        raise ValueError("Stage7 尚未完成或跨階段狀態改變")
    (OUT / "validation/result.json").unlink(missing_ok=True)
    runtime = OUT / "runtime"
    provenance = {}

    def copy_source(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        provenance[str(target.relative_to(OUT))] = {"source": str(source.relative_to(ROOT)), "sha256": sha(source), "transform": "byte-identical"}

    ultra = Path(ultralytics.__file__).resolve().parent
    for source in sorted(ultra.rglob("*")):
        if source.is_file() and source.suffix in {".py", ".yaml"} and "__pycache__" not in source.parts:
            copy_source(source, runtime / "ultralytics" / source.relative_to(ultra))
    for package, folder, names in (
        ("activation_lab", "yolo_activation/src/activation_lab", ("activations.py",)),
        ("binaryqk", "yolo_binaryqk/binaryqk", ("model.py", "basis.py")),
        ("yolo_combine", "yolo_combine/src/yolo_combine", ("fusion_model.py", "contracts.py")),
        ("yolo_attention", "yolo_attention/src/yolo_attention", ("__init__.py", "config.py", "attention.py", "normalization.py", "reciprocal.py", "score_affine.py", "bdcn.py", "binary_basis.py", "projection.py", "quantization.py", "relative_bias.py", "schedule.py")),
    ):
        for name in names:
            copy_source(ROOT / folder / name, runtime / package / name)
        if "__init__.py" not in names:
            (runtime / package / "__init__.py").write_text('"""凍結推論 runtime；不匯入訓練入口。"""\n')
    source = ROOT / "yolo_binaryqk/binaryqk/scale_checkpoint.py"
    code = source.read_text()
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == "AffineBranchScore")
    excerpt = ast.get_source_segment(code, node)
    target = runtime / "binaryqk/scale_checkpoint.py"
    target.write_text('"""原 AffineBranchScore 類別逐字抽取；不匯入 checkpoint／訓練流程。"""\n'
                      'import torch\nfrom yolo_attention.config import FixedScaleFormat, ScoreAffineConfig\n'
                      'from yolo_attention.score_affine import QuantizedScoreAffine\n'
                      'from .basis import BASES, branch_dots\n'
                      'CANDIDATES = {"S1": FixedScaleFormat.Q10, "S2": FixedScaleFormat.POWER_OF_TWO}\n\n' + excerpt + '\n')
    provenance[str(target.relative_to(OUT))] = {"source": str(source.relative_to(ROOT)), "sha256": sha(source), "transform": "AffineBranchScore class byte-identical AST extraction; minimal runtime imports"}
    copy_source(ultra.parent / "LICENSE", OUT / "licenses/ultralytics-AGPL-3.0.txt")
    write(OUT / "runtime-sources.json", provenance)
    models, references, metrics = {}, {}, {}
    for name in ("qsilu_pq", "poly_quality"):
        source = RESULT / "candidates" / f"{name}.pt"
        expected = selection["models"][name]
        if sha(source) != expected["sha256"]:
            raise ValueError("選定權重 SHA 不符")
        model = load(source)
        manifest_for(model)
        references[name] = {}
        for size in (160, 640):
            torch.manual_seed(20261103 + size)
            inputs = torch.rand(1, 3, size, size)
            with torch.inference_mode():
                outputs = model(inputs, task="both")
            references[name][str(size)] = tensor_records(outputs)
            print(f"reference {name} {size}: {len(references[name][str(size)])} tensors", flush=True)
        template = copy.deepcopy(model).cpu().eval()
        with torch.no_grad():
            for tensor in template.state_dict().values():
                tensor.zero_()
        for head in (template.detect_head, template.pose_head):
            head.shape = None
            head.anchors = torch.empty(0)
            head.strides = torch.empty(0)
        architecture = f"{name}.architecture.pt.gz"
        with gzip.open(OUT / architecture, "wb", compresslevel=6) as stream:
            torch.save(template, stream)
        destination = OUT / f"{name}.pt"
        # 單一實體模型；原候選路徑作相容入口，避免複製 100 MiB 權重。
        if not destination.exists():
            source.rename(destination)
            source.with_suffix(".json").rename(destination.with_suffix(".json"))
            source.symlink_to(os.path.relpath(destination, source.parent))
            source.with_suffix(".json").symlink_to(os.path.relpath(destination.with_suffix(".json"), source.parent))
        if sha(destination) != expected["sha256"]:
            raise ValueError("搬移改變 checkpoint")
        models[name] = {"checkpoint": f"{name}.pt", "sidecar": f"{name}.json", "architecture": architecture,
                        "sha256": expected["sha256"], "files": [f"{name}.pt", f"{name}.json", architecture],
                        "role": "RepConv 共同 parent" if name == "qsilu_pq" else "Activation 比較精度最佳；不可取代 qSiLU RepConv parent"}
        metrics[name] = {k: expected[k] for k in ("metrics", "epoch", "activation_relative_gate")}
        del template, model, outputs
    write(OUT / "validation/reference.json", {"seed": 20261103, "device": "cpu", "dtype": "FP32", "threads": 2, "models": references})
    write(OUT / "metrics.json", {"evaluator": "完整 COCO val5000、BBAT5 v1 val683；FP32 E2E conf .001 iou .7 max_det300", "models": metrics, "original_independent_gate": selection["original_independent_gate"]})
    packages = ("torch", "torchvision", "numpy", "opencv-python", "PyYAML", "Pillow", "scipy", "matplotlib", "requests", "psutil", "polars", "ultralytics-thop")
    versions, optional_missing = {}, []
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            optional_missing.append(name)
    write(OUT / "environment.json", {"python": sys.version, "third_party": versions, "optional_not_installed": optional_missing, "ultralytics": ultralytics.__version__, "runtime_source": str(ultra), "gpu_used_for_packaging": False})
    (OUT / "requirements.txt").write_text('# 先依硬體安裝 torch／torchvision；本包自帶客製 Ultralytics，勿以 pip ultralytics 覆蓋。\n' + ''.join(f'{k}=={v}\n' for k, v in versions.items()))
    files = {str(p.relative_to(OUT)): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(OUT.rglob("*")) if p.is_file() and p.suffix != ".pyc"
             and p.name != "manifest.json" and "validation" not in p.relative_to(OUT).parts}
    runtime_files = [n for n in files if n.startswith("runtime/") or n == "load_model.py"]
    write(OUT / "manifest.json", {"schema": "stage7-portable-handoff-v1", "stage": 7, "next_stage_started": False,
                                    "repconv_parent": "qsilu_pq", "models": models, "runtime_files": runtime_files, "files": files})
    print("交付模型已移入 handoff；需接續獨立 CPU 驗證。", flush=True)


if __name__ == "__main__":
    main()
