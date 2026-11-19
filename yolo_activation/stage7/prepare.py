"""第七階段 CPU 前置檢查：只讀 Stage6 parent，不替換或訓練模型。"""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from activation_lab.activations import get_profile
from stage6.checkpoint import load
from stage6.models import manifest_for
from stage6.verification import assert_nested_equal
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261003)
    review = json.loads((HERE / "reference-review.json").read_text())
    if not review["reviewed_before_implementation"]:
        raise ValueError("需先核對 optimize")
    for path, digest in review["sources_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"參考來源已變更：{path}")
    parent = Path(review["parent"]["checkpoint"])
    if sha256(parent) != review["parent"]["sha256"]:
        raise ValueError("Stage6 parent SHA 不符")
    model = load(parent).cpu().eval()
    manifest = manifest_for(model)
    if manifest["upstream"]["normalization"] != review["hardware_contract"]:
        raise ValueError("PWL／reciprocal 契約不符")
    profile = get_profile("qsilu_pq")
    expected_coefficients = (
        (57 / 256, 0, 0),
        (23 / 256, 17 / 64, -17 / 128),
        (-11 / 512, 91 / 128, -37 / 64),
        (-5 / 1024, 37 / 64, -5 / 16),
    )
    if profile is None or profile.quadratic_coefficients != expected_coefficients:
        raise ValueError("qSiLU 固定分段係數不符")
    aliases = defaultdict(list)
    sites = []
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, nn.SiLU):
            aliases[id(module)].append(name)
            sites.append(name)
    if not sites or manifest["repconv"] is not None:
        raise ValueError("預期尚未替換 activation／RepConv 的 Stage6 parent")

    # SiLU 本身可能共用同一物件；以擁有 .act 的 Conv 前向判斷 call-site，
    # 避免在共用 SiLU 上為每條 alias 加 hook 後誤算成全部都執行。
    calls = {task: defaultdict(int) for task in ("both", "detect", "pose")}
    owners = defaultdict(list)
    hook_targets = []
    non_conv_sites = []
    for path in sites:
        owner, _, child = path.rpartition(".")
        if child == "act":
            target = model.get_submodule(owner)
        else:
            target = model.get_submodule(path)
            if ".flow_model." not in path or len(aliases[id(target)]) != 1:
                raise ValueError(f"需另外稽核非 Conv.act 的 SiLU：{path}")
            non_conv_sites.append(path)
        owners[id(target)].append(path)
        hook_targets.append((path, target))
    ambiguous = [paths for paths in owners.values() if len(paths) != 1]
    if ambiguous:
        raise ValueError(f"需追查共用 owner 的實際路徑：{ambiguous}")
    handles = []
    active_task = "both"

    def hook(path, module, inputs):
        calls[active_task][path] += 1

    for path, target in hook_targets:
        handles.append(
            target.register_forward_pre_hook(
                lambda module, inputs, path=path: hook(path, module, inputs)
            )
        )
    image = torch.rand(1, 3, 160, 160)
    outputs = {}
    try:
        with torch.inference_mode():
            for task in calls:
                active_task = task
                outputs[task] = model(image, task=task)
                if not all(torch.isfinite(t).all() for t in tensors(outputs[task])):
                    raise ValueError(f"{task} 輸出非有限值")
    finally:
        for handle in handles:
            handle.remove()
    for task in ("detect", "pose"):
        assert_nested_equal(outputs["both"][task], outputs[task][task])
    if any(path in executed for executed in calls.values() for path in non_conv_sites):
        raise ValueError("訓練專用 RealNVP 不應在 eval 執行")
    if manifest_for(model) != manifest or sha256(parent) != review["parent"]["sha256"]:
        raise ValueError("檢查期間 parent／契約改變")
    report = {
        "status": "passed",
        "scope": "CPU parent 準備檢查；不是 activation 替換、AP 或訓練",
        "parent": review["parent"],
        "input": [1, 3, 160, 160],
        "dtype": "FP32",
        "device": "cpu",
        "parameters": sum(p.numel() for p in model.parameters()),
        "registered_silu_references": sites,
        "alias_groups": list(aliases.values()),
        "registered_references_count": len(sites),
        "unique_silu_objects": len(aliases),
        "executed_call_sites": {task: dict(value) for task, value in calls.items()},
        "executed_site_counts": {task: len(value) for task, value in calls.items()},
        "call_site_method": "Conv.act 在唯一 owner 記錄；獨立 RealNVP SiLU 直接記錄，不在共用 SiLU 物件重複加 hook",
        "training_only_realnvp_sites": non_conv_sites,
        "routing": "單任務與 both 對應輸出逐值相同",
        "qSiLU_reference_profile": asdict(profile),
        "qSiLU_installed": False,
        "normalization_preserved": True,
        "repconv_installed": False,
        "gpu_used": False,
        "script_sha256": sha256(__file__),
    }
    (HERE / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "registered_references_count",
                    "unique_silu_objects",
                    "executed_site_counts",
                    "gpu_used",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
