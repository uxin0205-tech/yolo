"""依已完成的完整驗證交付 Stage6；CPU 路由／成本稽核，不再加訓。"""

import csv
import gc
import math
import os
from pathlib import Path

import torch
from stage6 import checkpoint
from stage6.common import ARTIFACTS, PROJECT, read, review, sha256, write_json
from stage6.evaluate import expected, vector
from stage6.models import manifest_for, native_pose
from stage6.verification import assert_nested_equal
from torch import nn

from yolo_attention import stage3_checkpoint


@torch.inference_mode()
def count_convolutions(model, forward):
    rows, handles = [], []

    def hook(name, layer, inputs, output):
        per_output = layer.in_channels // layer.groups * math.prod(layer.kernel_size)
        rows.append(
            {
                "module": name,
                "output": list(output.shape),
                "MAC": int(output.numel() * per_output),
            }
        )

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(
                layer.register_forward_hook(
                    lambda layer, inputs, output, name=name: hook(
                        name, layer, inputs, output
                    )
                )
            )
    try:
        output = forward()
    finally:
        for handle in handles:
            handle.remove()
    return output, {"MAC": sum(row["MAC"] for row in rows), "calls": rows}


def write_csv(path, fields, rows):
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261003)
    source = ARTIFACTS / "pose-head-recovery-20261029"
    audit = ARTIFACTS / "final-review-20261029"
    result = read(source / "result.json")
    if (
        result["status"] != "candidate_complete"
        or not result["verification"]["full_ap_reproduced"]
    ):
        raise ValueError("complete independent validation required")
    selected = result["best"]
    optimize = read(audit / "optimize-result.json")
    if optimize["status"] != "passed" or not optimize["same_dataset_order_and_padding"]:
        raise ValueError("matched optimize validation required")
    if sha256(selected["checkpoint"]) != selected["sha256"]:
        raise ValueError("selected SHA changed")
    destination = PROJECT / "result/stage6"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("refusing to overwrite existing Stage6 handoff")
    model = checkpoint.load(selected["checkpoint"]).cpu().eval()
    manifest = manifest_for(model)
    parameters = sum(p.numel() for p in model.parameters())
    image = torch.zeros(1, 3, 640, 640)
    print("CPU routing and executed Conv2d cost audit", flush=True)
    outputs, costs = {}, {}
    for task in ("both", "detect", "pose"):
        outputs[task], costs[task] = count_convolutions(
            model, lambda task=task: model(image, task=task)
        )
    for task in ("detect", "pose"):
        assert_nested_equal(outputs["both"][task], outputs[task][task])
    site_rows = []
    for name in stage3_checkpoint.SITES:
        attn = model.graph.get_submodule(name)
        heads, kd, hd, tokens, bases = (
            attn.num_heads,
            attn.key_dim,
            attn.head_dim,
            400,
            2,
        )
        site_rows.append(
            {
                "site": name,
                "heads": heads,
                "key_dim": kd,
                "head_dim": hd,
                "tokens_at_640": tokens,
                "bases": bases,
                "binary_bit_products": bases * heads * tokens**2 * kd,
                "PV_float_MAC": heads * tokens**2 * hd,
                "hadamard_add_sub": 2 * heads * tokens * kd * int(math.log2(kd)),
                "reciprocal_rows": heads * tokens,
            }
        )
    del outputs
    for task in ("detect", "pose"):
        original = (
            (
                stage3_checkpoint.load(review()["detect_parent"])[0]
                if task == "detect"
                else native_pose()
            )
            .cpu()
            .eval()
        )
        _, costs["independent_" + task] = count_convolutions(
            original, lambda original=original: original(image)
        )
        del original
        gc.collect()
    subtotal = costs["independent_detect"]["MAC"] + costs["independent_pose"]["MAC"]
    cost = {
        "input": [1, 3, 640, 640],
        "dtype": "FP32",
        "device": "CPU",
        "classification": "executed-shape analytical Conv2d MAC subtotal; not hardware measurement",
        "parameters": parameters,
        "independent_parameters": manifest["independent_parameters"],
        "parameter_reduction_fraction": 1
        - parameters / manifest["independent_parameters"],
        "checkpoint_bytes": Path(selected["checkpoint"]).stat().st_size,
        "Conv2d": costs,
        "independent_Conv2d_MAC": subtotal,
        "Conv2d_reduction_fraction": 1 - costs["both"]["MAC"] / subtotal,
        "attention_640_analytical": site_rows,
        "excluded_from_conv_subtotal": [
            "QK binary",
            "Hadamard",
            "PWL/reciprocal",
            "PV matmul",
            "BN",
            "activations",
            "decode",
            "resize/concat",
        ],
        "QKV_and_projection_Conv2d_included": True,
        "target_latency_energy_RTL_HLS": "未量測",
    }
    current = read(source / "verification/evaluation/metrics.json")
    metrics = selected["metrics"]
    if current["metrics"] != metrics:
        raise ValueError("reloaded metrics changed")
    opt_official = vector(optimize["detect"], optimize["pose"])
    comparisons = []
    for label, opt, now in (
        (
            "COCO internal",
            optimize["detect"]["internal"]["metrics/mAP50-95(B)"],
            current["detect"]["internal"]["metrics/mAP50-95(B)"],
        ),
        (
            "person internal",
            optimize["detect"]["classes_internal"]["person"]["AP50_95"],
            current["detect"]["classes_internal"]["person"]["AP50_95"],
        ),
        *((key, opt_official[key], metrics[key]) for key in metrics),
    ):
        comparisons.append(
            {"metric": label, "optimize": opt, "selected": now, "delta": now - opt}
        )
    write_json(audit / "cost.json", cost)
    write_json(audit / "comparison.json", comparisons)
    # 搬移已停止的唯一成果資料夾，原入口保留相容連結，沒有複製第二套權重。
    destination.mkdir(parents=True, exist_ok=True)
    moved = destination / "pose-head-recovery"
    source.rename(moved)
    source.symlink_to(os.path.relpath(moved, source.parent), target_is_directory=True)
    actual = moved / "train/best-e001.pt"
    if sha256(actual) != selected["sha256"]:
        raise ValueError("moved checkpoint SHA differs")
    for suffix in (".pt", ".json"):
        target = destination / ("selected" + suffix)
        target.symlink_to(os.path.relpath(actual.with_suffix(suffix), destination))
    reloaded = checkpoint.load(destination / "selected.pt")
    if manifest_for(reloaded) != manifest or any(
        not torch.equal(v, reloaded.state_dict()[k])
        for k, v in model.state_dict().items()
    ):
        raise ValueError("handoff reload state/manifest differs")
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "cost.json", cost)
    write_json(destination / "comparison.json", comparisons)
    rows = []
    for phase in ("H0", "H1", "P", "M0", "J1"):
        rows.extend(
            {"phase": phase, "epoch": h["epoch"], **h["metrics"]}
            for h in read(ARTIFACTS / "rebuild-20261029" / phase / "history.json")
        )
    for phase, run in (
        ("KD", ARTIFACTS / "recovery-kd-20261029"),
        ("Pose-head", source),
    ):
        rows.extend(
            {"phase": phase, "epoch": h["epoch"], **h["metrics"]}
            for h in read(run / "train/history.json")
        )
    write_csv(destination / "history.csv", ["phase", "epoch", *metrics], rows)
    write_csv(
        destination / "comparison.csv",
        ["metric", "optimize", "selected", "delta"],
        comparisons,
    )
    handoff = {
        "status": "research_stage_complete",
        "stage": 6,
        "original_accuracy_gate_passed": result["accepted"],
        "automatically_promoted": False,
        "selection": "best observed balanced candidate after bounded recovery; not globally optimal guarantee",
        "selected": {**selected, "checkpoint": str(destination / "selected.pt")},
        "resume": {
            "checkpoint": str(moved / "train/resume.pt"),
            "epoch": 5,
            "note": "最新續訓狀態為 E5，並非選定 E1 的 exact resume；下一階段由 E1 自包含權重建立新 optimizer。",
        },
        "metrics": metrics,
        "independent_references": expected(),
        "remaining_target_deficits": result["remaining_target_deficits"],
        "matched_optimize_evaluation": str(audit / "optimize-result.json"),
        "full_AP_reload_reproduced": True,
        "handoff_reload_state_equal": True,
        "cpu_single_and_both_routing": "bit-identical",
        "deployment_contract": manifest["upstream"],
        "next_stage_started": False,
        "cost": {k: v for k, v in cost.items() if k != "Conv2d"},
    }
    write_json(destination / "handoff.json", handoff)
    print(
        {
            "result": str(destination),
            "parameters": parameters,
            "Conv2d_MAC": costs["both"]["MAC"],
            "original_accuracy_gate_passed": result["accepted"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
