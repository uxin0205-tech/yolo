"""完整 train macro 的任務梯度校準；零更新，每個 macro 恢復所有 tensor。"""

import statistics
from dataclasses import asdict
from pathlib import Path

import torch
from binaryqk.run import require_idle_gpu, seed, transfer
from binaryqk.train import state_cpu
from stage6.data import dataset
from yolo_combine.contracts import Task

from . import model as checkpoint
from .common import event, protocol, sha256, write_json
from .model import configure
from .ops import MacroEngine, ProbeOptimizer, gradient_groups, loss_for_recipe, macros


def calibrate(output, parent, recipe):
    output = Path(output)
    require_idle_gpu()
    config = protocol()["calibration"]
    seed(protocol()["seed"])
    if sha256(parent["checkpoint"]) != parent["sha256"]:
        raise ValueError("calibration parent checksum")
    model = checkpoint.load(parent["checkpoint"]).cuda()
    mutable = configure(model, recipe)
    initial = state_cpu(model)
    pds = dataset("pose", "train", 16, recipe["epochs"])[0]
    dds = dataset("detect", "train", 16, recipe["epochs"])[0]
    engine = MacroEngine(
        model=model,
        losses=loss_for_recipe(model, recipe),
        mutable_buffers=mutable,
        optimizer=ProbeOptimizer(p for p in model.parameters() if p.requires_grad),
        reference_batch_size=64,
        task_weights={Task.DETECT: 1.0, Task.POSE: config["initial_pose_weight"]},
        gradient_groups=gradient_groups(model),
        scaler=torch.amp.GradScaler("cuda", init_scale=1024),
        max_amp_retries=16,
        max_grad_norm=10.0,
        preprocess=lambda _task, batch: transfer(batch, "cuda:0"),
    )
    iterator = iter(macros(dds, pds, protocol()["seed"] + 90000))
    records = []

    def measure(count, phase):
        ratios = []
        for i in range(count):
            model.load_state_dict(initial, strict=True)
            db, pb = next(iterator)
            report = engine.run(
                detect_batches=db, pose_batches=pb, record_gradient_statistics=True
            )
            stats = asdict(report.gradient_statistics)
            if not stats["detect_norm"] > 0 or not stats["pose_norm"] > 0:
                raise ValueError("missing task gradients")
            ratio = stats["pose_norm"] / stats["detect_norm"]
            ratios.append(ratio)
            records.append(
                {
                    "phase": phase,
                    "index": i,
                    "ratio": ratio,
                    "pose_weight": engine.task_weights[Task.POSE],
                    **stats,
                    "amp_retries": report.amp_overflow_retries,
                }
            )
            event(output, kind="calibration", **records[-1])
        return statistics.median(ratios)

    def quantize(value):
        return max(
            config["bounds"][0],
            min(config["bounds"][1], round(value / config["step"]) * config["step"]),
        )

    initial_ratio = measure(config["macros"], "measure")
    selected = quantize(
        config["initial_pose_weight"] * config["target_ratio"] / initial_ratio
    )
    for attempt in range(config["attempts"]):
        engine.task_weights[Task.POSE] = selected
        observed = measure(config["confirmation_macros"], f"confirm-{attempt + 1}")
        if config["confirmation_band"][0] <= observed <= config["confirmation_band"][1]:
            break
        if attempt + 1 < config["attempts"]:
            selected = quantize(selected * config["target_ratio"] / observed)
    model.load_state_dict(initial, strict=True)
    if any(not torch.equal(v, model.state_dict()[n].cpu()) for n, v in initial.items()):
        raise ValueError("calibration changed parent")
    passed = (
        config["confirmation_band"][0] <= observed <= config["confirmation_band"][1]
    )
    result = {
        "status": "passed" if passed else "needs_review",
        "parent_sha256": parent["sha256"],
        "recipe": recipe,
        "selected_pose_weight": selected,
        "calibration_median_ratio": initial_ratio,
        "confirmation_median_ratio": observed,
        "validation_used": False,
        "checkpoint_restored_each_macro": True,
        "optimizer_step_noop": True,
        "records": records,
    }
    write_json(output / "result.json", result)
    return result
