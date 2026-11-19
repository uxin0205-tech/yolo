"""正式資料重現、實際 macro 更新與部署路由檢查。"""

import gc
import time
from dataclasses import asdict
from pathlib import Path

import torch
from binaryqk.run import require_idle_gpu, seed, transfer
from binaryqk.train import FrozenEMA, state_cpu
from masf_yolo.stage4.evaluate import evaluate_pose

from . import checkpoint
from .common import event, review, sha256, write_json
from .data import dataset
from .evaluate import COCO_KEYS, expected, pair
from .models import build, configure, manifest_for, native_pose, view
from .ops import (
    MacroEngine,
    ProgressiveLoss,
    fixed_hashes,
    gradient_groups,
    macros,
    optimizer_for,
)
from .policy import recipe_for


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("routing keys differ")
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError("routing lengths differ")
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    elif left != right:
        raise ValueError("routing values differ")


@torch.inference_mode()
def routing(model):
    model.eval()
    image = torch.zeros(1, 3, 640, 640, device=next(model.parameters()).device)
    both = model(image, task="both")
    for task in ("detect", "pose"):
        assert_nested_equal(both[task], model(image, task=task)[task])
        assert_nested_equal(both[task], view(model, task)(image))
    return {
        "shape": list(image.shape),
        "single_and_dual_and_native_view": "bit-identical",
        "attention_contract": manifest_for(model)["upstream"],
    }


def smoke(output, path, *, recipes=None, macro_count=1):
    output = Path(output)
    if macro_count not in (1, 2):
        raise ValueError("smoke requires one or two real macros")
    recipes = recipes or [recipe_for(name) for name in ("H0", "P", "J1", "J2", "J3")]
    records = []
    pds = dataset("pose", "train", 16, recipes[0]["epochs"])[0]
    dds = (
        dataset("detect", "train", 32)[0]
        if any(r["task"] == "joint" for r in recipes)
        else None
    )
    for recipe in recipes:
        name = recipe["name"]
        seed(review()["protocol"]["seed"])
        model = checkpoint.load(path).cuda()
        mutable = configure(model, recipe)
        active = {n for n, p in model.named_parameters() if p.requires_grad}
        before = state_cpu(model)
        fixed = fixed_hashes(model, active, mutable)
        ema = FrozenEMA(model, mutable_buffers=mutable)
        if ema.updates != 0:
            raise ValueError("EMA must start at zero")
        optimizer = optimizer_for(model, recipe)
        losses = ProgressiveLoss(model, recipe)
        engine = MacroEngine(
            model=model,
            losses=losses,
            optimizer=optimizer,
            mutable_buffers=mutable,
            reference_batch_size=64,
            task_weights={
                "detect": 1.0,
                "pose": 0.25 if recipe["task"] == "joint" else 1.0,
            },
            gradient_groups=gradient_groups(model),
            scaler=torch.amp.GradScaler("cuda", init_scale=1024),
            ema=ema,
            max_grad_norm=10.0,
            max_amp_retries=16,
            preprocess=lambda _task, batch: transfer(batch, "cuda:0"),
        )
        stream = macros(
            dds if recipe["task"] == "joint" else None,
            pds,
            review()["protocol"]["seed"],
        )
        started = time.time()
        reports = []
        for _ in range(macro_count):
            db, pb = next(stream)
            report = engine.run(
                detect_batches=db,
                pose_batches=pb,
                record_gradient_statistics=recipe["task"] == "joint",
            )
            reports.append(asdict(report))
        seconds = time.time() - started
        changed = [
            n
            for n, p in model.named_parameters()
            if n in active and not torch.equal(p.cpu(), before[n])
        ]
        if not changed or ema.updates != macro_count:
            raise ValueError("no actual parameter/EMA update")
        if not report.gradient_presence["pose_head"] or (
            recipe["task"] == "joint" and not all(report.gradient_presence.values())
        ):
            raise ValueError("missing task/shared gradient")
        for candidate in (model, ema.ema):
            if fixed_hashes(candidate, active, mutable) != fixed:
                raise ValueError("smoke changed frozen weights/buffers")
            manifest_for(candidate)
        records.append(
            {
                "phase": name,
                "recipe": recipe,
                "seconds": seconds,
                "updated_parameter_tensors": len(changed),
                "active_tensors": len(active),
                "mutable_bn_buffers": len(mutable),
                "frozen_verified": True,
                "smoke_updates_discarded": True,
                "macro_count": macro_count,
                "macro_reports": reports,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                **asdict(report),
            }
        )
        event(output, kind="smoke-pass", **records[-1])
        del model, ema, optimizer, losses, engine, stream, db, pb, before
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output / "smoke.json", records)
    return records


def initialize(output):
    output = Path(output)
    require_idle_gpu()
    torch.set_num_threads(2)
    seed(review()["protocol"]["seed"])
    path = output / "detect-anchor.pt"
    if not path.exists():
        model = build()
        checkpoint.save(path, model)
        del model
    smoke(output, path)
    seed(review()["protocol"]["seed"])
    model = checkpoint.load(path).cuda()
    checks = routing(model)
    metrics = pair(model, output / "anchor-evaluation", progress_root=output)["metrics"]
    if any(abs(metrics[k] - expected()[k]) > 1e-10 for k in COCO_KEYS):
        raise ValueError("Detect anchor does not reproduce upstream official AP")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    seed(review()["protocol"]["seed"])
    pose = native_pose().cuda()
    ds, data, _ = dataset("pose", "val", 8)
    result = evaluate_pose(
        pose,
        ds,
        data,
        output / "independent-pose",
        progress=lambda e: event(output, **e),
    )
    write_json(output / "independent-pose.json", result)
    if any(
        abs(result["metrics"][k] - v) > 1e-10
        for k, v in expected().items()
        if k not in COCO_KEYS
    ):
        raise ValueError("independent Pose does not reproduce upstream AP")
    result = {
        "status": "passed",
        "anchor": {
            "checkpoint": str(path.resolve()),
            "sha256": sha256(path),
            "metrics": metrics,
            "phase": "initialization",
            "epoch": 0,
        },
        "routing": checks,
        "independent_inputs_reproduced": True,
    }
    write_json(output / "result.json", result)
    event(output, kind="initialization-complete", status="passed")
    return result
