"""每個完整 epoch 可續跑；不讓較差後段覆蓋最佳 RepConv。"""

import copy
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from binaryqk.run import require_idle_gpu, seed, transfer
from binaryqk.train import FrozenEMA, atomic_torch, state_cpu
from stage6.data import dataset
from yolo_combine.contracts import Task
from yolo_combine.joint_trainer import StageWarmupCosineScheduler

from . import model as checkpoint
from .common import (
    event,
    passed,
    protocol,
    rank,
    read,
    sha256,
    source_guard,
    write_json,
)
from .evaluate import pair
from .model import configure, manifest_for
from .rep import branches
from .ops import (
    MacroEngine,
    fixed_hashes,
    gradient_groups,
    loss_for_recipe,
    macros,
    optimizer_for,
    restore_rng,
    rng_state,
)


def engine_for(model, recipe, pose_weight, mutable, optimizer, ema=None):
    losses = loss_for_recipe(model, recipe)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    engine = MacroEngine(
        model=model,
        losses=losses,
        optimizer=optimizer,
        mutable_buffers=mutable,
        reference_batch_size=64,
        task_weights={Task.DETECT: 1.0, Task.POSE: pose_weight},
        gradient_groups=gradient_groups(model),
        scaler=scaler,
        ema=ema,
        max_grad_norm=10.0,
        max_amp_retries=16,
        preprocess=lambda task, batch: transfer(batch, "cuda:0"),
    )
    return engine, losses, scaler


def smoke(output, parent, recipe, calibration):
    output = Path(output)
    seed(protocol()["seed"])
    model = checkpoint.load(parent["checkpoint"]).cuda()
    mutable = configure(model, recipe)
    active = {n for n, p in model.named_parameters() if p.requires_grad}
    before = state_cpu(model)
    fixed = fixed_hashes(model, active, mutable)
    ema = FrozenEMA(model, mutable_buffers=mutable)
    engine, _, _ = engine_for(
        model,
        recipe,
        calibration["selected_pose_weight"],
        mutable,
        optimizer_for(model, recipe),
        ema,
    )
    pds = dataset("pose", "train", 16, recipe["epochs"])[0]
    dds = dataset("detect", "train", 16, recipe["epochs"])[0]
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    iterator, records = iter(macros(dds, pds, protocol()["seed"] + 80000)), []
    for index in range(2):
        db, pb = next(iterator)
        report = engine.run(
            detect_batches=db, pose_batches=pb, record_gradient_statistics=True
        )
        if not all(report.gradient_presence.values()):
            raise ValueError("smoke 缺少共享層或任務 head 梯度")
        records.append(asdict(report))
        event(
            output, kind="training-smoke", macro=index + 1, seconds=time.time() - start
        )
    for candidate in (model, ema.ema):
        if fixed_hashes(candidate, active, mutable) != fixed:
            raise ValueError("smoke／EMA 改變 frozen tensors")
        manifest_for(candidate)
    updated = [
        n
        for n, p in model.named_parameters()
        if n in active and not torch.equal(before[n], p.detach().cpu())
    ]
    if not updated or ema.updates != 2:
        raise ValueError("smoke 未實際更新")
    for index in model.stage8_manifest["layers"]:
        suffixes = [f"conv{b}.{key}" for b in range(2, len(branches(model.graph.model[index])) + 1) for key in ("bn.weight", "conv.weight")]
        for suffix in suffixes:
            if f"graph.model.{index}.{suffix}" not in updated:
                raise ValueError(f"Rep{index} 新分支沒有實際更新：{suffix}")
    for head in (".detect_head.", ".pose_head."):
        if not any(head in name for name in updated):
            raise ValueError(f"任務 head 未實際更新：{head}")
    result = {
        "status": "passed",
        "parent_sha256": parent["sha256"],
        "recipe": recipe,
        "records": records,
        "updated_parameters": len(updated),
        "new_branch_and_both_heads_updated": True,
        "frozen_tensors_and_ema_verified": True,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
        "seconds_per_macro": (time.time() - start) / 2,
        "ema_updates": ema.updates,
        "smoke_weights_discarded": True,
    }
    write_json(output / "result.json", result)
    return result


def train(output, parent, recipe, calibration, root):
    output = Path(output)
    require_idle_gpu()
    seed(protocol()["seed"])
    source_guard(root)
    if sha256(parent["checkpoint"]) != parent["sha256"]:
        raise ValueError("訓練 parent SHA 改變")
    if (
        calibration["status"] != "passed"
        or calibration["parent_sha256"] != parent["sha256"]
        or calibration["recipe"] != recipe
    ):
        raise ValueError("需要相符且通過的梯度校準")
    config = {
        "parent": parent,
        "recipe": recipe,
        "calibration": calibration,
        "source_sha256": sha256(Path(root) / "source.json"),
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and read(output / "config.json") != config:
        raise ValueError("續跑配置改變")
    write_json(output / "config.json", config)
    model = checkpoint.load(parent["checkpoint"]).cuda()
    mutable = configure(model, recipe)
    active = {n for n, p in model.named_parameters() if p.requires_grad}
    fixed = fixed_hashes(model, active, mutable)
    model.stage8_manifest["training_lineage"].append(
        {
            "parent": parent["checkpoint"],
            "sha256": parent["sha256"],
            "recipe": recipe,
            "pose_weight": calibration["selected_pose_weight"],
            "initial_ema_updates": 0,
            "source_lock_sha256": config["source_sha256"],
        }
    )
    ema = FrozenEMA(model, mutable_buffers=mutable)
    optimizer = optimizer_for(model, recipe)
    engine, losses, scaler = engine_for(
        model, recipe, calibration["selected_pose_weight"], mutable, optimizer, ema
    )
    event(output, kind="prepare-training-data", phase=recipe["name"])
    pds = dataset("pose", "train", 16, recipe["epochs"])[0]
    dds = dataset("detect", "train", 16, recipe["epochs"])[0]
    steps = math.ceil(len(dds) / 256)
    scheduler = StageWarmupCosineScheduler(
        optimizer,
        stage=recipe["name"],
        epochs=recipe["epochs"],
        steps_per_epoch=steps,
        warmup_epochs=1,
        final_lr_factor=0.5,
    )
    best, history, next_epoch, reason = copy.deepcopy(parent), [], 1, None
    resume = output / "resume.pt"
    if resume.exists():
        if sha256(resume) != read(resume.with_suffix(".json"))["sha256"]:
            raise ValueError("resume SHA 不符")
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError("resume 配置不符")
        model.load_state_dict(saved["live"], strict=True)
        ema.ema.load_state_dict(saved["ema"], strict=True)
        ema.updates = saved["ema_updates"]
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        scheduler.load_state_dict(saved["scheduler"])
        losses.load_state_dict(saved["criterion"])
        history, best, next_epoch, reason = (
            saved[k] for k in ("history", "best", "next_epoch", "reason")
        )
        restore_rng(saved["rng"])
        del saved
    for candidate in (model, ema.ema):
        if fixed_hashes(candidate, active, mutable) != fixed:
            raise ValueError("resume frozen state 不符")
    for epoch in range(next_epoch, recipe["epochs"] + 1):
        if reason is not None:
            break
        source_guard(root)
        configure(model, recipe)
        seed(protocol()["seed"] + epoch)
        started = time.time()
        sums = {"detect_images": 0, "pose_images": 0, "amp_retries": 0}
        for step, (db, pb) in enumerate(
            macros(dds, pds, protocol()["seed"] + epoch), 1
        ):
            if (Path(root) / "STOP").exists() or (output / "STOP").exists():
                raise InterruptedError("STOP；保留最後完整 epoch resume")
            lrs = scheduler.prepare_step()
            report = engine.run(
                detect_batches=db,
                pose_batches=pb,
                record_gradient_statistics=step == 1 or step % 50 == 0,
            )
            scheduler.advance()
            sums["detect_images"] += report.detect_images
            sums["pose_images"] += report.pose_images
            sums["amp_retries"] += report.amp_overflow_retries
            if step == 1 or step % 10 == 0 or step == steps:
                elapsed = time.time() - started
                event(
                    output,
                    kind="training",
                    phase=recipe["name"],
                    epoch=epoch,
                    epochs=recipe["epochs"],
                    step=step,
                    steps=steps,
                    seconds=elapsed,
                    remaining_train_seconds=(steps - step) * elapsed / step,
                    lrs=lrs,
                    ema_updates=ema.updates,
                    criterion=losses.state_dict(),
                    **asdict(report),
                )
        if step != steps or sums["detect_images"] != len(dds):
            raise ValueError("epoch 未涵蓋完整 COCO train")
        for candidate in (model, ema.ema):
            if fixed_hashes(candidate, active, mutable) != fixed:
                raise ValueError("訓練／EMA 改變 frozen tensors")
            manifest_for(candidate)
        training_values = pair(
            ema.ema, output / "training-evaluation" / f"e{epoch:03d}", progress_root=output, operators="training"
        )["metrics"]
        values = pair(
            ema.ema, output / "evaluation" / f"e{epoch:03d}", progress_root=output
        )["metrics"]
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError("AP 非有限")
        violations = {
            k: values[k] - v
            for k, v in parent["metrics"].items()
            if values[k] < v - 0.05
        }
        if not violations and (
            best.get("epoch", 0) == 0 or rank(values) > rank(best["metrics"])
        ):
            path = output / f"best-e{epoch:03d}.pt"
            # Previous interrupted epoch may have created an uncommitted best.
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            checkpoint.save(path, ema.ema)
            best = {
                "checkpoint": str(path.resolve()),
                "sha256": sha256(path),
                "metrics": values,
                "training_metrics": training_values,
                "deployment_minus_training": {k: values[k] - training_values[k] for k in values},
                "epoch": epoch,
                "phase": recipe["name"],
                "candidate": model.stage8_manifest["candidate"],
                "rep_relative_gate": passed(values),
                "ema_updates": ema.updates,
            }
        history.append(
            {
                "epoch": epoch,
                "metrics": values,
                "training_metrics": training_values,
                "deployment_minus_training": {k: values[k] - training_values[k] for k in values},
                "seconds": time.time() - started,
                "lrs": lrs,
                "ema_updates": ema.updates,
                "criterion": losses.state_dict(),
                "violations": violations,
                **sums,
            }
        )
        engine.advance_epoch()
        reason = (
            "safety" if violations else "budget" if epoch == recipe["epochs"] else None
        )
        atomic_torch(
            resume,
            {
                "config": config,
                "live": state_cpu(model),
                "ema": state_cpu(ema.ema),
                "ema_updates": ema.updates,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "criterion": losses.state_dict(),
                "history": history,
                "best": best,
                "next_epoch": epoch + 1,
                "reason": reason,
                "rng": rng_state(),
            },
        )
        write_json(
            resume.with_suffix(".json"),
            {"sha256": sha256(resume), "next_epoch": epoch + 1},
        )
        write_json(output / "history.json", history)
        for old in output.glob("best-e*.pt"):
            if str(old.resolve()) != best["checkpoint"]:
                old.unlink()
                old.with_suffix(".json").unlink(missing_ok=True)
        event(
            output,
            kind="epoch-complete",
            phase=recipe["name"],
            epoch=epoch,
            metrics=values,
            best=best,
            reason=reason,
            seconds=history[-1]["seconds"],
        )
    result = {
        "status": "needs_review" if reason == "safety" else "complete",
        "reason": reason,
        "parent": parent,
        "recipe": recipe,
        "best": best,
        "history": history,
        "resume": str(resume.resolve()),
        "initial_ema_updates": 0,
        "all_frozen_tensors_verified": True,
    }
    write_json(output / "result.json", result)
    event(output, kind="phase-complete", phase=recipe["name"], status=result["status"])
    return result
