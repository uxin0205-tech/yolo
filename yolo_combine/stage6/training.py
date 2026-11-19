"""同血緣 Pose 適應及聯合訓練；每段獨立配方、每個完整 epoch 可續跑。"""

import copy
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from binaryqk.run import require_idle_gpu, seed, transfer
from binaryqk.train import FrozenEMA, atomic_torch, state_cpu
from yolo_combine.contracts import Task
from yolo_combine.joint_trainer import StageWarmupCosineScheduler

from . import checkpoint
from .common import event, read, review, sha256, source_guard, write_json
from .data import dataset
from .evaluate import accepted, pair
from .models import configure, manifest_for
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
from .policy import Plateau, eligible, rank, safety, selection_score


def train(output, parent, recipe, calibration=None):
    output = Path(output)
    protocol = review()["protocol"]
    require_idle_gpu()
    torch.set_num_threads(2)
    seed(protocol["seed"])
    config = {
        "parent": parent,
        "recipe": recipe,
        "calibration": calibration,
        "reference_review": sha256(
            Path(__file__).parents[1] / "artifacts/stage6/reference-review.json"
        ),
        "fresh_optimizer": True,
        "initial_ema_updates": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        if read(output / "config.json") != config:
            raise ValueError("resume parent/recipe/calibration changed")
        source_guard(output)
    else:
        write_json(output / "config.json", config)
        source_guard(output, create=True)
    if sha256(parent["checkpoint"]) != parent["sha256"]:
        raise ValueError("phase parent SHA changed")
    if (output / "result.json").exists():
        return read(output / "result.json")
    pose_weight = 1.0
    if recipe["task"] == "joint":
        if (
            not calibration
            or calibration["status"] != "passed"
            or calibration["parent_sha256"] != parent["sha256"]
            or calibration["recipe"] != recipe
        ):
            raise ValueError("joint phase requires matching passed calibration")
        pose_weight = calibration["selected_pose_weight"]
    model = checkpoint.load(parent["checkpoint"]).to("cuda:0")
    mutable = configure(model, recipe)
    active = {n for n, p in model.named_parameters() if p.requires_grad}
    fixed = fixed_hashes(model, active, mutable)
    lineage = copy.deepcopy(model.stage6_manifest.get("training_lineage", []))
    lineage.append(
        {
            "parent": parent["checkpoint"],
            "parent_sha256": parent["sha256"],
            "recipe": recipe,
            "pose_weight": pose_weight,
            "initial_ema_updates": 0,
            "execution_source_lock": str((output / "source.json").resolve()),
            "execution_source_lock_sha256": sha256(output / "source.json"),
        }
    )
    model.stage6_manifest["training_lineage"] = lineage
    ema = FrozenEMA(model, mutable_buffers=mutable)
    optimizer = optimizer_for(model, recipe)
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=protocol["optimizer"]["initial_scale"]
    )
    losses = loss_for_recipe(model, recipe)
    event(output, kind="prepare-training-data", phase=recipe["name"])
    pds = dataset("pose", "train", 16, recipe["epochs"])[0]
    dds = (
        dataset("detect", "train", 32, recipe["epochs"])[0]
        if recipe["task"] == "joint"
        else None
    )
    steps = math.ceil(len(dds) / 256) if dds is not None else math.ceil(len(pds) / 16)
    scheduler = StageWarmupCosineScheduler(
        optimizer,
        stage=recipe["name"],
        epochs=recipe["epochs"],
        steps_per_epoch=steps,
        warmup_epochs=recipe["warmup_epochs"],
        final_lr_factor=recipe["final_factor"],
    )
    plateau = Plateau(recipe)
    plateau.observe(selection_score(parent["metrics"], recipe))
    best, history, next_epoch, reason = copy.deepcopy(parent), [], 1, None
    resume = output / "resume.pt"
    if resume.exists():
        if sha256(resume) != read(resume.with_suffix(".json"))["sha256"]:
            raise ValueError("resume checksum mismatch")
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError("resume config mismatch")
        model.load_state_dict(saved["live"], strict=True)
        ema.ema.load_state_dict(saved["ema"], strict=True)
        ema.updates = saved["ema_updates"]
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        scheduler.load_state_dict(saved["scheduler"])
        losses.load_state_dict(saved["criterion"])
        plateau.load_state_dict(saved["plateau"])
        history, best, next_epoch, reason = (
            saved[k] for k in ("history", "best", "next_epoch", "reason")
        )
        restore_rng(saved["rng"])
        del saved
    for candidate in (model, ema.ema):
        if fixed_hashes(candidate, active, mutable) != fixed:
            raise ValueError("resume modified frozen state")
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
        preprocess=lambda _task, batch: transfer(batch, "cuda:0"),
    )
    for epoch in range(next_epoch, recipe["epochs"] + 1):
        if reason is not None:
            break
        source_guard(output)
        configure(model, recipe)
        seed(protocol["seed"] + epoch)
        started = time.time()
        sums = {"detect_images": 0, "pose_images": 0, "amp_retries": 0}
        for step, (db, pb) in enumerate(macros(dds, pds, protocol["seed"] + epoch), 1):
            if any(
                (p / "STOP").exists()
                for p in (output, output.parent, output.parent.parent)
            ):
                raise InterruptedError("STOP; only complete epoch resume retained")
            lrs = scheduler.prepare_step()
            for group in optimizer.param_groups:
                group["lr"] *= plateau.multiplier
                lrs[group["group_name"]] = group["lr"]
            report = engine.run(
                detect_batches=db,
                pose_batches=pb,
                record_gradient_statistics=dds is not None
                and (step == 1 or step % 50 == 0),
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
        if (
            step != steps
            or (dds is not None and sums["detect_images"] != len(dds))
            or (dds is None and sums["pose_images"] != len(pds))
        ):
            raise ValueError("epoch did not cover the canonical dataset")
        for candidate in (model, ema.ema):
            if fixed_hashes(candidate, active, mutable) != fixed:
                raise ValueError("training/EMA modified frozen tensors")
            manifest_for(candidate)
        eval_dir = output / "evaluation" / f"e{epoch:03d}"
        values = pair(ema.ema, eval_dir, progress_root=output)["metrics"]
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError("nonfinite AP")
        violation = safety(values, parent["metrics"], recipe, epoch)
        selectable = (
            recipe["scope"] == "pose_full"
            or recipe.get("selection") in ("balanced_recovery", "pose_recovery")
            or eligible(values)
        )
        better = (
            not violation
            and selectable
            and rank(values, recipe) > rank(best["metrics"], recipe)
        )
        candidate_path = None
        if better or recipe.get("retain_epoch_candidates"):
            prefix = "candidate" if recipe.get("retain_epoch_candidates") else "best"
            path = output / f"{prefix}-e{epoch:03d}.pt"
            # An interrupted epoch may have saved this candidate before its resume transaction.
            if path.exists():
                path.unlink()
                path.with_suffix(".json").unlink(missing_ok=True)
            checkpoint.save(path, ema.ema)
            candidate_path = str(path.resolve())
        if better:
            best = {
                "checkpoint": str(path.resolve()),
                "sha256": sha256(path),
                "metrics": values,
                "epoch": epoch,
                "phase": recipe["name"],
                "ema_updates": ema.updates,
            }
        exhausted = plateau.observe(selection_score(values, recipe))
        history.append(
            {
                "epoch": epoch,
                "metrics": values,
                "seconds": time.time() - started,
                "lrs": lrs,
                "ema_updates": ema.updates,
                "criterion": losses.state_dict(),
                "safety": violation,
                "candidate_checkpoint": candidate_path,
                **sums,
            }
        )
        engine.advance_epoch([Task.POSE] if recipe["task"] == "pose" else None)
        reason = (
            "safety: " + violation
            if violation
            else "accepted"
            if recipe["scope"] != "pose_full" and accepted(best["metrics"])
            else "plateau"
            if exhausted
            else "budget"
            if epoch == recipe["epochs"]
            else None
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
                "plateau": plateau.state_dict(),
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
        "status": "needs_review"
        if reason and reason.startswith("safety:")
        else "complete",
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
    event(
        output,
        kind="phase-complete",
        phase=recipe["name"],
        status=result["status"],
        reason=reason,
    )
    return result
