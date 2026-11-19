"""形狀適應：沿用 optimize macro／scheduler，固定 qSiLU／D1 部署前向。"""

import copy
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
from binaryqk.run import seed, loader, transfer
from binaryqk.train import FrozenEMA, atomic_torch, state_cpu
from stage7.ops import ProbeOptimizer, fixed_hashes, restore_rng, rng_state
from stage6.ops import MacroEngine
from yolo_combine.contracts import Task
from yolo_combine.joint_loss import NativeTaskLossRouter
from yolo_combine.joint_trainer import StageWarmupCosineScheduler
from activation_lab.training.full35 import _fp32_bbox_iou

from . import model as checkpoint
from .common import RUN, ROOT, event, protocol, read, write, digest, guard, rank, score
from .data import dataset
from .evaluate import pair


class ShapeLoss(NativeTaskLossRouter):
    def __init__(self, model, recipe, shape):
        from ultralytics.utils import loss
        if not getattr(loss.bbox_iou, "_stage7_fp32", False):
            loss.bbox_iou = _fp32_bbox_iou(loss.bbox_iou)
            loss.bbox_iou._stage7_fp32 = True
        super().__init__(model, epochs=recipe["loss_horizon"], imgsz=max(shape))

    def loss_for(self, task, batch):
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            return super().loss_for(task, batch)


def macros(dds, pds, epoch_seed):
    pi = iter(loader(pds, 8, epoch_seed))
    group, cycle = [], 0
    def pose_batch():
        nonlocal pi, cycle
        try:
            return next(pi)
        except StopIteration:
            cycle += 1
            pi = iter(loader(pds, 8, epoch_seed + cycle * 100000))
            return next(pi)
    for db in loader(dds, 8, epoch_seed):
        group.append(db)
        if len(group) == 32:
            yield tuple(group), (pose_batch(), pose_batch())
            group = []
    if group:
        yield tuple(group), (pose_batch(), pose_batch())


def make_engine(model, recipe, shape, weight, optimizer, ema=None):
    losses = ShapeLoss(model, recipe, shape)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    engine = MacroEngine(model=model, losses=losses, optimizer=optimizer, mutable_buffers=set(),
                         reference_batch_size=64, task_weights={Task.DETECT: 1., Task.POSE: weight},
                         gradient_groups=checkpoint.groups(model), scaler=scaler, ema=ema,
                         max_grad_norm=10., max_amp_retries=16,
                         preprocess=lambda task, batch: transfer(batch, "cuda:0"))
    return engine, losses, scaler


def sources(shape, epochs):
    return dataset("detect", "train", shape, epochs=epochs)[0], dataset("pose", "train", shape, epochs=epochs)[0]


def calibrate(spec):
    output, shape = Path(spec["output"]), spec["shape"]
    recipe, cfg = spec["recipe"], protocol()["calibration"]
    seed(protocol()["seed"])
    model = checkpoint.load(spec["parent"]["checkpoint"]).cuda()
    checkpoint.configure(model)
    initial = state_cpu(model)
    engine, _, _ = make_engine(model, recipe, shape, cfg["initial_pose_weight"],
                               ProbeOptimizer(p for p in model.parameters() if p.requires_grad))
    dds, pds = sources(shape, recipe["epochs"])
    iterator = iter(macros(dds, pds, protocol()["seed"] + 90000))
    records = []
    def measure(count, phase):
        ratios = []
        for index in range(count):
            model.load_state_dict(initial, strict=True)
            db, pb = next(iterator)
            result = engine.run(detect_batches=db, pose_batches=pb, record_gradient_statistics=True)
            stats = asdict(result.gradient_statistics)
            if not stats["detect_norm"] > 0 or not stats["pose_norm"] > 0:
                raise ValueError("缺少共享梯度")
            ratio = stats["pose_norm"] / stats["detect_norm"]
            ratios.append(ratio)
            records.append({"phase": phase, "index": index, "ratio": ratio,
                            "pose_weight": engine.task_weights[Task.POSE], **stats})
            event(output, kind="calibration", **records[-1])
        return statistics.median(ratios)
    def quantize(x):
        return max(cfg["bounds"][0], min(cfg["bounds"][1], round(x / cfg["step"]) * cfg["step"]))
    ratio = measure(cfg["macros"], "measure")
    weight = quantize(cfg["initial_pose_weight"] * cfg["target_ratio"] / ratio)
    for attempt in range(cfg["attempts"]):
        engine.task_weights[Task.POSE] = weight
        observed = measure(cfg["confirm"], f"confirm-{attempt}")
        if cfg["band"][0] <= observed <= cfg["band"][1]:
            break
        if attempt + 1 < cfg["attempts"]:
            weight = quantize(weight * cfg["target_ratio"] / observed)
    model.load_state_dict(initial, strict=True)
    if any(not torch.equal(v, model.state_dict()[k].cpu()) for k, v in initial.items()):
        raise ValueError("校準改變 parent")
    passed = cfg["band"][0] <= observed <= cfg["band"][1]
    result = {"status": "passed" if passed else "needs_review", "selected_pose_weight": weight,
              "parent_sha256": spec["parent"]["sha256"], "recipe": recipe, "shape": shape,
              "records": records, "validation_used": False, "parent_restored": True}
    write(output / "result.json", result)
    if not passed:
        raise ValueError("任務梯度比例未通過確認")
    return result


def smoke(spec):
    output, shape, recipe = Path(spec["output"]), spec["shape"], spec["recipe"]
    model = checkpoint.load(spec["parent"]["checkpoint"]).cuda()
    checkpoint.configure(model)
    active = {n for n, p in model.named_parameters() if p.requires_grad}
    initial, frozen = state_cpu(model), fixed_hashes(model, active, set())
    ema = FrozenEMA(model, mutable_buffers=set())
    engine, _, _ = make_engine(model, recipe, shape, spec["calibration"]["selected_pose_weight"],
                               checkpoint.optimizer(model, recipe), ema)
    dds, pds = sources(shape, recipe["epochs"])
    iterator = iter(macros(dds, pds, protocol()["seed"] + 80000))
    torch.cuda.reset_peak_memory_stats()
    start, reports = time.time(), []
    for step in range(2):
        db, pb = next(iterator)
        report = engine.run(detect_batches=db, pose_batches=pb, record_gradient_statistics=True)
        if not all(report.gradient_presence.values()):
            raise ValueError("smoke 缺少共享／雙 head 梯度")
        reports.append(asdict(report)); event(output, kind="smoke", macro=step + 1)
    for candidate in (model, ema.ema):
        if fixed_hashes(candidate, active, set()) != frozen:
            raise ValueError("smoke 改變固定 backbone／Attention／BN")
        checkpoint.verify(candidate)
    changed = [n for n, p in model.named_parameters() if n in active and not torch.equal(initial[n], p.detach().cpu())]
    if not all(any(checkpoint.role(n) == role for n in changed) for role in ("neck", "detect_head", "pose_head")):
        raise ValueError("smoke 未實際更新各訓練區域")
    result = {"status": "passed", "parent_sha256": spec["parent"]["sha256"], "shape": shape,
              "recipe": recipe, "seconds_per_macro": (time.time() - start) / 2,
              "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
              "updated_parameters": len(changed), "reports": reports, "smoke_weights_discarded": True}
    write(output / "result.json", result)
    return result


def train(spec):
    output, shape, recipe, parent = Path(spec["output"]), spec["shape"], spec["recipe"], spec["parent"]
    output.mkdir(parents=True, exist_ok=True)
    guard()
    if digest(ROOT / parent["checkpoint"]) != parent["sha256"]:
        raise ValueError("訓練 parent SHA 改變")
    calibration = spec["calibration"]
    if calibration["status"] != "passed" or calibration["shape"] != shape:
        raise ValueError("沒有相符的任務校準")
    config = {"parent": parent, "shape": shape, "recipe": recipe, "calibration": calibration,
              "source_sha256": digest(RUN / "source.json")}
    if (output / "config.json").exists() and read(output / "config.json") != config:
        raise ValueError("續跑配置改變")
    write(output / "config.json", config)
    seed(protocol()["seed"])
    model = checkpoint.load(parent["checkpoint"]).cuda()
    checkpoint.configure(model)
    active = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = fixed_hashes(model, active, set())
    ema = FrozenEMA(model, mutable_buffers=set())
    optimizer = checkpoint.optimizer(model, recipe)
    engine, losses, scaler = make_engine(model, recipe, shape, calibration["selected_pose_weight"], optimizer, ema)
    dds, pds = sources(shape, recipe["epochs"])
    steps = math.ceil(len(dds) / 256)
    scheduler = StageWarmupCosineScheduler(optimizer, stage=spec["name"], epochs=recipe["epochs"],
                                          steps_per_epoch=steps, warmup_epochs=1, final_lr_factor=.5)
    resume = output / "resume.pt"
    best, history, next_epoch, reason = copy.deepcopy(parent), [], 1, None
    if resume.exists():
        if digest(resume) != read(resume.with_suffix(".json"))["sha256"]:
            raise ValueError("resume SHA 不符")
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError("resume config 不符")
        model.load_state_dict(saved["live"], strict=True); ema.ema.load_state_dict(saved["ema"], strict=True)
        ema.updates = saved["ema_updates"]
        optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        scheduler.load_state_dict(saved["scheduler"]); losses.load_state_dict(saved["criterion"])
        history, best, next_epoch, reason = (saved[k] for k in ("history", "best", "next_epoch", "reason"))
        restore_rng(saved["rng"])
        del saved
    if digest(ROOT / best["checkpoint"]) != best["sha256"]:
        raise ValueError("best SHA 與已提交 resume 不符")
    for candidate in (model, ema.ema):
        if fixed_hashes(candidate, active, set()) != frozen:
            raise ValueError("resume 改變 frozen tensors")
    for epoch in range(next_epoch, recipe["epochs"] + 1):
        if reason is not None:
            break
        guard(); checkpoint.configure(model); seed(protocol()["seed"] + epoch)
        start = time.time(); count = 0
        for step, (db, pb) in enumerate(macros(dds, pds, protocol()["seed"] + epoch), 1):
            if (RUN / "STOP").exists() or (output / "STOP").exists():
                raise InterruptedError("STOP；保留完整 epoch resume")
            lrs = scheduler.prepare_step()
            report = engine.run(detect_batches=db, pose_batches=pb, record_gradient_statistics=step == 1 or step % 50 == 0)
            scheduler.advance(); count += report.detect_images
            if step == 1 or step % 10 == 0 or step == steps:
                event(output, kind="training", candidate=spec["candidate"], phase=spec["name"], epoch=epoch,
                      epochs=recipe["epochs"], step=step, steps=steps, seconds=time.time() - start,
                      lrs=lrs, ema_updates=ema.updates, **asdict(report))
        if step != steps or count != len(dds):
            raise ValueError("沒有完整覆蓋 COCO train")
        for candidate in (model, ema.ema):
            if fixed_hashes(candidate, active, set()) != frozen:
                raise ValueError("訓練／EMA 改變固定張量")
            checkpoint.verify(candidate)
        values = pair(ema.ema, spec["candidate"], shape, output / "evaluation" / f"e{epoch:03d}")["metrics"]
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError("非有限 AP")
        violations = {k: values[k] - v for k, v in parent["metrics"].items() if values[k] < v - .05}
        if not violations and rank(values) > rank(best["metrics"]):
            # A crash before the resume commit must not overwrite its referenced best.
            path = output / f"best-e{epoch:03d}.pt"
            shape_manifest = {"schema": "actshape-stage9-v1", "candidate": spec["candidate"], "shape": shape,
                              "parent": parent, "phase": spec["name"], "epoch": epoch,
                              "recipe": recipe, "pose_weight": calibration["selected_pose_weight"],
                              "source_sha256": config["source_sha256"], "preprocessing": protocol()["preprocessing"]}
            checkpoint.save(path, ema.ema, shape_manifest)
            best = {"checkpoint": str(path.resolve()), "sha256": digest(path), "metrics": values,
                    "epoch": epoch, "candidate": spec["candidate"], "phase": spec["name"]}
        history.append({"epoch": epoch, "metrics": values, "seconds": time.time() - start,
                        "ema_updates": ema.updates, "detect_images": count, "lrs": lrs, "violations": violations})
        engine.advance_epoch()
        reason = "safety" if violations else "budget" if epoch == recipe["epochs"] else None
        atomic_torch(resume, {"config": config, "live": state_cpu(model), "ema": state_cpu(ema.ema),
                             "ema_updates": ema.updates, "optimizer": optimizer.state_dict(),
                             "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
                             "criterion": losses.state_dict(), "history": history, "best": best,
                             "next_epoch": epoch + 1, "reason": reason, "rng": rng_state()})
        write(resume.with_suffix(".json"), {"sha256": digest(resume), "next_epoch": epoch + 1,
                                           "config_sha256": digest(output / "config.json"),
                                           "source_sha256": config["source_sha256"]})
        write(output / "history.json", history)
        # Only this phase's superseded EMA files; the committed best is protected.
        for old in output.glob("best-e*.pt"):
            if old.resolve() != Path(best["checkpoint"]).resolve():
                old.unlink()
                old.with_suffix(".json").unlink(missing_ok=True)
    result = {"status": "complete" if reason != "safety" else "safety_stop", "reason": reason,
              "parent": parent, "recipe": recipe, "shape": shape, "history": history, "best": best,
              "resume": str(resume), "all_frozen_tensors_verified": True}
    write(output / "result.json", result)
    return result
