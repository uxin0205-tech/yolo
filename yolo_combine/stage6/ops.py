"""與 optimize 相同的 macro loss 尺度、分組 AdamW、BN 重試及 fresh EMA。"""

import hashlib
import random

import numpy as np
import torch
from binaryqk.run import loader
from yolo_combine.joint_loss import MacroStepEngine, NativeTaskLossRouter

from .common import review
from .models import role


class ProgressiveLoss(NativeTaskLossRouter):
    def __init__(self, model, recipe, amp=True):
        super().__init__(
            model,
            epochs=recipe["loss_horizon"],
            imgsz=640,
            detect_overrides={"epochs": 0} if recipe["task"] == "pose" else None,
        )
        self.amp = amp

    def loss_for(self, task, batch):
        with torch.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.amp and self.device.type == "cuda",
        ):
            return super().loss_for(task, batch)


def loss_for_recipe(model, recipe):
    if "kd" in recipe:
        from .kd import SpatialLoss

        return SpatialLoss(model, recipe)
    return ProgressiveLoss(model, recipe)


class MacroEngine(MacroStepEngine):
    """AMP 重試恢復本 macro 開始前的 BN，避免重複累積統計。"""

    def __init__(self, *, mutable_buffers, **kwargs):
        super().__init__(**kwargs)
        self.mutable_buffers = set(mutable_buffers)
        self.before = None

    def restore(self):
        buffers = dict(self.model.named_buffers())
        with torch.no_grad():
            for name, value in self.before.items():
                buffers[name].copy_(value)

    def run(self, *, _amp_overflow_retries=0, **kwargs):
        outer = _amp_overflow_retries == 0
        if outer:
            self.before = {
                n: b.detach().clone()
                for n, b in self.model.named_buffers()
                if n in self.mutable_buffers
            }
        else:
            self.restore()
        try:
            return super().run(_amp_overflow_retries=_amp_overflow_retries, **kwargs)
        except BaseException:
            self.restore()
            raise
        finally:
            if outer:
                self.before = None


class ProbeOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {"lr": 0.0})

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("probe forbids closures")


def optimizer_for(model, recipe):
    config = review()["protocol"]["optimizer"]
    groups = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups.setdefault(
                (role(name), p.ndim >= 2 and not name.endswith(".bias")), []
            ).append(p)
    return torch.optim.AdamW(
        [
            {
                "params": params,
                "lr": recipe["lrs"][r],
                "weight_decay": config["weight_decay"] if decay else 0.0,
                "group_name": f"{r}-{int(decay)}",
            }
            for (r, decay), params in groups.items()
        ],
        betas=tuple(config["betas"]),
    )


def gradient_groups(model):
    groups = {"shared": [], "detect_head": [], "pose_head": []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            r = role(name)
            groups[r if r in ("detect_head", "pose_head") else "shared"].append(p)
    return groups


def fixed_hashes(model, trainable, mutable):
    return {
        n: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for n, v in model.state_dict().items()
        if n not in trainable and n not in mutable
    }


def macros(detect_ds, pose_ds, epoch_seed):
    pi = iter(loader(pose_ds, 16, epoch_seed))
    if detect_ds is None:
        for pb in pi:
            yield (), (pb,)
        return
    config = review()["protocol"]
    cycle, group = 0, []
    for db in loader(detect_ds, config["detect_microbatch"], epoch_seed):
        group.append(db)
        if len(group) < config["detect_microbatches"]:
            continue
        try:
            pb = next(pi)
        except StopIteration:
            cycle += 1
            pi = iter(loader(pose_ds, 16, epoch_seed + 100000 * cycle))
            pb = next(pi)
        yield tuple(group), (pb,)
        group = []
    if group:
        try:
            pb = next(pi)
        except StopIteration:
            pi = iter(loader(pose_ds, 16, epoch_seed + 100000 * (cycle + 1)))
            pb = next(pi)
        yield tuple(group), (pb,)


def rng_state():
    ns = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (ns[0], ns[1].tolist(), *ns[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(value):
    random.setstate(value["python"])
    ns = value["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])
