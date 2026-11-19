"""訓練限定的雙教師空間蒸餾；教師不進學生、EMA 或部署 checkpoint。"""

import copy
import io
import math
import statistics
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from binaryqk.run import loader, seed, transfer
from binaryqk.train import FrozenEMA, state_cpu
from yolo_combine.contracts import Task

from yolo_attention import stage3_checkpoint

from . import checkpoint
from .common import event, review, sha256, write_json
from .data import dataset
from .models import configure, manifest_for, native_pose
from .ops import (
    MacroEngine,
    ProgressiveLoss,
    fixed_hashes,
    gradient_groups,
    macros,
    optimizer_for,
)

SITES = (16, 19, 22)


def spatial_loss(student, teacher):
    """每圖三尺度平均；不同通道可比較，空間座標不允許插值。"""
    if len(student) != 3 or len(teacher) != 3:
        raise ValueError("exactly three feature scales required")
    terms = []
    for s, t in zip(student, teacher):
        if (
            s.ndim != 4
            or t.ndim != 4
            or s.shape[0] != t.shape[0]
            or s.shape[2:] != t.shape[2:]
        ):
            raise ValueError("teacher/student spatial coordinates must match")
        sa = F.normalize(s.float().square().mean(1).flatten(1), dim=1, eps=1e-6)
        ta = F.normalize(
            t.detach().float().square().mean(1).flatten(1), dim=1, eps=1e-6
        )
        terms.append((sa - ta).square().sum(1))
    result = torch.stack(terms).mean(0)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite spatial KD")
    return result


@contextmanager
def capture(layers):
    values, handles = {}, []
    try:
        for index in SITES:

            def hook(module, args, output, index=index):
                if index in values:
                    raise RuntimeError("repeated shared feature forward")
                values[index] = output

            handles.append(layers[index].register_forward_hook(hook))
        yield values
    finally:
        for handle in handles:
            handle.remove()
        values.clear()


class SpatialLoss(ProgressiveLoss):
    def __init__(self, model, recipe):
        super().__init__(model, recipe)
        self.kd = copy.deepcopy(recipe["kd"])
        if self.kd["sites"] != list(SITES) or self.kd["version"] != 1:
            raise ValueError("unknown spatial KD contract")
        self.mu = {Task(k): float(v) for k, v in self.kd["mu"].items()}
        if set(self.mu) != set(Task) or any(
            not math.isfinite(v) or v < 0 for v in self.mu.values()
        ):
            raise ValueError("invalid fixed KD coefficients")
        source = review()
        for task in ("detect", "pose"):
            info = self.kd["teachers"][task]
            if info["checkpoint"] != source[task + "_parent"] or info[
                "sha256"
            ] != sha256(info["checkpoint"]):
                raise ValueError("teacher source changed")
        with torch.random.fork_rng(devices=[self.device.index or 0]):
            detect, _ = stage3_checkpoint.load(
                self.kd["teachers"]["detect"]["checkpoint"]
            )
            self.teachers = {Task.DETECT: detect, Task.POSE: native_pose()}
            for teacher in self.teachers.values():
                teacher.to(self.device).eval().requires_grad_(False)

    def parts(self, task, batch):
        task = Task(task)
        teacher = self.teachers[task]
        if teacher.training or any(p.requires_grad for p in teacher.parameters()):
            raise ValueError("teacher must remain frozen/eval")
        with (
            torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp),
            capture(teacher.model) as tf,
        ):
            with torch.no_grad():
                teacher(batch["img"])
            with capture(self.model.graph.model) as sf:
                native = super().loss_for(task, batch)
                features = [sf[i] for i in SITES]
                kd = spatial_loss(features, [tf[i] for i in SITES]).sum()
        return native, kd, features

    def loss_for(self, task, batch):
        native, kd, _ = self.parts(task, batch)
        return replace(native, raw_total=native.raw_total + self.mu[Task(task)] * kd)

    def state_dict(self):
        return {**super().state_dict(), "spatial_kd": self.kd}

    def load_state_dict(self, state):
        if state.get("spatial_kd") != self.kd:
            raise ValueError("resume KD contract changed")
        super().load_state_dict(state)

    def verify_teachers(self, hashes):
        for task, teacher in self.teachers.items():
            if teacher.training or any(
                p.grad is not None for p in teacher.parameters()
            ):
                raise ValueError("teacher gradients or training state changed")
            if fixed_hashes(teacher, set(), set()) != hashes[task]:
                raise ValueError("teacher tensors changed")
        for layers in [
            self.model.graph.model,
            *(t.model for t in self.teachers.values()),
        ]:
            if any(layers[i]._forward_hooks for i in SITES):
                raise ValueError("temporary KD hook leaked")


def calibrate_and_smoke(output, parent, recipe, kd_spec):
    """先 μ 校準，再兩個真實更新；所有 probe 均丟棄，訓練另載 parent。"""
    output = Path(output)
    seed(review()["protocol"]["seed"])
    if sha256(parent["checkpoint"]) != parent["sha256"]:
        raise ValueError("KD calibration parent changed")
    model = checkpoint.load(parent["checkpoint"]).cuda()
    mutable = configure(model, recipe)
    initial = state_cpu(model)
    active = {n for n, p in model.named_parameters() if p.requires_grad}
    fixed = fixed_hashes(model, active, mutable)
    kd = {k: copy.deepcopy(kd_spec[k]) for k in ("sites", "teachers", "version")}
    kd["mu"] = {"detect": 0.0, "pose": 0.0}
    losses = SpatialLoss(model, {**recipe, "kd": kd})
    teacher_hashes = {
        t: fixed_hashes(m, set(), set()) for t, m in losses.teachers.items()
    }
    event(output, kind="prepare-kd-calibration-data")
    dds = dataset("detect", "train", 32, recipe["epochs"])[0]
    pds = dataset("pose", "train", 16, recipe["epochs"])[0]
    rows, mu = [], {}

    def norm(grads):
        return float(
            torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()
        )

    for task, ds, batch_size in ((Task.DETECT, dds, 32), (Task.POSE, pds, 16)):
        ratios = []
        for index, raw in enumerate(loader(ds, batch_size, 90123)):
            model.load_state_dict(initial, strict=True)
            native, kd_loss, features = losses.parts(task, transfer(raw, "cuda:0"))
            ng = norm(
                torch.autograd.grad(native.raw_total, features, retain_graph=True)
            )
            kg = norm(torch.autograd.grad(kd_loss, features, retain_graph=True))
            if not math.isfinite(ng / max(kg, 1e-30)) or min(ng, kg) <= 0:
                raise ValueError("nonfinite/zero KD calibration gradient")
            # KD 必須直接驅動共享 neck；不以 native loss 的更新代替這項證據。
            neck = model.graph.model[19].cv1.conv.weight
            neck_grad = torch.autograd.grad(kd_loss, neck)[0]
            if not torch.isfinite(neck_grad).all() or not neck_grad.abs().sum() > 0:
                raise ValueError("KD does not reach shared neck")
            ratios.append(ng / kg)
            rows.append(
                {
                    "task": task.value,
                    "batch": index,
                    "native_norm": ng,
                    "kd_norm": kg,
                    "kd_neck_norm": norm([neck_grad]),
                }
            )
            event(output, kind="kd-calibration", **rows[-1])
            del native, kd_loss, features, neck_grad
            if index + 1 == kd_spec["calibration_batches_per_task"]:
                break
        value = kd_spec["target_feature_gradient_ratio"] * statistics.median(ratios)
        fractions = [value / r for r in ratios]
        if not 0 < value < kd_spec["mu_upper_bound"] or any(
            not kd_spec["confirmation_band"][0] <= f <= kd_spec["confirmation_band"][1]
            for f in fractions
        ):
            raise ValueError("KD coefficient/gradient fraction outside reviewed bounds")
        mu[task.value] = value
    model.load_state_dict(initial, strict=True)
    losses.kd["mu"] = mu
    losses.mu = {Task(k): v for k, v in mu.items()}
    selected_recipe = {**recipe, "kd": copy.deepcopy(losses.kd)}
    ema = FrozenEMA(model, mutable_buffers=mutable)
    optimizer = optimizer_for(model, selected_recipe)
    engine = MacroEngine(
        model=model,
        losses=losses,
        optimizer=optimizer,
        mutable_buffers=mutable,
        reference_batch_size=64,
        task_weights={Task.DETECT: 1.0, Task.POSE: 0.075},
        gradient_groups=gradient_groups(model),
        scaler=torch.amp.GradScaler("cuda", init_scale=1024),
        ema=ema,
        max_grad_norm=10.0,
        max_amp_retries=16,
        preprocess=lambda _task, batch: transfer(batch, "cuda:0"),
    )
    from dataclasses import asdict

    reports = []
    for i, (db, pb) in enumerate(macros(dds, pds, 90124)):
        reports.append(
            asdict(
                engine.run(
                    detect_batches=db, pose_batches=pb, record_gradient_statistics=True
                )
            )
        )
        event(output, kind="kd-smoke", macro=i + 1, **reports[-1])
        if i == 1:
            break
    losses.verify_teachers(teacher_hashes)
    for candidate in (model, ema.ema):
        if fixed_hashes(candidate, active, mutable) != fixed:
            raise ValueError("KD smoke changed frozen state/EMA")
        manifest_for(candidate)
        if set(candidate.state_dict()) != set(initial):
            raise ValueError("KD leaked teacher tensors into student/EMA")
    changed = [
        n
        for n, p in model.named_parameters()
        if n in active and not torch.equal(p.detach().cpu(), initial[n])
    ]
    if not any(n.startswith("graph.model.19.") for n in changed):
        raise ValueError("smoke did not update shared neck")
    buffer = io.BytesIO()
    torch.save(losses.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    losses.load_state_dict(state)
    if state != losses.state_dict():
        raise ValueError("KD criterion roundtrip differs")
    result = {
        "status": "passed",
        "parent_sha256": parent["sha256"],
        "recipe": selected_recipe,
        "mu": mu,
        "calibration": rows,
        "smoke": reports,
        "ema_updates": ema.updates,
        "teacher_tensors_unchanged": True,
        "teacher_gradients": False,
        "hooks_removed": True,
        "criterion_weights_only_roundtrip": True,
        "student_and_ema_fixed_verified": True,
        "changed_parameters": changed,
        "validation_used": False,
        "smoke_updates_discarded": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(output / "result.json", result)
    return result
