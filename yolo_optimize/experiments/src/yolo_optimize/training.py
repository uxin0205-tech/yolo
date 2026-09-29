"""第一輪 BEST 延續、HOG 對照及物理 batch 測速；不改動 final。

此模組只在呼叫 run_training/run_benchmark 時執行工作。訓練快照完整保存，
不提供一般自動 resume；僅允許明示且嚴格核對的 EMA 診斷 E1 前綴採用，禁止覆寫既有 run。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any, Callable, Mapping

from . import runtime  # 先啟用唯讀 final 的 import path
from .hog import HOGAuxiliary
from .repconv import install_layer17, evaluation_copy
from .masf_bridge import disable_shared_masf
from .training_safety import FixedStateEMA, RetrySafeMacroStepEngine
from .ema_diagnostic import parent_ema_updates
import torch
from torch import nn
from yolo_combine.contracts import Task
from yolo_combine.early_stop import StageEarlyStopping
from yolo_combine.experiment_log import ExperimentLogger
from yolo_combine.formal_training import reseed_loader_for_epoch, seed_everything
from yolo_combine.hardware_contract import HardwareContractGuard
from yolo_combine.joint_data import JointEpochScheduler, TaskLoaderSettings, build_task_loader
from yolo_combine.joint_loss import JointMacroPlan, NativeTaskLossRouter, TaskLossResult
from yolo_combine.joint_trainer import JointEpochRunner, StageWarmupCosineScheduler
from yolo_combine.metrics import AccuracyGate, CheckpointSelectors, GATE_METRICS
from yolo_combine.resume import TrainingProgress, save_inference_weights, save_training_snapshot
from yolo_combine._resume_impl import _rng_state, _restore_rng
from yolo_combine.stage_policy import JointStage, apply_stage, build_joint_optimizer
from yolo_combine.validation import JointValidator, ValidationSettings


SCOPE = JointStage(
    name="j3", task_mode="joint", epochs=10, patience=4,
    backbone_start_layer=11, tune_attention=False, warmup_epochs=1,
    learning_rates={"backbone": 0.0, "neck": 1e-5, "masf": 0.0,
                    "attention": 0.0, "detect_head": 2.5e-5, "pose_head": 2.5e-5},
)


def _portable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    return value


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _new_run(path: str | Path) -> Path:
    path = runtime.output_path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


class EventSink:
    def __init__(self, root: Path, callback: Callable[[dict], None] | None):
        self.path = root / "progress.jsonl"
        self.callback = callback

    def __call__(self, event: dict) -> None:
        event = _portable({"time_unix": time.time(), **event})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        if self.callback is not None:
            self.callback(event)


class TrainingModel(nn.Module):
    """aux 歸訓練圖管理；部署與驗證只取 base。"""

    def __init__(self, base: nn.Module, aux: nn.Module | None = None):
        super().__init__()
        self.base = base
        self.aux = aux

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def contract(self):
        contract = {"model_kind": "direction1_training", "schema_version": 1,
                "base": self.base.contract(),
                "auxiliary": None if self.aux is None else {
                    "kind": "raw_p3_hog", "channels": self.aux.projection.in_channels,
                    "bins": self.aux.bins, "cell_size": self.aux.cell_size}}
        if hasattr(self.base, '_direction1_repconv'):
            contract['reparameterization'] = dict(self.base._direction1_repconv)
        if hasattr(self.base, '_direction1_masf_off'):
            contract['masf_intervention'] = dict(self.base._direction1_masf_off)
        return contract


@contextmanager
def capture_raw_p3(base):
    """短期 pre-hook：不修改特徵、不 detach、不殘留到 EMA／驗證。"""
    captured = []
    masf = base.graph.model[16].p3_masf

    def hook(_module, inputs):
        captured.append(inputs[0])

    handle = masf.register_forward_pre_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def continuation_horizons(payload: Mapping) -> dict[str, int]:
    """由 parent 的實際 stage 與原始政策重建 E2E horizon，核對當下比例。"""
    config = payload["resolved_config"]
    policies = config["stage_policies"]
    stages = list(config["stages"])
    active = payload["progress"]["stage"]
    if config.get("enable_j3") or active == "j3":
        if "j3" not in stages:
            stages.append("j3")
    if active not in stages:
        raise ValueError("parent active stage 無法由原 config 重建")
    horizons = {
        "detect": sum(int(policies[name]["epochs"]) for name in stages
                      if policies[name]["task_mode"] == "joint"),
        "pose": sum(int(policies[name]["epochs"]) for name in stages),
    }
    for task, horizon in horizons.items():
        state = payload["criteria_state"][task]
        if not state.get("end2end") or horizon < 1:
            raise ValueError("本輪只接受可核對的 YOLO26 progressive E2E criterion")
        expected = max(1 - int(state["updates"]) / max(horizon - 1, 1), 0)
        expected = expected * (float(state["o2m_copy"]) - float(state["final_o2m"])) + float(state["final_o2m"])
        if not math.isclose(expected, float(state["o2m"]), abs_tol=1e-10):
            raise ValueError(f"{task} parent E2E horizon 與 saved o2m 不符")
        if not math.isclose(float(state["o2o"]), float(state["total"]) - expected, abs_tol=1e-10):
            raise ValueError(f"{task} parent E2E o2o 狀態不符")
    return horizons


def _parent_criteria(path: Path, base) -> dict:
    # mmap 避免將未使用的舊 optimizer tensors 全部讀進實體記憶體；不安裝舊 optimizer。
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("checkpoint_kind") != "full_resume" or payload.get("contract") != base.contract():
        raise ValueError("paired parent 必須是相同架構的 full-resume snapshot")
    ema_state = payload["ema_state"]
    live = base.state_dict()
    if set(live) != set(ema_state) or any(
        not torch.equal(value.detach().cpu(), ema_state[name]) for name, value in live.items()
    ):
        raise ValueError("selected inference EMA 與 paired full-resume EMA 不一致")
    parent_record = payload["best_state"]["last"]
    if int(parent_record["epoch"]) != int(payload["progress"]["joint_epochs_completed"]) - 1:
        raise ValueError("paired snapshot 的 last metrics epoch 與 progress 不一致")
    parent_metrics = {name: float(parent_record["metrics"][name]) for name in GATE_METRICS}
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in parent_metrics.values()):
        raise ValueError("paired snapshot parent 八 AP 無效")
    return {"path": str(path), "horizons": continuation_horizons(payload),
            "state": _portable(payload["criteria_state"]),
            **({"ema_updates": payload["ema_updates"]} if "ema_updates" in payload else {}),
            "parent_metrics": parent_metrics,
            "parent_progress": _portable(payload["progress"])}


def ema_initial_updates(mode: str, paired_snapshot: Mapping) -> int:
    """age 與權重起點分離；parent age 僅接受已配對 snapshot 的原始頂層欄位。"""
    if mode not in ("fresh", "parent"):
        raise ValueError("ema_age_mode 必須是 fresh 或 parent")
    return 0 if mode == "fresh" else parent_ema_updates(paired_snapshot)


def hog_multiplier(epoch: int, macro: int, macros_per_epoch: int) -> float:
    """epoch 為 0-based：E1 關、E2 ramp、E3–8 hold、E9–10 關。"""
    if not 0 <= epoch < 10 or not 0 <= macro < macros_per_epoch:
        raise ValueError("HOG schedule counter 越界")
    if epoch == 1:
        return (macro + 1) / macros_per_epoch
    return 1.0 if 2 <= epoch <= 7 else 0.0


class HOGTaskLossRouter:
    def __init__(self, native, model: TrainingModel, *, device, amp: bool):
        self.native = native
        self.model = model
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.mu = 0.0
        self.calibration = None
        self.last_aux = {}

    def reset_attempt_statistics(self):
        """清除單次 forward attempt 的日誌，不改 μ、criterion 或梯度。"""
        self.last_aux = {}

    def _parts(self, task, batch, *, force_aux=False):
        active = self.model.aux is not None and (self.mu > 0 or force_aux)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            if not active:
                return self.native.loss_for(task, batch), None, None
            with capture_raw_p3(self.model.base) as captured:
                native = self.native.loss_for(task, batch)
            if len(captured) != 1 or not captured[0].requires_grad:
                raise RuntimeError("raw P3 必須恰好捕捉一次且可微分")
            raw = captured[0]
            aux = self.model.aux(raw, batch["img"], batch["bboxes"], batch["batch_idx"])
            return native, aux, raw

    def loss_for(self, task, batch):
        native, aux, _raw = self._parts(task, batch)
        if aux is None:
            return native
        if not bool(torch.isfinite(aux.loss_sum)):
            raise FloatingPointError("HOG loss 非有限")
        item = self.last_aux.setdefault(Task(task).value, {
            "loss_per_image_mean": 0.0, "mu": self.mu, "valid_cells": 0,
            "images_with_valid_cells": 0, "images": 0, "physical_batch_sizes": [],
        })
        count = item["images"] + native.actual_batch_size
        item["loss_per_image_mean"] = (item["loss_per_image_mean"] * item["images"]
            + float(aux.loss_sum.detach().cpu())) / count
        item["images"] = count
        item["physical_batch_sizes"].append(native.actual_batch_size)
        item["valid_cells"] += int(aux.targets.valid_mask.sum().item())
        item["images_with_valid_cells"] += int(aux.targets.valid_mask.flatten(1).any(1).sum().item())
        return TaskLossResult(task=native.task,
                              raw_total=native.raw_total + self.mu * aux.loss_sum,
                              components=native.components,
                              actual_batch_size=native.actual_batch_size)

    def advance_epoch(self, tasks=None):
        self.native.advance_epoch(tasks)

    def state_dict(self):
        return {"schema_version": 1, "native": self.native.state_dict(),
                "mu": self.mu, "calibration": self.calibration}

    def load_state_dict(self, state):
        if state.get("schema_version") != 1:
            raise ValueError("HOG router snapshot schema 不符")
        self.native.load_state_dict(state["native"])
        self.mu = float(state["mu"])
        self.calibration = state["calibration"]


@dataclass
class RunState:
    source: Any
    model: TrainingModel
    optimizer: Any
    ema: Any
    scaler: Any
    router: HOGTaskLossRouter
    detect: Any
    pose: Any
    guard: Any
    scheduler: Any
    engine: Any
    metadata: dict
    macros: int
    device: torch.device

    def training_mode(self):
        apply_stage(self.model.base, recovery_scope(self.metadata['variant'], self.metadata.get('base_lr_scale', 1.0)))
        if self.model.aux is not None:
            self.model.aux.train()


def scaled_base_scope(scale=1.0):
    if type(scale) not in (int, float) or scale not in (1.0, 0.25):
        raise ValueError("base_lr_scale 僅接受既定對照 1 或單變因候選 0.25")
    return replace(SCOPE, learning_rates={name: value * scale for name, value in SCOPE.learning_rates.items()})


def recovery_scope(variant, scale=1.0):
    scope = scaled_base_scope(scale)
    if variant == 'heads':
        scope = replace(scope, learning_rates={**scope.learning_rates, 'neck': 0.0})
    return scope


def _build(config, checkpoint, paired_snapshot, data, physical, variant, device, *, ema_age_mode="fresh",
           base_lr_scale=1.0) -> RunState:
    if physical not in (32, 64, 128) or variant not in ("native", "hog", "rep17", "masf_off", "heads"):
        raise ValueError("physical batch 必須 32/64/128；variant 必須 native/hog/rep17/masf_off")
    if ema_age_mode not in ("fresh", "parent"):
        raise ValueError("ema_age_mode 必須是 fresh 或 parent")
    device = torch.device(device)
    seed_everything(0)
    source, base, factory_report, loaded = runtime.load_model(config, checkpoint, device)
    parent = _parent_criteria(Path(paired_snapshot), base)
    parent_initial_digest = _state_digest(base)
    reparameterization = install_layer17(base) if variant == 'rep17' else None
    masf_intervention = disable_shared_masf(base) if variant == 'masf_off' else None
    if masf_intervention is not None:
        base._direction1_masf_off = masf_intervention
    initial_ema_age = ema_initial_updates(ema_age_mode, parent)
    scope = recovery_scope(variant, base_lr_scale)
    optimizer, optimizer_report = build_joint_optimizer(
        base, scope, optimizer_name="AdamW", weight_decay=0.00027, beta1=0.948, beta2=0.999,
    )
    aux = None
    if variant == "hog":
        # aux 初始化不消耗 native 臂的 RNG trace；CPU 建立後移至 device。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1701)
            aux = HOGAuxiliary(base.graph.model[16].p3_masf.channels).to(device)
        for decay in (True, False):
            selected = [(name, parameter) for name, parameter in aux.named_parameters()
                        if (parameter.ndim > 1 and not name.endswith(".bias")) == decay]
            if selected:
                optimizer.add_param_group({
                    "params": [parameter for _, parameter in selected],
                    "param_names": tuple(f"aux.{name}" for name, _ in selected),
                    "group_name": f"aux.{'decay' if decay else 'no_decay'}", "role": "aux",
                    "lr": 3e-4, "weight_decay": 0.00027 if decay else 0.0,
                })
    # 讓 full-resume optimizer manifest 與 wrapper 的正式參數名一致。
    for group in optimizer.param_groups:
        if group["role"] != "aux":
            group["param_names"] = tuple(f"base.{name}" for name in group["param_names"])
    model = TrainingModel(base, aux)
    horizons = parent["horizons"]
    native = NativeTaskLossRouter(base, epochs=horizons["pose"], imgsz=config.imgsz,
                                 detect_overrides={"epochs": horizons["detect"]},
                                 pose_overrides={"epochs": horizons["pose"]})
    native.load_state_dict(parent["state"])
    router = HOGTaskLossRouter(native, model, device=device, amp=config.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    ema = FixedStateEMA(model, decay=0.9999, tau=2000, updates=initial_ema_age)
    seed_everything(0)
    detect = build_task_loader(base, data_yaml=data[0],
                              settings=TaskLoaderSettings.for_detect(batch_size=physical,
                                  workers=config.detect_workers, imgsz=config.imgsz, seed=0),
                              device=device, registry=config.registry)
    pose = build_task_loader(base, data_yaml=data[1],
                            settings=TaskLoaderSettings.for_pose(batch_size=16,
                                workers=config.pose_workers, imgsz=config.imgsz, seed=0),
                            device=device, registry=config.registry)
    macros = math.ceil(len(detect.loader) / (256 // physical))
    scheduler = StageWarmupCosineScheduler(optimizer, stage="recovery", epochs=10,
        steps_per_epoch=macros, warmup_epochs=1, warmup_start_factor=0.1, final_lr_factor=0.5)
    groups = {"shared": [], "detect_head": [], "pose_head": [], "hog_auxiliary": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = ("hog_auxiliary" if name.startswith("aux.") else "detect_head" if ".detect_head." in name
                else "pose_head" if ".pose_head." in name else "shared")
        groups[role].append(parameter)
    guard = HardwareContractGuard.capture(base)
    engine = RetrySafeMacroStepEngine(model=model, losses=router, optimizer=optimizer,
        reference_batch_size=64, task_weights={Task.DETECT: 1.0, Task.POSE: 0.25},
        gradient_groups=groups, scaler=scaler, ema=ema, max_grad_norm=10.0,
        max_amp_retries=config.amp_max_overflow_retries,
        preprocess=lambda task, batch: detect.preprocess(batch) if task is Task.DETECT else pose.preprocess(batch))
    metadata = {"variant": variant, "parent": str(checkpoint), "loaded": asdict(loaded),
        "criteria_continuation": parent, "factory": factory_report.as_dict(),
        "optimizer": asdict(optimizer_report), "scope": asdict(scope), "base_lr_scale": base_lr_scale,
        "optimizer_hyperparameters": {
            "name": type(optimizer).__name__, "fresh_state": True,
            "betas": list(optimizer.defaults["betas"]), "eps": optimizer.defaults["eps"],
            "decay_weight_decay": 0.00027, "no_decay_weight_decay": 0.0,
            "auxiliary_lr": 3e-4 if aux is not None else None,
            "groups": [{"name": group["group_name"], "role": group["role"],
                        "base_lr": group["lr"], "weight_decay": group["weight_decay"]}
                       for group in optimizer.param_groups],
        },
        "loaders": {
            "detect": {"overrides": detect.settings.overrides(detect.data_yaml),
                       "runtime_num_workers": detect.loader.num_workers,
                       "runtime_batch_size": detect.loader.batch_size,
                       "dataset_images": len(detect.dataset)},
            "pose": {"overrides": pose.settings.overrides(pose.data_yaml),
                     "runtime_num_workers": pose.loader.num_workers,
                     "runtime_batch_size": pose.loader.batch_size,
                     "dataset_images": len(pose.dataset)},
        },
        "amp": {"requested": bool(config.amp), "enabled": scaler.is_enabled(),
                "autocast_dtype": "float16" if router.amp else None,
                "max_overflow_retries": engine.max_amp_retries,
                "retry_bn_rng_rollback": True},
        "gradient_clip_norm": engine.max_grad_norm,
        "ema": {"decay": 0.9999, "tau": 2000, "updates_at_start": initial_ema_age,
                "age_mode": ema_age_mode,
                "fixed_state_exact_copy": True,
                "fixed_state_names": list(ema.fixed_state_names)},
        "scheduler": {"horizon_epochs": 10, "warmup_epochs": 1,
                      "warmup_start_factor": 0.1, "final_lr_factor": 0.5,
                      "steps_per_epoch": macros},
        "physical_batch": physical, "detect_logical": 128, "detect_per_macro": 256,
        "pose_per_macro": 16, "reference_batch": 64, "seed": 0,
        "scheduler_horizon": 10, "warmup_epochs": 1,
        "initial_native_criterion": native.state_dict(),
        "base_state_sha256": _state_digest(base),
        "parent_initial_state_sha256": parent_initial_digest,
        "reparameterization": reparameterization,
        "masf_intervention": masf_intervention,
        "deployment_auxiliary": False, "automatic_resume_supported": False}
    return RunState(source, model, optimizer, ema, scaler, router, detect, pose, guard,
                    scheduler, engine, _portable(metadata), macros, device)


def _state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _reseed(state, epoch):
    return {"detect": reseed_loader_for_epoch(state.detect.loader, seed=0, epoch=epoch, offset=0),
            "pose": reseed_loader_for_epoch(state.pose.loader, seed=0, epoch=epoch, offset=1)}


def _raw_macros(state, physical):
    return JointEpochScheduler(detect_loader=state.detect.loader, pose_loader=state.pose.loader,
                               detect_batches_per_macro=256 // physical)


def calibration_coefficient(native_squared: float, auxiliary_squared: float, target=0.05,
                            *, task_squared=None) -> float:
    if not all(math.isfinite(value) and value > 1e-24 for value in (native_squared, auxiliary_squared)):
        raise FloatingPointError("HOG 校準 native/aux 梯度為零或非有限；禁止猜測 μ")
    mu = target * math.sqrt(native_squared / auxiliary_squared)
    if not math.isfinite(mu) or mu <= 0:
        raise FloatingPointError("HOG 校準 μ 無效")
    if task_squared is not None:
        lower, upper = 0.0, math.inf
        for native, auxiliary in task_squared:
            if not all(math.isfinite(x) and x > 1e-24 for x in (native, auxiliary)):
                raise FloatingPointError("HOG 任務校準梯度為零或非有限")
            ratio = math.sqrt(auxiliary / native)
            lower = max(lower, 0.02 / ratio)
            upper = min(upper, 0.10 / ratio)
        if lower > upper or not math.isfinite(upper):
            raise FloatingPointError("HOG 共用 μ 的任務 2–10% 區間無可行交集")
        # 保留原整體目標；只在必要時投影到任務共同可行區間。
        # 邊界向內一個 ULP，避免浮點乘回後落到觀察區間之外。
        if mu < lower:
            mu = min(math.nextafter(lower, math.inf), upper)
        elif mu > upper:
            mu = max(math.nextafter(upper, 0.0), lower)
    return mu


def probe_gradient(loss, raw, *, retain_graph=False, loss_scale=1024.0):
    """只供校準：兩個 loss 同倍放大，減少 raw FP16 grad 下溢，再以 FP32 除回。"""
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("probe loss scale 必須為正有限值")
    gradient = torch.autograd.grad(loss * loss_scale, raw, retain_graph=retain_graph)[0]
    result = gradient.detach().float() / loss_scale
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("校準 probe 梯度非有限；不調大正式 μ")
    return result


def calibrate_hog(state, physical, count, sink):
    if count < 1:
        raise ValueError("校準 macro 數必須為正")
    state.training_mode()
    _reseed(state, 0)
    native_squared = auxiliary_squared = 0.0
    tasks = {task.value: {"native_gradient_squared": 0.0, "auxiliary_gradient_squared": 0.0,
                         "valid_cells": 0, "images_with_valid_cells": 0, "images": 0} for task in Task}
    observed = 0
    for index, macro in enumerate(_raw_macros(state, physical)):
        if index >= count:
            break
        plan = JointMacroPlan.from_batch_sizes(
            detect_batch_sizes=[batch["img"].shape[0] for batch in macro.detect_batches],
            pose_batch_sizes=[batch["img"].shape[0] for batch in macro.pose_batches],
            reference_batch_size=64, detect_weight=1.0, pose_weight=0.25)
        for task, batches, factor in (
            (Task.DETECT, macro.detect_batches, plan.detect_backward_scale),
            (Task.POSE, macro.pose_batches, plan.pose_backward_scale),
        ):
            for batch in batches:
                prepared = state.detect.preprocess(batch) if task is Task.DETECT else state.pose.preprocess(batch)
                native, aux, raw = state.router._parts(task, prepared, force_aux=True)
                gn = probe_gradient(native.raw_total * factor, raw, retain_graph=True)
                gh = probe_gradient(aux.loss_sum * factor, raw)
                norm_n = float(gn.square().sum(dtype=torch.float64).cpu())
                norm_h = float(gh.square().sum(dtype=torch.float64).cpu())
                native_squared += norm_n
                auxiliary_squared += norm_h
                item = tasks[task.value]
                item["native_gradient_squared"] += norm_n
                item["auxiliary_gradient_squared"] += norm_h
                item["valid_cells"] += int(aux.targets.valid_mask.sum().item())
                item["images_with_valid_cells"] += int(aux.targets.valid_mask.flatten(1).any(1).sum().item())
                item["images"] += native.actual_batch_size
                del prepared, native, aux, raw, gn, gh
        observed += 1
        sink({"kind": "calibration_macro", "macro": observed,
              "native_gradient_squared": native_squared, "auxiliary_gradient_squared": auxiliary_squared})
    if observed != count:
        raise RuntimeError("完整 loader 的校準 prefix 不足指定 macro 數")
    unconstrained_mu = calibration_coefficient(native_squared, auxiliary_squared)
    mu = calibration_coefficient(native_squared, auxiliary_squared, task_squared=[
        (item["native_gradient_squared"], item["auxiliary_gradient_squared"])
        for item in tasks.values()])
    outside_band = []
    for task, item in tasks.items():
        if item["native_gradient_squared"] <= 1e-24 or item["auxiliary_gradient_squared"] <= 1e-24:
            raise FloatingPointError(f"{task} HOG 校準梯度為零；coverage={item}")
        ratio = mu * math.sqrt(item["auxiliary_gradient_squared"] / item["native_gradient_squared"])
        item["weighted_auxiliary_to_native_gradient_ratio"] = ratio
        if not 0.02 <= ratio <= 0.10:
            outside_band.append(task)
    result = {"mu": mu,
            "unconstrained_mu": unconstrained_mu,
            "policy": "global_target_projected_to_shared_task_band_v1",
            "achieved_global_gradient_ratio": mu * math.sqrt(auxiliary_squared / native_squared),
            "target_gradient_ratio": 0.05, "macros": observed,
            "native_gradient_squared": native_squared, "auxiliary_gradient_squared": auxiliary_squared,
            "gradient_space": "concatenated_raw_p3_per_image_with_native_macro_task_scaling",
            "probe_loss_scale": 1024.0, "probe_gradient_accumulation_dtype": "float32_with_float64_sum",
            "optimizer_steps": 0, "validation_used": False, "tasks": tasks,
            "task_ratios_outside_observation_band": outside_band}
    sink({"kind": "calibration_result", **result})
    if outside_band:
        raise FloatingPointError(f"HOG calibration 任務梯度超出 2–10% 觀察區間，先分析再訓練：{result}")
    return result


class TimedEngine:
    def __init__(self, state, sink, *, mu=0.0, schedule=True):
        self.state, self.sink, self.mu, self.schedule = state, sink, mu, schedule
        self.epoch = self.macro = 0
        self.durations = []

    def run(self, **kwargs):
        state = self.state
        if state.metadata.get('variant') == 'heads':
            # 沒有 shared 可訓練參數，shared task cosine 不適用；不改 loss 或反向。
            kwargs['record_gradient_statistics'] = False
        factor = hog_multiplier(self.epoch, self.macro, state.macros) if self.schedule else 1.0
        state.router.mu = self.mu * factor
        state.router.last_aux = {}
        files = [str(name) for batches in (kwargs["detect_batches"], kwargs["pose_batches"])
                 for batch in batches for name in batch.get("im_file", ())]
        _sync(state.device)
        started = time.perf_counter()
        report = state.engine.run(**kwargs)
        _sync(state.device)
        elapsed = time.perf_counter() - started
        self.durations.append(elapsed)
        self.sink({"kind": "macro", "epoch": self.epoch + 1, "macro": self.macro,
            "seconds": elapsed, "mu": state.router.mu, "report": asdict(report),
            "lrs": {group["group_name"]: group["lr"] for group in state.optimizer.param_groups},
            "auxiliary": state.router.last_aux,
            "image_order_sha256": hashlib.sha256("\n".join(files).encode()).hexdigest()})
        self.macro += 1
        return report

    def advance_epoch(self, tasks=None):
        self.state.engine.advance_epoch(tasks)


@contextmanager
def validation_boundary(*models):
    """守住來源與 RNG；只恢復已證實共享、無狀態的 exact SiLU mode。

    以 ``with validation_boundary(...) as audit`` 取得可 JSON 化的恢復紀錄。
    audit 在離開邊界時填入；其他 tensor／trainability／mode 漂移仍拒絕。
    """
    rng = _rng_state()
    def signature(model):
        return (_state_digest(model),
                tuple((name, module.training) for name, module in model.named_modules()),
                tuple((name, parameter.requires_grad) for name, parameter in model.named_parameters()))
    before = [signature(model) for model in models]
    modes = [{name: (module, module.training) for name, module in model.named_modules()}
             for model in models]
    audit = {"restored_training_modes": []}
    try:
        yield audit
    finally:
        _restore_rng(rng)
        for index, (model, expected_modes) in enumerate(zip(models, modes, strict=True)):
            current_modules = dict(model.named_modules())
            for name, (module, training) in expected_modes.items():
                if current_modules.get(name) is not module or module.training == training:
                    continue
                if (type(module) is nn.SiLU and not tuple(module.parameters())
                        and not tuple(module.buffers()) and not tuple(module.children())):
                    observed = module.training
                    # 不用遞迴 train()，避免破壞 shared BN eval／head BN train。
                    module.training = training
                    audit["restored_training_modes"].append({
                        "model_index": index, "module_name": name,
                        "module_type": f"{type(module).__module__}.{type(module).__name__}",
                        "before": training, "observed": observed, "restored": module.training,
                        "all_alias_paths": [path for path, candidate in model.named_modules(remove_duplicate=False)
                                            if candidate is module],
                    })
        if any(signature(model) != expected for model, expected in zip(models, before, strict=True)):
            raise RuntimeError("驗證修改了訓練／EMA 來源 state 或 training mode；拒絕保存並續訓")


def _validate(state, config, data, root, epoch, sink, *, live=False):
    event_prefix = "live_" if live else ""
    source_model = state.model.base if live else state.ema.ema.base
    source_model = evaluation_copy(source_model)
    validator = JointValidator(state.source, detect_data_yaml=data[0], pose_data_yaml=data[1],
        output_root=root / ("validation-live" if live else "validation"), settings=ValidationSettings(imgsz=config.imgsz,
            detect_batch_size=config.detect_val_batch_size, pose_batch_size=config.pose_val_batch_size,
            detect_workers=config.detect_workers, pose_workers=config.pose_workers,
            device=str(state.device), plots=config.validation_plots, save_coco_json=config.save_coco_json))
    metrics = {}
    for backend in (("bittrue",) if live else ("float", "bittrue")):
        sink({"kind": f"{event_prefix}validation_start", "epoch": epoch + 1, "backend": backend})
        result = validator.validate(source_model, epoch=epoch + 1, kind=backend)
        metrics[backend] = dict(result.metrics)
        if any(name not in metrics[backend] for name in GATE_METRICS):
            raise RuntimeError("驗證缺少八項必要 AP")
        sink({"kind": f"{event_prefix}validation", "epoch": epoch + 1, "backend": backend, "metrics": metrics[backend]})
        del result
        _cleanup()
    return metrics


def _save_epoch(state, root, epoch, global_macro, metadata, selectors, selection, metrics, early, seeds):
    labels = (f"epoch-{epoch + 1:04d}", *selection.selected)
    inference_model = evaluation_copy(state.ema.ema.base)
    progress = TrainingProgress(stage="recovery", next_epoch=epoch + 1,
        global_macro_step=global_macro, joint_epochs_completed=epoch + 1)
    for label in labels:
        saved = save_training_snapshot(root / "checkpoints" / f"{label}.pt", model=state.model,
            ema=state.ema, optimizer=state.optimizer, scheduler=state.scheduler, scaler=state.scaler,
            criteria=state.router, progress=progress, resolved_config=metadata,
            provenance={"parent": metadata["parent"], "criteria_continuation": metadata["criteria_continuation"]},
            loader_state={"snapshot_boundary": "formal_epoch_end", "seeds": seeds,
                          "early_stop": early.state_dict() if early else None},
            best_state=selectors.state_dict())
        save_inference_weights(root / "inference" / f"{label}.pt", model=inference_model,
            ema=SimpleNamespace(ema=inference_model), use_ema=True,
            metadata={"epoch": epoch + 1, "metrics": metrics, "training_auxiliary_removed": True,
                      "full_resume_sha256": saved.sha256,
                      "reparameterization_fused": metadata.get('reparameterization')})


def _save_pending_validation(state, root, epoch, global_macro, metadata, selectors, early, seeds):
    """先保全完成的 optimizer 邊界；未驗證快照不可當作新 BEST。"""
    saved = save_training_snapshot(root / "checkpoints" / f"epoch-{epoch + 1:04d}-validation-pending.pt",
        model=state.model, ema=state.ema, optimizer=state.optimizer,
        scheduler=state.scheduler, scaler=state.scaler, criteria=state.router,
        progress=TrainingProgress(stage="recovery", next_epoch=epoch + 1,
            global_macro_step=global_macro, joint_epochs_completed=epoch + 1),
        resolved_config=metadata,
        provenance={"parent": metadata["parent"],
                    "adopted_prefix": metadata.get("adopted_prefix")},
        loader_state={"snapshot_boundary": "epoch_end_before_validation",
                      "validation_pending": True, "seeds": seeds,
                      "early_stop": early.state_dict() if early else None},
        best_state=selectors.state_dict())
    return _portable(asdict(saved))


def severe_regressions(metrics, parent_metrics, maximum_drop=0.005):
    return {name: float(metrics[name]) - float(parent_metrics[name]) for name in GATE_METRICS
            if float(metrics[name]) < float(parent_metrics[name]) - maximum_drop - 1e-12}


def memory_headroom(total, budget, peak_reserved):
    required = max(math.ceil(total * 0.15), 2 * 1024**3)
    available = budget - peak_reserved
    return {"total_bytes": total, "allocator_budget_bytes": budget,
            "required_headroom_bytes": required, "remaining_headroom_bytes": available,
            "eligible": peak_reserved <= total * 0.85 and available >= required}


def run_training(*, config, checkpoint, paired_snapshot, run_dir, variant="native", physical_batch=32,
                 device="cuda:0", calibration_macros=2, callback=None,
                 ema_age_mode="fresh", validate_live=False, adopt_ema_prefix_dir=None,
                 base_lr_scale=1.0, pause_on_live_regression=False) -> dict:
    """明確指定的單臂正式訓練；不自動啟動下一臂。"""
    if ema_age_mode not in ("fresh", "parent"):
        raise ValueError("ema_age_mode 必須是 fresh 或 parent")
    if type(validate_live) is not bool:
        raise TypeError("validate_live 必須是 bool")
    scaled_base_scope(base_lr_scale)
    if type(pause_on_live_regression) is not bool or (pause_on_live_regression and not validate_live):
        raise ValueError("live 安全暫停必須明確啟用 live 驗證")
    if adopt_ema_prefix_dir is not None and (base_lr_scale != 1.0 or pause_on_live_regression):
        raise ValueError("診斷 E1 前綴不可套用不同 LR 或新增 live 暫停策略")
    if adopt_ema_prefix_dir is not None and (variant, ema_age_mode, physical_batch, validate_live) != ("native", "parent", 32, True):
        raise ValueError("EMA 前綴採用僅限 native/parent/32 並開啟 live 驗證")
    root = _new_run(run_dir)
    sink = EventSink(root, callback)
    state = None
    try:
        preflight = config.preflight()
        if not preflight.ready or preflight.baseline is None:
            raise RuntimeError(f"正式 baseline preflight 失敗：{preflight.blockers}")
        data = runtime.prepare_data(config, config.run_root / "datasets")
        calibration = None
        if variant == "hog":
            sink({"kind": "calibration_start"})
            state = _build(config, checkpoint, paired_snapshot, data, physical_batch, variant, device,
                           ema_age_mode=ema_age_mode, base_lr_scale=base_lr_scale)
            initial_digest = state.metadata["base_state_sha256"]
            calibration = calibrate_hog(state, physical_batch, calibration_macros, sink)
            runtime.write_json(root / "calibration.json", calibration)
            del state
            state = None
            _cleanup()
        state = _build(config, checkpoint, paired_snapshot, data, physical_batch, variant, device,
                       ema_age_mode=ema_age_mode, base_lr_scale=base_lr_scale)
        if calibration is not None and state.metadata["base_state_sha256"] != initial_digest:
            raise AssertionError("校準後 parent 重建不一致")
        state.router.calibration = calibration
        metadata = {**state.metadata, "max_epochs": 10 if variant == "hog" else 5,
                    "patience": 4 if variant == "hog" else 0, "early_stop_min_delta": 1e-4,
                    "training_epoch_numbering": "1-based in logs, validation, inference labels and selectors",
                    "validate_live": validate_live,
                    "pause_on_live_regression": pause_on_live_regression,
                    "live_metrics_role": ("diagnostic_and_safety_not_selection" if pause_on_live_regression
                                          else "diagnostic_only_not_selection_or_pause"),
                    "validation_preserves_training_rng_and_source_state": True,
                    "calibration": calibration, "data": [str(path) for path in data]}
        gate = AccuracyGate(preflight.baseline, maximum_drop=config.maximum_map_drop)
        selectors = CheckpointSelectors()
        early = StageEarlyStopping(stage="recovery", patience=4, min_delta=1e-4) if variant == "hog" else None
        timed = TimedEngine(state, sink, mu=calibration["mu"] if calibration else 0.0)
        global_macro = 0
        start_epoch = 0
        epochs = []
        if adopt_ema_prefix_dir is not None:
            from .control_prefix import adopt_ema_prefix
            adopted = adopt_ema_prefix(state, Path(adopt_ema_prefix_dir),
                metadata["criteria_continuation"]["parent_metrics"])
            start_epoch, global_macro = adopted["start_epoch"], adopted["global_macro_step"]
            metadata["adopted_prefix"] = adopted["provenance"]
            metadata["optimizer_state_at_run_entry"] = "restored_from_verified_diagnostic_e1"
            metadata["epochs_trained_in_this_process"] = metadata["max_epochs"] - start_epoch
            selected = adopted["metrics"]["bittrue"]
            gate_report = gate.evaluate(selected)
            selection = selectors.observe(epoch=start_epoch, metrics=selected, gate=gate_report)
            if severe_regressions(selected, metadata["criteria_continuation"]["parent_metrics"]):
                raise ValueError("採用 E1 未通過 parent 安全門檻；拒絕續訓")
            _save_epoch(state, root, start_epoch - 1, global_macro, metadata, selectors, selection,
                        selected, early, adopted["loader_state"]["seeds"])
            record = {"epoch": start_epoch, "reused_prefix": True,
                "training": adopted["training"], "metrics": adopted["metrics"],
                "live_metrics": adopted["live_metrics"],
                "live_regressions": severe_regressions(adopted["live_metrics"],
                    metadata["criteria_continuation"]["parent_metrics"]),
                "gate": asdict(gate_report), "scores": selection.scores, "early_stop": None,
                "adoption_provenance": adopted["provenance"]}
            epochs.append(record)
            sink({"kind": "prefix_adopted", **record})
            runtime.write_json(root / "summary.json", {"status": "running", "epochs": epochs,
                "best": selectors.state_dict(), "metadata": metadata})
        runtime.write_json(root / "resolved-config.json", metadata)
        with ExperimentLogger(root / "logs", tensorboard="off") as logger:
            runner = JointEpochRunner(engine=timed, detect_loader=state.detect.loader,
                pose_loader=state.pose.loader, scheduler=state.scheduler, logger=logger,
                apply_training_mode=state.training_mode,
                assert_hardware_contract=lambda: state.guard.assert_unchanged(state.model.base),
                detect_batches_per_macro=256 // physical_batch,
                gradient_statistics_interval=config.gradient_statistics_interval)
            for epoch in range(start_epoch, metadata["max_epochs"]):
                timed.epoch, timed.macro = epoch, 0
                seeds = _reseed(state, epoch)
                sink({"kind": "epoch_start", "epoch": epoch + 1, "seeds": seeds})
                report = runner.run_epoch(epoch=epoch + 1, global_macro_step=global_macro, stage="recovery")
                global_macro = report.next_global_macro_step
                pending = _save_pending_validation(state, root, epoch, global_macro,
                    metadata, selectors, early, seeds)
                sink({"kind": "validation_pending_snapshot", "epoch": epoch + 1, **pending})
                with validation_boundary(state.model.base, state.ema.ema.base) as validation_audit:
                    metrics = _validate(state, config, data, root, epoch, sink)
                    live_metrics = (_validate(state, config, data, root, epoch, sink, live=True)["bittrue"]
                                    if validate_live else None)
                state.guard.assert_unchanged(state.model.base)
                selected = metrics["bittrue"]
                gate_report = gate.evaluate(selected)
                selection = selectors.observe(epoch=epoch + 1, metrics=selected, gate=gate_report)
                decision = early.observe(selection.scores["best_joint"]) if early else None
                _save_epoch(state, root, epoch, global_macro, metadata, selectors, selection,
                            selected, early, seeds)
                record = {"epoch": epoch + 1, "training": asdict(report), "metrics": metrics,
                          "gate": asdict(gate_report), "scores": selection.scores,
                          "early_stop": asdict(decision) if decision else None,
                          "validation_boundary": validation_audit,
                          "pre_validation_snapshot": pending}
                if live_metrics is not None:
                    record["live_metrics"] = live_metrics
                    record["live_regressions"] = severe_regressions(
                        live_metrics, metadata["criteria_continuation"]["parent_metrics"])
                epochs.append(record)
                sink({"kind": "epoch_complete", **record})
                runtime.write_json(root / "summary.json", {"status": "running", "epochs": epochs,
                    "best": selectors.state_dict(), "metadata": metadata})
                degraded = severe_regressions(selected, metadata["criteria_continuation"]["parent_metrics"])
                live_degraded = record.get("live_regressions", {}) if pause_on_live_regression else {}
                if degraded or live_degraded:
                    result = {"status": "paused_for_analysis", "epochs": epochs,
                        "best": selectors.state_dict(), "metadata": metadata,
                        "reason": "EMA 或已啟用的 live 安全檢查：任一必要 AP 較 parent 下降超過 0.005；已保存再暫停",
                        "regressions": degraded, "live_regressions": live_degraded}
                    runtime.write_json(root / "summary.json", result)
                    sink({"kind": "paused_for_analysis", "epoch": epoch + 1,
                          "regressions": degraded, "live_regressions": live_degraded})
                    return result
                if decision is not None and decision.should_stop:
                    break
        result = {"status": "complete", "epochs": epochs, "best": selectors.state_dict(), "metadata": metadata}
        runtime.write_json(root / "summary.json", result)
        sink({"kind": "complete", "epochs_completed": len(epochs)})
        return result
    except Exception as error:
        sink({"kind": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        state = None
        _cleanup()


def _benchmark_case(config, checkpoint, paired_snapshot, data, physical, variant, device, warmup, timed_count, sink):
    state = _build(config, checkpoint, paired_snapshot, data, physical, variant, device)
    state.training_mode()
    _reseed(state, 0)
    timed = TimedEngine(state, sink, mu=1e-3 if variant == "hog" else 0.0, schedule=False)
    _sync(state.device)
    memory_budget = total_memory = None
    if state.device.type == "cuda":
        free_memory, total_memory = torch.cuda.mem_get_info(state.device)
        memory_budget = free_memory + torch.cuda.memory_reserved(state.device)
        torch.cuda.reset_peak_memory_stats(state.device)
    iterator = iter(_raw_macros(state, physical))
    durations, total_images = [], 0
    for index in range(warmup + timed_count):
        _sync(state.device)
        start = time.perf_counter()
        macro = next(iterator)
        state.scheduler.prepare_step()
        report = timed.run(detect_batches=macro.detect_batches, pose_batches=macro.pose_batches,
                           record_gradient_statistics=False)
        state.scheduler.advance()
        _sync(state.device)
        if index >= warmup:
            durations.append(time.perf_counter() - start)
            total_images += report.detect_images + report.pose_images
    state.guard.assert_unchanged(state.model.base)
    elapsed = sum(durations)
    result = {"status": "ok", "variant": variant, "physical_batch": physical,
        "logical_detect_batch": 128, "detect_images_per_macro": 256, "pose_images_per_macro": 16,
        "warmup_macros": warmup, "timed_macros": timed_count,
        "macro_seconds_end_to_end": durations, "macro_seconds_engine": timed.durations[warmup:],
        "images_per_second": total_images / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(state.device) if state.device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(state.device) if state.device.type == "cuda" else None,
        "benchmark_mu_not_formal": timed.mu, "updates_discarded": True,
        "base_state_sha256": state.metadata["base_state_sha256"]}
    result["memory_headroom"] = memory_headroom(total_memory, memory_budget,
        result["peak_reserved_bytes"]) if total_memory is not None else None
    return result


def run_benchmark(*, config, checkpoint, paired_snapshot, run_dir, device="cuda:0",
                  physical_batches=(32, 64, 128), warmup_macros=2, timed_macros=5, callback=None) -> dict:
    """測真實 train steps；只產生推薦，不啟動正式訓練。"""
    if warmup_macros < 0 or timed_macros < 1:
        raise ValueError("benchmark macro 設定無效")
    root = _new_run(run_dir)
    sink = EventSink(root, callback)
    data = runtime.prepare_data(config, config.run_root / "datasets")
    cases = []
    for physical in physical_batches:
        for variant in ("native", "hog"):
            sink({"kind": "benchmark_start", "physical_batch": physical, "variant": variant})
            try:
                record = _benchmark_case(config, checkpoint, paired_snapshot, data, physical, variant,
                                         device, warmup_macros, timed_macros, sink)
            except torch.cuda.OutOfMemoryError as error:
                record = {"status": "oom", "physical_batch": physical, "variant": variant, "error": str(error)}
            finally:
                _cleanup()
            cases.append(record)
            sink({"kind": "benchmark_case", **record})
            runtime.write_json(root / "benchmark.json", {"status": "running", "cases": cases})
    eligible = []
    for physical in physical_batches:
        pair = [item for item in cases if item["physical_batch"] == physical]
        if len(pair) == 2 and all(item["status"] == "ok" and item["memory_headroom"] is not None
                                  and item["memory_headroom"]["eligible"] for item in pair):
            eligible.append((min(item["images_per_second"] for item in pair), physical))
    result = {"status": "complete", "cases": cases,
              "recommended_physical_batch": max(eligible)[1] if eligible else None,
              "selection_rule": "兩臂成功且各保留 max(15% total VRAM, 2 GiB) 餘裕，再選較慢一臂 throughput 最大者；仍需驗證 BN/AP",
              "formal_training_started": False}
    runtime.write_json(root / "benchmark.json", result)
    return result
