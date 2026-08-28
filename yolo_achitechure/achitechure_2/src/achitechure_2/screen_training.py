"""Full35 C0–C3 固定20% Float 篩選訓練與公平 OOM gate。"""

from __future__ import annotations

import gc
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import ModelEMA

from .candidate import build_candidate
from .config import SPEC_PATH, SPEC_VERSION, file_sha256
from .freezing import FrozenStateGuard, enforce_frozen_eval
from .full35_adapter import Full35Release, Full35TrainingAPI
from .runtime_dataset import build_runtime_yolo_dataset
from .screen_validation import ScreenValidationResult, ScreenValidator, ThresholdSet
from .screening_data import validate_screening_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} 必須是 mapping")
    return dict(value)


def _path(value: Any, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True)
class ScreenRunConfig:
    """一份可直接執行且所有重要 knob 都在 YAML 的正式設定。"""

    path: Path
    payload: dict[str, Any]
    config_id: str
    handoff_manifest: Path
    full35_root: Path
    candidates: tuple[str, ...]
    screen_manifest: Path
    screen_root: Path
    detect_data: Path
    pose_data: Path
    diagnostic_detect_data: Path
    run_root: Path
    device: str
    pose_enabled: bool
    imgsz: int
    epochs: int
    seed: int
    deterministic: bool
    amp: bool
    fraction: float
    scale: float
    cache: bool | str
    detect_logical_batch: int
    detect_microbatch: int
    detect_oom_fallback_microbatch: int
    detect_logical_batches_per_macro: int
    pose_batch: int
    detect_val_batch: int
    pose_val_batch: int
    detect_workers: int
    pose_workers: int
    nbs: int
    optimizer: str
    momentum: float
    weight_decay: float
    warmup_epochs: int
    warmup_start_factor: float
    final_lr_factor: float
    max_grad_norm: float
    amp_max_overflow_retries: int
    reference_batch_size: int
    validation_interval: int
    task_weights: dict[str, float]
    learning_rates: dict[str, float]
    augmentation: dict[str, dict[str, float]]
    loss_overrides: dict[str, dict[str, float]]

    @classmethod
    def load(cls, path: str | Path) -> ScreenRunConfig:
        source = Path(path).expanduser().resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Float20 run YAML 根節點必須是 mapping")
        required = {
            "schema_version",
            "spec_version",
            "spec_sha256",
            "kind",
            "config_id",
            "title_zh",
            "status",
            "authorization",
            "handoff",
            "candidates",
            "datasets",
            "training",
            "queue",
            "lineage",
        }
        if set(payload) != required:
            raise ValueError(
                f"Float20 run YAML 欄位漂移：missing={sorted(required - set(payload))} "
                f"unknown={sorted(set(payload) - required)}"
            )
        if (
            payload["schema_version"] != 1
            or payload["kind"] != "float_screen_run"
            or payload["spec_version"] != SPEC_VERSION
            or payload["spec_sha256"] != file_sha256(SPEC_PATH)
        ):
            raise ValueError("Float20 run YAML schema/spec metadata 漂移")
        base = PROJECT_ROOT
        authorization = _mapping(payload["authorization"], "authorization")
        if authorization.get("gpu") is not True:
            raise PermissionError("此 run YAML 沒有 GPU 授權")
        handoff = _mapping(payload["handoff"], "handoff")
        datasets = _mapping(payload["datasets"], "datasets")
        training = _mapping(payload["training"], "training")
        batch = _mapping(training["batch"], "training.batch")
        workers = _mapping(training["workers"], "training.workers")
        scheduler = _mapping(training["scheduler"], "training.scheduler")
        candidates = tuple(str(value).upper() for value in payload["candidates"])
        if candidates != ("C0", "C1", "C2", "C3"):
            raise ValueError("本輪候選順序必須固定為 C0、C1、C2、C3")
        fraction = float(training["fraction"])
        if fraction != 1.0:
            raise ValueError("固定20%已由 manifest 決定，training.fraction 必須保持 1.0")
        cache = training["cache"]
        if cache not in {False, True, "ram"}:
            raise ValueError("screening cache 只允許 false/true/ram；禁止寫入唯讀來源的 disk cache")
        pose_decision = authorization.get("pose")
        if not isinstance(pose_decision, bool):
            raise PermissionError(
                "Pose gate 尚未決定；請在正式 YAML 同時把 "
                "authorization.pose 與 training.pose_enabled 設為 true 或 false"
            )
        pose_enabled = pose_decision
        if training["pose_enabled"] is not pose_enabled:
            raise ValueError("authorization.pose 與 training.pose_enabled 必須一致")
        learning_rates = {
            str(key): float(value)
            for key, value in _mapping(
                training["learning_rates"],
                "training.learning_rates",
            ).items()
        }
        expected_roles = {
            "backbone",
            "neck",
            "masf",
            "attention",
            "detect_head",
            "pose_head",
        }
        if set(learning_rates) != expected_roles:
            raise ValueError("learning_rates roles 不完整")
        if learning_rates["masf"] != 0.0 or learning_rates["attention"] != 0.0:
            raise ValueError("inherited MASF 與 attention 必須永久凍結")
        result = cls(
            path=source,
            payload=payload,
            config_id=str(payload["config_id"]),
            handoff_manifest=_path(handoff["manifest"], base=base),
            full35_root=_path(handoff["full35_root"], base=base),
            candidates=candidates,
            screen_manifest=_path(datasets["screen_manifest"], base=base),
            screen_root=_path(datasets["screen_root"], base=base),
            detect_data=_path(datasets["detect"], base=base),
            pose_data=_path(datasets["pose"], base=base),
            diagnostic_detect_data=_path(datasets["diagnostic_detect"], base=base),
            run_root=_path(training["run_root"], base=base),
            device=str(training["device"]),
            pose_enabled=pose_enabled,
            imgsz=int(training["imgsz"]),
            epochs=int(training["epochs"]),
            seed=int(training["seed"]),
            deterministic=bool(training["deterministic"]),
            amp=bool(training["amp"]),
            fraction=fraction,
            scale=float(training["scale"]),
            cache=cache,
            detect_logical_batch=int(batch["detect_logical"]),
            detect_microbatch=int(batch["detect_physical_microbatch"]),
            detect_oom_fallback_microbatch=int(batch["detect_oom_fallback_microbatch"]),
            detect_logical_batches_per_macro=int(batch["detect_logical_batches_per_macro"]),
            pose_batch=int(batch["pose_physical"]),
            detect_val_batch=int(batch["validation_detect"]),
            pose_val_batch=int(batch["validation_pose"]),
            detect_workers=int(workers["detect"]),
            pose_workers=int(workers["pose"]),
            nbs=int(training["nbs"]),
            optimizer=str(training["optimizer"]),
            momentum=float(training["momentum"]),
            weight_decay=float(training["weight_decay"]),
            warmup_epochs=int(scheduler["warmup_epochs"]),
            warmup_start_factor=float(scheduler["warmup_start_factor"]),
            final_lr_factor=float(scheduler["final_lr_factor"]),
            max_grad_norm=float(training["max_grad_norm"]),
            amp_max_overflow_retries=int(training["amp_max_overflow_retries"]),
            reference_batch_size=int(training["reference_batch_size"]),
            validation_interval=int(training["validation_interval"]),
            task_weights={
                str(key): float(value)
                for key, value in _mapping(
                    training["task_weights"],
                    "training.task_weights",
                ).items()
            },
            learning_rates=learning_rates,
            augmentation={
                str(task): {
                    str(key): float(value)
                    for key, value in _mapping(settings, f"augmentation.{task}").items()
                }
                for task, settings in _mapping(
                    training["augmentation"],
                    "training.augmentation",
                ).items()
            },
            loss_overrides={
                str(task): {
                    str(key): float(value) for key, value in _mapping(settings, f"losses.{task}").items()
                }
                for task, settings in _mapping(
                    training["loss_overrides"],
                    "training.loss_overrides",
                ).items()
            },
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.device != "0":
            raise ValueError("本輪只允許單卡 device=0")
        if self.seed != 0 or not self.deterministic:
            raise ValueError("第一輪必須 deterministic seed 0")
        if self.imgsz != 640 or self.epochs < 1:
            raise ValueError("本輪 imgsz 必須 640 且 epochs 必須為正")
        if self.optimizer != "MuSGD":
            raise ValueError("Full35 Float20 恢復 optimizer 必須是 MuSGD")
        if (
            self.detect_logical_batch < 1
            or self.detect_microbatch < 1
            or self.detect_oom_fallback_microbatch < 1
            or self.detect_logical_batch % self.detect_microbatch
            or self.detect_logical_batch % self.detect_oom_fallback_microbatch
        ):
            raise ValueError("Detect physical microbatch 必須整除 logical batch")
        if (
            min(
                self.pose_batch,
                self.detect_val_batch,
                self.pose_val_batch,
                self.nbs,
                self.reference_batch_size,
                self.validation_interval,
            )
            < 1
        ):
            raise ValueError("batch/nbs/reference/validation interval 必須為正")
        if self.detect_val_batch != 16 or self.pose_val_batch != 16:
            raise ValueError("依使用者 OOM 經驗，本輪 Detect/Pose validation 都固定 batch 16")
        if self.scale != 0.5:
            raise ValueError("本 revision augmentation scale 固定 0.5")
        if set(self.task_weights) != {"detect", "pose"}:
            raise ValueError("task_weights 必須包含 detect/pose")
        if set(self.augmentation) != {"detect", "pose"}:
            raise ValueError("augmentation 必須分開宣告 detect/pose")
        if set(self.loss_overrides) != {"detect", "pose"}:
            raise ValueError("loss_overrides 必須分開宣告 detect/pose")
        for path in (
            self.handoff_manifest,
            self.screen_manifest,
            self.detect_data,
            self.pose_data,
            self.diagnostic_detect_data,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        report = validate_screening_data(self.screen_root)
        if report["coco_train_count"] != 23657 or report["bbat5_train_count"] != 1073:
            raise ValueError("固定20% screening assignment count 漂移")

    def resolved_payload(self, *, candidate: str, microbatch: int) -> dict[str, Any]:
        return {
            "config": self.payload,
            "config_path": str(self.path),
            "config_sha256": file_sha256(self.path),
            "candidate": candidate,
            "effective_detect_microbatch": microbatch,
            "pose_enabled": self.pose_enabled,
        }


class _AutocastRouter:
    def __init__(
        self,
        router: Any,
        *,
        device: torch.device,
        enabled: bool,
    ) -> None:
        self.router = router
        self.device = device
        self.enabled = bool(enabled and device.type == "cuda")

    def loss_for(self, task: Any, batch: Mapping[str, Any]) -> Any:
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
            enabled=self.enabled,
        ):
            return self.router.loss_for(task, batch)

    def advance_epoch(self, tasks: Any = None) -> None:
        self.router.advance_epoch(tasks)

    def state_dict(self) -> dict[str, Any]:
        return self.router.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.router.load_state_dict(state)


def _device(value: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Float20 需要 CUDA，但目前不可用")
    index = int(value)
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} 不存在")
    return torch.device(f"cuda:{index}")


def _normalize_names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(index): str(name) for index, name in value.items()}
    raise TypeError("dataset names 必須是 list 或 mapping")


def _build_screen_loader(
    api: Full35TrainingAPI,
    model: nn.Module,
    *,
    task: Any,
    data_yaml: Path,
    batch_size: int,
    workers: int,
    config: ScreenRunConfig,
    device: torch.device,
) -> Any:
    data = check_det_dataset(str(data_yaml), autodownload=False)
    head = model.head_for(task)
    names = _normalize_names(data.get("names"))
    expected_names = model.detect_names if task is api.Task.DETECT else model.pose_names
    if int(data["nc"]) != int(head.nc) or names != expected_names:
        raise ValueError(f"{task.value} dataset schema nc/names 與 Full35 head 不一致")
    if task is api.Task.POSE:
        if tuple(int(value) for value in data.get("kpt_shape", ())) != (2, 3):
            raise ValueError("Pose kpt_shape 必須是 [2,3]")
        if tuple(int(value) for value in data.get("flip_idx", ())) != (0, 1):
            raise ValueError("Pose flip_idx 必須保持 [0,1]")
    settings = api.TaskLoaderSettings(
        task=task,
        batch_size=batch_size,
        workers=workers,
        imgsz=config.imgsz,
        mode="train",
        fraction=config.fraction,
        seed=config.seed,
        augmentation=config.augmentation[task.value],
    )
    overrides = settings.overrides(data_yaml)
    overrides["cache"] = config.cache
    overrides.update(config.augmentation[task.value])
    args = get_cfg(DEFAULT_CFG, overrides=overrides)
    stride = max(int(head.stride.max().item()), 32)
    dataset = build_runtime_yolo_dataset(
        args,
        data["train"],
        batch_size,
        data,
        label_cache_path=(
            config.screen_root / "runtime-cache" / f"{task.value}-{data_yaml.stem}-train.cache"
        ),
        mode="train",
        rect=False,
        stride=stride,
        fraction=config.fraction,
    )
    loader = build_dataloader(
        dataset,
        batch=batch_size,
        workers=workers,
        shuffle=True,
        rank=-1,
        drop_last=False,
    )
    return api.PreparedTaskLoader(
        task=task,
        data_yaml=data_yaml,
        data=data,
        dataset=dataset,
        loader=loader,
        settings=settings,
        device=device,
    )


def _gradient_groups(model: nn.Module) -> dict[str, tuple[nn.Parameter, ...]]:
    values: dict[str, list[nn.Parameter]] = {
        "shared": [],
        "detect_head": [],
        "pose_head": [],
    }
    for name, parameter in model.named_parameters():
        if ".detect_head." in name:
            values["detect_head"].append(parameter)
        elif ".pose_head." in name:
            values["pose_head"].append(parameter)
        else:
            values["shared"].append(parameter)
    if any(not group for group in values.values()):
        raise ValueError("Full35 gradient ownership group 不完整")
    return {name: tuple(group) for name, group in values.items()}


def _stage(api: Full35TrainingAPI, config: ScreenRunConfig) -> Any:
    rates = dict(config.learning_rates)
    if not config.pose_enabled:
        rates["pose_head"] = 0.0
    return api.JointStage(
        name=str(getattr(config, "stage_name", "float20")),
        task_mode="joint",
        epochs=config.epochs,
        patience=int(getattr(config, "patience", 0)),
        backbone_start_layer=0,
        tune_attention=False,
        learning_rates=rates,
        warmup_epochs=config.warmup_epochs,
    )


@dataclass
class _Runtime:
    release: Full35Release
    api: Full35TrainingAPI
    parent: Any
    resolved: Any
    model: nn.Module
    build_report: Any
    stage: Any
    optimizer: torch.optim.Optimizer
    optimizer_report: Any
    router: _AutocastRouter
    scaler: Any
    ema: Any
    guard: FrozenStateGuard
    detect_loader: Any
    pose_loader: Any | None
    scheduler: Any
    steps_per_epoch: int
    microbatches_per_macro: int
    device: torch.device


def _make_runtime(
    config: ScreenRunConfig,
    *,
    candidate: str,
    microbatch: int,
) -> _Runtime:
    release = Full35Release(config.full35_root)
    api = release.training_api()
    parent = release.load_parent()
    resolved = release.resolved_candidate(candidate)
    model, build_report = build_candidate(parent.model, resolved, seed=config.seed)
    device = _device(config.device)
    model = model.to(device)
    stage = _stage(api, config)
    optimizer, optimizer_report = api.build_joint_optimizer(
        model,
        stage,
        optimizer_name=config.optimizer,
        weight_decay=config.weight_decay,
        beta1=config.momentum,
        beta2=0.999,
    )
    enforce_frozen_eval(model, release.frozen_module_paths)
    guard = FrozenStateGuard.capture(
        model,
        release.frozen_module_paths,
        reset_trainable=False,
    )
    detect_loader = _build_screen_loader(
        api,
        model,
        task=api.Task.DETECT,
        data_yaml=config.detect_data,
        batch_size=microbatch,
        workers=config.detect_workers,
        config=config,
        device=device,
    )
    pose_loader = (
        _build_screen_loader(
            api,
            model,
            task=api.Task.POSE,
            data_yaml=config.pose_data,
            batch_size=config.pose_batch,
            workers=config.pose_workers,
            config=config,
            device=device,
        )
        if config.pose_enabled
        else None
    )
    microbatches_per_logical = config.detect_logical_batch // microbatch
    microbatches_per_macro = config.detect_logical_batches_per_macro * microbatches_per_logical
    steps_per_epoch = math.ceil(len(detect_loader.loader) / microbatches_per_macro)
    router = api.NativeTaskLossRouter(
        model,
        epochs=config.epochs,
        imgsz=config.imgsz,
        detect_overrides=config.loss_overrides["detect"],
        pose_overrides=config.loss_overrides["pose"],
    )
    autocast = _AutocastRouter(router, device=device, enabled=config.amp)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.amp,
    )
    ema = ModelEMA(model, decay=0.9999, tau=2000)
    scheduler = api.StageWarmupCosineScheduler(
        optimizer,
        stage=str(getattr(config, "stage_name", "float20")),
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=config.warmup_epochs,
        warmup_start_factor=config.warmup_start_factor,
        final_lr_factor=config.final_lr_factor,
    )
    return _Runtime(
        release=release,
        api=api,
        parent=parent,
        resolved=resolved,
        model=model,
        build_report=build_report,
        stage=stage,
        optimizer=optimizer,
        optimizer_report=optimizer_report,
        router=autocast,
        scaler=scaler,
        ema=ema,
        guard=guard,
        detect_loader=detect_loader,
        pose_loader=pose_loader,
        scheduler=scheduler,
        steps_per_epoch=steps_per_epoch,
        microbatches_per_macro=microbatches_per_macro,
        device=device,
    )


def _apply_training_mode(runtime: _Runtime) -> Any:
    report = runtime.api.apply_stage(runtime.model, runtime.stage)
    enforce_frozen_eval(runtime.model, runtime.release.frozen_module_paths)
    return report


def _engine(runtime: _Runtime, config: ScreenRunConfig) -> Any:
    return runtime.api.MacroStepEngine(
        model=runtime.model,
        losses=runtime.router,
        optimizer=runtime.optimizer,
        reference_batch_size=config.reference_batch_size,
        task_weights={
            runtime.api.Task.DETECT: config.task_weights["detect"],
            runtime.api.Task.POSE: (config.task_weights["pose"] if config.pose_enabled else 0.0),
        },
        gradient_groups=_gradient_groups(runtime.model),
        scaler=runtime.scaler,
        ema=runtime.ema,
        max_grad_norm=config.max_grad_norm,
        max_amp_retries=config.amp_max_overflow_retries,
        preprocess=lambda task, batch: (
            runtime.detect_loader.preprocess(batch)
            if task is runtime.api.Task.DETECT
            else runtime.pose_loader.preprocess(batch)
        ),
    )


@dataclass(frozen=True)
class _EpochState:
    next_global_macro_step: int


def _run_detect_only_epoch(
    runtime: _Runtime,
    config: ScreenRunConfig,
    *,
    engine: Any,
    logger: Any,
    epoch: int,
    global_macro_step: int,
) -> _EpochState:
    """Pose 未 opt-in 時只跑 Detect，並明確留下不完整排名證據。"""

    _apply_training_mode(runtime)
    pending: list[Mapping[str, Any]] = []
    weighted_loss = 0.0
    images = 0
    macros = 0

    def step(batches: tuple[Mapping[str, Any], ...]) -> None:
        nonlocal weighted_loss, images, macros, global_macro_step
        lrs = dict(runtime.scheduler.prepare_step())
        report = engine.run(
            detect_batches=batches,
            pose_batches=(),
            record_gradient_statistics=False,
        )
        runtime.scheduler.advance()
        weighted_loss += report.detect_mean_loss * report.detect_images
        images += report.detect_images
        logger.log(
            "macro",
            step=global_macro_step,
            values={
                "loss/detect_mean": report.detect_mean_loss,
                "loss/joint_mean": report.joint_mean_loss,
                "images/detect": report.detect_images,
                "images/pose": 0,
                **{f"lr/{name}": value for name, value in lrs.items()},
            },
            context={
                "stage": "float20",
                "epoch": epoch,
                "pose_status": "not_run_by_user_choice",
            },
        )
        global_macro_step += 1
        macros += 1

    for batch in runtime.detect_loader.loader:
        pending.append(batch)
        if len(pending) == runtime.microbatches_per_macro:
            step(tuple(pending))
            pending.clear()
    if pending:
        step(tuple(pending))
    if not macros or not images:
        raise RuntimeError("Detect-only epoch 沒有 optimizer macro")
    engine.advance_epoch((runtime.api.Task.DETECT,))
    runtime.guard.assert_unchanged(runtime.model)
    logger.log(
        "epoch",
        step=epoch,
        values={
            "loss/detect_mean": weighted_loss / images,
            "loss/pose_mean": 0.0,
            "images/detect": images,
            "images/pose": 0,
            "macros": macros,
        },
        context={
            "stage": "float20",
            "pose_status": "not_run_by_user_choice",
        },
    )
    return _EpochState(next_global_macro_step=global_macro_step)


def _take(loader: Any, count: int) -> tuple[Mapping[str, Any], ...]:
    iterator = iter(loader)
    values: list[Mapping[str, Any]] = []
    for _ in range(count):
        try:
            values.append(next(iterator))
        except StopIteration:
            break
    if not values:
        raise RuntimeError("loader 沒有任何 batch")
    return tuple(values)


def probe_screen_memory(
    config: ScreenRunConfig,
    *,
    microbatch: int,
) -> dict[str, Any]:
    """以 C0 一個完整 macro 做公平 memory gate；結果不進入正式權重。"""

    runtime: _Runtime | None = None
    try:
        runtime = _make_runtime(config, candidate="C0", microbatch=microbatch)
        # model.to(device) 會先初始化對應 CUDA allocator；若在此之前重設
        # peak stats，PyTorch 會以「Invalid device argument」失敗。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime.device)
        engine = _engine(runtime, config)
        _apply_training_mode(runtime)
        report = engine.run(
            detect_batches=_take(
                runtime.detect_loader.loader,
                runtime.microbatches_per_macro,
            ),
            pose_batches=(_take(runtime.pose_loader.loader, 1) if runtime.pose_loader is not None else ()),
            record_gradient_statistics=False,
        )
        runtime.guard.assert_unchanged(runtime.model)
        torch.cuda.synchronize(runtime.device)
        return {
            "passed": True,
            "candidate": "C0",
            "logical_detect_batch": config.detect_logical_batch,
            "physical_detect_microbatch": microbatch,
            "detect_physical_microbatches_per_macro": runtime.microbatches_per_macro,
            "pose_batch": config.pose_batch if config.pose_enabled else 0,
            "peak_allocated_mib": int(
                torch.cuda.max_memory_allocated(runtime.device) / 2**20
            ),
            "peak_reserved_mib": int(
                torch.cuda.max_memory_reserved(runtime.device) / 2**20
            ),
            "detect_images": report.detect_images,
            "pose_images": report.pose_images,
            "probe_weights_discarded": True,
        }
    finally:
        del runtime
        gc.collect()
        torch.cuda.empty_cache()


def _threshold_path(config: ScreenRunConfig) -> Path:
    return config.run_root / "shared-controls" / "c0-f1-thresholds.json"


def _read_thresholds(config: ScreenRunConfig) -> ThresholdSet:
    path = _threshold_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"C1～C3 啟動前必須先有 C0 固定 F1 thresholds：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ThresholdSet.from_mapping(_mapping(payload["thresholds"], "thresholds"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _scores(result: ScreenValidationResult, *, pose_enabled: bool) -> dict[str, float]:
    detect = float(result.metrics["detect"]["box"]["ap"]["map50_95"])
    pose = float(result.metrics["pose"]["keypoints"]["ap"]["map50_95"]) if pose_enabled else 0.0
    return {
        "detect": detect,
        "pose_research": pose,
        "pose_official": (
            float(result.metrics["pose"]["official_combined_fitness"]) if pose_enabled else 0.0
        ),
        "joint_screening": (detect + 0.25 * pose) / (1.25 if pose_enabled else 1.0),
    }


def _lineage(
    config: ScreenRunConfig,
    runtime: _Runtime,
    *,
    candidate: str,
    microbatch: int,
) -> dict[str, Any]:
    candidate_files = {
        "C0": PROJECT_ROOT / "configs/candidates/c0.yaml",
        "C1": PROJECT_ROOT / "configs/candidates/c1-e0375.yaml",
        "C2": PROJECT_ROOT / "configs/candidates/c2-n1.yaml",
        "C3": PROJECT_ROOT / "configs/candidates/c3-mixed.yaml",
    }
    return {
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "architecture_yaml": str(candidate_files[candidate]),
        "architecture_yaml_sha256": file_sha256(candidate_files[candidate]),
        "training_yaml": str(config.path),
        "training_yaml_sha256": file_sha256(config.path),
        "detect_dataset_yaml": str(config.detect_data),
        "detect_dataset_yaml_sha256": file_sha256(config.detect_data),
        "pose_dataset_yaml": str(config.pose_data),
        "pose_dataset_yaml_sha256": file_sha256(config.pose_data),
        "screening_manifest": str(config.screen_manifest),
        "screening_manifest_sha256": file_sha256(config.screen_manifest),
        "handoff_manifest": str(config.handoff_manifest),
        "handoff_manifest_sha256": file_sha256(config.handoff_manifest),
        "parent_checkpoint": str(runtime.release.checkpoint),
        "parent_checkpoint_sha256": file_sha256(runtime.release.checkpoint),
        "candidate": candidate,
        "seed": config.seed,
        "effective_detect_microbatch": microbatch,
    }


def _save_snapshot(
    runtime: _Runtime,
    config: ScreenRunConfig,
    run_dir: Path,
    *,
    label: str,
    next_epoch: int,
    global_macro_step: int,
    resolved_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    loader_state: Mapping[str, Any],
    best_state: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    checkpoint = run_dir / "checkpoints" / f"{label}.pt"
    saved = runtime.api.save_training_snapshot(
        checkpoint,
        model=runtime.model,
        ema=runtime.ema,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        scaler=runtime.scaler,
        criteria=runtime.router,
        progress=runtime.api.TrainingProgress(
            stage=str(getattr(config, "stage_name", "float20")),
            next_epoch=next_epoch,
            global_macro_step=global_macro_step,
            joint_epochs_completed=next_epoch,
        ),
        resolved_config=resolved_config,
        provenance=provenance,
        loader_state=loader_state,
        best_state=best_state,
    )
    if label != "last":
        runtime.api.save_inference_weights(
            run_dir / "inference" / f"{label}.pt",
            model=runtime.model,
            ema=runtime.ema,
            use_ema=True,
            metadata={
                "candidate": resolved_config["candidate"],
                "epoch": next_epoch - 1,
                "metrics": dict(metrics),
                "full_resume_sha256": saved.sha256,
            },
        )
    return saved.path


def run_screen_candidate(
    config: ScreenRunConfig,
    *,
    candidate: str,
    microbatch: int,
) -> dict[str, Any]:
    """訓練一個候選；每 epoch 保存 exact resume，完成後不自動選 C_best。"""

    candidate = candidate.upper()
    if candidate not in config.candidates:
        raise ValueError(f"候選不在正式矩陣：{candidate}")
    run_dir = config.run_root / f"{candidate.lower()}-control-seed{config.seed}"
    completed = run_dir / "complete.json"
    if completed.is_file():
        payload = json.loads(completed.read_text(encoding="utf-8"))
        payload["already_complete"] = True
        return payload
    runtime = _make_runtime(config, candidate=candidate, microbatch=microbatch)
    resolved_config = config.resolved_payload(candidate=candidate, microbatch=microbatch)
    lineage = _lineage(
        config,
        runtime,
        candidate=candidate,
        microbatch=microbatch,
    )
    provenance = {
        "lineage": lineage,
        "full35_factory": runtime.parent.factory_report,
        "parent_checkpoint": runtime.parent.checkpoint_report,
        "candidate_build": runtime.build_report.to_dict(),
        "optimizer": asdict(runtime.optimizer_report),
        "screening_only": True,
        "formal_split_used": False,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run-manifest.json", provenance)
    (run_dir / "resolved-config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    fixed_thresholds = None if candidate == "C0" else _read_thresholds(config)
    validator = ScreenValidator(
        runtime.release,
        runtime.parent.source,
        runtime.resolved,
        detect_data=config.detect_data,
        pose_data=config.pose_data,
        output_root=run_dir / "validation",
        imgsz=config.imgsz,
        detect_batch=config.detect_val_batch,
        pose_batch=config.pose_val_batch,
        detect_workers=config.detect_workers,
        pose_workers=config.pose_workers,
        device=config.device,
        pose_enabled=config.pose_enabled,
        runtime_cache_root=config.screen_root / "runtime-cache",
    )
    best_state: dict[str, Any] = {
        "detect": {"score": -1.0, "epoch": -1},
        "pose_research": {"score": -1.0, "epoch": -1},
        "pose_official": {"score": -1.0, "epoch": -1},
        "joint_screening": {"score": -1.0, "epoch": -1},
        "c0_thresholds": None,
    }
    start_epoch = 0
    global_macro = 0
    last_checkpoint = run_dir / "checkpoints/last.pt"
    if last_checkpoint.is_file():
        restored = runtime.api.load_training_snapshot(
            last_checkpoint,
            model=runtime.model,
            ema=runtime.ema,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            scaler=runtime.scaler,
            criteria=runtime.router,
            restore_rng=True,
        )
        if restored.resolved_config != resolved_config:
            raise ValueError("resume effective config 與目前 YAML/candidate/microbatch 不一致")
        start_epoch = restored.progress.next_epoch
        global_macro = restored.progress.global_macro_step
        best_state = restored.best_state
    else:
        runtime.api.seed_everything(config.seed)

    engine = _engine(runtime, config)
    loader_state: dict[str, Any] = {
        "effective_detect_microbatch": microbatch,
        "detect_logical_batch": config.detect_logical_batch,
        "detect_physical_microbatches_per_macro": runtime.microbatches_per_macro,
        "pose_batch": config.pose_batch if config.pose_enabled else 0,
        "validation_detect_batch": config.detect_val_batch,
        "validation_pose_batch": config.pose_val_batch,
    }
    with runtime.api.ExperimentLogger(
        run_dir / "logs",
        tensorboard="auto",
    ) as logger:
        runner = (
            runtime.api.JointEpochRunner(
                engine=engine,
                detect_loader=runtime.detect_loader.loader,
                pose_loader=runtime.pose_loader.loader,
                scheduler=runtime.scheduler,
                logger=logger,
                apply_training_mode=lambda: _apply_training_mode(runtime),
                assert_hardware_contract=lambda: runtime.guard.assert_unchanged(runtime.model),
                detect_batches_per_macro=runtime.microbatches_per_macro,
                gradient_statistics_interval=100,
            )
            if runtime.pose_loader is not None
            else None
        )
        for epoch in range(start_epoch, config.epochs):
            detect_seed = runtime.api.reseed_loader_for_epoch(
                runtime.detect_loader.loader,
                seed=config.seed,
                epoch=epoch,
                offset=0,
            )
            pose_seed = None
            if runtime.pose_loader is not None:
                pose_seed = runtime.api.reseed_loader_for_epoch(
                    runtime.pose_loader.loader,
                    seed=config.seed,
                    epoch=epoch,
                    offset=1,
                )
                if runner is None:
                    raise AssertionError("Pose loader 缺少 joint runner")
                training = runner.run_epoch(
                    epoch=epoch,
                    global_macro_step=global_macro,
                    stage="float20",
                )
            else:
                training = _run_detect_only_epoch(
                    runtime,
                    config,
                    engine=engine,
                    logger=logger,
                    epoch=epoch,
                    global_macro_step=global_macro,
                )
            global_macro = training.next_global_macro_step
            if (epoch + 1) % config.validation_interval:
                raise RuntimeError("本 revision 要求每 epoch validation")
            validation = validator.validate(
                runtime.ema.ema,
                epoch=epoch,
                fixed_thresholds=fixed_thresholds,
            )
            logger.log(
                "validation",
                step=epoch,
                values=validation.flat_metrics,
                context={
                    "candidate": candidate,
                    "backend": "float",
                    "screening_only": True,
                },
            )
            scores = _scores(validation, pose_enabled=config.pose_enabled)
            selected: list[str] = []
            for label, score in scores.items():
                if score > float(best_state[label]["score"]):
                    best_state[label] = {"score": score, "epoch": epoch}
                    selected.append(f"best-{label.replace('_', '-')}")
            if candidate == "C0" and "best-joint-screening" in selected:
                best_state["c0_thresholds"] = validation.thresholds.to_dict()
            loader_state.update(
                {
                    "next_epoch": epoch + 1,
                    "detect_seed": detect_seed,
                    "pose_seed": pose_seed,
                }
            )
            checkpoint_metrics = {
                "scores": scores,
                "thresholds": validation.thresholds.to_dict(),
            }
            _save_snapshot(
                runtime,
                config,
                run_dir,
                label="last",
                next_epoch=epoch + 1,
                global_macro_step=global_macro,
                resolved_config=resolved_config,
                provenance=provenance,
                loader_state=loader_state,
                best_state=best_state,
                metrics=checkpoint_metrics,
            )
            for label in selected:
                _save_snapshot(
                    runtime,
                    config,
                    run_dir,
                    label=label,
                    next_epoch=epoch + 1,
                    global_macro_step=global_macro,
                    resolved_config=resolved_config,
                    provenance=provenance,
                    loader_state=loader_state,
                    best_state=best_state,
                    metrics=checkpoint_metrics,
                )

    if candidate == "C0":
        thresholds = best_state.get("c0_thresholds")
        if not isinstance(thresholds, dict):
            raise RuntimeError("C0 完成但沒有 best-joint F1 thresholds")
        _write_json(
            _threshold_path(config),
            {
                "schema_version": 1,
                "source_candidate": "C0",
                "source_epoch": best_state["joint_screening"]["epoch"],
                "source_split": "train_only_search_val",
                "formal_split_used": False,
                "thresholds": thresholds,
                "training_yaml_sha256": file_sha256(config.path),
            },
        )
    summary = {
        "schema_version": 1,
        "candidate": candidate,
        "status": "completed_screening",
        "screening_only": True,
        "formal_split_used": False,
        "epochs_completed": config.epochs,
        "global_macro_steps": global_macro,
        "best_state": best_state,
        "selection_status": "pending_user_decision",
        "c_best": None,
        "quantization_eligible": (True if candidate == "C0" else "pending_user_decision"),
        "run_dir": str(run_dir),
        "lineage": lineage,
    }
    _write_json(completed, summary)
    return summary


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def run_screen_matrix(config_path: str | Path, *, execute: bool) -> dict[str, Any]:
    """先鎖定一個全矩陣 microbatch，再依序跑 C0～C3。"""

    config = ScreenRunConfig.load(config_path)
    plan = {
        "config": str(config.path),
        "config_sha256": file_sha256(config.path),
        "candidates": list(config.candidates),
        "pose_enabled": config.pose_enabled,
        "logical_detect_batch": config.detect_logical_batch,
        "requested_detect_microbatch": config.detect_microbatch,
        "oom_fallback_microbatch": config.detect_oom_fallback_microbatch,
        "validation_batch": {
            "detect": config.detect_val_batch,
            "pose": config.pose_val_batch,
        },
        "execute": execute,
    }
    if not execute:
        plan["status"] = "dry_run_only"
        return plan

    batch_plan_path = config.run_root / "shared-controls" / "batch-plan.json"
    if batch_plan_path.is_file():
        batch_plan = json.loads(batch_plan_path.read_text(encoding="utf-8"))
        microbatch = int(batch_plan["selected_detect_microbatch"])
    else:
        probes: list[dict[str, Any]] = []
        try:
            report = probe_screen_memory(
                config,
                microbatch=config.detect_microbatch,
            )
            probes.append(report)
            microbatch = config.detect_microbatch
        except Exception as error:
            if not _is_oom(error):
                raise
            probes.append(
                {
                    "passed": False,
                    "physical_detect_microbatch": config.detect_microbatch,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()
            report = probe_screen_memory(
                config,
                microbatch=config.detect_oom_fallback_microbatch,
            )
            probes.append(report)
            microbatch = config.detect_oom_fallback_microbatch
        batch_plan = {
            "schema_version": 1,
            "fairness_scope": "C0-C1-C2-C3",
            "logical_detect_batch": config.detect_logical_batch,
            "selected_detect_microbatch": microbatch,
            "detect_physical_microbatches_per_macro": (
                config.detect_logical_batches_per_macro * config.detect_logical_batch // microbatch
            ),
            "pose_batch": config.pose_batch if config.pose_enabled else 0,
            "validation_detect_batch": config.detect_val_batch,
            "validation_pose_batch": config.pose_val_batch,
            "probes": probes,
        }
        _write_json(batch_plan_path, batch_plan)

    results = [
        run_screen_candidate(config, candidate=candidate, microbatch=microbatch)
        for candidate in config.candidates
    ]
    matrix = {
        **plan,
        "status": "completed_screening_matrix",
        "selected_detect_microbatch": microbatch,
        "results": results,
        "selection_status": "pending_user_decision",
        "c_best": None,
        "next_gate": "user_reviews_float20_before_ptq_or_qat_lite",
    }
    _write_json(config.run_root / "matrix-complete.json", matrix)
    return matrix
