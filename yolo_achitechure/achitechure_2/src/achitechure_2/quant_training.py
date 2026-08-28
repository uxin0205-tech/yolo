"""C2/C3 正式權重的 Q0、W8A8 PTQ 與 200-step QAT-lite 模擬。"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from ultralytics.utils.torch_utils import ModelEMA

from .config import file_sha256
from .freezing import FrozenStateGuard, enforce_frozen_eval
from .full_training import FullRunConfig
from .profiling import _load_inference_checkpoint
from .quantization import (
    configure_qat_lite_step,
    prepare_w8a8_simulation,
    quant_scope_dict,
    quantization_gap_report,
    set_fake_quant,
    set_observers,
)
from .screen_training import (
    _AutocastRouter,
    _engine,
    _make_runtime,
    _write_json,
)
from .screen_validation import (
    ScreenValidationResult,
    ThresholdSet,
    _flat,
    _metric_payload,
    _RuntimeDetectionValidator,
    _RuntimePoseValidator,
)


@dataclass(frozen=True)
class _QuantRuntimeConfig:
    full: FullRunConfig
    epochs: int = 3
    patience: int = 0
    stage_name: str = "qat-lite"
    amp: bool = False
    warmup_epochs: int = 0

    def __getattr__(self, name: str) -> Any:
        if name == "path":
            return self.full.path
        if name == "payload":
            return self.full.payload
        if name == "candidates":
            return self.full.candidates
        return getattr(self.full.base, name)

    @property
    def learning_rates(self) -> dict[str, float]:
        return {
            name: (0.0 if value == 0.0 else float(value) * 0.1)
            for name, value in self.full.base.learning_rates.items()
        }


class QuantRouteModel(nn.Module):
    """讓官方 Ultralytics validator 只看到 shared model 的單一路由輸出。"""

    def __init__(self, shared: nn.Module, task: str) -> None:
        super().__init__()
        if task not in {"detect", "pose"}:
            raise ValueError("quant route 只允許 detect 或 pose")
        self.shared = shared
        self.task = task
        head = shared.detect_head if task == "detect" else shared.pose_head
        self.stride = head.stride
        self.names = dict(
            shared.detect_names if task == "detect" else shared.pose_names
        )
        self.end2end = bool(getattr(head, "end2end", False))
        if task == "pose":
            self.kpt_shape = tuple(int(value) for value in head.kpt_shape)

    def forward(
        self,
        images: torch.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed: Any = None,
        **kwargs: Any,
    ) -> Any:
        del augment, visualize, embed, kwargs
        return self.shared(images, task=self.task)[self.task]


def _validator_args(
    *,
    task: str,
    data: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> dict[str, Any]:
    return {
        "task": task,
        "data": str(data),
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "plots": False,
        "save_json": False,
        "compile": False,
        "rect": True,
        "split": "val",
        "mode": "val",
        "verbose": False,
    }


def validate_quant_routes(
    model: nn.Module,
    config: FullRunConfig,
    *,
    output_root: Path,
    backend: str,
    event: int,
    thresholds: ThresholdSet,
) -> ScreenValidationResult:
    """直接驗證 fake-quant shared graph，避免 materialize 遺失 wrapper state。"""

    root = output_root / backend / f"event-{event:04d}"
    trainability = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    detect_validator = _RuntimeDetectionValidator(
        save_dir=root / "detect",
        label_cache_path=config.screen_root / "quant-validation/detect-val.cache",
        args=_validator_args(
            task="detect",
            data=config.detect_data,
            imgsz=config.imgsz,
            batch=config.detect_val_batch,
            workers=config.detect_workers,
            device=config.device,
        ),
    )
    detect_raw = detect_validator(model=QuantRouteModel(model, "detect"))
    if not isinstance(detect_raw, dict):
        raise TypeError("Q0/Q1/Q2L Detect validator 未回傳 metrics mapping")
    detect_names = {
        int(key): str(value)
        for key, value in model.detect_names.items()
    }
    detect_support = np.asarray(detect_validator.metrics.nt_per_class)
    detect, detect_threshold = _metric_payload(
        detect_validator.metrics.box,
        names=detect_names,
        supports=detect_support,
        threshold=thresholds.detect_box,
    )

    pose_validator = _RuntimePoseValidator(
        save_dir=root / "pose",
        label_cache_path=config.screen_root / "quant-validation/pose-val.cache",
        args=_validator_args(
            task="pose",
            data=config.pose_data,
            imgsz=config.imgsz,
            batch=config.pose_val_batch,
            workers=config.pose_workers,
            device=config.device,
        ),
    )
    pose_raw = pose_validator(model=QuantRouteModel(model, "pose"))
    if not isinstance(pose_raw, dict):
        raise TypeError("Q0/Q1/Q2L Pose validator 未回傳 metrics mapping")
    pose_names = {
        int(key): str(value)
        for key, value in model.pose_names.items()
    }
    pose_support = np.asarray(pose_validator.metrics.nt_per_class)
    pose_box, pose_box_threshold = _metric_payload(
        pose_validator.metrics.box,
        names=pose_names,
        supports=pose_support,
        threshold=thresholds.pose_box,
    )
    pose_keypoints, pose_keypoint_threshold = _metric_payload(
        pose_validator.metrics.pose,
        names=pose_names,
        supports=pose_support,
        threshold=thresholds.pose_keypoints,
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(trainability[name])

    resolved = ThresholdSet(
        detect_box=detect_threshold,
        pose_box=pose_box_threshold,
        pose_keypoints=pose_keypoint_threshold,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "scope": "formal_coco_val2017_and_bbat5_val",
        "formal_split_used": True,
        "event": event,
        "backend": backend,
        "simulation_only": backend != "q0-float",
        "detect": {"box": detect},
        "pose": {
            "status": "measured",
            "box": pose_box,
            "keypoints": pose_keypoints,
            "official_combined_fitness": float(pose_validator.metrics.fitness),
        },
        "thresholds": resolved.to_dict(),
        "detect_raw": {str(key): float(value) for key, value in detect_raw.items()},
        "pose_raw": {str(key): float(value) for key, value in pose_raw.items()},
    }
    _write_json(root / "metrics.json", payload)
    return ScreenValidationResult(
        epoch=event,
        metrics=payload,
        flat_metrics=_flat({"detect": payload["detect"], "pose": payload["pose"]}),
        thresholds=resolved,
        output_dir=root,
    )


def _metric_vector(result: ScreenValidationResult) -> dict[str, float]:
    return {
        "coco_box_map50_95": float(
            result.metrics["detect"]["box"]["ap"]["map50_95"]
        ),
        "bbat5_pose_box_map50_95": float(
            result.metrics["pose"]["box"]["ap"]["map50_95"]
        ),
        "bbat5_keypoint_map50_95": float(
            result.metrics["pose"]["keypoints"]["ap"]["map50_95"]
        ),
        "macro_f1": float(result.metrics["pose"]["keypoints"]["f1"]["macro_f1"]),
        "micro_f1": float(result.metrics["pose"]["keypoints"]["f1"]["micro_f1"]),
    }


def _thresholds(config: FullRunConfig) -> ThresholdSet:
    path = config.base.run_root / "shared-controls/c0-f1-thresholds.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("thresholds")
    if not isinstance(value, dict):
        raise TypeError("C0 threshold payload 漂移")
    return ThresholdSet.from_mapping(value)


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        result: list[torch.Tensor] = []
        for child in value.values():
            result.extend(_tensor_leaves(child))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for child in value:
            result.extend(_tensor_leaves(child))
        return result
    return []


def _fusion_equivalence(runtime: Any, *, imgsz: int) -> dict[str, Any]:
    materialized = runtime.release.materialize_validation_models(
        runtime.model,
        runtime.parent.source,
        runtime.resolved,
        kind="float",
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    images = torch.rand(1, 3, imgsz, imgsz, generator=generator)
    report: dict[str, Any] = {}
    for task, source in (
        ("detect", materialized.detect),
        ("pose", materialized.pose),
    ):
        source = source.cpu().eval()
        fused = copy.deepcopy(source).eval()
        fused.fuse(verbose=False)
        with torch.inference_mode():
            before = _tensor_leaves(source(images))
            after = _tensor_leaves(fused(images))
        if len(before) != len(after) or not before:
            raise ValueError(f"Q0 {task} fuse 輸出結構漂移")
        maximum = max(
            float((left.float() - right.float()).abs().max())
            for left, right in zip(before, after)
        )
        if not math.isfinite(maximum) or maximum > 0.0001:
            raise ValueError(f"Q0 {task} fuse 等價性失敗：{maximum}")
        report[task] = {
            "max_abs_diff": maximum,
            "limit": 0.0001,
            "passed": True,
        }
    return report


def _calibrate(
    model: nn.Module,
    runtime: Any,
    *,
    max_batches_per_task: int,
) -> dict[str, int]:
    model.eval()
    set_fake_quant(model, enabled=False)
    set_observers(model, enabled=True)
    counts = {"detect": 0, "pose": 0}
    with torch.inference_mode():
        for raw in runtime.detect_loader.loader:
            batch = runtime.detect_loader.preprocess(raw)
            model(batch["img"], task="detect")
            counts["detect"] += 1
            if counts["detect"] >= max_batches_per_task:
                break
        for raw in runtime.pose_loader.loader:
            batch = runtime.pose_loader.preprocess(raw)
            model(batch["img"], task="pose")
            counts["pose"] += 1
            if counts["pose"] >= max_batches_per_task:
                break
    if min(counts.values()) < max_batches_per_task:
        raise RuntimeError(f"calibration batches 不足：{counts}")
    set_observers(model, enabled=False)
    set_fake_quant(model, enabled=True)
    return counts


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def _quant_lineage(
    config: FullRunConfig,
    *,
    candidate: str,
    full_checkpoint: Path,
) -> dict[str, Any]:
    """把完整Float lineage延伸成量化checkpoint的直接parent lineage。"""

    full_complete_path = full_checkpoint.parents[1] / "complete.json"
    full_complete = json.loads(full_complete_path.read_text(encoding="utf-8"))
    base = full_complete.get("lineage")
    if not isinstance(base, dict):
        raise TypeError(f"{candidate} full complete缺少lineage")
    required = {
        "spec_version",
        "spec_sha256",
        "architecture_yaml",
        "architecture_yaml_sha256",
        "training_yaml",
        "training_yaml_sha256",
        "detect_dataset_yaml",
        "detect_dataset_yaml_sha256",
        "pose_dataset_yaml",
        "pose_dataset_yaml_sha256",
        "handoff_manifest",
        "handoff_manifest_sha256",
        "parent_checkpoint",
        "parent_checkpoint_sha256",
        "candidate",
    }
    missing = sorted(required - set(base))
    if missing or base.get("candidate") != candidate:
        raise ValueError(
            f"{candidate} full lineage漂移：missing={missing} "
            f"lineage_candidate={base.get('candidate')}"
        )
    policy = Path(__file__).resolve().parents[2] / "configs/quant/w8a8-simulation.yaml"
    return {
        **base,
        "architecture_parent_checkpoint": base["parent_checkpoint"],
        "architecture_parent_checkpoint_sha256": base["parent_checkpoint_sha256"],
        "parent_checkpoint": str(full_checkpoint),
        "parent_checkpoint_sha256": file_sha256(full_checkpoint),
        "quant_run_yaml": str(config.path),
        "quant_run_yaml_sha256": file_sha256(config.path),
        "quantization_policy_yaml": str(policy),
        "quantization_policy_yaml_sha256": file_sha256(policy),
        "full_complete": str(full_complete_path),
        "full_complete_sha256": file_sha256(full_complete_path),
    }



def _qat_runtime(runtime: Any, config: _QuantRuntimeConfig, model: nn.Module) -> Any:
    rates = dict(config.learning_rates)
    stage = runtime.api.JointStage(
        name=config.stage_name,
        task_mode="joint",
        epochs=config.epochs,
        patience=0,
        backbone_start_layer=0,
        tune_attention=False,
        learning_rates=rates,
        warmup_epochs=0,
    )
    optimizer, optimizer_report = runtime.api.build_joint_optimizer(
        model,
        stage,
        optimizer_name=config.optimizer,
        weight_decay=config.weight_decay,
        beta1=config.momentum,
        beta2=0.999,
    )
    enforce_frozen_eval(model, runtime.release.frozen_module_paths)
    guard = FrozenStateGuard.capture(
        model,
        runtime.release.frozen_module_paths,
        reset_trainable=False,
    )
    router = _AutocastRouter(
        runtime.api.NativeTaskLossRouter(
            model,
            epochs=config.epochs,
            imgsz=config.imgsz,
            detect_overrides=config.loss_overrides["detect"],
            pose_overrides=config.loss_overrides["pose"],
        ),
        device=runtime.device,
        enabled=False,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = ModelEMA(model, decay=0.9999, tau=2000)
    scheduler = runtime.api.StageWarmupCosineScheduler(
        optimizer,
        stage=config.stage_name,
        epochs=1,
        steps_per_epoch=200,
        warmup_epochs=0,
        warmup_start_factor=1.0,
        final_lr_factor=config.final_lr_factor,
    )
    runtime.model = model
    runtime.stage = stage
    runtime.optimizer = optimizer
    runtime.optimizer_report = optimizer_report
    runtime.guard = guard
    runtime.router = router
    runtime.scaler = scaler
    runtime.ema = ema
    runtime.scheduler = scheduler
    return runtime


def _take_cycling(
    iterator: Any,
    loader: Any,
    count: int,
) -> tuple[tuple[Mapping[str, Any], ...], Any, int]:
    values: list[Mapping[str, Any]] = []
    wraps = 0
    for _ in range(count):
        try:
            values.append(next(iterator))
        except StopIteration:
            wraps += 1
            iterator = iter(loader)
            values.append(next(iterator))
    return tuple(values), iterator, wraps


def _run_qat_lite(
    runtime: Any,
    config: FullRunConfig,
    quant_config: _QuantRuntimeConfig,
    model: nn.Module,
    *,
    candidate: str,
    output_root: Path,
    thresholds: ThresholdSet,
    lineage: dict[str, Any],
) -> tuple[ScreenValidationResult, dict[str, Any]]:
    runtime = _qat_runtime(runtime, quant_config, model)
    runtime.api.seed_everything(config.seed)
    runtime.api.apply_stage(runtime.model, runtime.stage)
    enforce_frozen_eval(runtime.model, runtime.release.frozen_module_paths)
    engine = _engine(runtime, quant_config)
    detect_iterator = iter(runtime.detect_loader.loader)
    pose_iterator = iter(runtime.pose_loader.loader)
    total_steps = int(config.payload["quantization"]["qat_lite_steps"])
    observer_steps = int(config.payload["quantization"]["observer_update_steps"])
    validation_interval = int(
        config.payload["quantization"]["validation_interval_steps"]
    )
    events: list[dict[str, Any]] = []
    detect_wraps = 0
    pose_wraps = 0
    final_validation: ScreenValidationResult | None = None
    for step in range(1, total_steps + 1):
        observers_enabled = configure_qat_lite_step(
            runtime.model,
            step,
            observer_update_steps=observer_steps,
        )
        detect_batches, detect_iterator, wrapped = _take_cycling(
            detect_iterator,
            runtime.detect_loader.loader,
            runtime.microbatches_per_macro,
        )
        detect_wraps += wrapped
        if wrapped:
            engine.advance_epoch()
        pose_batches, pose_iterator, wrapped = _take_cycling(
            pose_iterator,
            runtime.pose_loader.loader,
            1,
        )
        pose_wraps += wrapped
        lrs = dict(runtime.scheduler.prepare_step())
        training = engine.run(
            detect_batches=detect_batches,
            pose_batches=pose_batches,
            record_gradient_statistics=(step == 1 or step % 100 == 0),
        )
        runtime.scheduler.advance()
        if step % validation_interval == 0:
            final_validation = validate_quant_routes(
                runtime.model,
                config,
                output_root=output_root / "validation",
                backend="q2l-qat-lite",
                event=step,
                thresholds=thresholds,
            )
            events.append(
                {
                    "step": step,
                    "observers_enabled": observers_enabled,
                    "metrics": _metric_vector(final_validation),
                    "joint_mean_loss": float(training.joint_mean_loss),
                    "learning_rates": lrs,
                }
            )
            runtime.api.apply_stage(runtime.model, runtime.stage)
            enforce_frozen_eval(runtime.model, runtime.release.frozen_module_paths)
            runtime.guard.assert_unchanged(runtime.model)
            _atomic_torch(
                output_root / "checkpoints/qat-lite-last.pt",
                {
                    "schema_version": 1,
                    "candidate": candidate,
                    "stage": "Q2L",
                    "simulation_only": True,
                    "step": step,
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in runtime.model.state_dict().items()
                    },
                    "optimizer_state_dict": runtime.optimizer.state_dict(),
                    "scheduler_state_dict": runtime.scheduler.state_dict(),
                    "observer_update_steps": observer_steps,
                    "lineage": lineage,
                },
            )
            _write_json(
                output_root / "qat-lite-progress.json",
                {
                    "schema_version": 1,
                    "candidate": candidate,
                    "status": "running" if step < total_steps else "completed",
                    "step": step,
                    "total_steps": total_steps,
                    "events": events,
                    "detect_wraps": detect_wraps,
                    "pose_wraps": pose_wraps,
                },
            )
    if final_validation is None:
        raise RuntimeError("QAT-lite 沒有產生 validation")
    return final_validation, {
        "steps": total_steps,
        "observer_update_steps": observer_steps,
        "validation_interval_steps": validation_interval,
        "events": events,
        "detect_wraps": detect_wraps,
        "pose_wraps": pose_wraps,
        "optimizer": asdict(runtime.optimizer_report),
    }


def run_quant_candidate(
    config: FullRunConfig,
    *,
    candidate: str,
    microbatch: int,
) -> dict[str, Any]:
    config.require_execution_enabled()
    output_root = (
        Path(str(config.payload["quantization"]["result_root"])).expanduser()
    )
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parents[2] / output_root
    output_root = output_root.resolve() / candidate.lower()
    completed = output_root / "complete.json"
    if completed.is_file():
        payload = json.loads(completed.read_text(encoding="utf-8"))
        payload["already_complete"] = True
        return payload

    quant_config = _QuantRuntimeConfig(config)
    runtime = _make_runtime(
        quant_config,
        candidate=candidate,
        microbatch=microbatch,
    )
    full_checkpoint = (
        config.run_root
        / f"{candidate.lower()}-full-seed{config.seed}"
        / "inference/best-joint-formal.pt"
    )
    if not full_checkpoint.is_file():
        raise FileNotFoundError(full_checkpoint)
    checkpoint_payload = _load_inference_checkpoint(runtime.model, full_checkpoint)
    if checkpoint_payload["metadata"].get("candidate") != candidate:
        raise ValueError("full checkpoint candidate metadata 漂移")
    thresholds = _thresholds(config)
    lineage = _quant_lineage(
        config,
        candidate=candidate,
        full_checkpoint=full_checkpoint,
    )
    q0 = validate_quant_routes(
        runtime.model,
        config,
        output_root=output_root / "validation",
        backend="q0-float",
        event=0,
        thresholds=thresholds,
    )
    fusion_equivalence = _fusion_equivalence(runtime, imgsz=64)

    float_model = runtime.model.cpu()
    # Q0 後不再使用預建的 optimizer/EMA/router；先釋放，避免量化 deep-copy
    # 時同時保留多份 GPU model。
    runtime.optimizer = None
    runtime.ema = None
    runtime.router = None
    runtime.scheduler = None
    runtime.scaler = None
    runtime.guard = None
    gc.collect()
    torch.cuda.empty_cache()
    q1_model, scope = prepare_w8a8_simulation(float_model)
    del float_model
    runtime.model = q1_model.to(runtime.device)
    calibration = _calibrate(
        runtime.model,
        runtime,
        max_batches_per_task=int(
            config.payload["quantization"]["calibration_batches_per_task"]
        ),
    )
    q1 = validate_quant_routes(
        runtime.model,
        config,
        output_root=output_root / "validation",
        backend="q1-ptq",
        event=0,
        thresholds=thresholds,
    )
    _atomic_torch(
        output_root / "checkpoints/ptq-calibrated.pt",
        {
            "schema_version": 1,
            "candidate": candidate,
            "stage": "Q1",
            "simulation_only": True,
            "scope": quant_scope_dict(scope),
            "calibration": calibration,
            "lineage": lineage,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in runtime.model.state_dict().items()
            },
        },
    )
    q2l, training = _run_qat_lite(
        runtime,
        config,
        quant_config,
        runtime.model,
        candidate=candidate,
        output_root=output_root,
        thresholds=thresholds,
        lineage=lineage,
    )
    gap = quantization_gap_report(
        q0=_metric_vector(q0),
        q1=_metric_vector(q1),
        q2=_metric_vector(q2l),
    )
    max_drop = float(config.payload["quantization"]["max_accuracy_drop"])
    q1_compatible = all(value <= max_drop for value in gap.q1_drop.values())
    q2l_compatible = all(value <= max_drop for value in gap.q2_drop.values())
    report = {
        "schema_version": 1,
        "candidate": candidate,
        "status": "completed_q0_q1_q2l",
        "simulation_only": True,
        "full_checkpoint": str(full_checkpoint),
        "full_checkpoint_sha256": file_sha256(full_checkpoint),
        "lineage": lineage,
        "q0_fusion_equivalence": fusion_equivalence,
        "scope": quant_scope_dict(scope),
        "calibration": calibration,
        "qat_lite": training,
        "metrics": {
            "q0": _metric_vector(q0),
            "q1": _metric_vector(q1),
            "q2l": _metric_vector(q2l),
        },
        "gaps": asdict(gap),
        "max_accuracy_drop": max_drop,
        "ptq_compatible": q1_compatible,
        "qat_lite_compatible": q2l_compatible,
        "deployment_int8_validated": False,
    }
    _write_json(completed, report)
    return report


def run_quant_matrix(config_path: str | Path, *, execute: bool) -> dict[str, Any]:
    config = FullRunConfig.load(config_path)
    if execute:
        config.require_execution_enabled()
    matrix_path = config.run_root / "matrix-complete.json"
    if not matrix_path.is_file():
        raise FileNotFoundError("正式 C2/C3 full matrix 尚未完成")
    full_matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    status = full_matrix.get("status")
    candidates = tuple(str(value) for value in full_matrix.get("eligible_candidates", []))
    plan = {
        "config": str(config.path),
        "candidates": list(candidates),
        "stages": ["Q0", "Q1-PTQ", "Q2L-QAT-lite"],
        "simulation_only": True,
        "execute": execute,
    }
    if not execute:
        return {**plan, "status": "dry_run_only"}
    if status == "completed_no_eligible_candidates":
        report = {
            **plan,
            "status": "completed_no_eligible_candidates",
            "results": [],
        }
        _write_json(
            Path(str(config.payload["quantization"]["result_root"]))
            / "matrix-complete.json",
            report,
        )
        return report
    if status != "completed_formal_training_matrix":
        raise ValueError(f"full matrix status 漂移：{status}")
    batch_plan = json.loads(
        (config.run_root / "shared-controls/batch-plan.json").read_text(
            encoding="utf-8"
        )
    )
    microbatch = int(batch_plan["selected_detect_microbatch"])
    results = [
        run_quant_candidate(
            config,
            candidate=candidate,
            microbatch=microbatch,
        )
        for candidate in candidates
    ]
    report = {
        **plan,
        "status": "completed_q0_q1_q2l_matrix",
        "selected_detect_microbatch": microbatch,
        "results": results,
    }
    destination = Path(str(config.payload["quantization"]["result_root"]))
    if not destination.is_absolute():
        destination = Path(__file__).resolve().parents[2] / destination
    _write_json(destination.resolve() / "matrix-complete.json", report)
    return report
