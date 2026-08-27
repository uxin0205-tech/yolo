"""Real-data, two-macro CPU acceptance for the graph-shared trainer."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from ultralytics.utils.torch_utils import ModelEMA

from .contracts import Task
from .data import prepare_bbt5_view, prepare_coco_detect_subset
from .factory import FusionModelFactory
from .hardware_contract import HardwareContractGuard
from .joint_config import JointExperimentConfig
from .joint_data import TaskLoaderSettings, build_task_loader
from .joint_loss import MacroStepEngine, NativeTaskLossRouter
from .source import SourceBundle
from .stage_policy import JOINT_STAGES, apply_stage, build_joint_optimizer
from .xnor import XNORExecutionConfig


def _bn_state(model: nn.Module, *, heads: bool) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        is_head = ".detect_head." in name or ".pose_head." in name
        if is_head != heads or not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        state[name] = module.running_mean.detach().cpu().clone()
    if not state:
        raise ValueError("expected BatchNorm state is empty")
    return state


def _changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> tuple[str, ...]:
    if set(before) != set(after):
        raise ValueError("BatchNorm paths changed during smoke")
    return tuple(name for name in before if not torch.equal(before[name], after[name]))


def run_real_cpu_smoke(
    config: JointExperimentConfig,
    *,
    output_root: str | Path,
    pose_checkpoint: str | Path | None = None,
    steps: int = 2,
    imgsz: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Exercise real COCO/BBAT batches without changing formal readiness."""

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU smoke refuses to run after CUDA initialization")
    if steps < 2:
        raise ValueError("formal CPU smoke requires at least two macro-steps")
    if imgsz < 64 or imgsz % 32:
        raise ValueError("CPU smoke imgsz must be >=64 and divisible by 32")
    torch.manual_seed(seed)
    output = Path(output_root).expanduser().resolve()
    selected_pose = (
        Path(pose_checkpoint).expanduser().resolve()
        if pose_checkpoint is not None
        else config.provisional_pose_checkpoint
    )
    if selected_pose is None or not selected_pose.is_file():
        raise FileNotFoundError(
            "CPU smoke needs an architecture-compatible Pose26 checkpoint"
        )
    source = SourceBundle(config.source_bundle, architecture=config.architecture)
    built = FusionModelFactory(
        source,
        detect_data_yaml=config.detect_data,
        pose_data_yaml=config.pose_data,
        xnor=XNORExecutionConfig(token_tile=config.xnor_token_tile),
    ).build(
        pose_head_checkpoint=selected_pose,
        checkpoint_kind="float",
    )
    model = built.model.cpu()
    stage = JOINT_STAGES["j1"]
    stage_report = apply_stage(model, stage)
    guard = HardwareContractGuard.capture(model)
    pose_view = prepare_bbt5_view(
        config.registry,
        output / "datasets" / "bbat5-v1-runtime",
    )
    detect_view = prepare_coco_detect_subset(
        config.detect_data,
        output / "datasets" / "coco-smoke",
        limit=steps * 2,
    )
    detect = build_task_loader(
        model,
        data_yaml=detect_view.yaml,
        settings=TaskLoaderSettings.for_detect(
            batch_size=1,
            workers=0,
            imgsz=imgsz,
            fraction=1.0,
            seed=seed,
        ),
        device=torch.device("cpu"),
        registry=config.registry,
    )
    pose = build_task_loader(
        model,
        data_yaml=pose_view.yaml,
        settings=TaskLoaderSettings.for_pose(
            batch_size=1,
            workers=0,
            imgsz=imgsz,
            fraction=0.001,
            seed=seed,
        ),
        device=torch.device("cpu"),
        registry=config.registry,
    )
    optimizer, optimizer_report = build_joint_optimizer(
        model,
        stage,
        optimizer_name="AdamW",
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
    )
    router = NativeTaskLossRouter(model, epochs=1, imgsz=imgsz)
    ema = ModelEMA(model, decay=0.9999, tau=2000)
    groups = {
        "shared": tuple(
            parameter
            for name, parameter in model.named_parameters()
            if ".detect_head." not in name and ".pose_head." not in name
        ),
        "detect_head": tuple(
            parameter for name, parameter in model.named_parameters() if ".detect_head." in name
        ),
        "pose_head": tuple(
            parameter for name, parameter in model.named_parameters() if ".pose_head." in name
        ),
    }
    engine = MacroStepEngine(
        model=model,
        losses=router,
        optimizer=optimizer,
        reference_batch_size=config.reference_batch_size,
        task_weights={Task.DETECT: config.detect_weight, Task.POSE: config.pose_weight},
        gradient_groups=groups,
        ema=ema,
        max_grad_norm=config.gradient_clip_norm,
        preprocess=lambda task, batch: (
            detect.preprocess(batch) if task is Task.DETECT else pose.preprocess(batch)
        ),
    )
    shared_before = _bn_state(model, heads=False)
    heads_before = _bn_state(model, heads=True)
    detect_iterator = iter(detect.loader)
    pose_iterator = iter(pose.loader)
    reports = []
    for step in range(steps):
        detect_batches = (next(detect_iterator), next(detect_iterator))
        try:
            pose_batch = next(pose_iterator)
        except StopIteration:
            pose_iterator = iter(pose.loader)
            pose_batch = next(pose_iterator)
        report = engine.run(
            detect_batches=detect_batches,
            pose_batches=(pose_batch,),
            record_gradient_statistics=step == 0,
        )
        if not all(report.gradient_presence.values()):
            raise AssertionError(f"silent missing macro gradients: {report.gradient_presence}")
        reports.append(asdict(report))
    engine.advance_epoch()
    guard.assert_unchanged(model)
    shared_after = _bn_state(model, heads=False)
    heads_after = _bn_state(model, heads=True)
    changed_shared = _changed(shared_before, shared_after)
    changed_heads = _changed(heads_before, heads_after)
    if changed_shared:
        raise AssertionError(f"shared BN drifted: {changed_shared[:10]}")
    if not changed_heads:
        raise AssertionError("no head BatchNorm running statistics changed")
    return {
        "schema_version": 1,
        "device": "cpu",
        "architecture": config.architecture,
        "seed": seed,
        "imgsz": imgsz,
        "steps": steps,
        "pose_checkpoint": str(selected_pose),
        "factory": built.report.as_dict(),
        "stage": asdict(stage_report),
        "optimizer": asdict(optimizer_report),
        "datasets": {
            "pose": asdict(pose_view),
            "detect": asdict(detect_view),
        },
        "macros": reports,
        "bn": {
            "shared_modules": len(shared_before),
            "shared_changed": list(changed_shared),
            "head_modules": len(heads_before),
            "head_changed": list(changed_heads),
        },
        "criterion": router.state_dict(),
        "ema_updates": int(ema.updates),
    }

