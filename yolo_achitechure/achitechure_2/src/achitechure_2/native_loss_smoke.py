"""以正式資料 loader 與 native criteria 執行 CPU loss smoke。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .candidate import build_candidate
from .config import SPEC_PATH, SPEC_VERSION, file_sha256
from .freezing import FrozenStateGuard, enforce_frozen_eval
from .full35_adapter import Full35Release
from .screen_training import (
    ScreenRunConfig,
    _build_screen_loader,
    _stage,
)


def _gradient_summary(model: torch.nn.Module) -> tuple[int, bool]:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    return len(gradients), finite


def run_cpu_native_loss_smoke(config_path: str | Path) -> dict[str, Any]:
    """用 C0-J3 與各一個 real-data batch 驗證 Detect/Pose native loss。"""

    source = ScreenRunConfig.load(config_path)
    if not source.pose_enabled:
        raise PermissionError("native Pose loss smoke 需要本輪 pose_enabled=true")
    config = replace(
        source,
        imgsz=64,
        amp=False,
        cache=False,
        detect_microbatch=1,
        pose_batch=1,
        detect_workers=0,
        pose_workers=0,
    )
    release = Full35Release(config.full35_root)
    api = release.training_api()
    parent = release.load_parent()
    resolved = release.resolved_candidate("C0")
    model, build_report = build_candidate(parent.model, resolved, seed=config.seed)
    model = model.cpu()
    stage = _stage(api, config)
    api.apply_stage(model, stage)
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
        batch_size=1,
        workers=0,
        config=config,
        device=torch.device("cpu"),
    )
    pose_loader = _build_screen_loader(
        api,
        model,
        task=api.Task.POSE,
        data_yaml=config.pose_data,
        batch_size=1,
        workers=0,
        config=config,
        device=torch.device("cpu"),
    )
    router = api.NativeTaskLossRouter(
        model,
        epochs=config.epochs,
        imgsz=config.imgsz,
        detect_overrides=config.loss_overrides["detect"],
        pose_overrides=config.loss_overrides["pose"],
    )

    losses: dict[str, Any] = {}
    for task, prepared in (
        (api.Task.DETECT, detect_loader),
        (api.Task.POSE, pose_loader),
    ):
        try:
            raw_batch = next(iter(prepared.loader))
        except StopIteration as error:
            raise RuntimeError(f"{task.value} loader 沒有 real-data batch") from error
        batch = prepared.preprocess(raw_batch)
        model.zero_grad(set_to_none=True)
        result = router.loss_for(task, batch)
        result.mean_total.backward()
        gradient_tensors, gradients_finite = _gradient_summary(model)
        if not gradients_finite:
            raise FloatingPointError(f"{task.value} native loss gradient 非有限值")
        losses[task.value] = {
            "batch_size": result.actual_batch_size,
            "raw_total": float(result.raw_total.detach().cpu()),
            "mean_total": float(result.mean_total.detach().cpu()),
            "components": [
                float(value)
                for value in result.components.detach().cpu().flatten().tolist()
            ],
            "finite": bool(torch.isfinite(result.raw_total).item()),
            "gradient_tensors": gradient_tensors,
            "gradients_finite": gradients_finite,
            "runtime_label_cache": str(
                config.screen_root
                / "runtime-cache"
                / f"{task.value}-{prepared.data_yaml.stem}-train.cache"
            ),
        }
    guard.assert_unchanged(model)

    pose_wrapper = router.criteria[api.Task.POSE]
    pose_criterion = getattr(pose_wrapper, "one2one", pose_wrapper)
    rle_loss = getattr(pose_criterion, "rle_loss", None)
    flow_model = getattr(pose_criterion, "flow_model", None)
    rle_weight = float(getattr(pose_criterion.hyp, "rle", 0.0))
    rle_active = (
        pose_criterion.__class__.__name__ == "PoseLoss26"
        and rle_loss is not None
        and flow_model is not None
        and rle_weight == 1.0
    )
    if not rle_active:
        raise RuntimeError("PoseLoss26/RLE contract 未啟用")

    source_cache_paths = (
        Path("/home/uxin/yolo/coco2017/labels/train2017.cache"),
        Path(
            "/home/uxin/yolo/original/pose/derived/bbat5-v1/"
            "pose/labels/train.cache"
        ),
    )
    runtime_cache_root = config.screen_root / "runtime-cache"
    source_caches_absent = not any(path.exists() for path in source_cache_paths)
    if not source_caches_absent:
        raise RuntimeError("native loss smoke 不得在 canonical/source 旁建立 label cache")

    return {
        "schema_version": 2,
        "scope": "CPU real-data native loss smoke; not accuracy or performance",
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "run_config": {
            "path": str(source.path),
            "sha256": file_sha256(source.path),
        },
        "handoff": {
            "revision_id": "full35-final-j3-seed0-d67fb45c",
            "manifest_sha256": file_sha256(config.handoff_manifest),
            "parent_checkpoint_sha256": parent.checkpoint_report[
                "checkpoint_sha256"
            ],
        },
        "candidate": {
            "resolved_id": build_report.resolved_id,
            "parent_unchanged": build_report.parent_unchanged,
            "contract_unchanged": build_report.model_contract_unchanged,
        },
        "device": "cpu",
        "imgsz": config.imgsz,
        "datasets": {
            "detect": str(config.detect_data),
            "pose": str(config.pose_data),
        },
        "losses": losses,
        "criterion_state": router.state_dict(),
        "pose_rle": {
            "active": rle_active,
            "criterion_class": pose_criterion.__class__.__name__,
            "flow_model_class": flow_model.__class__.__name__,
            "rle_loss_class": rle_loss.__class__.__name__,
            "rle_weight": rle_weight,
        },
        "cache_policy": {
            "runtime_root": str(runtime_cache_root),
            "runtime_caches_exist": runtime_cache_root.is_dir(),
            "source_adjacent_paths": [str(path) for path in source_cache_paths],
            "source_adjacent_caches_absent": source_caches_absent,
        },
    }
