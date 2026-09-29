"""Formal trainer with executable task augmentation and shared-BN policy."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from torch import nn

from . import _formal_training_impl as _impl
from . import _formal_training_public_v1 as _v1
from .data import prepare_bbt5_view
from .joint_data import TaskLoaderSettings, build_task_loader

FormalRunReport = _v1.FormalRunReport
reseed_loader_for_epoch = _v1.reseed_loader_for_epoch
seed_everything = _v1.seed_everything


def _apply_shared_bn_affine(model: nn.Module, *, trainable: bool) -> None:
    if trainable:
        return
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if ".detect_head." in name or ".pose_head." in name:
            continue
        if module.affine:
            if module.weight is not None:
                module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)


def _reconcile_stage_report(report, model):
    parameters = dict(model.named_parameters())
    trainable_names = tuple(
        name for name in report.trainable_names if parameters[name].requires_grad
    )
    frozen_names = tuple(
        name for name in parameters if not parameters[name].requires_grad
    )
    return replace(
        report,
        trainable_names=trainable_names,
        frozen_names=frozen_names,
        trainable_parameters=sum(parameters[name].numel() for name in trainable_names),
        frozen_parameters=sum(parameters[name].numel() for name in frozen_names),
    )


class FormalJointTrainingSession(_v1.FormalJointTrainingSession):
    def _loaders(self, model):
        pose_view = prepare_bbt5_view(
            self.config.registry,
            self.run_dir / "datasets" / "bbat5-v1-runtime",
        )
        detect_settings = TaskLoaderSettings.for_detect(
            batch_size=self.detect_microbatch_size,
            workers=self.config.detect_workers,
            imgsz=self.config.imgsz,
            seed=self.config.seed,
        )
        detect_settings = replace(
            detect_settings,
            augmentation=MappingProxyType(
                {
                    **dict(detect_settings.augmentation),
                    "mosaic": self.config.detect_mosaic,
                    "fliplr": self.config.detect_fliplr,
                }
            ),
        )
        pose_settings = TaskLoaderSettings.for_pose(
            batch_size=self.config.pose_batch_size,
            workers=self.config.pose_workers,
            imgsz=self.config.imgsz,
            seed=self.config.seed,
        )
        pose_settings = replace(
            pose_settings,
            augmentation=MappingProxyType(
                {
                    **dict(pose_settings.augmentation),
                    "mosaic": self.config.pose_mosaic,
                    "fliplr": self.config.pose_fliplr,
                }
            ),
        )
        detect = build_task_loader(
            model,
            data_yaml=self.config.detect_data,
            settings=detect_settings,
            device=self.device,
            registry=self.config.registry,
        )
        pose = build_task_loader(
            model,
            data_yaml=pose_view.yaml,
            settings=pose_settings,
            device=self.device,
            registry=self.config.registry,
        )
        return detect, pose, pose_view

    def run(self, **kwargs):
        original_apply = _impl.apply_stage
        original_build = _impl.build_joint_optimizer
        original_update = _impl.update_optimizer_stage
        affine = self.config.shared_bn_affine_trainable

        def configured_apply(model, stage):
            report = original_apply(model, stage)
            _apply_shared_bn_affine(model, trainable=affine)
            return _reconcile_stage_report(report, model)

        def configured_build(model, stage, **build_kwargs):
            optimizer, report = original_build(model, stage, **build_kwargs)
            _apply_shared_bn_affine(model, trainable=affine)
            return optimizer, report

        def configured_update(optimizer, model, stage):
            report = original_update(optimizer, model, stage)
            _apply_shared_bn_affine(model, trainable=affine)
            return _reconcile_stage_report(report, model)

        _impl.apply_stage = configured_apply
        _impl.build_joint_optimizer = configured_build
        _impl.update_optimizer_stage = configured_update
        try:
            return super().run(**kwargs)
        finally:
            _impl.apply_stage = original_apply
            _impl.build_joint_optimizer = original_build
            _impl.update_optimizer_stage = original_update


__all__ = (
    "FormalJointTrainingSession",
    "FormalRunReport",
    "reseed_loader_for_epoch",
    "seed_everything",
)
