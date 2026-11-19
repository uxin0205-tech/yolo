"""Ultralytics trainer adapter; importing it never starts training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from .attention import HardwareFriendlyAttention
from .config import VariantConfig
from .integration import convert_yolo26_model, freeze_for_stage, set_progressive_epoch

CUSTOM_PARENT_MINIMUM_STATE_COVERAGE = 0.95


def fail_on_nonfinite_loss(trainer: Any) -> None:
    """Stop the current run after the first batch that reports NaN or Inf."""

    loss = getattr(trainer, "loss", None)
    if isinstance(loss, torch.Tensor) and not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("non-finite training loss detected; refusing to continue the run")


def _custom_parent_state_coverage(model: DetectionModel, weights: Any) -> tuple[int, int]:
    source = weights["model"] if isinstance(weights, dict) else weights
    source_state = source.state_dict()
    target_state = model.state_dict()
    compatible = sum(
        key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
        for key, value in source_state.items()
    )
    return compatible, len(target_state)


def build_hardware_attention_model(
    *,
    cfg: str | None,
    nc: int,
    channels: int,
    weights: Any,
    variant: VariantConfig,
    verbose: bool,
) -> DetectionModel:
    """Build the target structure before loading custom-checkpoint-only state."""

    model = DetectionModel(cfg, nc=nc, ch=channels, verbose=verbose)
    source_is_custom = bool(
        weights
        and hasattr(weights, "modules")
        and any(isinstance(module, HardwareFriendlyAttention) for module in weights.modules())
    )
    if source_is_custom:
        convert_yolo26_model(model, variant)
        compatible, total = _custom_parent_state_coverage(model, weights)
        coverage = compatible / total
        if coverage < CUSTOM_PARENT_MINIMUM_STATE_COVERAGE:
            raise RuntimeError(
                "incompatible custom parent checkpoint: "
                f"only {compatible}/{total} state entries match ({coverage:.1%}); "
                "the checkpoint may have been fused for inference"
            )
        model.load(weights)
    else:
        if weights:
            model.load(weights)
        convert_yolo26_model(model, variant)
    return model


class HardwareAttentionTrainer(DetectionTrainer):
    """Rebuild YOLO26, load source weights, then install custom Attention."""

    def __init__(
        self,
        *args: Any,
        variant_config: VariantConfig,
        training_stage: str,
        **kwargs: Any,
    ) -> None:
        self.variant_config = variant_config
        self.training_stage = training_stage
        super().__init__(*args, **kwargs)
        self.add_callback(
            "on_train_epoch_start",
            lambda trainer: set_progressive_epoch(trainer.model, trainer.epoch),
        )
        self.add_callback("on_train_batch_end", fail_on_nonfinite_loss)

    def get_model(self, cfg: str | None = None, weights: Any = None, verbose: bool = True) -> DetectionModel:
        return build_hardware_attention_model(
            cfg=cfg,
            nc=self.data["nc"],
            channels=self.data["channels"],
            weights=weights,
            variant=self.variant_config,
            verbose=verbose,
        )

    def build_optimizer(self, model, *args: Any, **kwargs: Any):
        # BaseTrainer re-enables non-standard frozen parameters before this seam.
        # Reapply the research scope immediately before optimizer construction.
        freeze_for_stage(model, self.training_stage)
        return super().build_optimizer(model, *args, **kwargs)


@dataclass(frozen=True)
class TrainerFactory:
    """Pickle-friendly callable accepted by YOLO.train(trainer=...)."""

    variant_config: VariantConfig
    training_stage: str
    trainer_class: type[HardwareAttentionTrainer] = HardwareAttentionTrainer

    def __call__(self, *args: Any, **kwargs: Any) -> HardwareAttentionTrainer:
        return self.trainer_class(
            *args,
            variant_config=self.variant_config,
            training_stage=self.training_stage,
            **kwargs,
        )


def make_trainer(variant: VariantConfig, *, stage: str) -> TrainerFactory:
    return TrainerFactory(variant_config=variant, training_stage=stage)
