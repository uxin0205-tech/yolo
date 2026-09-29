"""嚴格限制 scope 的 Float-PWL 訓練 Ultralytics adapter。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from .attention import HardwareFriendlyAttention
from .config import VariantConfig
from .integration import convert_yolo26_model
from .optimizer_scope import apply_layerwise_learning_rates, restrict_optimizer_to_trainable
from .scopes import apply_trainable_scope, enforce_frozen_batchnorm, sync_progressive_epoch

CUSTOM_PARENT_MINIMUM_STATE_COVERAGE = 0.95
MAP_SELECTION_METRIC = "metrics/mAP50-95(B)"


def _assert_finite_state(model: Any, label: str) -> None:
    state = model.state_dict()
    bad = [
        name
        for name, value in state.items()
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
    ]
    if bad:
        raise FloatingPointError(f"non-finite {label} tensors: {bad[:5]}")


def fail_on_nonfinite_loss(trainer: Any) -> None:
    loss = getattr(trainer, "loss", None)
    if isinstance(loss, torch.Tensor) and not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("non-finite training loss detected; refusing to continue")


def _custom_parent_state_coverage(model: DetectionModel, weights: Any) -> tuple[int, int]:
    source = weights["model"] if isinstance(weights, dict) else weights
    source_state, target_state = source.state_dict(), model.state_dict()
    compatible = sum(
        key in target_state and value.shape == target_state[key].shape for key, value in source_state.items()
    )
    return compatible, len(target_state)


def build_hardware_attention_model(
    *, cfg: str | None, nc: int, channels: int, weights: Any, variant: VariantConfig, verbose: bool
) -> DetectionModel:
    model = DetectionModel(cfg, nc=nc, ch=channels, verbose=verbose)
    source_is_custom = bool(
        weights
        and hasattr(weights, "modules")
        and any(isinstance(module, HardwareFriendlyAttention) for module in weights.modules())
    )
    if source_is_custom:
        convert_yolo26_model(model, variant)
        compatible, total = _custom_parent_state_coverage(model, weights)
        if compatible / total < CUSTOM_PARENT_MINIMUM_STATE_COVERAGE:
            raise RuntimeError(f"incompatible custom parent: {compatible}/{total} state entries match")
        model.load(weights)
    else:
        if weights:
            model.load(weights)
        convert_yolo26_model(model, variant)
    return model


class HardwareAttentionTrainer(DetectionTrainer):
    def __init__(
        self,
        *args: Any,
        variant_config: VariantConfig,
        training_stage: str,
        layer_lrs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.variant_config = variant_config
        self.training_stage = training_stage
        self.layer_lrs = layer_lrs
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._on_epoch_start)
        self.add_callback("on_train_epoch_end", self._on_epoch_end)
        self.add_callback("on_train_batch_end", fail_on_nonfinite_loss)
        self.add_callback("on_model_save", self._on_model_save)

    def _on_epoch_start(self, trainer: Any) -> None:
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        if ema is not None:
            sync_progressive_epoch(trainer.model, ema, trainer.epoch)
        enforce_frozen_batchnorm(trainer.model, self.training_stage)

    def _on_epoch_end(self, trainer: Any) -> None:
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        if ema is not None:
            sync_progressive_epoch(trainer.model, ema, trainer.epoch)

    @staticmethod
    def _on_model_save(trainer: Any) -> None:
        _assert_finite_state(trainer.model, "live checkpoint")
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        if ema is not None:
            _assert_finite_state(ema, "EMA checkpoint")

    def get_model(self, cfg: str | None = None, weights: Any = None, verbose: bool = True) -> DetectionModel:
        return build_hardware_attention_model(
            cfg=cfg,
            nc=self.data["nc"],
            channels=self.data["channels"],
            weights=weights,
            variant=self.variant_config,
            verbose=verbose,
        )

    def build_optimizer(self, model: Any, *args: Any, **kwargs: Any):
        apply_trainable_scope(model, self.training_stage)
        optimizer = super().build_optimizer(model, *args, **kwargs)
        restrict_optimizer_to_trainable(optimizer, model)
        apply_layerwise_learning_rates(optimizer, model, self.layer_lrs)
        return optimizer

    def validate(self):
        """只依 mAP50-95 選擇 best.pt，不受 composite fitness 變更影響。"""

        if self.ema and self.world_size > 1:
            import torch.distributed as dist

            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        metrics.pop("fitness", None)
        if MAP_SELECTION_METRIC not in metrics:
            raise KeyError(f"validator omitted required selection metric {MAP_SELECTION_METRIC}")
        fitness = float(metrics[MAP_SELECTION_METRIC])
        if not torch.isfinite(torch.tensor(fitness)):
            raise FloatingPointError("non-finite formal validation metric")
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness


@dataclass(frozen=True)
class TrainerFactory:
    variant_config: VariantConfig
    training_stage: str
    layer_lrs: dict[str, float] | None = None
    trainer_class: type[HardwareAttentionTrainer] = HardwareAttentionTrainer

    def __call__(self, *args: Any, **kwargs: Any) -> HardwareAttentionTrainer:
        return self.trainer_class(
            *args,
            variant_config=self.variant_config,
            training_stage=self.training_stage,
            layer_lrs=self.layer_lrs,
            **kwargs,
        )


def make_trainer(
    variant: VariantConfig, *, stage: str, layer_lrs: dict[str, float] | None = None
) -> TrainerFactory:
    return TrainerFactory(variant_config=variant, training_stage=stage, layer_lrs=layer_lrs)
