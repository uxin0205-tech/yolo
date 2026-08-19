"""Ultralytics trainer adapter for staged MASF fine-tuning."""

from __future__ import annotations

import copy
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model

from .checkpoint import build_training_model, file_sha256
from .config import CommonTrainingConfig
from .phases import (
    PhaseSpec,
    apply_phase_scope,
    assert_frozen_state_unchanged,
    build_phase_optimizer,
    enforce_frozen_modules_eval,
    snapshot_frozen_state,
)
from .runtime import nvidia_driver_version

MAP_SELECTION_METRIC = "metrics/mAP50-95(B)"


def make_bittrue_validation_copy(model: Any, bittrue_config: Path):
    """Deep-copy Float training state and reconfigure only its attention backend."""

    validation_model = copy.deepcopy(unwrap_model(model))
    convert_yolo26_model(validation_model, VariantConfig.from_yaml(bittrue_config))
    return validation_model


def _assert_finite(model: Any, label: str) -> None:
    bad = [
        name
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
    ]
    if bad:
        raise FloatingPointError(f"non-finite {label} tensors: {bad[:10]}")


class MASFTrainer(DetectionTrainer):
    """Rebuild optimizer/scheduler/EMA per phase and enforce immutable frozen state."""

    def __init__(
        self,
        *args: Any,
        masf_variant: str,
        attention_config: Path,
        bittrue_config: Path,
        phase_spec: PhaseSpec,
        **kwargs: Any,
    ) -> None:
        self.masf_variant = masf_variant
        self.attention_config = attention_config
        self.bittrue_config = bittrue_config
        self.phase_spec = phase_spec
        self._frozen_snapshot: dict[str, torch.Tensor] = {}
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._disable_oom_fallback)
        self.add_callback("on_train_epoch_end", self._on_epoch_end)
        self.add_callback("on_train_batch_end", self._on_batch_end)
        self.add_callback("on_model_save", self._on_model_save)

    def get_model(self, cfg: str | dict[str, Any] | None = None, weights: Any = None, verbose: bool = True):
        model, _ = build_training_model(
            cfg=cfg,
            nc=self.data["nc"],
            channels=self.data["channels"],
            weights=weights,
            masf_variant=self.masf_variant,
            attention_config=self.attention_config,
            verbose=verbose,
        )
        return model

    def get_dataloader(self, dataset_path: str, batch_size: int, rank: int = 0, mode: str = "train"):
        """Keep batch fixed and prevent simultaneous validation workers."""

        if mode == "val":
            batch_size = min(batch_size, self.batch_size)
            configured_workers = self.args.workers
            self.args.workers = 0
            try:
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = configured_workers
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def build_optimizer(self, model: Any, *args: Any, **kwargs: Any):
        apply_phase_scope(model, self.phase_spec.name)
        optimizer = build_phase_optimizer(model, self.phase_spec)
        enforce_frozen_modules_eval(model)
        self._frozen_snapshot = snapshot_frozen_state(model)
        return optimizer

    def _model_train(self) -> None:
        """Enter train mode, then restore immutable frozen BN and attention state."""

        super()._model_train()
        enforce_frozen_modules_eval(unwrap_model(self.model))

    @staticmethod
    def _disable_oom_fallback(trainer: Any) -> None:
        """Prevent upstream first-epoch OOM recovery from changing the fixed batch."""

        trainer._oom_retries = 3

    def _on_epoch_end(self, trainer: Any) -> None:
        assert_frozen_state_unchanged(unwrap_model(trainer.model), self._frozen_snapshot)

    @staticmethod
    def _on_batch_end(trainer: Any) -> None:
        loss = getattr(trainer, "loss", None)
        if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite training loss")

    @staticmethod
    def _on_model_save(trainer: Any) -> None:
        _assert_finite(unwrap_model(trainer.model), "live checkpoint")
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        if ema is not None:
            _assert_finite(ema, "EMA checkpoint")

    def validate(self):
        """Select best.pt and drive early stopping with actual Bit-True mAP50-95(B)."""

        float_ema = self.ema.ema
        self.ema.ema = make_bittrue_validation_copy(float_ema or self.model, self.bittrue_config)
        try:
            metrics = self.validator(self)
        finally:
            self.ema.ema = float_ema
        if metrics is None:
            return None, None
        metrics.pop("fitness", None)
        if MAP_SELECTION_METRIC not in metrics:
            raise KeyError(f"validator omitted {MAP_SELECTION_METRIC}")
        fitness = float(metrics[MAP_SELECTION_METRIC])
        if not torch.isfinite(torch.tensor(fitness)):
            raise FloatingPointError("non-finite validation mAP50-95")
        if not self.best_fitness or fitness > self.best_fitness:
            self.best_fitness = fitness
        return metrics, fitness


@dataclass(frozen=True)
class TrainerFactory:
    masf_variant: str
    attention_config: Path
    bittrue_config: Path
    phase_spec: PhaseSpec

    def __call__(self, *args: Any, **kwargs: Any) -> MASFTrainer:
        return MASFTrainer(
            *args,
            masf_variant=self.masf_variant,
            attention_config=self.attention_config,
            bittrue_config=self.bittrue_config,
            phase_spec=self.phase_spec,
            **kwargs,
        )


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_training_manifest(
    destination: Path,
    *,
    project_root: Path,
    weights: Path,
    variant: str,
    phase: PhaseSpec,
    common: CommonTrainingConfig,
) -> Path:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "phase": asdict(phase),
        "common": asdict(common),
        "parent": {"path": str(weights.resolve()), "sha256": file_sha256(weights)},
        "fresh_optimizer_scheduler_ema": True,
        "automatic_batch_fallback": False,
        "gradient_accumulation": common.gradient_accumulation,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver": nvidia_driver_version(),
            "gpu_vram_bytes": (
                torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
            ),
            "gpu_free_vram_bytes_at_start": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None,
            "ultralytics": ultralytics.__version__,
            "ultralytics_source": ultralytics.__path__[0],
            "git_revision": _git_revision(project_root),
        },
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def launch_phase(
    *,
    project_root: Path,
    weights: Path,
    masf_variant: str,
    attention_config: Path,
    bittrue_config: Path,
    phase: PhaseSpec,
    common: CommonTrainingConfig,
    run_id: str,
) -> Any:
    """Launch exactly one phase from its accepted parent checkpoint."""

    run = project_root / "artifacts" / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    write_training_manifest(
        run / "manifest.json",
        project_root=project_root,
        weights=weights,
        variant=masf_variant,
        phase=phase,
        common=common,
    )
    args = common.to_ultralytics_args(phase, project=run, name="ultralytics")
    data = Path(args["data"])
    if not data.is_absolute():
        args["data"] = str((project_root / data).resolve())
    model = YOLO(str(weights.resolve()))
    started = time.perf_counter()
    try:
        result = model.train(
            trainer=TrainerFactory(
                masf_variant,
                attention_config.resolve(),
                bittrue_config.resolve(),
                phase,
            ),
            **args,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"formal batch={common.batch} does not fit; stopped without batch reduction or accumulation"
        ) from exc
    elapsed = time.perf_counter() - started
    completed_epochs = int(getattr(model.trainer, "epoch", -1)) + 1
    (run / "training-complete.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "requested_epochs": phase.epochs,
                "completed_epochs": completed_epochs,
                "patience": phase.patience,
                "stop_reason": "early-stopping" if completed_epochs < phase.epochs else "max-epochs",
                "elapsed_seconds": elapsed,
                "best_fitness": float(model.trainer.best_fitness),
                "best_checkpoint": str((run / "ultralytics/weights/best.pt").resolve()),
                "epoch_metrics_csv": str((run / "ultralytics/results.csv").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
