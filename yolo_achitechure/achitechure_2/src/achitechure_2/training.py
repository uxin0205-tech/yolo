"""Ultralytics adapter for fair candidate training with Bit-True ranking."""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from .config import load_yaml
from .freezing import FrozenStateGuard, apply_frozen_scope, enforce_frozen_eval
from .graph import inspect_graph
from .intake import file_sha256, require_accepted_intake

MAP_SELECTION_METRIC = "metrics/mAP50-95(B)"
STAGE_RULES = {
    "smoke": {"epochs": 3, "patience": 3},
    "formal": {"epochs": 100, "patience": 20},
    "extension": {"epochs": 140, "patience": 15},
    "qat": {"epochs": 15, "patience": 5},
}


def _source_model(weights: Any) -> nn.Module:
    source = weights.get("model") if isinstance(weights, dict) else weights
    if not isinstance(source, nn.Module):
        raise TypeError("training weights must contain the materialized candidate model")
    return source


def make_bittrue_validation_copy(model: nn.Module, bittrue_config: Path) -> nn.Module:
    """Deep-copy Float EMA and change only the inherited attention normalization."""

    from yolo_attention.config import VariantConfig
    from yolo_attention.integration import convert_yolo26_model

    validation_model = copy.deepcopy(unwrap_model(model))
    convert_yolo26_model(validation_model, VariantConfig.from_yaml(bittrue_config))
    report = inspect_graph(validation_model)
    if report.attention_normalizations != ("bit_true_pwl", "bit_true_pwl"):
        raise AssertionError("Bit-True validation copy was not fully converted")
    return validation_model


class LiteC3k2Trainer(DetectionTrainer):
    """Preserve runtime replacements and keep inherited modules immutable."""

    def __init__(self, *args: Any, bittrue_config: Path, qat: bool = False, **kwargs: Any) -> None:
        self.bittrue_config = bittrue_config
        self.qat = qat
        self._frozen_guard: FrozenStateGuard | None = None
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._epoch_start)
        self.add_callback("on_train_epoch_end", self._epoch_end)
        self.add_callback("on_train_batch_end", self._batch_end)

    def check_resume(self, overrides: dict[str, Any]) -> None:
        """Allow the gated 100-to-140 total-epoch extension on last.pt."""

        super().check_resume(overrides)
        if self.args.resume and "epochs" in overrides:
            requested = int(overrides["epochs"])
            checkpoint_epochs = int(self.args.epochs)
            if requested <= checkpoint_epochs:
                raise ValueError("resume extension epochs must exceed checkpoint epochs")
            self.args.epochs = requested

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> nn.Module:
        model = copy.deepcopy(_source_model(weights)).float()
        inspect_graph(model)
        return model

    def build_optimizer(self, model: nn.Module, *args: Any, **kwargs: Any):
        apply_frozen_scope(model)
        optimizer = super().build_optimizer(model, *args, **kwargs)
        self._frozen_guard = FrozenStateGuard.capture(model)
        return optimizer

    def _epoch_start(self, trainer: Any) -> None:
        model = unwrap_model(trainer.model)
        enforce_frozen_eval(model)
        if self.qat:
            from .quantization import configure_qat_epoch

            configure_qat_epoch(model, int(trainer.epoch) + 1)

    def _epoch_end(self, trainer: Any) -> None:
        if self._frozen_guard is None:
            raise AssertionError("frozen state guard was not initialized")
        self._frozen_guard.assert_unchanged(unwrap_model(trainer.model))

    @staticmethod
    def _batch_end(trainer: Any) -> None:
        loss = getattr(trainer, "loss", None)
        if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite training loss")

    def validate(self):
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
            raise FloatingPointError("non-finite Bit-True validation metric")
        if not self.best_fitness or fitness > self.best_fitness:
            self.best_fitness = fitness
        return metrics, fitness


@dataclass(frozen=True)
class TrainerFactory:
    bittrue_config: Path
    qat: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> LiteC3k2Trainer:
        return LiteC3k2Trainer(*args, bittrue_config=self.bittrue_config, qat=self.qat, **kwargs)


def _git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _best_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or MAP_SELECTION_METRIC not in rows[0]:
        return None
    return max(range(len(rows)), key=lambda index: float(rows[index][MAP_SELECTION_METRIC])) + 1


def launch_training(
    *,
    project_root: Path,
    checkpoint: Path,
    candidate_id: str,
    stage: str,
    run_id: str,
    smoke_epochs: int = 3,
) -> Path:
    """Launch one gated stage; formal work never uses the development fixture."""

    intake = require_accepted_intake(project_root)
    stage = stage.lower()
    if stage not in STAGE_RULES:
        raise ValueError(f"unknown stage {stage}")
    if stage == "smoke" and smoke_epochs not in (3, 4, 5):
        raise ValueError("smoke_epochs must be 3, 4, or 5")
    run = project_root / "artifacts/runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    common_path = project_root / "configs/training/common.yaml"
    args = load_yaml(common_path)
    args.update(STAGE_RULES[stage])
    if stage == "smoke":
        args["epochs"] = smoke_epochs
    args.update(project=str(run), name="ultralytics", exist_ok=False)
    data = Path(str(args["data"]))
    if not data.is_absolute():
        args["data"] = str((project_root / data).resolve())
    if stage == "extension":
        args["resume"] = str(checkpoint.resolve())

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "stage": stage,
        "parent": {"path": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint)},
        "handoff_manifest_sha256": intake["manifest_sha256"],
        "requested_args": args,
        "config_sha256": file_sha256(common_path),
        "git_revision": _git_revision(project_root),
        "selection_backend": "bit_true_pwl",
        "frozen_modules": ["model.16.p3_masf", "model.10.m.0.attn", "model.22.m.0.1.attn"],
    }
    model = YOLO(str(checkpoint.resolve()))
    inspect_graph(model.model)
    if stage == "qat":
        from .quantization import Conv2dSimulationAdapter

        if not any(isinstance(module, Conv2dSimulationAdapter) for module in model.model.modules()):
            raise ValueError("QAT requires a quant-prepare simulation checkpoint")
        inherited_lr = float(model.overrides.get("lr0", 0.01))
        args["lr0"] = inherited_lr * 0.1
        manifest["qat_lr_ratio"] = 0.1
        manifest["inherited_architecture_lr0"] = inherited_lr
    manifest["requested_args"] = args
    (run / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    started = time.perf_counter()
    model.train(
        trainer=TrainerFactory(
            project_root.parent / "achitechure_1/configs/attention/bittrue-pwl-final.yaml",
            qat=stage == "qat",
        ),
        **args,
    )
    completed_epochs = int(getattr(model.trainer, "epoch", -1)) + 1
    resolved_args_path = run / "resolved-args.json"
    resolved_args_path.write_text(
        json.dumps(vars(model.trainer.args), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    metrics_csv = run / "ultralytics/results.csv"
    completion = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_epochs": completed_epochs,
        "requested_epochs": int(args["epochs"]),
        "best_epoch": _best_epoch(metrics_csv),
        "best_fitness": float(model.trainer.best_fitness),
        "stop_reason": "early-stopping" if completed_epochs < int(args["epochs"]) else "max-epochs",
        "best_checkpoint": str((run / "ultralytics/weights/best.pt").resolve()),
        "last_checkpoint": str((run / "ultralytics/weights/last.pt").resolve()),
        "metrics_csv": str(metrics_csv.resolve()),
        "resolved_args": str(resolved_args_path.resolve()),
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "simulation_only": stage == "qat",
    }
    destination = run / "training-complete.json"
    destination.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
