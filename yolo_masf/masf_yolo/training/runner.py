"""Ultralytics trainer adapter that preserves repository-owned model slots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from ultralytics.engine.trainer import LOCAL_RANK, RANK
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first

from .resume import TransientTrainingError


class RepositoryDetectionTrainer(DetectionTrainer):
    """Return serialized custom models directly instead of rebuilding their YAML."""

    def get_model(
        self,
        cfg: str | None = None,
        weights: str | nn.Module | None = None,
        verbose: bool = True,
    ) -> nn.Module:
        if isinstance(weights, nn.Module) and hasattr(weights, "masf_variant"):
            if hasattr(self, "args"):
                weights.args = self.args
            weights.criterion = None
            return weights
        return super().get_model(cfg=cfg, weights=weights, verbose=verbose)

    def final_eval(self) -> None:
        """Strip native checkpoints; strict fresh-process val runs afterward."""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                checkpoint = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    strip_optimizer(
                        self.best,
                        updates={"train_results": checkpoint.get("train_results")},
                    )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    best: Path
    last: Path
    save_dir: Path


def run_training(
    model: nn.Module | None,
    profile: Mapping[str, Any],
    *,
    resume_path: Path | None = None,
) -> TrainingResult:
    overrides = dict(profile)
    if resume_path is not None:
        overrides["model"] = str(resume_path)
        overrides["resume"] = str(resume_path)
    try:
        trainer = RepositoryDetectionTrainer(overrides=overrides)
        if resume_path is None:
            if model is None:
                raise ValueError("a new training run requires an initialized model")
            trainer.model = model
        trainer.train()
    except torch.cuda.OutOfMemoryError as error:
        raise TransientTrainingError(f"CUDA out of memory: {error}") from error
    best = trainer.best if trainer.best.is_file() else trainer.last
    if not best.is_file() or not trainer.last.is_file():
        raise RuntimeError("training ended without best.pt and last.pt")
    return TrainingResult(best=best.resolve(), last=trainer.last.resolve(), save_dir=trainer.save_dir.resolve())
