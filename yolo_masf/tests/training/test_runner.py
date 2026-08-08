from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from ultralytics.utils.torch_utils import torch_load

from masf_yolo.training.runner import RepositoryDetectionTrainer


class _ValidatorMustNotRun:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("redundant final validation must not run")


def _native_checkpoint(path: Path, train_results: dict[str, list[float]]) -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 1))
    model.args = {}
    torch.save(
        {
            "model": model,
            "ema": None,
            "optimizer": {"state": {}},
            "epoch": 2,
            "train_args": {"epochs": 3},
            "train_results": train_results,
        },
        path,
    )


def test_repository_final_eval_strips_without_duplicate_validation(tmp_path: Path) -> None:
    last = tmp_path / "last.pt"
    best = tmp_path / "best.pt"
    train_results = {"epoch": [1.0, 2.0, 3.0]}
    _native_checkpoint(last, train_results)
    _native_checkpoint(best, {"epoch": [1.0]})
    validator = _ValidatorMustNotRun()
    trainer = RepositoryDetectionTrainer.__new__(RepositoryDetectionTrainer)
    trainer.last = last
    trainer.best = best
    trainer.validator = validator

    trainer.final_eval()

    assert validator.calls == 0
    stripped_last = torch_load(last, map_location="cpu")
    stripped_best = torch_load(best, map_location="cpu")
    assert stripped_last["optimizer"] is None
    assert stripped_last["epoch"] == -1
    assert stripped_best["train_results"] == train_results
