from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from ultralytics.utils.torch_utils import ModelEMA

from yolo_combine.resume import (
    TrainingProgress,
    load_training_snapshot,
    save_inference_weights,
    save_training_snapshot,
)


class TinyResumeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)

    def contract(self) -> dict[str, object]:
        return {"model_kind": "tiny_resume", "shape": [2, 1]}


class CriterionState:
    def __init__(self) -> None:
        self.updates = 0

    def state_dict(self):
        return {"detect": {"end2end": True, "updates": self.updates}}

    def load_state_dict(self, state):
        self.updates = int(state["detect"]["updates"])


class FakeScaler:
    def __init__(self) -> None:
        self.scale_value = 8.0

    def state_dict(self):
        return {"scale_value": self.scale_value}

    def load_state_dict(self, state):
        self.scale_value = float(state["scale_value"])


def _system(seed: int = 7):
    torch.manual_seed(seed)
    model = TinyResumeModel()
    group = {
        "params": list(model.parameters()),
        "param_names": tuple(name for name, _ in model.named_parameters()),
        "group_name": "tiny.decay",
        "role": "backbone",
        "lr": 0.05,
    }
    optimizer = torch.optim.AdamW([group], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    ema = ModelEMA(model, decay=0.9999, tau=2000)
    criterion = CriterionState()
    scaler = FakeScaler()
    return model, optimizer, scheduler, ema, criterion, scaler


def _step(model, optimizer, scheduler, ema, value, target):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(value) - target).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    ema.update(model)
    return float(loss.detach())


def test_full_snapshot_restores_exact_continuation_and_rng(tmp_path: Path) -> None:
    first = _system()
    model, optimizer, scheduler, ema, criterion, scaler = first
    x1 = torch.tensor([[0.5, -1.0]])
    y1 = torch.tensor([[0.25]])
    x2 = torch.tensor([[1.0, 2.0]])
    y2 = torch.tensor([[-0.5]])
    _step(model, optimizer, scheduler, ema, x1, y1)
    criterion.updates = 1
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    checkpoint = save_training_snapshot(
        tmp_path / "last.pt",
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        criteria=criterion,
        progress=TrainingProgress(
            stage="j1",
            next_epoch=1,
            global_macro_step=1,
            joint_epochs_completed=1,
        ),
        resolved_config={"ratio": "2:1", "reference_batch_size": 64},
        provenance={"dataset_id": "fixture", "source_sha256": "a" * 64},
        loader_state={"detect_images": 6, "pose_images": 3},
        best_state={"best_joint": 0.5},
    )
    expected_rng = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
    )
    expected_loss = _step(model, optimizer, scheduler, ema, x2, y2)
    expected_model = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    expected_ema = {
        name: tensor.detach().clone() for name, tensor in ema.ema.state_dict().items()
    }

    second = _system(seed=999)
    restored = load_training_snapshot(
        checkpoint.path,
        model=second[0],
        ema=second[3],
        optimizer=second[1],
        scheduler=second[2],
        scaler=second[5],
        criteria=second[4],
        restore_rng=True,
    )
    actual_rng = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
    )
    actual_loss = _step(second[0], second[1], second[2], second[3], x2, y2)

    assert actual_rng == pytest.approx(expected_rng)
    assert actual_loss == pytest.approx(expected_loss, rel=0, abs=0)
    for name, tensor in second[0].state_dict().items():
        assert torch.equal(tensor, expected_model[name]), name
    for name, tensor in second[3].ema.state_dict().items():
        assert torch.equal(tensor, expected_ema[name]), name
    assert restored.progress.next_epoch == 1
    assert restored.progress.global_macro_step == 1
    assert restored.loader_state == {"detect_images": 6, "pose_images": 3}
    assert restored.best_state == {"best_joint": 0.5}
    assert second[4].updates == 1
    assert len(checkpoint.sha256) == 64


def test_snapshot_rejects_pending_gradients(tmp_path: Path) -> None:
    model, optimizer, scheduler, ema, criterion, scaler = _system()
    model(torch.ones(1, 2)).sum().backward()

    with pytest.raises(ValueError, match="pending gradients"):
        save_training_snapshot(
            tmp_path / "bad.pt",
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criteria=criterion,
            progress=TrainingProgress("j1", 0, 0, 0),
            resolved_config={},
            provenance={},
            loader_state={},
            best_state={},
        )


def test_inference_weights_exclude_training_state(tmp_path: Path) -> None:
    model, _, _, ema, _, _ = _system()
    path = save_inference_weights(tmp_path / "inference.pt", model=model, ema=ema)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["schema_version"] == 1
    assert "state_dict" in payload
    assert "optimizer" not in payload
    assert "scheduler" not in payload
    assert "rng" not in payload
