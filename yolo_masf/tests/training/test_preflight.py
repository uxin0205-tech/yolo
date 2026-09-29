from __future__ import annotations

import torch

from masf_yolo.models.builder import build_model
from masf_yolo.training.preflight import (
    probe_common_batch,
    run_finite_loss_batch,
    run_optimizer_step,
)


def test_real_model_forward_backward_produces_finite_loss() -> None:
    model = build_model("M3")
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
    }

    loss = run_finite_loss_batch(model, batch, device=torch.device("cpu"))

    assert loss > 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_optimizer_probe_allocates_state_and_updates_real_model() -> None:
    model = build_model("M3")
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
    }
    before = model.model[0].conv.weight.detach().clone()

    loss = run_optimizer_step(model, batch, device=torch.device("cpu"), amp=False)

    assert loss > 0
    assert not torch.equal(before, model.model[0].conv.weight)


def test_common_batch_requires_every_variant_to_pass() -> None:
    calls: list[tuple[str, int]] = []

    def probe(variant: str, batch: int) -> bool:
        calls.append((variant, batch))
        return not (variant == "M0" and batch > 4)

    selected = probe_common_batch(probe)

    assert selected == 4
    assert ("M0", 16) in calls
    assert ("M0", 8) in calls
    assert all((variant, 4) in calls for variant in ("B1", "M0", "M1", "M2", "M3"))


def test_common_batch_fails_if_batch_one_is_not_shared() -> None:
    def probe(variant: str, batch: int) -> bool:
        return variant != "M0"

    try:
        probe_common_batch(probe)
    except RuntimeError as error:
        assert "no common batch" in str(error)
    else:
        raise AssertionError("probe must fail closed")
