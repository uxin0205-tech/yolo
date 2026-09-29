from __future__ import annotations

import pytest
import torch

from yolo_combine.xnor import (
    XNORExecutionConfig,
    broadcast_xnor_popcount_dot,
    estimate_xnor_workspace,
    tiled_xnor_popcount_dot,
)


def test_tiled_xnor_returns_known_exact_scores() -> None:
    q = torch.tensor([[[[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]]]])
    k = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]]])

    actual = tiled_xnor_popcount_dot(q, k, token_tile=1)

    expected = torch.tensor([[[[1.0, 1.0], [-1.0, 3.0]]]])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("token_tile", [1, 2, 3, 8])
def test_tiled_xnor_preserves_values_and_existing_gradient_semantics(
    token_tile: int,
) -> None:
    generator = torch.Generator().manual_seed(20260824)
    q_reference = torch.randn(2, 3, 5, 7, generator=generator, requires_grad=True)
    k_reference = torch.randn(2, 3, 5, 9, generator=generator, requires_grad=True)
    q_tiled = q_reference.detach().clone().requires_grad_(True)
    k_tiled = k_reference.detach().clone().requires_grad_(True)

    reference_scores = broadcast_xnor_popcount_dot(q_reference, k_reference)
    tiled_scores = tiled_xnor_popcount_dot(q_tiled, k_tiled, token_tile=token_tile)
    assert torch.equal(tiled_scores, reference_scores)
    assert not reference_scores.requires_grad
    assert not tiled_scores.requires_grad

    # The retained Full35 path receives Q/K gradients only through its magnitude
    # coefficient. The tiled execution must preserve that exact behavior.
    reference_magnitude = q_reference.abs().mean(dim=-2).unsqueeze(-1)
    tiled_magnitude = q_tiled.abs().mean(dim=-2).unsqueeze(-1)
    reference_loss = (reference_scores * reference_magnitude).sum()
    tiled_loss = (tiled_scores * tiled_magnitude).sum()
    reference_loss.backward()
    tiled_loss.backward()

    assert torch.equal(q_tiled.grad, q_reference.grad)
    assert k_tiled.grad is None
    assert k_reference.grad is None


def test_workspace_estimate_exposes_untiled_oom_and_tiled_peak() -> None:
    untiled = estimate_xnor_workspace(
        batch=128,
        heads=4,
        query_tokens=400,
        key_tokens=400,
        channels=32,
        token_tile=None,
    )
    tiled = estimate_xnor_workspace(
        batch=128,
        heads=4,
        query_tokens=400,
        key_tokens=400,
        channels=32,
        token_tile=32,
    )

    assert untiled.reduction_bytes == 20_971_520_000
    assert tiled.reduction_bytes == 134_217_728
    assert untiled.reduction_bytes / tiled.reduction_bytes == 156.25
    assert tiled.query_tile == 32
    assert tiled.key_tile == 32


def test_xnor_config_rejects_non_integer_or_non_positive_tiles() -> None:
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises((TypeError, ValueError)):
            XNORExecutionConfig(token_tile=invalid)  # type: ignore[arg-type]
