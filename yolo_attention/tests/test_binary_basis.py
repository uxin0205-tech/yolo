from __future__ import annotations

import pytest
import torch

from yolo_attention.binary_basis import BinaryScore, fast_hadamard_transform, xnor_popcount_dot
from yolo_attention.config import BasisKind, ScaleMode


def test_xnor_popcount_matches_signed_matrix_product() -> None:
    q = torch.tensor([[[[1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]]])
    k = torch.tensor([[[[1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]]])

    packed_reference = xnor_popcount_dot(q, k)
    signed_reference = q.transpose(-2, -1) @ k

    torch.testing.assert_close(packed_reference, signed_reference)


def test_normalized_hadamard_preserves_sign_and_energy() -> None:
    x = torch.tensor([[[[1.0], [2.0], [3.0], [5.0]]]])

    raw = fast_hadamard_transform(x, dim=-2, normalize=False)
    normalized = fast_hadamard_transform(x, dim=-2, normalize=True)

    torch.testing.assert_close(normalized.square().sum(), x.square().sum())
    assert torch.equal(torch.sign(raw), torch.sign(normalized))


@pytest.mark.parametrize("basis", [BasisKind.IDENTITY, BasisKind.HADAMARD, BasisKind.T5])
def test_binary_score_has_global_attention_shape_and_gradients(basis: BasisKind) -> None:
    q = torch.randn(2, 2, 4, 6, requires_grad=True)
    k = torch.randn(2, 2, 4, 6, requires_grad=True)
    module = BinaryScore(num_heads=2, basis=basis, scale_mode=ScaleMode.DYNAMIC, use_ste=True)

    scores = module(q, k)
    scores.mean().backward()

    assert scores.shape == (2, 2, 6, 6)
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()


def test_fixed_head_scale_removes_input_dependent_magnitude() -> None:
    module = BinaryScore(
        num_heads=2,
        basis=BasisKind.HADAMARD,
        scale_mode=ScaleMode.FIXED_HEAD,
        use_ste=False,
    )
    module.set_fixed_coefficients(torch.tensor([[0.25, 0.5], [0.75, 1.0]]))
    q, k = torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3)

    first = module(q, k)
    second = module(10 * q, 10 * k)

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("scale_mode", [ScaleMode.FIXED_HEAD, ScaleMode.POWER_OF_TWO])
def test_fixed_scale_zero_shot_calibration_freezes_per_head_coefficients(
    scale_mode: ScaleMode,
) -> None:
    module = BinaryScore(
        num_heads=2,
        basis=BasisKind.HADAMARD,
        scale_mode=scale_mode,
        use_ste=False,
    ).eval()
    q = torch.tensor([[[[1.0, 3.0], [1.0, 3.0]], [[2.0, 4.0], [2.0, 4.0]]]])
    k = q.clone()

    module.begin_calibration()
    calibration_scores = module(q, k)
    coefficients = module.finish_calibration()
    fixed_scores = module(q, k)

    assert bool(module.fixed_coefficients_ready)
    assert coefficients.shape == (2, 2)
    assert torch.isfinite(coefficients).all()
    if scale_mode is ScaleMode.FIXED_HEAD:
        torch.testing.assert_close(fixed_scores, calibration_scores)
    else:
        logarithms = torch.log2(coefficients.abs())
        torch.testing.assert_close(logarithms, logarithms.round())
        torch.testing.assert_close(fixed_scores, module(q, k))


def test_hadamard_dynamic_scale_is_stable_between_train_and_eval_without_ste() -> None:
    module = BinaryScore(
        num_heads=2,
        basis=BasisKind.HADAMARD,
        scale_mode=ScaleMode.DYNAMIC,
        use_ste=False,
    )
    q, k = torch.randn(1, 2, 4, 3), torch.randn(1, 2, 4, 3)

    training_scores = module.train()(q, k)
    evaluation_scores = module.eval()(q, k)

    torch.testing.assert_close(training_scores, evaluation_scores)
