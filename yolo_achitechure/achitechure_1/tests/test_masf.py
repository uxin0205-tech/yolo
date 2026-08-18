from __future__ import annotations

import torch

from achitechure_1.masf import P3MASFFull35, P3MASFPartial75


def test_full35_preserves_p3_shape_and_initializes_alpha() -> None:
    module = P3MASFFull35(16)
    sample = torch.randn(2, 16, 12, 12)

    output = module(sample)

    assert output.shape == sample.shape
    assert module.alpha.detach().item() == torch.tensor(0.01).item()


def test_partial75_keeps_three_quarters_of_channels_bit_exact() -> None:
    module = P3MASFPartial75(16)
    sample = torch.randn(2, 16, 12, 12)

    output = module(sample)

    assert module.context_channels == 4
    assert module.bypass_channels == 12
    assert torch.equal(output[:, 4:], sample[:, 4:])


def test_new_branch_parameters_receive_finite_gradients() -> None:
    for module in (P3MASFFull35(16), P3MASFPartial75(16)):
        sample = torch.randn(2, 16, 12, 12, requires_grad=True)

        module(sample).square().mean().backward()

        assert sample.grad is not None and torch.isfinite(sample.grad).all()
        assert all(parameter.grad is not None for parameter in module.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())
