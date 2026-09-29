from __future__ import annotations

import torch
from torch import nn

from masf_yolo.models.mfam import MFAM, PartialMFAM, PaperFormulaMFAM, PartialPaperFormulaMFAM


def _make_conv_identity(conv_module: nn.Module) -> None:
    conv = conv_module.conv
    bn = conv_module.bn
    conv.weight.data.zero_()
    center_y = conv.weight.shape[-2] // 2
    center_x = conv.weight.shape[-1] // 2
    for output_channel in range(conv.weight.shape[0]):
        input_channel = 0 if conv.groups == conv.weight.shape[0] else output_channel
        conv.weight.data[output_channel, input_channel, center_y, center_x] = 1.0
    bn.weight.data.fill_(1.0)
    bn.bias.data.zero_()
    bn.running_mean.zero_()
    bn.running_var.fill_(1.0 - bn.eps)


def test_m0_sums_identity_and_all_four_branches_before_fusion() -> None:
    block = MFAM(1, kernels=(3, 5, 7, 9)).eval()
    for branch in block.branches:
        for module in branch.modules():
            if hasattr(module, "conv") and hasattr(module, "bn"):
                _make_conv_identity(module)
    _make_conv_identity(block.fuse)
    value = torch.tensor([[[[-1.0, 0.5], [1.0, 2.0]]]])

    actual = block(value)
    first = torch.nn.functional.silu(value)
    expected_sum = value + 2 * first + 2 * torch.nn.functional.silu(first)
    expected = torch.nn.functional.silu(expected_sum)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert len(block.branches) == 4


def test_m1_has_only_identity_three_and_five_paths() -> None:
    block = MFAM(4, kernels=(3, 5))

    assert block.kernels == (3, 5)
    assert len(block.branches) == 2
    assert block(torch.randn(2, 4, 9, 11)).shape == (2, 4, 9, 11)


def test_m7_sums_identity_three_five_and_one_7_equivalent_path() -> None:
    block = MFAM(1, kernels=(3, 5, 7)).eval()
    for branch in block.branches:
        for module in branch.modules():
            if hasattr(module, "conv") and hasattr(module, "bn"):
                _make_conv_identity(module)
    _make_conv_identity(block.fuse)
    value = torch.tensor([[[[-1.0, 0.5], [1.0, 2.0]]]], requires_grad=True)

    actual = block(value)
    first = torch.nn.functional.silu(value)
    expected = torch.nn.functional.silu(value + 2 * first + torch.nn.functional.silu(first))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert block.kernels == (3, 5, 7)
    assert len(block.branches) == 3
    actual.sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_partial_mfam_processes_leading_half_and_preserves_bypass() -> None:
    block = PartialMFAM(8, processed_ratio=0.5, kernels=(3, 5)).eval()
    value = torch.randn(2, 8, 7, 9, requires_grad=True)

    output = block(value)

    assert block.processed_channels == 4
    torch.testing.assert_close(output[:, 4:], value[:, 4:], rtol=0, atol=0)
    output.sum().backward()
    assert value.grad is not None
    assert torch.all(value.grad[:, 4:] == 1)
    for parameter in block.process.parameters():
        assert parameter.grad is not None


def test_partial_mfam_processes_leading_quarter() -> None:
    block = PartialMFAM(8, processed_ratio=0.25, kernels=(3, 5))

    output = block(torch.randn(1, 8, 5, 5))

    assert block.processed_channels == 2
    assert output.shape == (1, 8, 5, 5)


def test_paper_formula_mfam_has_explicit_branches_and_two_fusions() -> None:
    block = PaperFormulaMFAM(8)
    assert block.kernels == (3, 5, 7, 9)
    assert len(block.branches) == 4
    assert hasattr(block, "pre_fuse") and hasattr(block, "post_fuse")
    assert not any("gate" in name or "branch_weight" in name for name, _ in block.named_parameters())
    assert block(torch.randn(1, 8, 16, 16)).shape == (1, 8, 16, 16)


def test_partial_paper_formula_mfam_bypass_is_exact() -> None:
    block = PartialPaperFormulaMFAM(8, 0.5)
    value = torch.randn(1, 8, 16, 16)
    output = block(value)
    torch.testing.assert_close(output[:, 4:], value[:, 4:], rtol=0, atol=0)
