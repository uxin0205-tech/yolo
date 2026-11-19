import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from binaryqk import basis
from binaryqk.checkpoint import CONTRACT, manifest_for, validate_manifest
from binaryqk.model import BranchAttention, SplitQKV
from ultralytics.nn.modules.block import Attention


def test_exact_dot_zero_ties_rectangular_tokens_and_backward():
    q = torch.tensor([[[[0., -1., 1.], [-1., 1., 1.]]]], requires_grad=True)
    k = torch.tensor([[[[1., -1.], [-1., -1.]]]], requires_grad=True)
    a, b = basis.ClippedSign.apply(q), basis.ClippedSign.apply(k)
    out = basis.ExactDotSurrogate.apply(a, b)
    expected = a.transpose(-2, -1) @ b
    assert torch.equal(out, expected)
    assert torch.equal(out, basis.xnor_dot(a, b, tile=1))
    gradient = torch.arange(6.).reshape_as(out)
    out.backward(gradient)
    assert torch.equal(q.grad, b.detach() @ gradient.transpose(-2, -1))
    assert torch.equal(k.grad, a.detach() @ gradient)


def test_clipped_gradient_and_hadamard_channel_axis():
    x = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
    basis.ClippedSign.apply(x).sum().backward()
    assert torch.equal(x.grad, torch.tensor([0., 1., 1., 1., 0.]))
    value = torch.randn(2, 4, 32, 15)
    torch.testing.assert_close(basis.hadamard(basis.hadamard(value, normalize=True), normalize=True), value)
    with pytest.raises(ValueError):
        basis.hadamard(torch.ones(1, 1, 7, 16))


@pytest.mark.parametrize("variant", basis.BASES)
def test_branch_reference_single_combination_and_train_eval(variant):
    torch.manual_seed(1)
    q, k = torch.randn(1, 4, 32, 15), torch.randn(1, 4, 32, 15)
    coefficients = torch.rand(4, len(basis.BASES[variant])) + .01
    module = basis.BranchScore(variant, coefficients)
    sign = lambda x: torch.where(x >= 0, 1., -1.)
    pairs = {"sign": (q, k),
             "residual": (q - q.abs().mean(-2, keepdim=True) * sign(q),
                          k - k.abs().mean(-2, keepdim=True) * sign(k)),
             "hadamard": (basis.hadamard(q), basis.hadamard(k))}
    expected = sum((sign(pairs[b][0]).transpose(-2, -1) @ sign(pairs[b][1]))
                   * coefficients[:, i].view(1, 4, 1, 1) for i, b in enumerate(basis.BASES[variant]))
    with patch.object(basis, "xnor_dot", wraps=basis.xnor_dot) as dot:
        assert torch.equal(module(q, k), expected)
        assert dot.call_count == len(basis.BASES[variant])
    assert torch.equal(module.eval()(q, k), expected)


def test_split_qkv_layout_preserves_native_attention_output():
    torch.manual_seed(2)
    native = Attention(256, 4).eval()
    split = SplitQKV(native).eval()
    x = torch.randn(2, 256, 3, 5)
    with torch.no_grad():
        expected = native.qkv(x).view(2, 4, 128, 15).split((32, 32, 64), 2)
        actual = tuple(v.reshape(2, 4, -1, 15) for v in split(x))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        q, k, v = actual
        p = ((q * native.scale).transpose(-2, -1) @ k).softmax(-1)
        output = native.proj((v @ p.transpose(-2, -1)).reshape(2, 256, 3, 5)
                             + native.pe(v.reshape(2, 256, 3, 5)))
        torch.testing.assert_close(native(x), output, rtol=0, atol=0)
    candidate = BranchAttention(native, "B3", torch.ones(4, 3))
    for part in ("pe", "proj"):
        for key, tensor in getattr(native, part).state_dict().items():
            assert torch.equal(tensor, getattr(candidate, part).state_dict()[key])


def test_calibration_matches_explicit_pair_mean():
    q, k = torch.randn(2, 4, 32, 7), torch.randn(2, 4, 32, 7)
    explicit = (q.abs().mean(-2).unsqueeze(-1) * k.abs().mean(-2).unsqueeze(-2)).mean((0, 2, 3)) / 32**.5
    torch.testing.assert_close(basis.calibration_moments(q, k)[:, 0], explicit)


@pytest.mark.parametrize("field", list(CONTRACT))
def test_contract_fails_closed_for_missing_field(field):
    manifest = copy.deepcopy(CONTRACT)
    del manifest[field]
    with pytest.raises(ValueError, match=field):
        validate_manifest(manifest)


def test_invalid_coefficients_and_unsupported_variants():
    for value in (torch.zeros(4, 2), torch.full((4, 2), float("nan")), torch.ones(4, 3)):
        with pytest.raises(ValueError):
            basis.BranchScore("B1", value)
    with pytest.raises(ValueError):
        basis.BranchScore("sign-only", torch.ones(4, 1))


def test_manifest_rejects_silent_normalization_replacement():
    attention = BranchAttention(Attention(256, 4).eval(), "B1", torch.ones(4, 2))
    attention.normalizer = torch.nn.Identity()
    model = SimpleNamespace(get_submodule=lambda name: attention)
    with pytest.raises(ValueError, match="normalization"):
        manifest_for(model, "0" * 64)
