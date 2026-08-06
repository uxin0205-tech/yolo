import copy

import torch

from binary_attention.attention.base import FPAttention, ScaledBinaryAttention, SignOnlyBinaryAttention
from binary_attention.attention.quantizers import fake_quant_p8
from binary_attention.variants.definitions import get_variant
from binary_attention.verification.checkpoint import save_checkpoint, strict_reload
from binary_attention.kd import calibrate_weight, distillation_loss


def test_sign_forward_and_counter():
    module = SignOnlyBinaryAttention(16, 2)
    assert module(torch.randn(2, 16, 4, 4)).shape == (2, 16, 4, 4)
    assert module.binary_forward_count.item() == 1


def test_clipped_ste_reaches_qkv():
    module = ScaledBinaryAttention(16, 2, use_qat=True).train()
    module(torch.randn(2, 16, 4, 4)).mean().backward()
    assert module.qkv.weight.grad is not None
    assert torch.isfinite(module.qkv.weight.grad).all()


def test_ema_deepcopy_drops_forward_only_kd_graph():
    module = ScaledBinaryAttention(16, 2, use_qat=True).train()
    module(torch.randn(1, 16, 4, 4, requires_grad=True))
    assert module.kd_probabilities is not None
    assert module.kd_probabilities.grad_fn is not None

    ema = copy.deepcopy(module)

    assert ema.kd_probabilities is None
    assert ema.last_scores is None
    assert ema.last_probabilities is None
    assert torch.equal(ema.binary_forward_count, module.binary_forward_count)


def test_p8_is_not_renormalized():
    values = torch.tensor([[0.1, 0.1, 0.8]])
    assert not torch.equal(fake_quant_p8(values).sum(-1), torch.ones(1))


def test_strict_reload(tmp_path):
    variant = get_variant("T2")
    first = ScaledBinaryAttention(16, 2, use_qat=True)
    first(torch.randn(1, 16, 4, 4))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, first, variant)
    restored = ScaledBinaryAttention(16, 2, use_qat=True)
    manifest = strict_reload(path, restored, variant)
    assert manifest.attention_class == "ScaledBinaryAttention"


def test_kd_detaches_teacher_and_calibrates_weight():
    student = torch.randn(2, 3, requires_grad=True); teacher = torch.randn(2, 3, requires_grad=True)
    loss = distillation_loss(student, teacher, "output"); loss.backward()
    assert teacher.grad is None and student.grad is not None
    assert calibrate_weight(10, 5, .1) == .2
