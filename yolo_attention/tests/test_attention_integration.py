from __future__ import annotations

import copy

import torch
from ultralytics.nn.modules.block import C2PSA, Attention

from yolo_attention.attention import HardwareFriendlyAttention
from yolo_attention.config import (
    BasisKind,
    BiasKind,
    NormalizationKind,
    ScaleMode,
    VariantConfig,
)
from yolo_attention.integration import convert_c2psa, freeze_for_stage


def fp_config() -> VariantConfig:
    return VariantConfig(
        name="P0",
        basis=BasisKind.FP,
        bias=BiasKind.NONE,
        scale_mode=ScaleMode.DYNAMIC,
        normalization=NormalizationKind.EXACT,
    )


def test_fp_modular_attention_preserves_official_attention_output() -> None:
    source = Attention(dim=32, num_heads=2, attn_ratio=0.5).eval()
    converted = HardwareFriendlyAttention.from_ultralytics(copy.deepcopy(source), fp_config()).eval()
    x = torch.randn(1, 32, 4, 4)

    torch.testing.assert_close(converted(x), source(x), rtol=1e-5, atol=1e-6)


def test_c2psa_conversion_preserves_pe_ffn_residual_and_csp_path() -> None:
    source = C2PSA(128, 128, n=1, e=0.5).eval()
    converted = copy.deepcopy(source)
    paths = convert_c2psa(converted, fp_config())
    x = torch.randn(1, 128, 4, 4)

    assert paths == ["m.0.attn"]
    torch.testing.assert_close(converted(x), source(x), rtol=1e-5, atol=1e-6)


def test_screening_scope_only_unfreezes_attention() -> None:
    model = C2PSA(128, 128, n=1, e=0.5)
    convert_c2psa(model, fp_config())

    summary = freeze_for_stage(model, "screening")

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(".attn." in f".{name}." for name in trainable)
    assert summary.trainable_parameters < summary.total_parameters
