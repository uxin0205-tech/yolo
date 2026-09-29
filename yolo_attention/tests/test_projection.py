from __future__ import annotations

import torch
from ultralytics.nn.modules.block import Attention

from yolo_attention.projection import (
    ModularQKVProjection,
    fold_conv_bn,
    interleave_qkv_maps,
    split_folded_qkv,
)


def test_modular_qkv_matches_fused_conv_bn_in_eval_mode() -> None:
    source = Attention(dim=32, num_heads=2, attn_ratio=0.5).eval()
    projection = ModularQKVProjection.from_fused(
        source.qkv,
        q_channels=source.key_dim * source.num_heads,
        k_channels=source.key_dim * source.num_heads,
        v_channels=32,
        num_heads=source.num_heads,
    ).eval()
    x = torch.randn(2, 32, 5, 5)

    expected = source.qkv(x)
    actual = interleave_qkv_maps(*projection(x), num_heads=source.num_heads)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_bn_fold_and_channel_split_are_equivalent() -> None:
    source = Attention(dim=32, num_heads=2, attn_ratio=0.5).eval()
    x = torch.randn(1, 32, 4, 4)
    folded = fold_conv_bn(source.qkv).eval()
    q_conv, k_conv, v_conv = split_folded_qkv(
        folded,
        num_heads=source.num_heads,
        key_dim=source.key_dim,
        head_dim=source.head_dim,
    )

    torch.testing.assert_close(folded(x), source.qkv(x), rtol=1e-5, atol=1e-6)
    reconstructed = interleave_qkv_maps(q_conv(x), k_conv(x), v_conv(x), num_heads=source.num_heads)
    torch.testing.assert_close(reconstructed, folded(x))


def test_modular_qkv_preserves_checkpoint_bn_hyperparameters() -> None:
    source = Attention(dim=32, num_heads=2, attn_ratio=0.5).eval()
    source.qkv.bn.eps = 1e-3
    source.qkv.bn.momentum = 0.03
    projection = ModularQKVProjection.from_fused(
        source.qkv,
        q_channels=source.num_heads * source.key_dim,
        k_channels=source.num_heads * source.key_dim,
        v_channels=32,
        num_heads=source.num_heads,
    ).eval()
    x = torch.randn(1, 32, 4, 4)

    reconstructed = interleave_qkv_maps(*projection(x), num_heads=source.num_heads)

    assert projection.q.bn.eps == source.qkv.bn.eps
    assert projection.q.bn.momentum == source.qkv.bn.momentum
    torch.testing.assert_close(reconstructed, source.qkv(x), rtol=1e-5, atol=1e-6)
