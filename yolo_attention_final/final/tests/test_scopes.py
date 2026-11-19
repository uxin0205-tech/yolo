from __future__ import annotations

import torch
from torch import nn

from yolo_attention.attention import HardwareFriendlyAttention
from yolo_attention.integration import YOLO26M_ATTENTION_PATHS
from yolo_attention.scopes import apply_trainable_scope, enforce_frozen_batchnorm, sync_progressive_epoch


def _attention() -> HardwareFriendlyAttention:
    module = HardwareFriendlyAttention.__new__(HardwareFriendlyAttention)
    nn.Module.__init__(module)
    module.qkv = nn.Module()
    for part in ("q", "k", "v"):
        branch = nn.Module()
        branch.conv = nn.Conv2d(8, 8, 1, bias=False)
        branch.bn = nn.BatchNorm2d(8)
        setattr(module.qkv, part, branch)
    module.pe = nn.Conv2d(8, 8, 1)
    module.proj = nn.Sequential(nn.Conv2d(8, 8, 1), nn.BatchNorm2d(8))
    module.score = nn.Module()
    module.score.gamma = nn.Parameter(torch.ones(2))
    module.score.register_buffer("fixed_coefficients", torch.ones(4, 2))
    module.bias = nn.Module()
    module.bias.table_x = nn.Parameter(torch.zeros(4, 63))
    module.bias.table_y = nn.Parameter(torch.zeros(4, 63))
    module.normalize = nn.Module()
    module.normalize.register_buffer("endpoint_table", torch.arange(21))
    module.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long))
    return module


def _model() -> nn.Module:
    root = nn.Module()
    root.stem = nn.Sequential(nn.Conv2d(8, 8, 1), nn.BatchNorm2d(8))
    root.model = nn.ModuleList(nn.Identity() for _ in range(23))
    site10 = nn.Module()
    site10.m = nn.ModuleList([nn.Module()])
    site10.m[0].attn = _attention()
    root.model[10] = site10
    site22 = nn.Module()
    site22.m = nn.ModuleList([nn.ModuleList([nn.Identity(), nn.Module()])])
    site22.m[0][1].attn = _attention()
    root.model[22] = site22
    return root


def test_bias_scope_is_exactly_1008_and_excludes_dead_state() -> None:
    model = _model()
    summary = apply_trainable_scope(model, "bias_only")
    assert summary.trainable_parameters == 1008
    assert all(".bias.table_" in name for name in summary.trainable_names)


def test_qk_and_refinement_scope_boundaries() -> None:
    model = _model()
    qk = apply_trainable_scope(model, "qk_recovery")
    assert all(
        any(token in name for token in (".qkv.q.", ".qkv.k.", ".bias.table_")) for name in qk.trainable_names
    )
    full = apply_trainable_scope(model, "attention_refinement")
    assert any(".qkv.v." in name for name in full.trainable_names)
    assert any(".proj." in name for name in full.trainable_names)
    assert not any(name.endswith("score.gamma") for name in full.trainable_names)


def test_frozen_batchnorm_buffers_and_ema_epoch() -> None:
    model = _model().train()
    enforce_frozen_batchnorm(model, "qk_recovery")
    assert not model.stem[1].training
    for path in YOLO26M_ATTENTION_PATHS:
        attention = model.get_submodule(path)
        assert attention.qkv.q.bn.training
        assert attention.qkv.k.bn.training
        assert not attention.qkv.v.bn.training
    ema = _model()
    sync_progressive_epoch(model, ema, 7)
    assert all(int(model.get_submodule(path).current_epoch) == 7 for path in YOLO26M_ATTENTION_PATHS)
    assert all(int(ema.get_submodule(path).current_epoch) == 7 for path in YOLO26M_ATTENTION_PATHS)
