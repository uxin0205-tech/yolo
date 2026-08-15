from __future__ import annotations

import pytest
import torch

from yolo_attention.config import NormalizationKind, VariantConfig
from yolo_attention.normalization import Multimax, ProgressiveNormalizer, build_normalizer


@pytest.mark.parametrize(
    "kind",
    [
        NormalizationKind.EXACT,
        NormalizationKind.LUT,
        NormalizationKind.PIECEWISE_LINEAR,
        NormalizationKind.POWER_OF_TWO,
        NormalizationKind.HARD_SIGMOID,
        NormalizationKind.RELU,
        NormalizationKind.MULTIMAX,
    ],
)
def test_normalization_candidates_emit_valid_rows(kind: NormalizationKind) -> None:
    config = VariantConfig(name=f"test-{kind.value}", normalization=kind, multimax_top_k=3)
    module = build_normalizer(config).eval()
    scores = torch.tensor([[[[2.0, 0.5, -1.0, -4.0, 1.0]]]])

    probabilities = module(scores)

    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]))


@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_multimax_only_keeps_requested_top_k(top_k: int) -> None:
    scores = torch.tensor([[[[0.1, 3.0, 2.0, -1.0, 1.0, 0.0]]]])

    probabilities = Multimax(top_k=top_k)(scores)

    assert torch.count_nonzero(probabilities).item() == top_k
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]))


def test_progressive_normalizer_blends_exact_to_candidate() -> None:
    scores = torch.tensor([[[[2.0, 0.5, -1.0, -4.0]]]])
    candidate = Multimax(top_k=1)
    module = ProgressiveNormalizer(candidate, transition_epochs=5)

    module.set_epoch(0)
    torch.testing.assert_close(module(scores), scores.softmax(dim=-1))
    module.set_epoch(5)
    torch.testing.assert_close(module(scores), candidate(scores))


def test_n0_lut_does_not_quantize_probability_to_u8_grid() -> None:
    module = build_normalizer(VariantConfig(name="N0-LUT", normalization=NormalizationKind.LUT)).eval()
    scores = torch.tensor([[[[0.0, -0.3, -1.2]]]])

    probabilities = module(scores)

    assert not torch.allclose(probabilities * 255.0, torch.round(probabilities * 255.0))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]))
