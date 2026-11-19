import pytest
import torch

from yolo_attention.config import ReciprocalBackend, ReciprocalConfig
from yolo_attention.normalization import BitTruePiecewiseLinearSoftmax, CompletePWLSoftmax


@pytest.mark.parametrize('backend', list(ReciprocalBackend))
def test_exp_boundaries_full_probability_and_backward(backend):
    unit = CompletePWLSoftmax(ReciprocalConfig(backend=backend))
    scores = torch.tensor([[0., -0.5, -1., -9.5, -10., -10.5, -128.],
                           [1., 1., 1., 1., 1., 1., 1.]], requires_grad=True)
    reference = BitTruePiecewiseLinearSoftmax(score_floor=-10, segments=20)
    exact_weights = reference.approximate_weights(scores-scores.amax(-1, keepdim=True))
    weights = unit.weight_codes(scores)
    assert torch.equal(weights, (exact_weights*32768).long())
    assert weights[:, 0].tolist() == [32768, 32768]
    assert weights.min() >= 0 and weights.max() <= 32768
    trained = unit(scores)
    deployed = unit.eval()(scores)
    assert torch.equal(trained, deployed)
    assert deployed.min() >= 0 and deployed.max() <= 1
    assert (deployed.sum(-1)-1).abs().max() < .0022
    (trained*torch.arange(7)).sum().backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0


@pytest.mark.parametrize('tokens', [1, 3, 400, 1024])
def test_uniform_ties_and_dominant_rows(tokens):
    scores = torch.zeros(2, tokens)
    scores[1, 1:] = -100
    for backend in ReciprocalBackend:
        unit = CompletePWLSoftmax(ReciprocalConfig(backend=backend)).eval()
        output = unit(scores)
        assert torch.isfinite(output).all()
        assert (output.sum(-1)-1).abs().max() < .0022
        assert output[0].max() == output[0].min()
        assert unit.weight_codes(scores).sum(-1).max() <= 1 << 25
    with pytest.raises(ValueError):
        unit(torch.zeros(1, 1025))


def test_row_sum_error_is_not_hidden_by_float_renormalization():
    unit = CompletePWLSoftmax(ReciprocalConfig(backend='lut256')).eval()
    scores = torch.tensor([[0., -1., -2.]])
    w = unit.weight_codes(scores)
    p = unit(scores)
    expected_codes = unit.reciprocal.probability_codes(w)
    assert torch.equal(p, expected_codes.float()*(2.**-24))
    assert abs(p.sum().item()-1) > 1e-6
