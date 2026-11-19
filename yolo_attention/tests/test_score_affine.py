import pytest
import torch

from yolo_attention.config import FixedScaleFormat, ScoreAffineConfig
from yolo_attention.score_affine import QuantizedScoreAffine


@pytest.mark.parametrize("kind", list(FixedScaleFormat))
def test_integer_score_matches_quantized_float_and_propagates_gradients(kind):
    module = QuantizedScoreAffine(ScoreAffineConfig(format=kind), torch.full((4, 2), .07))
    with torch.no_grad():
        module.table_x.copy_(torch.randn_like(module.table_x) * .1)
        module.table_y.copy_(torch.randn_like(module.table_y) * .1)
    counts = [torch.randint(-16, 17, (1, 4, 6, 6)).float().mul(2).requires_grad_() for _ in range(2)]
    output = module(counts, height=2, width=3)
    direct = module.integer_score([x.detach().int() for x in counts], height=2, width=3).float() / 1024
    assert torch.equal(output, direct)
    weights = torch.randn_like(output)
    (output * weights).sum().backward()
    for tensor in [*counts, module.coefficients, module.table_x, module.table_y]:
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().max() > 0
    module.eval()
    assert torch.equal(module(counts, height=2, width=3), output.detach())


def test_q10_rounding_saturation_and_fixed_inference_constants():
    values = torch.tensor([[0., .0001], [1.5 / 1024, 2.5 / 1024], [.5, 1.], [2., .25]])
    module = QuantizedScoreAffine(ScoreAffineConfig(), values).eval()
    assert module.export_constants()["scale_codes"] == [[1, 1], [2, 2], [512, 1024], [1024, 256]]
    with torch.no_grad():
        module.table_x.fill_(40)
        module.table_y.fill_(-40)
    module.eval()
    constants = module.export_constants()
    assert {v for row in constants["bias_x_codes"] for v in row} == {32767}
    assert {v for row in constants["bias_y_codes"] for v in row} == {-32768}
    assert "cached_scale_codes" not in module.state_dict()


def test_power_of_two_shift_matches_q10_at_all_exponents():
    cfg = ScoreAffineConfig(format=FixedScaleFormat.POWER_OF_TWO)
    counts = [torch.tensor([-32, -2, 0, 32], dtype=torch.int32).reshape(1, 1, 2, 2).expand(1, 4, 2, 2)] * 2
    for exponent in range(-10, 1):
        module = QuantizedScoreAffine(cfg, torch.full((4, 2), 2. ** exponent)).eval()
        codes = module.integer_score(counts, height=1, width=2)
        assert torch.equal(codes, sum(counts) * (1 << (exponent + 10)))
        assert {v for row in module.export_constants()["scale_codes"] for v in row} == {exponent}


def test_bias_orientation_and_no_basis_duplication():
    module = QuantizedScoreAffine(ScoreAffineConfig(), torch.ones(4, 2)).eval()
    with torch.no_grad():
        module.table_y[:, 31 + 1] = 2
        module.table_x[:, 31 - 2] = 3
    module.eval()
    counts = [torch.zeros(1, 4, 6, 6, dtype=torch.int32)] * 2
    score = module.integer_score(counts, height=2, width=3)
    assert score[0, 0, 3, 2] == 5 * 1024  # query (1,0), key (0,2)
    assert score[0, 0, 2, 3] == 0
    assert module.table_x.numel() + module.table_y.numel() == 504


def test_shape_bounds_and_illegal_formats_fail_closed():
    module = QuantizedScoreAffine(ScoreAffineConfig(), torch.ones(4, 2))
    counts = [torch.zeros(1, 4, 6, 6)] * 2
    with pytest.raises(ValueError, match="shape"):
        module(counts, height=2, width=2)
    with pytest.raises(ValueError, match="size"):
        module(counts, height=1, width=33)
    with pytest.raises(ValueError):
        ScoreAffineConfig(format="float")
    with pytest.raises(ValueError):
        QuantizedScoreAffine(ScoreAffineConfig(), torch.full((4, 2), float('nan')))


def test_worst_case_score_requires_signed18_and_maximum_tokens():
    module = QuantizedScoreAffine(ScoreAffineConfig(), torch.ones(4, 2)).eval()
    with torch.no_grad():
        module.table_x.fill_(-32)
        module.table_y.fill_(-32)
    module.eval()
    counts = [torch.full((1, 4, 1024, 1024), -32, dtype=torch.int32)] * 2
    score = module.integer_score(counts, height=32, width=32)
    assert score.min() == -(1 << 17) and score.max() == -(1 << 17)
    assert module.config.accumulator_bits == 18
