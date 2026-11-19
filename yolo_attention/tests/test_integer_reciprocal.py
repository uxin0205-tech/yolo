import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from yolo_attention.config import ReciprocalBackend, ReciprocalConfig
from yolo_attention.reciprocal import IntegerReciprocal


@pytest.mark.parametrize('backend,limit', [('lut256', .0021), ('pwl16', .0011), ('newton64', .00012)])
def test_dense_denominators_and_power_boundaries(backend, limit):
    unit = IntegerReciprocal(ReciprocalConfig(backend=backend))
    powers = torch.tensor([1 << k for k in range(26)], dtype=torch.int64)
    dense = torch.arange(1, 65537, dtype=torch.int64)
    large = torch.arange(65537, (1 << 25) + 1, 509, dtype=torch.int64)
    values = torch.unique(torch.cat([dense, large, powers, (powers-1).clamp_min(1),
                                     (powers+1).clamp_max(1 << 25)]))
    codes, exponent = unit(values)
    relative = values.double() * codes.double() * torch.pow(2., -16-exponent.double()) - 1
    assert relative.abs().max() < limit
    c, e = unit(powers)
    assert torch.equal(c, torch.full_like(c, 65536))
    assert torch.equal(e, torch.arange(26))


@pytest.mark.parametrize('backend', list(ReciprocalBackend))
def test_invalid_zero_overflow_and_fractional_denominator(backend):
    unit = IntegerReciprocal(ReciprocalConfig(backend=backend))
    for invalid in [torch.tensor([0]), torch.tensor([-1]), torch.tensor([(1 << 25)+1]), torch.tensor([1.5])]:
        with pytest.raises(ValueError):
            unit(invalid)


class Operations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.names = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.names.append(str(func))
        return func(*args, **(kwargs or {}))


def test_integer_path_has_no_runtime_division_log_exp_or_frexp():
    for backend in ReciprocalBackend:
        unit = IntegerReciprocal(ReciprocalConfig(backend=backend))
        weights = torch.tensor([[32768, 100, 231, 1]], dtype=torch.int64)
        with Operations() as observed:
            output = unit.probability_codes(weights)
        assert output.dtype == torch.int64
        assert not any(any(key in name for key in ('aten.div.', 'aten.log', 'aten.exp.', 'aten.frexp', 'aten.ldexp'))
                       for name in observed.names)
