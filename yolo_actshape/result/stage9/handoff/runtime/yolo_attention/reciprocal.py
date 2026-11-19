"""專案延伸的整數 reciprocal reference；不含動態除法、frexp 或浮點 log。"""

import torch
from torch import nn

from .config import ReciprocalBackend, ReciprocalConfig


def round_shift(value: torch.Tensor, shift: int | torch.Tensor) -> torch.Tensor:
    """完整 signed product 捨入到指定邊界；nearest，ties 向正無窮。"""
    shifts = torch.as_tensor(shift, dtype=torch.int64, device=value.device)
    if (shifts < 0).any():
        raise ValueError('round_shift requires nonnegative shifts')
    half = torch.bitwise_left_shift(torch.ones_like(shifts), (shifts-1).clamp_min(0))
    return torch.where(shifts == 0, value, torch.bitwise_right_shift(value+half, shifts))


class IntegerReciprocal(nn.Module):
    """回傳 Q1.16 mantissa 與 exponent，使 reciprocal = code × 2^(-16-e)。"""

    def __init__(self, config: ReciprocalConfig):
        super().__init__()
        self.config = config
        size = 256 if config.backend is ReciprocalBackend.LUT256 else 64
        if config.backend is ReciprocalBackend.PWL16:
            points = 1 + torch.arange(17, dtype=torch.float64)/16
        else:
            points = 1 + (torch.arange(size, dtype=torch.float64)+.5)/size
        # 離線常數生成；部署只讀表，完整常數嵌入 checkpoint／manifest。
        values = torch.floor(65536/points+.5).to(torch.int64)
        self.register_buffer('table', values, persistent=True)

    @staticmethod
    def leading_one(denominator: torch.Tensor) -> torch.Tensor:
        working = denominator.clone()
        exponent = torch.zeros_like(working)
        for shift in (16, 8, 4, 2, 1):
            active = working >= (1 << shift)
            working = torch.where(active, working >> shift, working)
            exponent = exponent + active.to(torch.int64)*shift
        return exponent

    def forward(self, denominator: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (denominator.is_floating_point() or denominator.dtype == torch.bool
                or denominator.numel() == 0 or (denominator <= 0).any()
                or (denominator > (1 << 25)).any()):
            raise ValueError('reciprocal requires integer denominator in [1, 2^25]')
        denominator = denominator.to(torch.int64)
        exponent = self.leading_one(denominator)
        mantissa = torch.where(exponent <= 16, denominator << (16-exponent).clamp_min(0),
                               denominator >> (exponent-16).clamp_min(0))
        fraction = mantissa - 65536
        if self.config.backend is ReciprocalBackend.PWL16:
            index, residual = fraction >> 12, fraction & 4095
            lower = self.table[index]
            slope = self.table[index+1]-lower
            code = lower + round_shift(slope*residual, 12)
        else:
            index_bits = 8 if self.config.backend is ReciprocalBackend.LUT256 else 6
            code = self.table[fraction >> (16-index_bits)]
            if self.config.backend is ReciprocalBackend.NEWTON64:
                product = round_shift(mantissa*code, 16)
                code = round_shift(code*((2 << 16)-product), 16)
        is_power = (denominator & (denominator-1)) == 0
        code = torch.where(is_power, torch.full_like(code, 65536), code).clamp(0, 65536)
        return code, exponent

    def probability_codes(self, weights: torch.Tensor) -> torch.Tensor:
        if (weights.is_floating_point() or weights.dtype == torch.bool or weights.ndim < 1
                or not 1 <= weights.shape[-1] <= self.config.max_tokens
                or (weights < 0).any() or (weights > 32768).any()):
            raise ValueError('weights require UQ1.15 and 1..1024 tokens')
        weights = weights.to(torch.int64)
        denominator = weights.sum(-1, keepdim=True)
        code, exponent = self(denominator)
        product = weights*code
        shift = exponent + 16 - self.config.probability_fraction_bits
        result = torch.where(shift >= 0, round_shift(product, shift.clamp_min(0)),
                             product << (-shift).clamp_min(0))
        return result.clamp(0, 1 << self.config.probability_fraction_bits)

    def contract(self) -> dict:
        return {'backend': self.config.backend.value, 'mantissa_format': 'unsigned-Q1.16',
                'exponent': 'integer-leading-one; reciprocal=code*2^(-16-e)',
                'denominator_bits': 26, 'denominator_range': [1, 1 << 25],
                'normalization_mantissa': 'Q1.16 truncated after integer normalization',
                'probability_format': 'unsigned-Q1.24', 'max_tokens': self.config.max_tokens,
                'table': self.table.tolist(), 'table_storage_bits': self.table.numel()*17,
                'slope_format': 'signed17/65536' if self.config.backend is ReciprocalBackend.PWL16 else None,
                'newton_steps': int(self.config.backend is ReciprocalBackend.NEWTON64),
                'intermediate_container': 'int64, full products (up to 34 bits)',
                'rounding': 'nearest-ties-positive-infinity', 'output_saturation': [0, 1 << 24],
                'power_of_two_denominator': 'exact mantissa 65536', 'final_renormalization': False}
