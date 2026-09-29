from __future__ import annotations

import torch
from torch import nn

from yolo_combine.standalone_baseline import copy_float_state_to_bittrue


class FloatNormalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("knots", torch.tensor([0.0, 1.0]))
        self.register_buffer("values", torch.tensor([0.25, 0.75]))


class BitTrueNormalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("endpoint_table", torch.tensor([11, 22]))


class TinyRepresentation(nn.Module):
    def __init__(self, *, bittrue: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.attn = nn.Module()
        self.attn.normalize = BitTrueNormalize() if bittrue else FloatNormalize()


def test_standalone_bittrue_conversion_copies_common_state_and_preserves_table() -> None:
    source = TinyRepresentation(bittrue=False)
    target = TinyRepresentation(bittrue=True)
    with torch.no_grad():
        source.weight.copy_(torch.tensor([7.0, 9.0]))
    original_table = target.attn.normalize.endpoint_table.clone()

    report = copy_float_state_to_bittrue(source, target)

    assert report.complete
    assert report.copied_tensors == 1
    assert report.preserved_bittrue_tensors == 1
    assert report.ignored_float_tensors == 2
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.attn.normalize.endpoint_table, original_table)

