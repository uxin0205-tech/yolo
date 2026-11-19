from __future__ import annotations

import pytest
import torch
from torch import nn

from yolo_quantize.qat_optimizer import split_quantizer_parameter_groups


class _ToyOptimizerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2))
        self.weight_quantizer = nn.Module()
        self.weight_quantizer.scale = nn.Parameter(torch.ones(2))
        self.output_quantizer = nn.Module()
        self.output_quantizer.offset = nn.Parameter(torch.zeros(()))


def _optimizer(model: nn.Module) -> torch.optim.AdamW:
    named = tuple(model.named_parameters())
    return torch.optim.AdamW(
        [
            {
                "params": [parameter for _, parameter in named],
                "param_names": tuple(name for name, _ in named),
                "group_name": "backbone.no_decay",
                "role": "backbone",
                "lr": 1e-3,
                "weight_decay": 0.0,
            }
        ]
    )


def test_split_quantizer_parameter_groups_applies_explicit_lr_ratio() -> None:
    model = _ToyOptimizerModel()
    optimizer = _optimizer(model)

    report = split_quantizer_parameter_groups(optimizer, qparam_lr_ratio=2.0)

    groups = {str(group["group_name"]): group for group in optimizer.param_groups}
    assert set(groups) == {"backbone.no_decay", "backbone.no_decay.qparam"}
    assert groups["backbone.no_decay"]["param_names"] == ("bias",)
    assert groups["backbone.no_decay.qparam"]["param_names"] == (
        "weight_quantizer.scale",
        "output_quantizer.offset",
    )
    assert groups["backbone.no_decay"]["lr"] == pytest.approx(1e-3)
    assert groups["backbone.no_decay.qparam"]["lr"] == pytest.approx(2e-3)
    assert groups["backbone.no_decay.qparam"]["weight_decay"] == 0.0
    assert report.model_parameters == 1
    assert report.quantizer_parameters == 2


def test_split_quantizer_group_never_uses_muon_matrix_update() -> None:
    model = _ToyOptimizerModel()
    optimizer = _optimizer(model)
    optimizer.param_groups[0]["use_muon"] = True

    split_quantizer_parameter_groups(optimizer, qparam_lr_ratio=1.0)

    quantizer = next(
        group
        for group in optimizer.param_groups
        if str(group["group_name"]).endswith(".qparam")
    )
    assert quantizer["use_muon"] is False


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("inf"), float("nan")])
def test_split_quantizer_parameter_groups_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        split_quantizer_parameter_groups(
            _optimizer(_ToyOptimizerModel()),
            qparam_lr_ratio=ratio,
        )
