from __future__ import annotations

import torch
from torch import nn

from yolo_attention.optimizer_scope import restrict_optimizer_to_trainable


def test_optimizer_contains_only_trainable_parameter_identities() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    restrict_optimizer_to_trainable(optimizer, model)
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert actual == expected
    assert not any(
        parameter is model[0].weight for group in optimizer.param_groups for parameter in group["params"]
    )
