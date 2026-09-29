from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from yolo_quantize import ActivationOutputAdapter, ActivationOutputPolicy


class _ToyActivationGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.SiLU()
        self.second = nn.Hardswish()
        self.protected = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.protected(self.second(self.first(value)))


def test_adapter_observes_exact_float_then_freezes_all_sites_for_fake_quant() -> None:
    original = _ToyActivationGraph().eval()
    policy = ActivationOutputPolicy(
        policy_id="toy--lsq-plus-a3",
        bits=3,
    )

    applied = ActivationOutputAdapter().apply(
        original,
        module_paths=("first", "second"),
        policy=policy,
    )
    values = torch.linspace(-4.0, 4.0, 257)

    observed = applied.model(values)

    assert torch.equal(observed, original(values))
    assert applied.wrapped_paths == ("first", "second")
    assert applied.quantizer_count == 2
    assert set(applied.observer_ranges()) == {"first", "second"}

    applied.disable_quantization()
    assert applied.mode == "disabled"
    assert torch.equal(applied.model(values), original(values))

    applied.freeze_observers()
    quantized = applied.model(values)

    assert torch.isfinite(quantized).all()
    assert not torch.equal(quantized, observed)
    assert applied.mode == "fake_quant"
    assert isinstance(applied.model.protected, nn.Identity)


def test_adapter_preflight_fails_before_in_place_mutation() -> None:
    original = _ToyActivationGraph().eval()
    values = torch.linspace(-2.0, 2.0, 33)
    expected = original(values)

    with pytest.raises(ValueError, match="activation-output preflight failed"):
        ActivationOutputAdapter().apply(
            original,
            module_paths=("first", "missing"),
            policy=ActivationOutputPolicy("toy--lsq-plus-a8", bits=8),
            clone_model=False,
        )

    assert isinstance(original.first, nn.SiLU)
    assert torch.equal(original(values), expected)


def test_adapter_state_dict_roundtrip_restores_ranges_scales_and_mode() -> None:
    values = torch.linspace(-4.0, 4.0, 257)
    policy = ActivationOutputPolicy("toy--lsq-plus-a6", bits=6)
    first = ActivationOutputAdapter().apply(
        _ToyActivationGraph().eval(),
        module_paths=("first", "second"),
        policy=policy,
    )
    first.model(values)
    first.freeze_observers()
    expected = first.model(values)

    resumed = ActivationOutputAdapter().apply(
        _ToyActivationGraph().eval(),
        module_paths=("first", "second"),
        policy=policy,
    )
    resumed.model.load_state_dict(first.model.state_dict(), strict=True)

    assert resumed.mode == "fake_quant"
    assert resumed.observer_ranges() == first.observer_ranges()
    assert torch.equal(resumed.model(values), expected)


def test_applied_quantization_rebinds_only_to_an_equivalent_wrapped_graph() -> None:
    policy = ActivationOutputPolicy("toy--lsq-plus-a8", bits=8)
    applied = ActivationOutputAdapter().apply(
        _ToyActivationGraph().eval(),
        module_paths=("first", "second"),
        policy=policy,
    )
    deployment_clone = copy.deepcopy(applied.model)

    rebound = applied.rebind(deployment_clone)

    assert rebound.model is deployment_clone
    assert rebound.policy == policy
    assert rebound.wrapped_paths == applied.wrapped_paths
    assert rebound.quantizer_count == 2

    deployment_clone.first.output_quantizer.spec = type(
        deployment_clone.first.output_quantizer.spec
    )(bits=7)
    with pytest.raises(RuntimeError, match="policy differs"):
        applied.rebind(deployment_clone)
