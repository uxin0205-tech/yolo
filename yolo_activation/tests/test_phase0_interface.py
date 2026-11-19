from __future__ import annotations

import pytest
import torch

from activation_lab import (
    ValidationConfig,
    available_activations,
    build_activation,
    default_validation_plan,
    validate_activation,
)


def test_default_plan_keeps_only_two_existing_baseline_roles() -> None:
    plan = default_validation_plan()
    assert plan.accuracy_reference == "silu"
    assert plan.hardware_neighbor_baseline == "hardswish"
    assert plan.cheap_primitive_control == "relu"
    assert "relu" not in plan.recovery_queue
    assert plan.proposed_candidates == (
        "qsilu_pq",
        "poly_shift",
        "poly_quality",
    )


@pytest.mark.parametrize("name", available_activations())
def test_registry_builds_fresh_parameter_free_modules(name: str) -> None:
    first = build_activation(name)
    second = build_activation(name)
    assert first is not second
    assert list(first.parameters()) == []
    assert first.state_dict() == {}


@pytest.mark.parametrize("name", ["qsilu_pq", "poly_shift", "poly_quality"])
def test_proposed_float_reference_has_identity_and_exact_tails(name: str) -> None:
    report = validate_activation(name, ValidationConfig(samples=4_097))
    assert report.mathematics is not None
    assert report.mathematics.difference_identity_max_error < 1e-12
    assert report.mathematics.exact_relu_tail_max_error < 1e-12
    assert report.mathematics.zero_value_abs_error == 0.0
    assert report.curve.derivative_minimum > -0.2
    assert report.curve.derivative_maximum < 1.2


def test_qsilu_pq_improves_curve_error_with_one_shared_square() -> None:
    report = validate_activation("qsilu_pq", ValidationConfig(samples=4_097))
    assert report.curve.mse_vs_silu < 3e-5
    assert report.curve.max_abs_error_vs_silu < 0.011
    assert report.hardware_proxy.variable_multiplications == 1
    assert report.fixed_point is not None
    assert report.fixed_point.difference_identity_max_error_lsb == 0
    assert report.fixed_point.exact_relu_tail_max_error_lsb == 0
    assert report.fixed_point.max_abs_error_lsb <= 2.0


def test_qsilu_pq_is_c1_at_internal_knots_and_tail() -> None:
    module = build_activation("qsilu_pq")
    epsilon = 1e-7
    for knot in (1.0, 2.0, 4.0, 8.0):
        probe = torch.tensor(
            [knot - epsilon, knot + epsilon],
            dtype=torch.float64,
            requires_grad=True,
        )
        output = module(probe)
        gradient = torch.autograd.grad(output.sum(), probe)[0]
        assert float((output[1] - output[0]).abs().detach()) < 3e-7
        assert float((gradient[1] - gradient[0]).abs()) < 3e-7


def test_shift_profile_fixed_point_contract_is_bit_exact_where_required() -> None:
    report = validate_activation("poly_shift", ValidationConfig(samples=4_097))
    assert report.fixed_point is not None
    assert report.hardware_proxy.power_of_two_threshold
    assert report.hardware_proxy.dyadic_or_apot_constants
    assert report.hardware_proxy.variable_multiplications == 4
    assert report.fixed_point.difference_identity_max_error_lsb == 0
    assert report.fixed_point.exact_relu_tail_max_error_lsb == 0
    assert report.fixed_point.max_abs_error_lsb <= 4.0


@pytest.mark.parametrize("name", ["qsilu_pq", "poly_shift"])
def test_shift_friendly_onnx_graph_excludes_exp_sigmoid_and_div(
    tmp_path, name: str
) -> None:
    pytest.importorskip("onnx")
    report = validate_activation(
        name,
        ValidationConfig(samples=1_025),
        onnx_path=tmp_path / f"{name}.onnx",
    )
    assert report.onnx is not None
    assert report.onnx.forbidden_activation_ops == ()
    assert {"Abs", "Mul", "Add"}.issubset(report.onnx.node_counts)
    if name == "qsilu_pq":
        assert {"Less", "Where"}.issubset(report.onnx.node_counts)


def test_unknown_activation_fails_with_registry_choices() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        build_activation("unknown")


def test_autograd_remains_finite_around_both_boundaries() -> None:
    x = torch.tensor([-8.001, -8.0, -7.999, 0.0, 7.999, 8.0, 8.001], requires_grad=True)
    y = build_activation("poly_shift")(x)
    gradient = torch.autograd.grad(y.sum(), x)[0]
    assert torch.isfinite(y).all()
    assert torch.isfinite(gradient).all()
