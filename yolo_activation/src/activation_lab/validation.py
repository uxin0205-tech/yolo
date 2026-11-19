"""One deep validation entry point for curve, integer, and ONNX checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .activations import ActivationName, build_activation, get_profile
from .hardware import (
    FixedPointConfig,
    HardwareProxy,
    emulate_poly_shift_int,
    emulate_qsilu_pq_int,
    hardware_proxy,
)


@dataclass(frozen=True)
class ValidationConfig:
    grid_min: float = -12.0
    grid_max: float = 12.0
    samples: int = 24_001
    fixed_total_bits: int = 16
    fixed_fraction_bits: int = 10

    def __post_init__(self) -> None:
        if self.grid_min >= self.grid_max:
            raise ValueError("grid_min must be smaller than grid_max")
        if self.samples < 33 or self.samples % 2 == 0:
            raise ValueError(
                "samples must be an odd integer >= 33 so the grid includes zero"
            )


@dataclass(frozen=True)
class CurveMetrics:
    mse_vs_silu: float
    mae_vs_silu: float
    max_abs_error_vs_silu: float
    minimum_output: float
    derivative_minimum: float
    derivative_maximum: float


@dataclass(frozen=True)
class MathematicalMetrics:
    difference_identity_max_error: float
    exact_relu_tail_max_error: float
    zero_value_abs_error: float


@dataclass(frozen=True)
class FixedPointMetrics:
    total_bits: int
    fraction_bits: int
    mse_lsb: float
    max_abs_error_lsb: float
    difference_identity_max_error_lsb: int
    exact_relu_tail_max_error_lsb: int
    saturated_output_count: int


@dataclass(frozen=True)
class OnnxMetrics:
    path: str
    opset: int
    node_counts: dict[str, int]
    forbidden_activation_ops: tuple[str, ...]


@dataclass(frozen=True)
class ValidationReport:
    activation: ActivationName
    curve: CurveMetrics
    hardware_proxy: HardwareProxy
    mathematics: MathematicalMetrics | None
    fixed_point: FixedPointMetrics | None
    onnx: OnnxMetrics | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _curve_metrics(
    name: ActivationName, config: ValidationConfig
) -> tuple[CurveMetrics, MathematicalMetrics | None, torch.Tensor]:
    x = torch.linspace(
        config.grid_min,
        config.grid_max,
        config.samples,
        dtype=torch.float64,
        requires_grad=True,
    )
    module = build_activation(name).to(dtype=torch.float64)
    y = module(x)
    target = F.silu(x.detach())
    error = y.detach() - target
    derivative = torch.autograd.grad(y.sum(), x)[0].detach()
    curve = CurveMetrics(
        mse_vs_silu=float(error.square().mean()),
        mae_vs_silu=float(error.abs().mean()),
        max_abs_error_vs_silu=float(error.abs().max()),
        minimum_output=float(y.detach().min()),
        derivative_minimum=float(derivative.min()),
        derivative_maximum=float(derivative.max()),
    )

    profile = get_profile(name)
    mathematics = None
    if profile is not None:
        detached_x = x.detach()
        with torch.no_grad():
            symmetry_error = module(detached_x) - module(-detached_x) - detached_x
            tail_probe = torch.tensor(
                [
                    -2.0 * profile.threshold,
                    -profile.threshold,
                    profile.threshold,
                    2.0 * profile.threshold,
                ],
                dtype=torch.float64,
            )
            tail_error = module(tail_probe) - torch.relu(tail_probe)
            zero = torch.zeros(1, dtype=torch.float64)
            mathematics = MathematicalMetrics(
                difference_identity_max_error=float(symmetry_error.abs().max()),
                exact_relu_tail_max_error=float(tail_error.abs().max()),
                zero_value_abs_error=float(module(zero).abs().max()),
            )
    return curve, mathematics, x.detach()


def _fixed_point_metrics(
    name: ActivationName, x: torch.Tensor, config: ValidationConfig
) -> FixedPointMetrics | None:
    if name not in {"poly_shift", "qsilu_pq"}:
        return None
    fixed_config = FixedPointConfig(config.fixed_total_bits, config.fixed_fraction_bits)
    scale = fixed_config.scale
    x_q = (
        torch.round(x * scale)
        .to(torch.int64)
        .clamp(fixed_config.qmin, fixed_config.qmax)
    )
    quantized_x = x_q.to(torch.float64) / scale
    reference = build_activation(name).to(dtype=torch.float64)(quantized_x)
    emulator = emulate_qsilu_pq_int if name == "qsilu_pq" else emulate_poly_shift_int
    output_q = emulator(x_q, fixed_config)
    output = output_q.to(torch.float64) / scale
    error_lsb = (output - reference) * scale

    identity_mask = x_q != fixed_config.qmin
    mirrored_q = emulator(-x_q[identity_mask], fixed_config)
    identity_error = output_q[identity_mask] - mirrored_q - x_q[identity_mask]
    tail_probe_q = torch.tensor(
        [-(8 * scale + 7), -8 * scale, 8 * scale, 8 * scale + 7],
        dtype=torch.int64,
    )
    tail_output_q = emulator(tail_probe_q, fixed_config)
    tail_error = tail_output_q - torch.relu(tail_probe_q)
    saturation_count = int(
        ((output_q == fixed_config.qmin) | (output_q == fixed_config.qmax)).sum()
    )
    return FixedPointMetrics(
        total_bits=fixed_config.total_bits,
        fraction_bits=fixed_config.fraction_bits,
        mse_lsb=float(error_lsb.square().mean()),
        max_abs_error_lsb=float(error_lsb.abs().max()),
        difference_identity_max_error_lsb=int(identity_error.abs().max()),
        exact_relu_tail_max_error_lsb=int(tail_error.abs().max()),
        saturated_output_count=saturation_count,
    )


def _onnx_metrics(name: ActivationName, path: Path) -> OnnxMetrics:
    try:
        import onnx
    except (
        ImportError
    ) as error:  # pragma: no cover - exercised only in minimal environments
        raise RuntimeError(
            "ONNX graph validation requires the optional 'onnx' package"
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    module = build_activation(name).eval()
    sample = torch.linspace(-12.0, 12.0, 3 * 8 * 8, dtype=torch.float32).reshape(
        1, 3, 8, 8
    )
    torch.onnx.export(
        module,
        (sample,),
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    model = onnx.load(str(path))
    counts = Counter(node.op_type for node in model.graph.node)
    forbidden = tuple(sorted(set(counts) & {"Div", "Erf", "Exp", "Sigmoid", "Tanh"}))
    return OnnxMetrics(
        path=str(path),
        opset=18,
        node_counts=dict(sorted(counts.items())),
        forbidden_activation_ops=forbidden,
    )


def validate_activation(
    name: ActivationName,
    config: ValidationConfig | None = None,
    *,
    onnx_path: str | Path | None = None,
) -> ValidationReport:
    """Validate one activation without touching model checkpoints or datasets."""

    actual_config = config or ValidationConfig()
    curve, mathematics, x = _curve_metrics(name, actual_config)
    fixed_point = _fixed_point_metrics(name, x, actual_config)
    onnx = _onnx_metrics(name, Path(onnx_path)) if onnx_path is not None else None
    return ValidationReport(
        activation=name,
        curve=curve,
        hardware_proxy=hardware_proxy(name),
        mathematics=mathematics,
        fixed_point=fixed_point,
        onnx=onnx,
    )
