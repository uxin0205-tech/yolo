"""Deterministic convolution/activation/traffic profiling."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    params: int
    macs: int
    gflops: float
    p2_activation_bytes: int | None
    peak_live_activation_bytes: int
    depthwise_conv_count: int
    pointwise_conv_count: int
    feature_traffic_bytes: int


@dataclass(frozen=True, slots=True)
class GpuLatencyProfile:
    precision: str
    batch: int
    warmup: int
    iterations: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


def profile_gpu_latency(
    module: nn.Module,
    *,
    device: torch.device,
    imgsz: int,
    precision: str,
    batch: int,
    warmup: int,
    iterations: int,
) -> GpuLatencyProfile:
    """Measure synchronized end-to-end model forward latency under a fixed policy."""
    if device.type != "cuda" or precision != "fp16":
        raise ValueError("GPU latency profiling requires a CUDA device and fp16")
    if batch != 1 or warmup < 1 or iterations < 1:
        raise ValueError("invalid fixed latency policy")
    module = module.eval().half().to(device)
    sample = torch.zeros(batch, 3, imgsz, imgsz, device=device, dtype=torch.float16)
    timings: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            module(sample)
        torch.cuda.synchronize(device)
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            module(sample)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    ordered = sorted(timings)
    return GpuLatencyProfile(
        precision=precision,
        batch=batch,
        warmup=warmup,
        iterations=iterations,
        mean_ms=sum(timings) / len(timings),
        p50_ms=ordered[math.ceil(0.50 * len(ordered)) - 1],
        p95_ms=ordered[math.ceil(0.95 * len(ordered)) - 1],
    )


def _bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def profile_module(module: nn.Module, sample: Tensor) -> HardwareProfile:
    macs = 0
    peak_activation = 0
    traffic = 0
    depthwise = 0
    pointwise = 0
    p2_activation: int | None = None
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def convolution_hook(conv: nn.Conv2d, args: tuple[object, ...], output: Tensor) -> None:
        nonlocal macs, peak_activation, traffic, depthwise, pointwise
        input_tensor = args[0]
        if not isinstance(input_tensor, Tensor):
            raise TypeError("Conv2d input must be a tensor")
        kernel_macs = (conv.in_channels // conv.groups) * conv.kernel_size[0] * conv.kernel_size[1]
        macs += output.numel() * kernel_macs
        output_bytes = _bytes(output)
        peak_activation = max(peak_activation, output_bytes)
        traffic += _bytes(input_tensor) + output_bytes
        if conv.groups == conv.in_channels == conv.out_channels:
            depthwise += 1
        if conv.kernel_size == (1, 1):
            pointwise += 1

    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            handles.append(child.register_forward_hook(convolution_hook))

    model_layers = getattr(module, "model", None)
    if model_layers is not None and len(model_layers):
        def detect_pre_hook(_detect: nn.Module, args: tuple[object, ...]) -> None:
            nonlocal p2_activation
            features = args[0]
            if isinstance(features, list) and features and isinstance(features[0], Tensor):
                p2_activation = _bytes(features[0])

        handles.append(model_layers[-1].register_forward_pre_hook(detect_pre_hook))

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            module(sample)
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)
    return HardwareProfile(
        params=sum(parameter.numel() for parameter in module.parameters()),
        macs=macs,
        gflops=2 * macs / 1e9,
        p2_activation_bytes=p2_activation,
        peak_live_activation_bytes=peak_activation,
        depthwise_conv_count=depthwise,
        pointwise_conv_count=pointwise,
        feature_traffic_bytes=traffic,
    )
