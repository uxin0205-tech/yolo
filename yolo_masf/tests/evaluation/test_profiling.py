from __future__ import annotations

import pytest
import torch
from torch import nn

from masf_yolo.evaluation.profiling import profile_gpu_latency, profile_module


def test_profile_counts_macs_params_operators_and_traffic() -> None:
    module = nn.Sequential(
        nn.Conv2d(4, 4, 3, padding=1, groups=4),
        nn.Conv2d(4, 8, 1),
    )

    result = profile_module(module, torch.zeros(1, 4, 8, 8))

    assert result.params == 80
    assert result.macs == 4352
    assert result.gflops == pytest.approx(2 * 4352 / 1e9)
    assert result.depthwise_conv_count == 1
    assert result.pointwise_conv_count == 1
    assert result.peak_live_activation_bytes == 2048
    assert result.feature_traffic_bytes == 5120


def test_gpu_latency_policy_fails_closed_without_cuda_or_fp16() -> None:
    module = nn.Identity()

    with pytest.raises(ValueError, match="CUDA device and fp16"):
        profile_gpu_latency(
            module,
            device=torch.device("cpu"),
            imgsz=640,
            precision="fp16",
            batch=1,
            warmup=100,
            iterations=1000,
        )
    with pytest.raises(ValueError, match="CUDA device and fp16"):
        profile_gpu_latency(
            module,
            device=torch.device("cuda:0"),
            imgsz=640,
            precision="fp32",
            batch=1,
            warmup=100,
            iterations=1000,
        )
