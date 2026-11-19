"""Deterministic two-task activation-output calibration for QAT setup."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .activation_adapter import AppliedActivationQuantization


@dataclass(frozen=True)
class QATActivationCalibrationReport:
    """Observer evidence captured before any optimizer is constructed."""

    samples: dict[str, int]
    quantizers: int
    valid_observers: int
    invalid_observers: tuple[str, ...]
    observed_minimum: float
    observed_maximum: float
    seconds: float
    peak_gpu_memory_mib: float

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": dict(self.samples),
            "quantizers": self.quantizers,
            "valid_observers": self.valid_observers,
            "invalid_observers": list(self.invalid_observers),
            "observed_minimum": self.observed_minimum,
            "observed_maximum": self.observed_maximum,
            "seconds": self.seconds,
            "peak_gpu_memory_mib": self.peak_gpu_memory_mib,
        }


def calibrate_activation_outputs(
    applied: AppliedActivationQuantization,
    *,
    samples: Mapping[str, Sequence[Tensor]],
    device: torch.device,
) -> QATActivationCalibrationReport:
    """Observe Detect and Pose exemplars, then atomically enable A8 fake quant."""

    if set(samples) != {"detect", "pose"}:
        raise ValueError("QAT calibration samples must contain detect and pose")
    if any(not values for values in samples.values()):
        raise ValueError("each QAT calibration task requires at least one sample")
    if applied.mode != "observe":
        raise RuntimeError(
            f"activation calibration must begin in observe mode: {applied.mode}"
        )
    model = applied.model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for task in ("detect", "pose"):
            for value in samples[task]:
                model(value.to(device, non_blocking=device.type == "cuda"), task=task)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ranges = applied.observer_ranges()
    invalid = tuple(
        path
        for path, observed in ranges.items()
        if observed is None or observed[1] <= observed[0]
    )
    if invalid:
        raise RuntimeError(
            "activation observers are missing or degenerate: " + ", ".join(invalid)
        )
    applied.freeze_observers()
    valid_ranges = tuple(value for value in ranges.values() if value is not None)
    return QATActivationCalibrationReport(
        samples={task: len(samples[task]) for task in ("detect", "pose")},
        quantizers=applied.quantizer_count,
        valid_observers=len(valid_ranges),
        invalid_observers=(),
        observed_minimum=min(value[0] for value in valid_ranges),
        observed_maximum=max(value[1] for value in valid_ranges),
        seconds=time.perf_counter() - started,
        peak_gpu_memory_mib=(
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else 0.0
        ),
    )


__all__ = ("QATActivationCalibrationReport", "calibrate_activation_outputs")
