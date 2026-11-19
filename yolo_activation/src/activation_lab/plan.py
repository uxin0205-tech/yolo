"""Small, explicit validation plan for the first model-bearing delivery."""

from __future__ import annotations

from dataclasses import dataclass

from .activations import ActivationName


@dataclass(frozen=True)
class ValidationPlan:
    accuracy_reference: ActivationName
    hardware_neighbor_baseline: ActivationName
    cheap_primitive_control: ActivationName
    proposed_candidates: tuple[ActivationName, ...]
    recovery_queue: tuple[ActivationName, ...]
    notes: tuple[str, ...]


def default_validation_plan() -> ValidationPlan:
    """Return the narrow Phase 0-to-Phase 1 promotion plan.

    ReLU stays outside the expensive recovery queue unless its cheap smoke test
    reveals information not already supplied by Hardswish.
    """

    return ValidationPlan(
        accuracy_reference="silu",
        hardware_neighbor_baseline="hardswish",
        cheap_primitive_control="relu",
        proposed_candidates=("qsilu_pq", "poly_shift", "poly_quality"),
        recovery_queue=(
            "hardswish",
            "qsilu_pq",
            "poly_shift",
            "poly_quality",
        ),
        notes=(
            "SiLU 只作原始精度與收斂參考。",
            "Hardswish 是唯一需要等成本 recovery 的既有硬體鄰近 baseline。",
            "ReLU 預設只跑 zero-shot、export 與 primitive cost floor。",
            "qsilu_pq 以單一平方作低 DSP 候選；poly_shift 保留無分段 exact-tail 對照。",
        ),
    )
