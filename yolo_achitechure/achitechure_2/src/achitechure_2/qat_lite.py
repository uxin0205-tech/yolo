"""QAT-lite 的資格、固定短預算與描述性 gap 報告。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .quantization import require_quantization_stage

QAT_LITE_OPTIMIZER_STEPS = 200
QAT_LITE_OBSERVER_UPDATE_STEPS = 50
QAT_LITE_VALIDATION_INTERVAL = 50


def require_qat_lite_stage(
    candidate_id: str,
    *,
    user_approved: Iterable[str] = (),
    gpu_authorized: bool = False,
) -> str:
    """沿用 Q1 eligibility，並額外要求 QAT-lite GPU 執行授權。"""

    require_quantization_stage(
        candidate_id,
        "Q1",
        user_approved=user_approved,
    )
    if not gpu_authorized:
        raise PermissionError("QAT-lite 需要使用者明確 GPU 訓練授權")
    return "allowed"


@dataclass(frozen=True)
class QATLiteGapReport:
    simulation_only: bool
    q0: dict[str, float]
    q1: dict[str, float]
    q2_lite: dict[str, float]
    ptq_drop: dict[str, float]
    qat_lite_drop: dict[str, float]
    qat_lite_recovery: dict[str, float]
    selection_status: str
    accepted: None


def qat_lite_gap_report(
    *,
    q0: Mapping[str, float],
    q1: Mapping[str, float],
    q2_lite: Mapping[str, float],
) -> QATLiteGapReport:
    """描述短 QAT 是否恢復 PTQ gap；不自動核准候選或正式量化。"""

    if not q0 or set(q0) != set(q1) or set(q0) != set(q2_lite):
        raise ValueError("Q0/Q1/Q2L 必須提供相同且非空的 metric keys")
    normalized = {
        stage: {name: float(value) for name, value in values.items()}
        for stage, values in (("q0", q0), ("q1", q1), ("q2_lite", q2_lite))
    }
    invalid = {
        f"{stage}.{name}": value
        for stage, values in normalized.items()
        for name, value in values.items()
        if not 0 <= value <= 1
    }
    if invalid:
        raise ValueError(f"量化精度指標必須介於 [0,1]：{invalid}")
    return QATLiteGapReport(
        simulation_only=True,
        q0=normalized["q0"],
        q1=normalized["q1"],
        q2_lite=normalized["q2_lite"],
        ptq_drop={
            name: normalized["q0"][name] - normalized["q1"][name]
            for name in normalized["q0"]
        },
        qat_lite_drop={
            name: normalized["q0"][name] - normalized["q2_lite"][name]
            for name in normalized["q0"]
        },
        qat_lite_recovery={
            name: normalized["q2_lite"][name] - normalized["q1"][name]
            for name in normalized["q0"]
        },
        selection_status="screening_only_pending_user_decision",
        accepted=None,
    )
