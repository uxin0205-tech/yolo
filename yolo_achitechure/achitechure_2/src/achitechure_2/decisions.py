"""第一輪 Float 結果的描述性比較；不自動淘汰或選出 C_best。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ClassMetrics:
    """一個 BBAT5 class 的 box/keypoint AP 與偵測 F1。"""

    ap50: float
    ap50_95: float
    keypoint_ap50: float
    keypoint_ap50_95: float
    precision: float
    recall: float
    f1: float

    def __post_init__(self) -> None:
        _validate_rates(asdict(self), "class metrics")


@dataclass(frozen=True)
class CandidateMetrics:
    """正式報告要求的精度、F1 與成本欄位。"""

    candidate_id: str
    coco_box_map50: float
    coco_box_map50_95: float
    coco_person_ap50: float
    coco_person_ap50_95: float
    bbat5_pose_box_map50: float
    bbat5_pose_box_map50_95: float
    bbat5_keypoint_map50: float
    bbat5_keypoint_map50_95: float
    pose_official_combined_fitness: float
    classes: dict[str, ClassMetrics]
    macro_f1: float
    micro_f1: float
    f1_confidence_threshold: float
    params: int
    gflops: float
    latency_ms: float | None
    peak_vram_mb: float | None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id 不得為空")
        accuracy = {
            "coco_box_map50": self.coco_box_map50,
            "coco_box_map50_95": self.coco_box_map50_95,
            "coco_person_ap50": self.coco_person_ap50,
            "coco_person_ap50_95": self.coco_person_ap50_95,
            "bbat5_pose_box_map50": self.bbat5_pose_box_map50,
            "bbat5_pose_box_map50_95": self.bbat5_pose_box_map50_95,
            "bbat5_keypoint_map50": self.bbat5_keypoint_map50,
            "bbat5_keypoint_map50_95": self.bbat5_keypoint_map50_95,
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "f1_confidence_threshold": self.f1_confidence_threshold,
        }
        _validate_rates(accuracy, self.candidate_id)
        if not 0 <= self.pose_official_combined_fitness <= 2:
            raise ValueError("Pose official combined fitness 必須介於 [0,2]")
        if set(self.classes) != {"ball", "bat"}:
            raise ValueError("classes 必須正好包含 ball 與 bat")
        expected_macro = sum(item.f1 for item in self.classes.values()) / 2
        if abs(self.macro_f1 - expected_macro) > 1e-6:
            raise ValueError(
                f"Macro F1 必須是 ball/bat 未加權平均：{expected_macro} != {self.macro_f1}"
            )
        if self.params <= 0 or self.gflops <= 0:
            raise ValueError("Params 與 GFLOPs 必須為正數")
        for name, value in (
            ("latency_ms", self.latency_ms),
            ("peak_vram_mb", self.peak_vram_mb),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} 必須為正數或 null")


def _validate_rates(values: dict[str, float], label: str) -> None:
    invalid = {name: value for name, value in values.items() if not 0 <= value <= 1}
    if invalid:
        raise ValueError(f"{label} 的比例欄位必須介於 [0,1]：{invalid}")


ACCURACY_FIELDS = (
    "coco_box_map50",
    "coco_box_map50_95",
    "coco_person_ap50",
    "coco_person_ap50_95",
    "bbat5_pose_box_map50",
    "bbat5_pose_box_map50_95",
    "bbat5_keypoint_map50",
    "bbat5_keypoint_map50_95",
    "macro_f1",
    "micro_f1",
)

DESCRIPTIVE_BAND_FIELDS = (
    "coco_box_map50_95",
    "bbat5_pose_box_map50_95",
    "bbat5_keypoint_map50_95",
)


@dataclass(frozen=True)
class CandidateAssessment:
    metrics: CandidateMetrics
    metric_delta_vs_c0: dict[str, float]
    cost_reduction_vs_c0: dict[str, float | None]
    descriptive_bands: dict[str, str]
    decision: None
    quantization_eligible: bool | str


@dataclass(frozen=True)
class FloatEvaluationReport:
    candidates: tuple[CandidateAssessment, ...]
    f1_threshold_source: str
    f1_confidence_threshold: float
    pareto_front: tuple[str, ...]
    pareto_pending: tuple[str, ...]
    selection_status: str
    c_best: None
    quantization_eligibility: dict[str, bool | str]
    notes_zh: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _band(drop: float) -> str:
    epsilon = 1e-12
    if drop <= epsilon:
        return "no_drop_or_gain"
    if drop <= 0.005 + epsilon:
        return "drop_le_0.005"
    if drop <= 0.008 + epsilon:
        return "drop_0.005_to_0.008"
    return "drop_gt_0.008"


def _reduction(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return 1 - float(candidate) / float(baseline)


def _assessment(candidate: CandidateMetrics, c0: CandidateMetrics) -> CandidateAssessment:
    deltas = {
        field: float(getattr(candidate, field)) - float(getattr(c0, field))
        for field in ACCURACY_FIELDS
    }
    bands = {
        field: _band(float(getattr(c0, field)) - float(getattr(candidate, field)))
        for field in DESCRIPTIVE_BAND_FIELDS
    }
    quantization: bool | str = (
        True if candidate.candidate_id == "C0" else "pending_user_decision"
    )
    return CandidateAssessment(
        metrics=candidate,
        metric_delta_vs_c0=deltas,
        cost_reduction_vs_c0={
            "params": _reduction(candidate.params, c0.params),
            "gflops": _reduction(candidate.gflops, c0.gflops),
            "latency_ms": _reduction(candidate.latency_ms, c0.latency_ms),
            "peak_vram_mb": _reduction(candidate.peak_vram_mb, c0.peak_vram_mb),
        },
        descriptive_bands=bands,
        decision=None,
        quantization_eligible=quantization,
    )


def _dominates(left: CandidateMetrics, right: CandidateMetrics) -> bool:
    accuracy_fields = (
        "coco_box_map50_95",
        "bbat5_pose_box_map50_95",
        "bbat5_keypoint_map50_95",
        "macro_f1",
    )
    cost_fields = ("params", "gflops", "latency_ms")
    accuracy_pairs = [
        (float(getattr(left, field)), float(getattr(right, field)))
        for field in accuracy_fields
    ]
    cost_pairs = [
        (float(getattr(left, field)), float(getattr(right, field)))
        for field in cost_fields
    ]
    no_worse = all(first >= second for first, second in accuracy_pairs) and all(
        first <= second for first, second in cost_pairs
    )
    strictly_better = any(first > second for first, second in accuracy_pairs) or any(
        first < second for first, second in cost_pairs
    )
    return no_worse and strictly_better


def evaluate_float_results(
    metrics: Iterable[CandidateMetrics],
    *,
    c0_id: str = "C0",
) -> FloatEvaluationReport:
    """產生完整比較與 Pareto 描述，永遠把 C_best 留給使用者。"""

    values = tuple(metrics)
    if not values:
        raise ValueError("至少需要一個 CandidateMetrics")
    ids = tuple(item.candidate_id for item in values)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id 不得重複")
    try:
        c0 = next(item for item in values if item.candidate_id == c0_id)
    except StopIteration as error:
        raise ValueError(f"結果缺少 {c0_id} reference") from error
    thresholds = {round(item.f1_confidence_threshold, 12) for item in values}
    if len(thresholds) != 1:
        raise ValueError("所有候選必須共用 C0-Control search-val 決定的 F1 threshold")

    complete = tuple(item for item in values if item.latency_ms is not None)
    pending = tuple(item.candidate_id for item in values if item.latency_ms is None)
    pareto = tuple(
        item.candidate_id
        for item in complete
        if not any(
            other.candidate_id != item.candidate_id and _dominates(other, item)
            for other in complete
        )
    )
    eligibility = {
        item.candidate_id: (
            True if item.candidate_id == c0_id else "pending_user_decision"
        )
        for item in values
    }
    return FloatEvaluationReport(
        candidates=tuple(_assessment(item, c0) for item in values),
        f1_threshold_source="c0_control_search_val",
        f1_confidence_threshold=c0.f1_confidence_threshold,
        pareto_front=pareto,
        pareto_pending=pending,
        selection_status="pending_user_decision",
        c_best=None,
        quantization_eligibility=eligibility,
        notes_zh=(
            "0.005/0.008 只作描述性敏感度，不是 PASS/REJECT gate。",
            "Pareto 不會自動選出 C_best。",
            "C1/C2/C3 是否量化、是否新增 C3-P5/R1/組合都等待使用者決定。",
        ),
    )
