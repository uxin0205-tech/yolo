"""Immutable validation-only BEST_PARTIAL selection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from masf_yolo.artifacts.io import atomic_write_json
from masf_yolo.contracts import canonical_json, sha256_value


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    variant_id: str
    map50_95: float | None
    ap_s: float | None
    ball_recall: float | None
    ball_ap_s: float | None
    tiny_recall: float | None
    blur_recall: float | None
    gflops: float
    params: float
    peak_activation: float
    traffic: float


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected: str
    reason: str
    ranking: tuple[str, str]
    candidate_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "reason": self.reason,
            "ranking": list(self.ranking),
            "candidate_hashes": self.candidate_hashes,
        }


def _quality(value: float | None) -> float:
    return value if value is not None and math.isfinite(value) else -math.inf


def _ranking_key(candidate: CandidateMetrics) -> tuple[float, ...]:
    return (
        _quality(candidate.map50_95),
        _quality(candidate.ap_s),
        _quality(candidate.ball_recall),
        _quality(candidate.ball_ap_s),
        _quality(candidate.tiny_recall),
        _quality(candidate.blur_recall),
        -candidate.gflops,
        -candidate.params,
        -candidate.peak_activation,
        -candidate.traffic,
    )


def select_best_partial(m2: CandidateMetrics, m3: CandidateMetrics) -> SelectionResult:
    if (m2.variant_id, m3.variant_id) != ("M2", "M3"):
        raise ValueError("BEST_PARTIAL selection requires M2 and M3 in order")
    if m2.map50_95 is None or m3.map50_95 is None or m2.ball_recall is None or m3.ball_recall is None:
        raise ValueError("M2 and M3 require mAP50-95 and Ball Recall")
    map_difference = abs(m2.map50_95 - m3.map50_95)
    recall_difference = abs(m2.ball_recall - m3.ball_recall)
    ranked = tuple(
        candidate.variant_id
        for candidate in sorted((m2, m3), key=_ranking_key, reverse=True)
    )
    if map_difference <= 0.002 + 1e-12 and recall_difference < 0.01:
        selected = "M3"
        reason = "efficiency_equivalent"
    else:
        selected = ranked[0]
        reason = "quality_hardware_ranking"
    return SelectionResult(
        selected=selected,
        reason=reason,
        ranking=ranked,
        candidate_hashes={
            m2.variant_id: sha256_value(asdict(m2)),
            m3.variant_id: sha256_value(asdict(m3)),
        },
    )


def freeze_selection(
    path: Path,
    result: SelectionResult,
    *,
    val_hashes: dict[str, str],
) -> None:
    if set(val_hashes) != {"M2", "M3"}:
        raise ValueError("selection requires M2 and M3 validation hashes")
    payload = result.to_dict() | {"val_hashes": dict(sorted(val_hashes.items()))}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError("selection is immutable once frozen")
        return
    atomic_write_json(path, payload)


def require_selection_before_test(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("test evaluation requires frozen selection.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selected") not in {"M2", "M3"} or set(payload.get("val_hashes", {})) != {"M2", "M3"}:
        raise RuntimeError("selection.json is incomplete")
    return payload
