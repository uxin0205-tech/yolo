"""Bit-True phase gates and deterministic winner selection."""

from __future__ import annotations

from dataclasses import dataclass

MAP_TOLERANCE = 0.001


@dataclass(frozen=True)
class Candidate:
    name: str
    map50_95: float
    fp16_p50_ms: float
    gflops: float
    parameters: int
    peak_vram_gib: float


def phase_gate(
    parent_name: str,
    parent_map: float,
    child_name: str,
    child_map: float,
    tolerance: float = MAP_TOLERANCE,
) -> str:
    """Accept a child unless its Bit-True mAP loses by more than the tolerance."""

    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    return parent_name if child_map < parent_map - tolerance else child_name


def phase_c_candidate(parent_name: str, parent_map: float, child_name: str, child_map: float) -> str:
    """Retain the Phase-C parent unless the Bit-True child strictly improves it."""

    return child_name if child_map > parent_map else parent_name

def choose_architecture(candidates: tuple[Candidate, ...]) -> Candidate:
    """Select by Bit-True mAP, then latency/GFLOPs/params/VRAM inside 0.001."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    best_map = max(candidate.map50_95 for candidate in candidates)
    tied = tuple(candidate for candidate in candidates if best_map - candidate.map50_95 <= MAP_TOLERANCE)
    return min(
        tied,
        key=lambda item: (
            item.fp16_p50_ms,
            item.gflops,
            item.parameters,
            item.peak_vram_gib,
            item.name,
        ),
    )
