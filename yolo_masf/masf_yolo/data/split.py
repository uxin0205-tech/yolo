"""Deterministic multi-objective 80/10/10 group assignment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping


SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass(frozen=True, slots=True)
class GroupStats:
    group_id: str
    unique_frames: int
    ball_instances: int
    bat_instances: int
    ball_bins: tuple[int, int, int]
    content_hashes: tuple[str, ...]

    @property
    def vector(self) -> tuple[int, ...]:
        return (
            self.unique_frames,
            self.ball_instances,
            self.bat_instances,
            *self.ball_bins,
        )


@dataclass(frozen=True, slots=True)
class SplitReport:
    ok: bool
    frame_counts: dict[str, int]
    ball_counts: dict[str, int]
    bat_counts: dict[str, int]
    group_overlap: tuple[str, ...]
    hash_overlap: tuple[str, ...]


def _tie_key(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def assign_groups(groups: list[GroupStats], seed: int = 42) -> dict[str, str]:
    if not groups:
        raise ValueError("cannot split an empty dataset")
    if len({group.group_id for group in groups}) != len(groups):
        raise ValueError("group IDs must be unique")
    totals = tuple(sum(group.vector[index] for group in groups) for index in range(6))
    current = {split: [0.0] * 6 for split in SPLITS}
    assignment: dict[str, str] = {}
    ordered = sorted(
        groups,
        key=lambda group: (
            -sum(group.vector),
            -group.ball_instances,
            _tie_key(seed, group.group_id),
        ),
    )

    for group in ordered:
        candidates: list[tuple[float, int, str]] = []
        for split_index, split in enumerate(SPLITS):
            score = 0.0
            for feature, total in enumerate(totals):
                if total == 0:
                    continue
                target = total * RATIOS[split]
                after = current[split][feature] + group.vector[feature]
                score += ((after - target) / max(target, 1.0)) ** 2
                for other in SPLITS:
                    if other != split:
                        other_target = total * RATIOS[other]
                        score += ((current[other][feature] - other_target) / max(other_target, 1.0)) ** 2
            candidates.append((score, split_index, split))
        _, _, selected = min(candidates)
        assignment[group.group_id] = selected
        for index, value in enumerate(group.vector):
            current[selected][index] += value
    return assignment


def verify_split(
    groups: list[GroupStats],
    assignment: Mapping[str, str],
    minimum_ball_count: int = 50,
) -> SplitReport:
    group_ids = {group.group_id for group in groups}
    if set(assignment) != group_ids:
        missing = sorted(group_ids - set(assignment))
        extra = sorted(set(assignment) - group_ids)
        raise ValueError(f"assignment mismatch: missing={missing}, extra={extra}")
    if any(split not in SPLITS for split in assignment.values()):
        raise ValueError("assignment contains an unsupported split")

    frame_counts = {split: 0 for split in SPLITS}
    ball_counts = {split: 0 for split in SPLITS}
    bat_counts = {split: 0 for split in SPLITS}
    hashes = {split: set() for split in SPLITS}
    for group in groups:
        split = assignment[group.group_id]
        frame_counts[split] += group.unique_frames
        ball_counts[split] += group.ball_instances
        bat_counts[split] += group.bat_instances
        hashes[split].update(group.content_hashes)

    hash_overlap: set[str] = set()
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            hash_overlap.update(hashes[left] & hashes[right])
    if hash_overlap:
        raise ValueError(f"hash overlap between splits: {sorted(hash_overlap)}")
    if any(ball_counts[split] == 0 or bat_counts[split] == 0 for split in SPLITS):
        raise ValueError("both ball and bat classes must exist in every split")
    if ball_counts["val"] < minimum_ball_count or ball_counts["test"] < minimum_ball_count:
        raise ValueError(f"val and test must each contain at least {minimum_ball_count} ball instances")

    return SplitReport(
        ok=True,
        frame_counts=frame_counts,
        ball_counts=ball_counts,
        bat_counts=bat_counts,
        group_overlap=(),
        hash_overlap=(),
    )
