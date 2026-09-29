from __future__ import annotations

import pytest

from masf_yolo.data.split import GroupStats, assign_groups, verify_split


def _balanced_groups(count: int = 100) -> list[GroupStats]:
    return [
        GroupStats(
            group_id=f"g{index:03d}",
            unique_frames=1,
            ball_instances=1,
            bat_instances=1,
            ball_bins=(1, 0, 0),
            content_hashes=(f"h{index:03d}",),
        )
        for index in range(count)
    ]


def test_seed42_split_is_reproducible_and_leak_free() -> None:
    groups = _balanced_groups()

    first = assign_groups(groups, seed=42)
    second = assign_groups(list(reversed(groups)), seed=42)
    report = verify_split(groups, first, minimum_ball_count=10)

    assert first == second
    assert report.ok
    assert report.frame_counts == {"train": 80, "val": 10, "test": 10}
    assert report.group_overlap == ()
    assert report.hash_overlap == ()


def test_val_and_test_minimum_ball_gate_fails_closed() -> None:
    groups = _balanced_groups(20)
    assignment = assign_groups(groups, seed=42)

    with pytest.raises(ValueError, match="at least 50 ball"):
        verify_split(groups, assignment, minimum_ball_count=50)


def test_hash_overlap_is_detected_even_for_distinct_groups() -> None:
    groups = _balanced_groups()
    groups[0] = GroupStats("g000", 1, 1, 1, (1, 0, 0), ("shared",))
    groups[-1] = GroupStats("g099", 1, 1, 1, (1, 0, 0), ("shared",))
    assignment = assign_groups(groups, seed=42)

    if assignment["g000"] == assignment["g099"]:
        assignment["g099"] = "val" if assignment["g000"] != "val" else "test"

    with pytest.raises(ValueError, match="hash overlap"):
        verify_split(groups, assignment, minimum_ball_count=1)
