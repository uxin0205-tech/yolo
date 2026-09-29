from __future__ import annotations

from masf_yolo.data.grouping import (
    GroupingRecord,
    build_union_groups,
    derive_source_key,
    roboflow_original_stem,
)


def test_roboflow_hash_is_removed_without_losing_original_frame() -> None:
    name = "clip_mp4-0037_jpg.rf.0123456789abcdef0123456789abcdef.jpg"

    assert roboflow_original_stem(name) == "clip_mp4-0037_jpg"
    assert derive_source_key("clip_mp4-0037_jpg") == "clip"


def test_video_and_mpeg_frame_names_get_conservative_source_keys() -> None:
    assert derive_source_key("20250312_143547_mp4-0552_jpg") == "20250312_143547"
    assert derive_source_key("3H2A0030-MPEG-4289_jpg") == "3H2A0030-MPEG"
    assert derive_source_key("unstructured_camera_name") == "unstructured_camera_name"


def test_augmented_siblings_and_exact_duplicates_share_union_group() -> None:
    records = [
        GroupingRecord("a", "frame_1", "video_a", "hash-a"),
        GroupingRecord("b", "frame_1", "video_a", "hash-b"),
        GroupingRecord("c", "frame_2", "video_b", "hash-a"),
        GroupingRecord("d", "frame_3", "video_c", "hash-d"),
    ]

    groups = build_union_groups(records)

    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]
