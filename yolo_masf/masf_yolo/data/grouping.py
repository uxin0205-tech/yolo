"""Union grouping for original frames, source sequences, and exact images."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-fA-F]{16,64}$")
_VIDEO_FRAME = re.compile(r"^(?P<source>.+?)[_-](?:mp4|mov)[_-]?\d+(?:_jpg)?$", re.IGNORECASE)
_MPEG_FRAME = re.compile(r"^(?P<source>.+?-MPEG)-\d+(?:_jpg)?$", re.IGNORECASE)
_GENERIC_FRAME = re.compile(r"^(?P<source>frame|baseball|image)\d+(?:_jpg)?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class GroupingRecord:
    record_id: str
    original_stem: str
    source_key: str
    content_hash: str


def roboflow_original_stem(filename: str | Path) -> str:
    stem = Path(filename).stem
    return _ROBOFLOW_SUFFIX.sub("", stem)


def derive_source_key(original_stem: str) -> str:
    for pattern in (_VIDEO_FRAME, _MPEG_FRAME, _GENERIC_FRAME):
        match = pattern.match(original_stem)
        if match:
            return match.group("source")
    return original_stem


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def build_union_groups(records: list[GroupingRecord]) -> dict[str, str]:
    record_ids = [record.record_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("record IDs must be unique")
    union = _UnionFind(record_ids)
    seen: dict[tuple[str, str], str] = {}
    for record in records:
        keys = (
            ("original", record.original_stem),
            ("source", record.source_key),
            ("content", record.content_hash),
        )
        for key in keys:
            if key in seen:
                union.union(record.record_id, seen[key])
            else:
                seen[key] = record.record_id
    roots = {record_id: union.find(record_id) for record_id in record_ids}
    canonical = {
        root: min(item for item, item_root in roots.items() if item_root == root)
        for root in set(roots.values())
    }
    return {record_id: canonical[root] for record_id, root in roots.items()}
