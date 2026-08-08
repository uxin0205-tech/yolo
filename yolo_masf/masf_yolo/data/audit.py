"""End-to-end dataset audit and reproducible manifest generation."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from masf_yolo.contracts import DatasetManifest, sha256_file, sha256_value

from .export import LetterboxCocoBox, box_to_letterbox_coco
from .grouping import GroupingRecord, build_union_groups, derive_source_key, roboflow_original_stem
from .labels import Box, parse_yolo_label
from .split import GroupStats, SPLITS, assign_groups, verify_split


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class _Record:
    record_id: str
    image_path: Path
    label_path: Path
    original_stem: str
    source_key: str
    content_hash: str
    width: int
    height: int
    boxes: tuple[Box, ...]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _index_dataset(source_root: Path) -> list[_Record]:
    records: list[_Record] = []
    for partition in ("train", "valid"):
        image_root = source_root / partition / "images"
        label_root = source_root / partition / "labels"
        if not image_root.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(f"missing dataset partition: {partition}")
        images = sorted(
            path for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"dataset partition has no images: {partition}")
        for image_path in images:
            label_path = label_root / f"{image_path.stem}.txt"
            boxes = parse_yolo_label(label_path)
            original_stem = roboflow_original_stem(image_path.name)
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            records.append(
                _Record(
                    record_id=f"{partition}/{image_path.name}",
                    image_path=image_path.resolve(),
                    label_path=label_path.resolve(),
                    original_stem=original_stem,
                    source_key=derive_source_key(original_stem),
                    content_hash=sha256_file(image_path.resolve()),
                    width=width,
                    height=height,
                    boxes=boxes,
                )
            )
    return records


def _group_stats(records: list[_Record], group_ids: dict[str, str]) -> list[GroupStats]:
    members: dict[str, list[_Record]] = defaultdict(list)
    for record in records:
        members[group_ids[record.record_id]].append(record)
    groups: list[GroupStats] = []
    for group_id, group_records in sorted(members.items()):
        class_counts: Counter[int] = Counter()
        bins = Counter()
        for record in group_records:
            for box in record.boxes:
                class_counts[box.class_id] += 1
                if box.class_id == 0:
                    bins[box_to_letterbox_coco(box, record.width, record.height).size_bin] += 1
        groups.append(
            GroupStats(
                group_id=group_id,
                unique_frames=len({record.original_stem for record in group_records}),
                ball_instances=class_counts[0],
                bat_instances=class_counts[1],
                ball_bins=(bins["tiny"], bins["small"], bins["large"]),
                content_hashes=tuple(sorted({record.content_hash for record in group_records})),
            )
        )
    return groups


def _coco(records: list[_Record]) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, record in enumerate(sorted(records, key=lambda item: item.record_id), 1):
        images.append(
            {
                "id": image_id,
                "file_name": str(record.image_path),
                "width": 640,
                "height": 640,
                "source_width": record.width,
                "source_height": record.height,
            }
        )
        for box in record.boxes:
            converted = box_to_letterbox_coco(box, record.width, record.height)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": box.class_id,
                    "bbox": list(converted.bbox),
                    "area": converted.area,
                    "iscrowd": 0,
                    "segmentation": [],
                    "short_side": converted.short_side,
                    "baseball_size_bin": converted.size_bin,
                    "blur_proxy": converted.blur_proxy,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "ball"}, {"id": 1, "name": "bat"}],
    }


def _write_histograms(output_root: Path, converted: list[LetterboxCocoBox]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/masf-yolo-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    values = {
        "bbox_width_hist.png": [box.bbox[2] for box in converted],
        "bbox_height_hist.png": [box.bbox[3] for box in converted],
        "bbox_area_hist.png": [box.area for box in converted],
        "p2_cell_hist.png": [box.short_side / 4 for box in converted],
        "p3_cell_hist.png": [box.short_side / 8 for box in converted],
        "p4_cell_hist.png": [box.short_side / 16 for box in converted],
    }
    for filename, series in values.items():
        figure, axis = plt.subplots(figsize=(6, 4))
        data_range = max(series) - min(series)
        representable_step = math.ulp(max(abs(value) for value in series))
        bins = 1 if data_range <= representable_step * 30 else 30
        axis.hist(series, bins=bins)
        axis.set_title(filename.removesuffix(".png").replace("_", " "))
        figure.tight_layout()
        figure.savefig(output_root / filename, dpi=120)
        plt.close(figure)


def audit_dataset(
    source_root: Path,
    output_root: Path,
    *,
    seed: int = 42,
    minimum_ball_count: int = 50,
) -> DatasetManifest:
    """Audit the train/valid union and write deterministic Phase 1 views."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    records = _index_dataset(source_root)
    grouping_records = [
        GroupingRecord(
            record.record_id,
            record.original_stem,
            record.source_key,
            record.content_hash,
        )
        for record in records
    ]
    group_ids = build_union_groups(grouping_records)
    groups = _group_stats(records, group_ids)
    assignment = assign_groups(groups, seed=seed)
    split_report = verify_split(groups, assignment, minimum_ball_count=minimum_ball_count)
    split_records = {split: [] for split in SPLITS}
    for record in records:
        split_records[assignment[group_ids[record.record_id]]].append(record)

    dataset_payload = [
        {
            "record_id": record.record_id,
            "original_stem": record.original_stem,
            "source_key": record.source_key,
            "content_hash": record.content_hash,
            "boxes": [asdict(box) for box in record.boxes],
            "group_id": group_ids[record.record_id],
            "split": assignment[group_ids[record.record_id]],
        }
        for record in sorted(records, key=lambda item: item.record_id)
    ]
    dataset_hash = sha256_value(dataset_payload)
    output_root.mkdir(parents=True, exist_ok=True)
    for split, selected in split_records.items():
        (output_root / f"{split}.txt").write_text(
            "".join(f"{record.image_path}\n" for record in sorted(selected, key=lambda item: item.record_id)),
            encoding="utf-8",
        )
    (output_root / "data.yaml").write_text(
        "path: .\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnc: 2\nnames: [ball, bat]\n",
        encoding="utf-8",
    )
    for split in ("val", "test"):
        _write_json(output_root / f"{split}.coco.json", _coco(split_records[split]))

    ball_boxes = [
        box_to_letterbox_coco(box, record.width, record.height)
        for record in records for box in record.boxes if box.class_id == 0
    ]
    profile = {
        "dataset_hash": dataset_hash,
        "records": len(records),
        "unique_frames": len({record.original_stem for record in records}),
        "groups": len(groups),
        "class_instances": {
            "ball": sum(len([box for box in record.boxes if box.class_id == 0]) for record in records),
            "bat": sum(len([box for box in record.boxes if box.class_id == 1]) for record in records),
        },
        "ball_size_bins": dict(Counter(box.size_bin for box in ball_boxes)),
        "ball_blur_proxy_count": sum(box.blur_proxy for box in ball_boxes),
    }
    audit = {
        "ok": split_report.ok,
        "dataset_hash": dataset_hash,
        "frame_counts": split_report.frame_counts,
        "ball_counts": split_report.ball_counts,
        "bat_counts": split_report.bat_counts,
        "group_overlap": list(split_report.group_overlap),
        "hash_overlap": list(split_report.hash_overlap),
    }
    manifest = DatasetManifest(
        source_root=source_root,
        output_root=output_root,
        dataset_hash=dataset_hash,
        split_ratios=(0.8, 0.1, 0.1),
        split_counts={split: len(split_records[split]) for split in SPLITS},
        class_names=("ball", "bat"),
        group_count=len(groups),
    )
    _write_json(output_root / "dataset_profile.json", profile)
    _write_json(output_root / "audit.json", audit)
    _write_json(output_root / "manifest.json", manifest.to_dict())
    _write_histograms(output_root, ball_boxes)
    return manifest
