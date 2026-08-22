"""從唯讀 BBAT5 Pose 權威標註重建 Pose/Detect 兩個資料 view。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import PROJECT_ROOT, SPEC_PATH, SPEC_VERSION, file_sha256

DEFAULT_POSE_SOURCE = Path("/home/uxin/yolo/original/pose/dataset")
DEFAULT_DETECT_SOURCE = Path("/home/uxin/yolo/original/pose/detect_dataset")
DEFAULT_DESTINATION = Path("/home/uxin/yolo/original/pose/derived/bbat5-v1")
DEFAULT_COCO_TRAIN_LIST = Path("/home/uxin/yolo/coco2017/train2017.txt")
POSE_DATASET_YAML = PROJECT_ROOT / "configs/data/bbat5-pose.yaml"
DETECT_DATASET_YAML = PROJECT_ROOT / "configs/data/bbat5-detect.yaml"

BBAT5_V1_HISTORICAL_SPEC_LINEAGES = frozenset(
    {
        (
            "2.0.0",
            (
                "75db239262de75998171a05a85c8755d"
                "ea10c2bc76920dd2aabe8ff0dabb7a3b"
            ),
        )
    }
)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
COORDINATE_TOKEN_INDICES = frozenset({1, 2, 3, 4, 5, 6, 8, 9})
VISIBILITY_TOKEN_INDICES = (7, 10)


def source_group(filename: str) -> str:
    """以 `.rf.` 前綴作為同源影像群組。"""

    return filename.split(".rf.", 1)[0] if ".rf." in filename else Path(filename).stem


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _yaml_write(path: Path, payload: Any) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class PoseRecord:
    image: Path
    label: Path
    source_split: str
    group: str


@dataclass(frozen=True)
class CoordinatePatch:
    source: str
    pose_output: str
    detect_output: str
    line: int
    token: int
    old: float
    new: float
    source_sha256: str


@dataclass(frozen=True)
class Bbat5PreparationReport:
    pose_source: str
    detect_source: str
    destination: str
    seed: int
    train_ratio: float
    search_val_ratio: float
    images: int
    groups: int
    formal_train_images: int
    formal_val_images: int
    search_train_images: int
    search_val_images: int
    patched_coordinates: int
    coco_train_overlap_groups: int
    coco_exclusion_status: str
    formal_ready: bool
    executed: bool
    manifests: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SourceAudit:
    records: tuple[PoseRecord, ...]
    pose_tokens: dict[str, tuple[tuple[str, ...], ...]]
    patches: tuple[tuple[str, int, int, float], ...]
    pose_label_tree_sha256: str
    detect_label_tree_sha256: str
    empty_labels: int
    instances_by_class: dict[str, int]


def _discover_records(source: Path) -> tuple[PoseRecord, ...]:
    records: list[PoseRecord] = []
    seen: set[str] = set()
    for source_split in ("train", "valid", "val"):
        images = source / source_split / "images"
        labels = source / source_split / "labels"
        if not images.is_dir():
            continue
        for image in sorted(
            path for path in images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        ):
            if image.name in seen:
                raise ValueError(f"Pose 來源 split 間有重複影像檔名：{image.name}")
            label = labels / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Pose 影像缺少標註：{image}")
            seen.add(image.name)
            records.append(
                PoseRecord(
                    image=image.resolve(),
                    label=label.resolve(),
                    source_split=source_split,
                    group=source_group(image.name),
                )
            )
    if not records:
        raise ValueError(f"在 {source} 找不到 BBAT5 Pose 影像")
    if len({record.group for record in records}) < 2:
        raise ValueError("grouped split 至少需要兩個 source groups")
    return tuple(records)


def _detect_files(source: Path) -> tuple[set[str], dict[str, Path]]:
    images: set[str] = set()
    labels: dict[str, Path] = {}
    for split in ("train", "valid", "val"):
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        if image_dir.is_dir():
            for path in image_dir.iterdir():
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    if path.name in images:
                        raise ValueError(f"Detect 來源 split 間有重複影像檔名：{path.name}")
                    images.add(path.name)
        if label_dir.is_dir():
            for path in label_dir.glob("*.txt"):
                if path.name in labels:
                    raise ValueError(f"Detect 來源 split 間有重複標籤檔名：{path.name}")
                labels[path.name] = path.resolve()
    return images, labels


def _parse_pose_label(
    path: Path,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[int, int, float], ...], dict[str, int]]:
    parsed: list[tuple[str, ...]] = []
    patches: list[tuple[int, int, float]] = []
    classes: dict[str, int] = {"0": 0, "1": 0}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        tokens = line.split()
        if len(tokens) != 11:
            raise ValueError(f"{path}:{line_number}: Pose label 必須正好 11 欄")
        try:
            values = [float(token) for token in tokens]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: Pose label 含非數字欄位") from error
        if tokens[0] not in classes or values[0] != int(values[0]):
            raise ValueError(f"{path}:{line_number}: class 必須是 ball=0 或 bat=1")
        classes[tokens[0]] += 1
        for index in COORDINATE_TOKEN_INDICES:
            value = values[index]
            if value < 0:
                patches.append((line_number, index, value))
            elif value > 1:
                raise ValueError(f"{path}:{line_number}: token {index} 超出 [0,1]")
        for index in VISIBILITY_TOKEN_INDICES:
            if values[index] not in {0.0, 1.0, 2.0}:
                raise ValueError(f"{path}:{line_number}: visibility 必須是 0/1/2")
        parsed.append(tuple(tokens))
    return tuple(parsed), tuple(patches), classes


def _numeric_lines(path: Path, expected_columns: int) -> tuple[tuple[float, ...], ...]:
    parsed: list[tuple[float, ...]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        tokens = line.split()
        if len(tokens) != expected_columns:
            raise ValueError(f"{path}:{line_number}: Detect label 必須正好 5 欄")
        try:
            parsed.append(tuple(float(token) for token in tokens))
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: Detect label 含非數字欄位") from error
    return tuple(parsed)


def _audit_sources(pose_source: Path, detect_source: Path) -> _SourceAudit:
    records = _discover_records(pose_source)
    pose_images = {record.image.name for record in records}
    pose_labels = {record.label.name for record in records}
    detect_images, detect_labels = _detect_files(detect_source)
    problems = {
        "缺少 Detect images": sorted(pose_images - detect_images),
        "多出 Detect images": sorted(detect_images - pose_images),
        "缺少 Detect labels": sorted(pose_labels - set(detect_labels)),
        "多出 Detect labels": sorted(set(detect_labels) - pose_labels),
    }
    if any(problems.values()):
        summary = {name: values[:5] for name, values in problems.items() if values}
        raise ValueError(f"Pose/Detect 一對一一致性失敗：{summary}")

    pose_tokens: dict[str, tuple[tuple[str, ...], ...]] = {}
    patches: list[tuple[str, int, int, float]] = []
    instances = {"0": 0, "1": 0}
    mismatches: list[str] = []
    empty_labels = 0
    for record in records:
        lines, label_patches, classes = _parse_pose_label(record.label)
        pose_tokens[record.image.name] = lines
        empty_labels += not lines
        for class_id, count in classes.items():
            instances[class_id] += count
        patches.extend(
            (record.image.name, line, token, old)
            for line, token, old in label_patches
        )
        detect_lines = _numeric_lines(detect_labels[record.label.name], 5)
        authoritative = tuple(
            tuple(float(token) for token in line[:5]) for line in lines
        )
        if detect_lines != authoritative:
            mismatches.append(record.label.name)
    if mismatches:
        raise ValueError(
            "原始 Detect labels 與 Pose 前五欄一致性失敗："
            f"count={len(mismatches)} examples={mismatches[:5]}"
        )

    pose_label_paths = tuple(record.label for record in records)
    detect_label_paths = tuple(detect_labels[name] for name in sorted(detect_labels))
    return _SourceAudit(
        records=records,
        pose_tokens=pose_tokens,
        patches=tuple(patches),
        pose_label_tree_sha256=_tree_sha256(pose_source, pose_label_paths),
        detect_label_tree_sha256=_tree_sha256(detect_source, detect_label_paths),
        empty_labels=empty_labels,
        instances_by_class=instances,
    )


def _coco_overlap(
    groups: set[str], coco_train_list: Path
) -> tuple[str, set[str], str | None, int]:
    if not coco_train_list.is_file():
        return "blocked", set(), None, 0
    lines = [line.strip() for line in coco_train_list.read_text(encoding="utf-8").splitlines()]
    coco_ids = {Path(line).stem for line in lines if line}
    overlap = {
        group
        for group in groups
        if (match := re.match(r"^(\d{12})_jpg$", group))
        and match.group(1) in coco_ids
    }
    return "passed", overlap, _sha256(coco_train_list), len(coco_ids)


def _grouped_assignment(
    groups: Iterable[str],
    *,
    train_ratio: float,
    seed: int,
    force_train: Iterable[str] = (),
) -> dict[str, str]:
    ordered = sorted(set(groups))
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必須介於 0 與 1 之間")
    if len(ordered) < 2:
        raise ValueError("grouped split 至少需要兩個 groups")
    forced = set(force_train)
    if not forced.issubset(ordered):
        raise ValueError("force_train 含未知 groups")
    available = [group for group in ordered if group not in forced]
    if not available:
        raise ValueError("所有 groups 都被排除於 validation，無法建立 val")
    random.Random(seed).shuffle(available)
    target_train = min(max(round(len(ordered) * train_ratio), 1), len(ordered) - 1)
    take = min(max(target_train - len(forced), 0), len(available) - 1)
    train = forced | set(available[:take])
    return {group: ("train" if group in train else "val") for group in ordered}


def _render_pose(
    lines: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, ...], tuple[tuple[int, int, float], ...]]:
    rendered: list[str] = []
    patches: list[tuple[int, int, float]] = []
    for line_number, source_tokens in enumerate(lines, 1):
        tokens = list(source_tokens)
        for index in COORDINATE_TOKEN_INDICES:
            value = float(tokens[index])
            if value < 0:
                patches.append((line_number, index, value))
                tokens[index] = "0"
        rendered.append(" ".join(tokens))
    return tuple(rendered), tuple(patches)


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    materialized = tuple(lines)
    path.write_text(
        "\n".join(materialized) + ("\n" if materialized else ""),
        encoding="utf-8",
    )


def _counts(records: tuple[PoseRecord, ...], assignment: dict[str, str]) -> tuple[int, int]:
    train = sum(assignment[record.group] == "train" for record in records)
    return train, len(records) - train


def _common_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "derived_version": "bbat5-v1",
    }


def _dataset_configs(destination: Path) -> dict[str, dict[str, Any]]:
    names = {0: "ball", 1: "bat"}
    pose = {
        "path": str(destination / "pose"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
        "kpt_shape": [2, 3],
        "flip_idx": [0, 1],
    }
    detect = {
        "path": str(destination / "detect"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    pose_search = dict(pose)
    pose_search.update(
        train=str(destination / "pose/splits/search-train.txt"),
        val=str(destination / "pose/splits/search-val.txt"),
    )
    detect_search = dict(detect)
    detect_search.update(
        train=str(destination / "detect/splits/search-train.txt"),
        val=str(destination / "detect/splits/search-val.txt"),
    )
    return {
        "pose.yaml": pose,
        "detect.yaml": detect,
        "pose-search.yaml": pose_search,
        "detect-search.yaml": detect_search,
    }


def _readme(
    pose_source: Path,
    detect_source: Path,
    destination: Path,
    report: Bbat5PreparationReport,
) -> str:
    return f"""# BBAT5 v1 衍生資料

此目錄是 architecture_2 的不可覆寫資料版本。建立資料不會啟動 Pose 訓練；是否執行 Pose 仍由使用者另外決定。

## 我們做了什麼

- 唯讀 Pose 來源：`{pose_source}`。
- 唯讀 Detect audit 來源：`{detect_source}`。
- Pose labels 是唯一權威；Detect labels 由修補後每列前五欄產生。
- 依 `.rf.` 前 prefix、seed {report.seed} 做 grouped {report.train_ratio:.0%}/{1-report.train_ratio:.0%} formal train/val。
- search split 僅從 formal train 內再分組，formal val 完全不參與搜尋。
- 將 {report.patched_coordinates} 個負座標在衍生 label clamp 成 0；原始檔不變。
- COCO train 重疊群組數：{report.coco_train_overlap_groups}；排除狀態：`{report.coco_exclusion_status}`。
- Pose/Detect 共用相同 assignment；影像為 symlink；沒有建立 test split。

## 目錄

- `pose/`：正式 Pose view 與 search 清單。
- `detect/`：ball/bat 2-class 診斷 view，不取代 COCO80 Detect。
- `configs/`：可直接交給 Ultralytics 的 formal/search dataset YAML。
- `manifests/`：來源稽核、split、patch、COCO 排除與重建 lineage。

## 重建

```bash
python -m achitechure_2.cli prepare-pose-data --execute
```

此 v1 不可覆寫；規則或資料有任何變更時，請建立 `bbat5-v2`。
"""


def _build_dataset(
    temporary: Path,
    destination: Path,
    pose_source: Path,
    detect_source: Path,
    coco_train_list: Path,
    audit: _SourceAudit,
    formal: dict[str, str],
    search: dict[str, str],
    overlap: set[str],
    coco_status: str,
    coco_sha256: str | None,
    coco_images: int,
    report: Bbat5PreparationReport,
) -> None:
    for view in ("pose", "detect"):
        for split in ("train", "val"):
            (temporary / view / "images" / split).mkdir(parents=True, exist_ok=True)
            (temporary / view / "labels" / split).mkdir(parents=True, exist_ok=True)
        (temporary / view / "splits").mkdir(parents=True, exist_ok=True)
    (temporary / "configs").mkdir(parents=True, exist_ok=True)
    (temporary / "manifests").mkdir(parents=True, exist_ok=True)

    patch_items: list[CoordinatePatch] = []
    for record in audit.records:
        split = formal[record.group]
        pose_image = temporary / "pose/images" / split / record.image.name
        detect_image = temporary / "detect/images" / split / record.image.name
        os.symlink(str(record.image), pose_image)
        os.symlink(str(record.image), detect_image)
        pose_label = temporary / "pose/labels" / split / record.label.name
        detect_label = temporary / "detect/labels" / split / record.label.name
        rendered, patches = _render_pose(audit.pose_tokens[record.image.name])
        _write_lines(pose_label, rendered)
        _write_lines(detect_label, (" ".join(line.split()[:5]) for line in rendered))
        for line, token, old in patches:
            patch_items.append(
                CoordinatePatch(
                    source=str(record.label),
                    pose_output=str(destination / "pose/labels" / split / record.label.name),
                    detect_output=str(destination / "detect/labels" / split / record.label.name),
                    line=line,
                    token=token,
                    old=old,
                    new=0.0,
                    source_sha256=_sha256(record.label),
                )
            )

    for view in ("pose", "detect"):
        view_root = destination / view
        temporary_root = temporary / view
        for split_name in ("train", "val"):
            selected = [
                str(view_root / "images" / formal[record.group] / record.image.name)
                for record in audit.records
                if formal[record.group] == split_name
            ]
            _write_lines(temporary_root / "splits" / f"formal-{split_name}.txt", selected)
        for split_name in ("train", "val"):
            selected = [
                str(view_root / "images/train" / record.image.name)
                for record in audit.records
                if formal[record.group] == "train" and search[record.group] == split_name
            ]
            _write_lines(temporary_root / "splits" / f"search-{split_name}.txt", selected)

    for name, payload in _dataset_configs(destination).items():
        _yaml_write(temporary / "configs" / name, payload)

    formal_train_groups = sorted(group for group, split in formal.items() if split == "train")
    formal_val_groups = sorted(set(formal) - set(formal_train_groups))
    search_train_groups = sorted(group for group, split in search.items() if split == "train")
    search_val_groups = sorted(set(search) - set(search_train_groups))
    common = _common_manifest()
    source_manifest = {
        **common,
        "pose_source_read_only": str(pose_source),
        "detect_source_read_only": str(detect_source),
        "pose_source_yaml_sha256": _sha256(pose_source / "data.yaml"),
        "detect_source_yaml_sha256": _sha256(detect_source / "data.yaml"),
        "pose_label_tree_sha256": audit.pose_label_tree_sha256,
        "detect_label_tree_sha256": audit.detect_label_tree_sha256,
        "images": len(audit.records),
        "groups": len(formal),
        "empty_labels": audit.empty_labels,
        "instances_by_class": audit.instances_by_class,
        "detect_audit": {
            "status": "passed",
            "rule": "每個原始 Detect label 必須與 Pose label 前五欄數值一致",
            "matched_images": len(audit.records),
            "missing": 0,
            "unexpected": 0,
            "mismatched": 0,
        },
    }
    split_manifest = {
        **common,
        "group_key": "prefix_before_.rf.",
        "seed": report.seed,
        "formal": {
            "train_ratio": report.train_ratio,
            "assignment": formal,
            "train_groups": formal_train_groups,
            "val_groups": formal_val_groups,
            "train_images": report.formal_train_images,
            "val_images": report.formal_val_images,
            "leakage": [],
        },
        "search": {
            "scope": "formal_train_only",
            "val_ratio": report.search_val_ratio,
            "assignment": search,
            "train_groups": search_train_groups,
            "val_groups": search_val_groups,
            "train_images": report.search_train_images,
            "val_images": report.search_val_images,
            "leakage": [],
        },
        "shared_by_views": ["pose", "detect"],
        "test_split": None,
    }
    patch_manifest = {
        **common,
        "rule": "只在衍生 Pose label 將負座標 clamp 成 0；Detect 再由修補後前五欄產生",
        "patch_count": len(patch_items),
        "patches": [asdict(item) for item in patch_items],
    }
    exclusion_manifest = {
        **common,
        "status": coco_status,
        "coco_train_list": str(coco_train_list),
        "coco_train_list_sha256": coco_sha256,
        "coco_train_images": coco_images,
        "overlap_rule": "BBAT5 group 的 12 位 COCO ID 是否存在 COCO train2017",
        "overlap_groups": sorted(overlap),
        "overlap_group_count": len(overlap),
        "excluded_from_formal_val": sorted(overlap),
        "excluded_from_search_val": sorted(overlap & set(formal_train_groups)),
        "formal_val_overlap_after_exclusion": sorted(overlap & set(formal_val_groups)),
        "search_val_overlap_after_exclusion": sorted(overlap & set(search_val_groups)),
    }
    rebuild_manifest = {
        **common,
        "entrypoint": "prepare-pose-data",
        "pose_source": str(pose_source),
        "detect_source": str(detect_source),
        "destination": str(destination),
        "coco_train_list": str(coco_train_list),
        "seed": report.seed,
        "train_ratio": report.train_ratio,
        "search_val_ratio": report.search_val_ratio,
        "expected_patch_count": report.patched_coordinates,
        "repository_dataset_yamls": {
            "pose": {
                "path": str(POSE_DATASET_YAML),
                "sha256": file_sha256(POSE_DATASET_YAML),
            },
            "detect": {
                "path": str(DETECT_DATASET_YAML),
                "sha256": file_sha256(DETECT_DATASET_YAML),
            },
        },
        "images_in_git": False,
        "labels_in_git": False,
    }
    manifests = {
        "source-audit-manifest.json": source_manifest,
        "split-manifest.json": split_manifest,
        "patch-manifest.json": patch_manifest,
        "coco-exclusion-manifest.json": exclusion_manifest,
        "rebuild-manifest.json": rebuild_manifest,
    }
    for name, payload in manifests.items():
        _json_write(temporary / "manifests" / name, payload)
    (temporary / "README.md").write_text(
        _readme(pose_source, detect_source, destination, report),
        encoding="utf-8",
    )


def prepare_bbat5_dataset(
    pose_source: str | Path = DEFAULT_POSE_SOURCE,
    detect_source: str | Path = DEFAULT_DETECT_SOURCE,
    destination: str | Path = DEFAULT_DESTINATION,
    *,
    coco_train_list: str | Path = DEFAULT_COCO_TRAIN_LIST,
    train_ratio: float = 0.9,
    search_val_ratio: float = 0.1,
    seed: int = 0,
    execute: bool = False,
    expected_patch_count: int | None = 4,
) -> Bbat5PreparationReport:
    """先稽核／規劃；只有 `execute=True` 才建立 immutable bbat5-v1。"""

    pose_root = Path(pose_source).resolve()
    detect_root = Path(detect_source).resolve()
    output_root = Path(destination).resolve()
    coco_list = Path(coco_train_list).resolve()
    if seed < 0:
        raise ValueError("seed 必須為非負整數")
    if not 0 < search_val_ratio < 1:
        raise ValueError("search_val_ratio 必須介於 0 與 1 之間")
    audit = _audit_sources(pose_root, detect_root)
    if expected_patch_count is not None and len(audit.patches) != expected_patch_count:
        raise ValueError(
            f"預期修補 {expected_patch_count} 個負座標，實際找到 {len(audit.patches)} 個"
        )
    groups = {record.group for record in audit.records}
    coco_status, overlap, coco_sha256, coco_images = _coco_overlap(groups, coco_list)
    formal = _grouped_assignment(
        groups,
        train_ratio=train_ratio,
        seed=seed,
        force_train=overlap,
    )
    formal_train_groups = {group for group, split in formal.items() if split == "train"}
    search = _grouped_assignment(
        formal_train_groups,
        train_ratio=1 - search_val_ratio,
        seed=seed,
        force_train=overlap & formal_train_groups,
    )
    formal_train_images, formal_val_images = _counts(audit.records, formal)
    search_records = tuple(
        record for record in audit.records if formal[record.group] == "train"
    )
    search_train_images, search_val_images = _counts(search_records, search)
    formal_ready = (
        coco_status == "passed"
        and not overlap.intersection(group for group, split in formal.items() if split == "val")
        and not overlap.intersection(group for group, split in search.items() if split == "val")
    )
    manifest_names = (
        "source-audit-manifest.json",
        "split-manifest.json",
        "patch-manifest.json",
        "coco-exclusion-manifest.json",
        "rebuild-manifest.json",
    )
    report = Bbat5PreparationReport(
        pose_source=str(pose_root),
        detect_source=str(detect_root),
        destination=str(output_root),
        seed=seed,
        train_ratio=train_ratio,
        search_val_ratio=search_val_ratio,
        images=len(audit.records),
        groups=len(groups),
        formal_train_images=formal_train_images,
        formal_val_images=formal_val_images,
        search_train_images=search_train_images,
        search_val_images=search_val_images,
        patched_coordinates=len(audit.patches),
        coco_train_overlap_groups=len(overlap),
        coco_exclusion_status=coco_status,
        formal_ready=formal_ready,
        executed=execute,
        manifests=tuple(str(output_root / "manifests" / name) for name in manifest_names),
    )
    if not execute:
        return report
    if output_root.exists():
        raise FileExistsError(f"衍生資料版本不可覆寫：{output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-building-", dir=output_root.parent)
    )
    try:
        _build_dataset(
            temporary,
            output_root,
            pose_root,
            detect_root,
            coco_list,
            audit,
            formal,
            search,
            overlap,
            coco_status,
            coco_sha256,
            coco_images,
            report,
        )
        if _tree_sha256(pose_root, (record.label for record in audit.records)) != audit.pose_label_tree_sha256:
            raise RuntimeError("Pose 原始 labels 在重建期間被改動")
        _, detect_labels = _detect_files(detect_root)
        if _tree_sha256(detect_root, detect_labels.values()) != audit.detect_label_tree_sha256:
            raise RuntimeError("Detect 原始 labels 在重建期間被改動")
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def validate_bbat5_dataset(destination: str | Path = DEFAULT_DESTINATION) -> dict[str, Any]:
    """驗證衍生 view、lineage、group isolation、patch 與 COCO 排除。"""

    root = Path(destination).resolve()
    manifests = root / "manifests"
    source = json.loads((manifests / "source-audit-manifest.json").read_text(encoding="utf-8"))
    split = json.loads((manifests / "split-manifest.json").read_text(encoding="utf-8"))
    patch = json.loads((manifests / "patch-manifest.json").read_text(encoding="utf-8"))
    exclusion = json.loads(
        (manifests / "coco-exclusion-manifest.json").read_text(encoding="utf-8")
    )
    rebuild = json.loads(
        (manifests / "rebuild-manifest.json").read_text(encoding="utf-8")
    )
    named_manifests = (
        ("source", source),
        ("split", split),
        ("patch", patch),
        ("exclusion", exclusion),
        ("rebuild", rebuild),
    )
    observed_lineages = {
        (payload.get("spec_version"), payload.get("spec_sha256"))
        for _, payload in named_manifests
    }
    if len(observed_lineages) != 1:
        raise AssertionError("BBAT5 manifests 混用了不同 spec lineage")
    manifest_spec_version, manifest_spec_sha256 = observed_lineages.pop()
    allowed_lineages = BBAT5_V1_HISTORICAL_SPEC_LINEAGES | {
        (SPEC_VERSION, file_sha256(SPEC_PATH))
    }
    if (manifest_spec_version, manifest_spec_sha256) not in allowed_lineages:
        raise AssertionError("BBAT5 manifest 的 authoritative spec lineage 不受支援")

    formal_train = set(split["formal"]["train_groups"])
    formal_val = set(split["formal"]["val_groups"])
    search_train = set(split["search"]["train_groups"])
    search_val = set(split["search"]["val_groups"])
    if formal_train & formal_val:
        raise AssertionError("formal train/val 發生 group leakage")
    if search_train & search_val or search_train | search_val != formal_train:
        raise AssertionError("search split 不是 formal train 的無洩漏分割")
    overlap = set(exclusion["overlap_groups"])
    if overlap & formal_val or overlap & search_val:
        raise AssertionError("COCO train 重疊群組進入 formal/search validation")

    view_names: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for view in ("pose", "detect"):
        names: set[str] = set()
        for split_name in ("train", "val"):
            image_dir = root / view / "images" / split_name
            label_dir = root / view / "labels" / split_name
            images = tuple(sorted(image_dir.iterdir()))
            labels = tuple(sorted(label_dir.glob("*.txt")))
            if not all(image.is_symlink() for image in images):
                raise AssertionError(f"{view}/{split_name} 含非 symlink 影像")
            if {image.stem for image in images} != {label.stem for label in labels}:
                raise AssertionError(f"{view}/{split_name} 的影像與 label 不一對一")
            names.update(image.name for image in images)
            counts[f"{view}_{split_name}"] = len(images)
        if (root / view / "images/test").exists() or (root / view / "labels/test").exists():
            raise AssertionError("原始資料沒有 test，不得建立 test split")
        view_names[view] = names
    if view_names["pose"] != view_names["detect"]:
        raise AssertionError("Pose/Detect view 沒有共用相同影像 assignment")

    for split_name in ("train", "val"):
        for pose_label in (root / "pose/labels" / split_name).glob("*.txt"):
            detect_label = root / "detect/labels" / split_name / pose_label.name
            pose_lines = pose_label.read_text(encoding="utf-8").splitlines()
            detect_lines = detect_label.read_text(encoding="utf-8").splitlines()
            expected = [" ".join(line.split()[:5]) for line in pose_lines]
            if detect_lines != expected:
                raise AssertionError(f"Detect view 不是 Pose 前五欄：{pose_label.name}")
            for line in pose_lines:
                tokens = line.split()
                if any(float(tokens[index]) < 0 for index in COORDINATE_TOKEN_INDICES):
                    raise AssertionError(f"衍生 Pose label 仍有負座標：{pose_label.name}")

    for item in patch["patches"]:
        if _sha256(Path(item["source"])) != item["source_sha256"]:
            raise AssertionError(f"原始 patch source 已改變：{item['source']}")
        if float(item["old"]) >= 0 or float(item["new"]) != 0:
            raise AssertionError("patch manifest 含無效修補")
    pose_source = Path(source["pose_source_read_only"])
    detect_source = Path(source["detect_source_read_only"])
    records = _discover_records(pose_source)
    _, detect_labels = _detect_files(detect_source)
    if _tree_sha256(pose_source, (record.label for record in records)) != source["pose_label_tree_sha256"]:
        raise AssertionError("Pose 原始 label tree 已改變")
    if _tree_sha256(detect_source, detect_labels.values()) != source["detect_label_tree_sha256"]:
        raise AssertionError("Detect 原始 label tree 已改變")

    formal_ready = (
        exclusion.get("status") == "passed"
        and not exclusion.get("formal_val_overlap_after_exclusion")
        and not exclusion.get("search_val_overlap_after_exclusion")
    )
    return {
        "valid": formal_ready,
        "formal_ready": formal_ready,
        "destination": str(root),
        "pose_images": len(view_names["pose"]),
        "detect_images": len(view_names["detect"]),
        "formal_train_images": counts["pose_train"],
        "formal_val_images": counts["pose_val"],
        "patched_coordinates": len(patch["patches"]),
        "coco_train_overlap_groups": len(overlap),
        "source_untouched": True,
        "test_split": None,
        "spec_version": manifest_spec_version,
        "spec_sha256": manifest_spec_sha256,
    }


def export_bbat5_metadata(
    source: str | Path = DEFAULT_DESTINATION,
    destination: str | Path = PROJECT_ROOT / "artifacts/datasets/bbat5-v1",
    *,
    execute: bool = False,
) -> dict[str, Any]:
    """只匯出 Git 所需 README、YAML 與 manifests；永不匯出影像或 labels。"""

    source_root = Path(source).resolve()
    output_root = Path(destination).resolve()
    relative_files = (
        Path("README.md"),
        Path("configs/pose.yaml"),
        Path("configs/detect.yaml"),
        Path("configs/pose-search.yaml"),
        Path("configs/detect-search.yaml"),
        Path("manifests/source-audit-manifest.json"),
        Path("manifests/split-manifest.json"),
        Path("manifests/patch-manifest.json"),
        Path("manifests/coco-exclusion-manifest.json"),
        Path("manifests/rebuild-manifest.json"),
    )
    missing = [str(path) for path in relative_files if not (source_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"BBAT5 metadata 不完整：{missing}")
    report = {
        "source": str(source_root),
        "destination": str(output_root),
        "execute": execute,
        "files": [str(path) for path in relative_files],
        "excluded": ["pose/images", "pose/labels", "detect/images", "detect/labels"],
    }
    if not execute:
        return report
    if output_root.exists():
        raise FileExistsError(f"Git metadata 目的地不可覆寫：{output_root}")
    for relative in relative_files:
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    if any(
        part in {"images", "labels"}
        for path in output_root.rglob("*")
        for part in path.relative_to(output_root).parts
    ):
        raise AssertionError("metadata export 意外包含 images/labels")
    return report


# 中文 CLI 名稱仍為 prepare-pose-data；函式別名方便既有自動化逐步遷移。
prepare_pose_data = prepare_bbat5_dataset
