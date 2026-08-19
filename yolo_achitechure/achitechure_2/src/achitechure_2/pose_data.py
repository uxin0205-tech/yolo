"""Read-only Pose source audit and deterministic grouped dataset derivation."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, SPEC_PATH, SPEC_VERSION, file_sha256

DEFAULT_DATASET_YAML = PROJECT_ROOT / "configs/data/pose-grouped.yaml"

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def source_group(filename: str) -> str:
    """Return the unaugmented source key used to prevent Roboflow leakage."""

    marker = ".rf."
    return filename.split(marker, 1)[0] if marker in filename else Path(filename).stem


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PoseRecord:
    image: Path
    label: Path
    group: str


@dataclass(frozen=True)
class CoordinatePatch:
    source: str
    output: str
    line: int
    token: int
    old: float
    new: float
    source_sha256: str


@dataclass(frozen=True)
class GroupedSplitReport:
    source: str
    destination: str
    seed: int
    train_ratio: float
    groups: int
    train_groups: int
    val_groups: int
    images: int
    train_images: int
    val_images: int
    leakage: tuple[str, ...]
    patched_coordinates: int
    patch_manifest: str | None
    split_manifest: str | None
    executed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_pose_records(source: str | Path) -> tuple[PoseRecord, ...]:
    """Collect image/label pairs from source train/valid trees without changing them."""

    root = Path(source).resolve()
    records: list[PoseRecord] = []
    seen_images: set[str] = set()
    for split in ("train", "valid", "val"):
        images = root / split / "images"
        labels = root / split / "labels"
        if not images.is_dir():
            continue
        for image in sorted(path for path in images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            if image.name in seen_images:
                raise ValueError(f"來源 split 之間有重複影像檔名：{image.name}")
            label = labels / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"找不到影像對應的 Pose 標籤：{image}")
            seen_images.add(image.name)
            records.append(PoseRecord(image.resolve(), label.resolve(), source_group(image.name)))
    if not records:
        raise ValueError(f"在 {root} 找不到 Pose 影像")
    return tuple(records)


def grouped_assignment(
    records: tuple[PoseRecord, ...],
    *,
    train_ratio: float = 0.9,
    seed: int = 0,
) -> dict[str, str]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必須介於 0 與 1 之間")
    groups = sorted({record.group for record in records})
    random.Random(seed).shuffle(groups)
    train_count = round(len(groups) * train_ratio)
    train_count = min(max(train_count, 1), len(groups) - 1)
    train = set(groups[:train_count])
    return {group: ("train" if group in train else "val") for group in groups}


def _clip_label(source: Path, output: Path) -> tuple[CoordinatePatch, ...]:
    lines = source.read_text(encoding="utf-8").splitlines()
    patched: list[CoordinatePatch] = []
    rendered: list[str] = []
    digest = _sha256(source)
    for line_number, line in enumerate(lines, 1):
        tokens = line.split()
        for index in range(1, len(tokens)):
            try:
                value = float(tokens[index])
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: 標籤含非數字 token") from error
            if value < 0:
                patched.append(
                    CoordinatePatch(
                        str(source),
                        str(output),
                        line_number,
                        index,
                        value,
                        0.0,
                        digest,
                    )
                )
                tokens[index] = "0"
        rendered.append(" ".join(tokens))
    output.write_text("\n".join(rendered) + ("\n" if rendered else ""), encoding="utf-8")
    return tuple(patched)


def prepare_grouped_pose_dataset(
    source: str | Path,
    destination: str | Path,
    *,
    train_ratio: float = 0.9,
    seed: int = 0,
    execute: bool = False,
    expected_patch_count: int | None = None,
    dataset_yaml: str | Path = DEFAULT_DATASET_YAML,
) -> GroupedSplitReport:
    """Plan or create a grouped split using image symlinks and copied/patched labels."""

    source_root = Path(source).resolve()
    destination_root = Path(destination).resolve()
    dataset_yaml_path = Path(dataset_yaml).resolve()
    spec_sha256 = file_sha256(SPEC_PATH)
    dataset_sha256 = file_sha256(dataset_yaml_path)
    records = discover_pose_records(source_root)
    assignment = grouped_assignment(records, train_ratio=train_ratio, seed=seed)
    train_groups = {group for group, split in assignment.items() if split == "train"}
    val_groups = set(assignment) - train_groups
    leakage = tuple(sorted(train_groups & val_groups))
    if leakage:
        raise AssertionError(f"偵測到來源群組洩漏：{leakage[:10]}")
    train_images = sum(assignment[record.group] == "train" for record in records)
    val_images = len(records) - train_images
    patch_manifest: Path | None = None
    split_manifest: Path | None = None
    patches: list[CoordinatePatch] = []
    negative_count = sum(
        float(token) < 0
        for record in records
        for line in record.label.read_text(encoding="utf-8").splitlines()
        for token in line.split()[1:]
    )
    if expected_patch_count is not None and negative_count != expected_patch_count:
        raise ValueError(
            f"預期修補 {expected_patch_count} 個負座標，實際找到 {negative_count} 個"
        )

    if execute:
        if destination_root.exists() and any(destination_root.iterdir()):
            raise FileExistsError(f"衍生 Pose 目的地不是空目錄：{destination_root}")
        for split in ("train", "val"):
            (destination_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (destination_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for record in records:
            split = assignment[record.group]
            image_output = destination_root / "images" / split / record.image.name
            label_output = destination_root / "labels" / split / record.label.name
            os.symlink(record.image, image_output)
            patches.extend(_clip_label(record.label, label_output))
        if expected_patch_count is not None and len(patches) != expected_patch_count:
            raise ValueError(
                f"預期修補 {expected_patch_count} 個負座標，實際找到 {len(patches)} 個"
            )
        patch_manifest = destination_root / "patch-manifest.json"
        patch_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "certification_history": [],
                    "spec_version": SPEC_VERSION,
                    "spec_sha256": spec_sha256,
                    "dataset_yaml": str(dataset_yaml_path),
                    "dataset_yaml_sha256": dataset_sha256,
                    "source_read_only": str(source_root),
                    "patches": [asdict(item) for item in patches],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        split_manifest = destination_root / "split-manifest.json"
        split_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "certification_history": [],
                    "spec_version": SPEC_VERSION,
                    "spec_sha256": spec_sha256,
                    "dataset_yaml": str(dataset_yaml_path),
                    "dataset_yaml_sha256": dataset_sha256,
                    "source_read_only": str(source_root),
                    "seed": seed,
                    "train_ratio": train_ratio,
                    "group_key": "prefix_before_.rf.",
                    "assignment": assignment,
                    "train_groups": sorted(train_groups),
                    "val_groups": sorted(val_groups),
                    "test_split": None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        # Count source negatives without creating any derived file.
        for record in records:
            for line_number, line in enumerate(record.label.read_text(encoding="utf-8").splitlines(), 1):
                for index, token in enumerate(line.split()[1:], 1):
                    if float(token) < 0:
                        patches.append(
                            CoordinatePatch(
                                str(record.label),
                                "",
                                line_number,
                                index,
                                float(token),
                                0.0,
                                _sha256(record.label),
                            )
                        )
        if expected_patch_count is not None and len(patches) != expected_patch_count:
            raise ValueError(
                f"預期修補 {expected_patch_count} 個負座標，實際找到 {len(patches)} 個"
            )

    return GroupedSplitReport(
        str(source_root),
        str(destination_root),
        seed,
        train_ratio,
        len(assignment),
        len(train_groups),
        len(val_groups),
        len(records),
        train_images,
        val_images,
        leakage,
        len(patches),
        str(patch_manifest) if patch_manifest else None,
        str(split_manifest) if split_manifest else None,
        execute,
    )



def recertify_grouped_pose_dataset(
    destination: str | Path,
    *,
    dataset_yaml: str | Path = DEFAULT_DATASET_YAML,
) -> dict[str, Any]:
    """保留舊 provenance，並用目前 spec/dataset YAML 重新認證相同資料內容。"""

    root = Path(destination).resolve()
    dataset_path = Path(dataset_yaml).resolve()
    for filename in ("split-manifest.json", "patch-manifest.json"):
        path = root / filename
        manifest = json.loads(path.read_text(encoding="utf-8"))
        previous = {
            "spec_version": manifest.get("spec_version"),
            "spec_sha256": manifest.get("spec_sha256"),
            "dataset_yaml": manifest.get("dataset_yaml"),
            "dataset_yaml_sha256": manifest.get("dataset_yaml_sha256"),
        }
        history = list(manifest.get("certification_history", []))
        current = {
            "spec_version": SPEC_VERSION,
            "spec_sha256": file_sha256(SPEC_PATH),
            "dataset_yaml": str(dataset_path),
            "dataset_yaml_sha256": file_sha256(dataset_path),
        }
        if previous != current and previous not in history:
            history.append(previous)
        manifest.update(current)
        manifest["schema_version"] = 3
        manifest["certification_history"] = history
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return validate_grouped_pose_dataset(root)


def validate_grouped_pose_dataset(destination: str | Path) -> dict[str, Any]:
    """驗證已建立資料集的隔離、數量、連結、patch 與規格來源。"""

    root = Path(destination).resolve()
    split_path = root / "split-manifest.json"
    patch_path = root / "patch-manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    expected_spec_sha = file_sha256(SPEC_PATH)
    expected_dataset_sha = file_sha256(DEFAULT_DATASET_YAML)
    for name, manifest in (("split", split), ("patch", patch)):
        if manifest.get("spec_version") != SPEC_VERSION or manifest.get("spec_sha256") != expected_spec_sha:
            raise AssertionError(f"{name} manifest 的規格來源不符")
        if manifest.get("dataset_yaml_sha256") != expected_dataset_sha:
            raise AssertionError(f"{name} manifest 的 dataset YAML 雜湊不符")
    train_groups = set(split["train_groups"])
    val_groups = set(split["val_groups"])
    leakage = sorted(train_groups & val_groups)
    if leakage:
        raise AssertionError(f"偵測到來源群組洩漏：{leakage[:10]}")
    counts: dict[str, int] = {}
    for split_name in ("train", "val"):
        images = tuple((root / "images" / split_name).iterdir())
        labels = tuple((root / "labels" / split_name).glob("*.txt"))
        if len(images) != len(labels):
            raise AssertionError(f"{split_name} 的影像與標籤數量不相等")
        if not all(image.is_symlink() for image in images):
            raise AssertionError(f"{split_name} 含有非 symlink 影像")
        counts[f"{split_name}_images"] = len(images)
        counts[f"{split_name}_labels"] = len(labels)
    if (root / "images/test").exists() or (root / "labels/test").exists():
        raise AssertionError("來源沒有 test split，不得建立 test 目錄")
    for item in patch["patches"]:
        if file_sha256(item["source"]) != item["source_sha256"]:
            raise AssertionError(f"來源標籤在分割後被改動：{item['source']}")
        if float(item["old"]) >= 0 or float(item["new"]) != 0:
            raise AssertionError("patch manifest 含無效座標修補")
    return {
        "valid": True,
        "destination": str(root),
        "groups": len(split["assignment"]),
        "leakage": leakage,
        "patched_coordinates": len(patch["patches"]),
        "image_symlinks": counts["train_images"] + counts["val_images"],
        **counts,
        "test_split": None,
        "source_untouched": True,
        "spec_version": SPEC_VERSION,
        "spec_sha256": expected_spec_sha,
        "dataset_yaml_sha256": expected_dataset_sha,
    }
