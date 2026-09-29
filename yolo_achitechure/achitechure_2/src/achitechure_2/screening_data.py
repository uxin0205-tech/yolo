"""建立不可變的 20% 架構篩選清單，不複製或修改任何資料。"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_image_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError(f"影像清單是空的：{path}")
    resolved: list[str] = []
    for line in lines:
        image = Path(line)
        if not image.is_absolute():
            image = path.parent / image
        resolved.append(os.path.abspath(image))
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"影像清單包含重複項目：{path}")
    return resolved


def _group_key(image: str) -> str:
    name = Path(image).name
    prefix, marker, _ = name.partition(".rf.")
    if not marker or not prefix:
        raise ValueError(f"BBAT5 檔名缺少 .rf. 群組標記：{name}")
    return prefix


def _label_path(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"影像路徑缺少 images 目錄：{path}") from error
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _class_statistics(images: list[str]) -> dict[str, Any]:
    instances: Counter[int] = Counter()
    images_with_class: Counter[int] = Counter()
    missing_labels = 0
    invalid_rows = 0
    for image in images:
        label = _label_path(image)
        if not label.is_file():
            missing_labels += 1
            continue
        classes_in_image: set[int] = set()
        for row in label.read_text(encoding="utf-8").splitlines():
            fields = row.split()
            if not fields:
                continue
            try:
                class_id = int(float(fields[0]))
            except ValueError:
                invalid_rows += 1
                continue
            instances[class_id] += 1
            classes_in_image.add(class_id)
        images_with_class.update(classes_in_image)
    return {
        "instances_by_class": {
            str(class_id): count for class_id, count in sorted(instances.items())
        },
        "images_by_class": {
            str(class_id): count
            for class_id, count in sorted(images_with_class.items())
        },
        "missing_label_files": missing_labels,
        "invalid_label_rows": invalid_rows,
    }


def _select_grouped(
    images: list[str], fraction: float, seed: int
) -> tuple[list[str], int]:
    groups: dict[str, list[str]] = defaultdict(list)
    for image in images:
        groups[_group_key(image)].append(image)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = round(len(images) * fraction)
    selected_keys: list[str] = []
    selected_count = 0
    for key in keys:
        next_count = selected_count + len(groups[key])
        if (
            selected_count >= target
            and abs(selected_count - target) <= abs(next_count - target)
        ):
            break
        selected_keys.append(key)
        selected_count = next_count
    selected = {image for key in selected_keys for image in groups[key]}
    return [image for image in images if image in selected], len(selected_keys)


@dataclass(frozen=True)
class ScreeningDataReport:
    destination: str
    executed: bool
    seed: int
    requested_fraction: float
    coco_source_count: int
    coco_train_count: int
    coco_search_val_count: int
    bbat5_source_count: int
    bbat5_train_count: int
    bbat5_group_count: int
    bbat5_search_val_count: int
    canonical_data_modified: bool
    pose_detect_assignment_equal: bool
    formal_validation_used: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_screening_data(
    *,
    coco_train_list: str | Path,
    bbat5_pose_search_train: str | Path,
    bbat5_detect_search_train: str | Path,
    bbat5_pose_search_val: str | Path,
    destination: str | Path,
    fraction: float = 0.2,
    coco_search_val_size: int = 5000,
    seed: int = 0,
    execute: bool = False,
) -> ScreeningDataReport:
    """從既有 train assignment 產生固定清單；正式 val 永遠不參與。"""

    if not 0 < fraction < 1:
        raise ValueError("fraction 必須介於 0 與 1 之間")
    if coco_search_val_size < 1:
        raise ValueError("coco_search_val_size 必須為正整數")

    coco_path = Path(coco_train_list).resolve()
    pose_train_path = Path(bbat5_pose_search_train).resolve()
    detect_train_path = Path(bbat5_detect_search_train).resolve()
    pose_val_path = Path(bbat5_pose_search_val).resolve()
    output = Path(destination).resolve()

    coco = _read_image_list(coco_path)
    pose_train = _read_image_list(pose_train_path)
    detect_train = _read_image_list(detect_train_path)
    pose_val = _read_image_list(pose_val_path)

    pose_by_name = {Path(path).name: path for path in pose_train}
    detect_by_name = {Path(path).name: path for path in detect_train}
    if set(pose_by_name) != set(detect_by_name):
        raise ValueError("BBAT5 Pose/Detect search-train assignment 不一致")

    coco_shuffled = list(coco)
    random.Random(seed).shuffle(coco_shuffled)
    coco_train_count = round(len(coco) * fraction)
    if coco_train_count + coco_search_val_size > len(coco):
        raise ValueError("COCO 20% train 與 search-val 超過來源 train 清單")
    coco_selected = sorted(coco_shuffled[:coco_train_count])
    coco_search_val = sorted(
        coco_shuffled[coco_train_count : coco_train_count + coco_search_val_size]
    )

    pose_selected, group_count = _select_grouped(pose_train, fraction, seed)
    selected_names = {Path(path).name for path in pose_selected}
    detect_selected = [
        path for path in detect_train if Path(path).name in selected_names
    ]
    if {Path(path).name for path in detect_selected} != selected_names:
        raise ValueError("BBAT5 20% Pose/Detect manifest 無法保持相同 assignment")
    selected_groups = {_group_key(path) for path in pose_selected}
    val_groups = {_group_key(path) for path in pose_val}
    leakage = selected_groups & val_groups
    if leakage:
        raise ValueError(
            f"BBAT5 20% train/search-val 發生 group leakage：{sorted(leakage)[:5]}"
        )

    report = ScreeningDataReport(
        destination=str(output),
        executed=execute,
        seed=seed,
        requested_fraction=fraction,
        coco_source_count=len(coco),
        coco_train_count=len(coco_selected),
        coco_search_val_count=len(coco_search_val),
        bbat5_source_count=len(pose_train),
        bbat5_train_count=len(pose_selected),
        bbat5_group_count=group_count,
        bbat5_search_val_count=len(pose_val),
        canonical_data_modified=False,
        pose_detect_assignment_equal=True,
        formal_validation_used=False,
    )
    if not execute:
        return report
    if output.exists():
        raise FileExistsError(f"screening view 已存在，禁止覆寫：{output}")

    from .config import SPEC_PATH, SPEC_VERSION, file_sha256

    files = {
        "coco/train.txt": coco_selected,
        "coco/search-val.txt": coco_search_val,
        "bbat5/pose-train.txt": pose_selected,
        "bbat5/detect-train.txt": detect_selected,
    }
    for relative, entries in files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(entries) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "view_id": "architecture-screen-20-v1",
        "purpose": "C0-Control/C1/C2/C3 Float 20% 與 QAT-lite 篩選",
        "screening_only": True,
        "canonical_data_modified": False,
        "formal_validation_used": False,
        "sampling": {
            "seed": seed,
            "requested_fraction": fraction,
            "coco": "random_without_replacement",
            "bbat5": "grouped_random_without_replacement_by_prefix_before_.rf.",
        },
        "sources": {
            "coco_train": {"path": str(coco_path), "sha256": _sha256(coco_path)},
            "bbat5_pose_search_train": {
                "path": str(pose_train_path),
                "sha256": _sha256(pose_train_path),
            },
            "bbat5_detect_search_train": {
                "path": str(detect_train_path),
                "sha256": _sha256(detect_train_path),
            },
            "bbat5_pose_search_val": {
                "path": str(pose_val_path),
                "sha256": _sha256(pose_val_path),
            },
        },
        "outputs": {
            relative: {
                "count": len(entries),
                "sha256": _sha256(output / relative),
                "class_statistics": _class_statistics(entries),
            }
            for relative, entries in files.items()
        },
        "bbat5_group_count": group_count,
        "pose_detect_assignment_equal": True,
        "bbat5_train_search_val_group_leakage": 0,
    }
    (output / "screening-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# architecture-screen-20-v1

這是 architecture_2 的可重建篩選 View，不是新的 canonical dataset，也沒有複製影像或標註。

- 目的：供 C0-Control、C1、C2、C3 使用相同資料做 Float 20% 初篩，並供通過候選做 PTQ／QAT-lite 模擬。
- COCO：從 train2017 以 seed {seed} 隨機抽取 {len(coco_selected):,} 張作訓練，另取互斥的 {len(coco_search_val):,} 張 train-only search-val；官方 val2017 未使用。
- BBAT5：從 canonical bbat5-v1 的 search-train 依 `.rf.` 前 prefix 整組抽取 {len(pose_selected):,} 張，共 {group_count:,} groups；沿用既有 search-val {len(pose_val):,} 張。
- Pose 與二類 Detect 使用相同影像 assignment。
- `fraction` 在 training YAML 保持 1.0；20% 由這些固定 manifest 決定。
- 禁止用此 View 宣稱正式 C_best，正式 val 仍保留給完整資料確認。
- 原始與 canonical 資料修改：無。

完整來源雜湊、輸出雜湊、類別統計與 leakage 證據見 `screening-manifest.json`。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return report


def validate_screening_data(destination: str | Path) -> dict[str, Any]:
    """驗證已生成清單的 hash、任務 assignment 與 BBAT5 group leakage。"""

    root = Path(destination).resolve()
    manifest_path = root / "screening-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("view_id") != "architecture-screen-20-v1"
        or manifest.get("screening_only") is not True
        or manifest.get("canonical_data_modified") is not False
        or manifest.get("formal_validation_used") is not False
    ):
        raise ValueError("screening manifest 固定語意漂移")
    for relative, metadata in manifest["outputs"].items():
        path = root / relative
        if not path.is_file() or _sha256(path) != metadata["sha256"]:
            raise ValueError(f"screening output hash 不符：{relative}")
        if len(_read_image_list(path)) != metadata["count"]:
            raise ValueError(f"screening output count 不符：{relative}")

    pose = _read_image_list(root / "bbat5/pose-train.txt")
    detect = _read_image_list(root / "bbat5/detect-train.txt")
    pose_names = {Path(path).name for path in pose}
    detect_names = {Path(path).name for path in detect}
    if pose_names != detect_names:
        raise ValueError("screening Pose/Detect assignment 不一致")
    val_source = Path(manifest["sources"]["bbat5_pose_search_val"]["path"])
    if _sha256(val_source) != manifest["sources"]["bbat5_pose_search_val"]["sha256"]:
        raise ValueError("BBAT5 search-val source hash 已改變")
    val = _read_image_list(val_source)
    leakage = {_group_key(path) for path in pose} & {
        _group_key(path) for path in val
    }
    if leakage:
        raise ValueError(f"screening BBAT5 group leakage：{sorted(leakage)[:5]}")
    coco_train = set(_read_image_list(root / "coco/train.txt"))
    coco_val = set(_read_image_list(root / "coco/search-val.txt"))
    if coco_train & coco_val:
        raise ValueError("screening COCO train/search-val 重疊")
    return {
        "valid": True,
        "view_id": manifest["view_id"],
        "coco_train_count": len(coco_train),
        "coco_search_val_count": len(coco_val),
        "bbat5_train_count": len(pose),
        "bbat5_group_count": len({_group_key(path) for path in pose}),
        "bbat5_search_val_count": len(val),
        "pose_detect_assignment_equal": True,
        "bbat5_train_search_val_group_leakage": 0,
        "coco_train_search_val_overlap": 0,
        "canonical_data_modified": False,
        "formal_validation_used": False,
    }
