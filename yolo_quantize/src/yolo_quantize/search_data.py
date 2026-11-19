"""Rebuildable BBAT5 search Runtime Dataset View."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_generated(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"generated file differs: {path}")
        return
    path.write_text(content, encoding="utf-8")


def _source_group(path: Path) -> str:
    return path.name.split(".rf.", maxsplit=1)[0]


def _listed_images(list_path: Path, *, root: Path) -> tuple[Path, ...]:
    images: list[Path] = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        configured = Path(value).expanduser()
        image = configured if configured.is_absolute() else root / configured
        image = Path(os.path.abspath(image))
        if not image.is_file():
            raise FileNotFoundError(image)
        if image.parent != root / "images/train":
            raise ValueError(f"search image is outside canonical formal train: {image}")
        images.append(image)
    if len(set(images)) != len(images):
        raise ValueError(f"search split contains duplicate images: {list_path}")
    return tuple(images)


def _label_for(image: Path, *, root: Path) -> Path:
    label = root / "labels/train" / image.with_suffix(".txt").name
    if not label.is_file():
        raise FileNotFoundError(label)
    return label


@dataclass(frozen=True)
class PreparedBBAT5SearchView:
    """Identity and counts of one symlink-only search View."""

    root: Path
    yaml: Path
    manifest: Path
    train_images: int
    val_images: int


def prepare_bbat5_search_view(
    source_yaml: str | Path,
    destination: str | Path,
    *,
    expected_train: int = 5364,
    expected_val: int = 600,
) -> PreparedBBAT5SearchView:
    """Materialize the fixed BBAT5 search assignment without copying data."""

    yaml_path = Path(source_yaml).expanduser().resolve()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("BBAT5 search YAML must be a mapping")
    if payload.get("names") != {0: "ball", 1: "bat"}:
        raise ValueError("BBAT5 search class schema must be ball=0, bat=1")
    if payload.get("kpt_shape") != [2, 3]:
        raise ValueError("BBAT5 search kpt_shape must be [2, 3]")
    root = Path(str(payload.get("path", ""))).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    def split_list(name: str) -> Path:
        configured = Path(str(payload.get(name, ""))).expanduser()
        result = configured if configured.is_absolute() else root / configured
        result = result.resolve()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result

    train_list = split_list("train")
    val_list = split_list("val")
    split_images = {
        "train": _listed_images(train_list, root=root),
        "val": _listed_images(val_list, root=root),
    }
    actual_counts = {name: len(paths) for name, paths in split_images.items()}
    expected_counts = {"train": expected_train, "val": expected_val}
    if actual_counts != expected_counts:
        raise ValueError(
            f"BBAT5 search counts changed: {actual_counts} != {expected_counts}"
        )
    if set(split_images["train"]) & set(split_images["val"]):
        raise ValueError("BBAT5 search train and val images overlap")
    train_groups = {_source_group(path) for path in split_images["train"]}
    val_groups = {_source_group(path) for path in split_images["val"]}
    overlap = sorted(train_groups & val_groups)
    if overlap:
        raise ValueError(f"BBAT5 search source groups overlap: {overlap[:5]}")

    target = Path(destination).expanduser().resolve()
    if target == root or root in target.parents:
        raise ValueError("runtime destination must not be inside canonical BBAT5")
    target.mkdir(parents=True, exist_ok=True)
    assignment: dict[str, str] = {}
    for split, images in split_images.items():
        image_dir = target / split / "images"
        label_dir = target / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        expected_image_names: set[str] = set()
        expected_label_names: set[str] = set()
        for image in images:
            label = _label_for(image, root=root)
            expected_image_names.add(image.name)
            expected_label_names.add(label.name)
            assignment[str(image)] = split
            for source, link in (
                (image, image_dir / image.name),
                (label, label_dir / label.name),
            ):
                if link.exists() or link.is_symlink():
                    if not link.is_symlink() or link.readlink() != source:
                        raise FileExistsError(f"unexpected runtime entry: {link}")
                else:
                    link.symlink_to(source)
        if {path.name for path in image_dir.iterdir()} != expected_image_names:
            raise FileExistsError(
                f"runtime image View contains stale entries: {image_dir}"
            )
        if {path.name for path in label_dir.iterdir()} != expected_label_names:
            raise FileExistsError(
                f"runtime label View contains stale entries: {label_dir}"
            )

    runtime_payload = {
        "path": str(target),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "ball", 1: "bat"},
        "kpt_shape": [2, 3],
        "flip_idx": [0, 1],
        "dataset_id": "bbat5-v1",
        "view_role": "canonical-search-runtime-view",
        "source_yaml": str(yaml_path),
    }
    output_yaml = target / "data.yaml"
    _write_generated(
        output_yaml,
        yaml.safe_dump(runtime_payload, sort_keys=False, allow_unicode=True),
    )
    manifest_payload = {
        "schema_version": 1,
        "dataset_id": "bbat5-v1",
        "view_role": "canonical-search-runtime-view",
        "source_yaml": str(yaml_path),
        "source_yaml_sha256": _sha256(yaml_path),
        "source_lists": {
            "train": {"path": str(train_list), "sha256": _sha256(train_list)},
            "val": {"path": str(val_list), "sha256": _sha256(val_list)},
        },
        "split_counts": actual_counts,
        "source_group_overlap": [],
        "assignment_changed": False,
        "assignment_sha256": _json_sha256(assignment),
        "storage": "symlink-only-runtime-view",
    }
    output_manifest = target / "manifest.json"
    _write_generated(
        output_manifest,
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    return PreparedBBAT5SearchView(
        root=target,
        yaml=output_yaml,
        manifest=output_manifest,
        train_images=actual_counts["train"],
        val_images=actual_counts["val"],
    )
