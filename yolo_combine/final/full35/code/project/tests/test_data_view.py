from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from yolo_combine.data import (
    CanonicalBBAT5,
    DatasetContractError,
    prepare_bbt5_view,
    prepare_coco_detect_subset,
)

POSE_LABEL = "0 0.5 0.5 0.1 0.1 0.4 0.5 2 0.6 0.5 2\n"
DETECT_LABEL = "0 0.5 0.5 0.1 0.1\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pair(
    root: Path, split: str, stem: str, pose_label: str = POSE_LABEL
) -> None:
    for task, label in (("pose", pose_label), ("detect", DETECT_LABEL)):
        image_dir = root / task / "images" / split
        label_dir = root / task / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{stem}.jpg").write_bytes(f"image-{stem}".encode())
        (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")


def _write_reference(path: Path, content: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "sha256": _sha256(path)}


def _registry(tmp_path: Path, *, leaked: bool = False, negative: bool = False) -> Path:
    root = tmp_path / "canonical" / "bbat5-v1"
    train_stem = "shared.rf.train" if leaked else "train-source.rf.train"
    val_stem = "shared.rf.val" if leaked else "val-source.rf.val"
    train_label = (
        "0 0.5 0.5 0.1 0.1 -0.001 0.5 2 0.6 0.5 2\n" if negative else POSE_LABEL
    )
    _write_pair(root, "train", train_stem, train_label)
    _write_pair(root, "val", val_stem)

    configs = root / "configs"
    configs.mkdir(parents=True)
    pose_yaml = configs / "pose.yaml"
    detect_yaml = configs / "detect.yaml"
    pose_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root / "pose"),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "ball", 1: "bat"},
                "kpt_shape": [2, 3],
                "flip_idx": [0, 1],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    detect_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root / "detect"),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "ball", 1: "bat"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pose_search = configs / "pose-search.yaml"
    detect_search = configs / "detect-search.yaml"
    pose_search.write_text("kind: pose-search\n", encoding="utf-8")
    detect_search.write_text("kind: detect-search\n", encoding="utf-8")

    manifest_names = ("source_audit", "patch", "coco_exclusion", "rebuild")
    manifests = {
        name: _write_reference(root / "manifests" / f"{name}.json", "{}\n")
        for name in manifest_names
    }
    split_assignment = (
        {"shared": "train"}
        if leaked
        else {
            "train-source": "train",
            "val-source": "val",
        }
    )
    manifests["split"] = _write_reference(
        root / "manifests" / "split.json",
        json.dumps({"formal": {"assignment": split_assignment}}, sort_keys=True) + "\n",
    )
    registry = tmp_path / "bbat5-v1.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset_id": "bbat5-v1",
                "aliases": ["BBAT5", "BBT5"],
                "status": "canonical",
                "root": str(root),
                "spec": {"version": "test", "sha256": "a" * 64},
                "source_archives": {
                    "pose": str(tmp_path / "source-pose"),
                    "detect_audit": str(tmp_path / "source-detect"),
                },
                "tasks": {
                    "pose": {
                        "data_yaml": str(pose_yaml),
                        "data_yaml_sha256": _sha256(pose_yaml),
                    },
                    "detect_2class": {
                        "data_yaml": str(detect_yaml),
                        "data_yaml_sha256": _sha256(detect_yaml),
                    },
                },
                "search_tasks": {
                    "pose": {
                        "data_yaml": str(pose_search),
                        "data_yaml_sha256": _sha256(pose_search),
                    },
                    "detect_2class": {
                        "data_yaml": str(detect_search),
                        "data_yaml_sha256": _sha256(detect_search),
                    },
                },
                "manifests": manifests,
                "counts": {
                    "images": 2,
                    "source_groups": 1 if leaked else 2,
                    "patched_coordinates": 1,
                    "instances": {"ball": 2, "bat": 0},
                },
                "splits": {
                    "formal": {
                        "train_images": 1,
                        "val_images": 1,
                        "source_group_overlap": 0,
                    },
                    "search": {
                        "train_images": 1,
                        "val_images": 1,
                        "source_group_overlap": 0,
                        "scope": "formal_train_only",
                    },
                    "test": None,
                },
                "policy": {
                    "canonical_mutability": "immutable",
                    "source_archives": "read_only",
                    "runtime_views": "generated_symlink_only",
                    "github_snapshot": "distribution_only",
                    "training_default": "formal",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return registry


def test_runtime_view_is_idempotent_symlink_only_and_traceable(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    destination = tmp_path / "runtime"

    first = prepare_bbt5_view(registry, destination)
    second = prepare_bbt5_view(registry, destination)

    assert first == second
    assert first.dataset_id == "bbat5-v1"
    assert first.images == first.labels == 2
    assert first.source_patched_coordinates == 1
    for relative in (
        "train/images/train-source.rf.train.jpg",
        "train/labels/train-source.rf.train.txt",
        "val/images/val-source.rf.val.jpg",
        "val/labels/val-source.rf.val.txt",
    ):
        assert (destination / relative).is_symlink()
    data = yaml.safe_load(first.yaml.read_text(encoding="utf-8"))
    assert data["dataset_id"] == "bbat5-v1"
    assert data["source_registry"] == str(registry)
    manifest = json.loads(first.manifest.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["storage"] == "symlink-only-runtime-view"
    assert manifest["source_group_overlap"] == []


def test_registry_fails_closed_on_yaml_hash_drift(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    pose_yaml = Path(payload["tasks"]["pose"]["data_yaml"])
    pose_yaml.write_text(pose_yaml.read_text() + "changed: true\n", encoding="utf-8")

    with pytest.raises(DatasetContractError, match="SHA256 drifted"):
        CanonicalBBAT5.load(registry)


@pytest.mark.parametrize(
    ("leaked", "negative", "message"),
    ((True, False, "leaks source groups"), (False, True, "negative coordinates")),
)
def test_runtime_view_rejects_noncanonical_content(
    tmp_path: Path, leaked: bool, negative: bool, message: str
) -> None:
    registry = _registry(tmp_path, leaked=leaked, negative=negative)

    with pytest.raises(DatasetContractError, match=message):
        prepare_bbt5_view(registry, tmp_path / "runtime")


def test_coco_subset_keeps_images_and_labels_read_only_and_cache_local(tmp_path):
    source_parent = tmp_path / "source"
    source = source_parent / "coco"
    image_dir = source / "images" / "train"
    label_dir = source / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for stem in ("one", "two"):
        (image_dir / f"{stem}.jpg").write_bytes(f"image-{stem}".encode())
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    (source / "train.txt").write_text(
        "./images/train/one.jpg\n./images/train/two.jpg\n",
        encoding="utf-8",
    )
    source_yaml = source_parent / "coco.yaml"
    source_yaml.write_text(
        yaml.safe_dump(
            {
                "path": "coco",
                "train": "train.txt",
                "val": "train.txt",
                "names": {0: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "variant" / "artifacts" / "coco_smoke"

    first = prepare_coco_detect_subset(source_yaml, destination, limit=2)
    second = prepare_coco_detect_subset(source_yaml, destination, limit=2)

    assert first == second
    assert first.images == first.labels == 2
    assert first.backgrounds == 1
    assert (destination / "train/images/one.jpg").is_symlink()
    assert (destination / "train/labels/one.txt").is_symlink()
    assert (destination / "train/labels/two.txt").read_text() == ""
    assert not (source / "labels/train.cache").exists()
    data = yaml.safe_load(first.yaml.read_text(encoding="utf-8"))
    assert data["path"] == str(destination)
    manifest = json.loads(first.manifest.read_text(encoding="utf-8"))
    assert manifest["source_images"] == manifest["selected_images"] == 2
