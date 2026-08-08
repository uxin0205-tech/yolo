from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from ultralytics.data.utils import img2label_paths
import yaml

from masf_yolo.data.audit import audit_dataset


def _write_sample(root: Path, split: str, index: int) -> None:
    images = root / split / "images"
    labels = root / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    name = f"source{index:03d}_mp4-0001_jpg.rf.{index:032x}"
    Image.new("RGB", (80 + index, 48), (index, index * 3 % 255, index * 7 % 255)).save(
        images / f"{name}.jpg"
    )
    (labels / f"{name}.txt").write_text(
        "0 0.5 0.5 0.01 0.01\n1 0.4 0.5 0.2 0.3\n", encoding="utf-8"
    )


def test_audit_writes_reproducible_manifests_profile_and_coco(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for index in range(20):
        _write_sample(source, "train" if index < 15 else "valid", index)
    output = tmp_path / "artifacts" / "dataset"

    first = audit_dataset(source, output, seed=42, minimum_ball_count=2)
    second = audit_dataset(source, output, seed=42, minimum_ball_count=2)

    assert first == second
    assert first.split_counts == {"train": 16, "val": 2, "test": 2}
    for name in ("train.txt", "val.txt", "test.txt", "data.yaml", "dataset_profile.json", "audit.json", "manifest.json", "val.coco.json", "test.coco.json"):
        assert (output / name).is_file()
    data_config = yaml.safe_load((output / "data.yaml").read_text())
    assert Path(data_config["path"]) == output.resolve()
    assert data_config["val"] == "val.txt"
    audit = json.loads((output / "audit.json").read_text())
    assert audit["ok"] is True
    assert audit["group_overlap"] == []
    assert audit["hash_overlap"] == []
    coco = json.loads((output / "val.coco.json").read_text())
    assert coco["categories"] == [{"id": 0, "name": "ball"}, {"id": 1, "name": "bat"}]
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 4
    assert all(image["width"] == 640 and image["height"] == 640 for image in coco["images"])


def test_audit_preserves_detect_view_for_symlinked_images(tmp_path: Path) -> None:
    source = tmp_path / "detect-view"
    originals = tmp_path / "pose-source"
    for index in range(20):
        split = "train" if index < 15 else "valid"
        _write_sample(source, split, index)
        image = next((source / split / "images").glob(f"source{index:03d}_*"))
        original = originals / split / "images" / image.name
        original.parent.mkdir(parents=True, exist_ok=True)
        image.replace(original)
        image.symlink_to(original)

    output = tmp_path / "artifacts" / "dataset"
    audit_dataset(source, output, seed=42, minimum_ball_count=2)

    manifest_images = [
        Path(line)
        for split in ("train", "val", "test")
        for line in (output / f"{split}.txt").read_text().splitlines()
    ]
    assert manifest_images
    assert all(str(path).startswith(str(source.absolute())) for path in manifest_images)
    assert all(not str(path).startswith(str(originals.absolute())) for path in manifest_images)
    inferred_labels = [Path(path) for path in img2label_paths([str(path) for path in manifest_images])]
    assert all(path.parent.name == "labels" for path in inferred_labels)
    assert all(str(path).startswith(str(source.absolute())) for path in inferred_labels)
    assert all(path.is_file() for path in inferred_labels)
