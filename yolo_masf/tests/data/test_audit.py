from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

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
    audit = json.loads((output / "audit.json").read_text())
    assert audit["ok"] is True
    assert audit["group_overlap"] == []
    assert audit["hash_overlap"] == []
    coco = json.loads((output / "val.coco.json").read_text())
    assert coco["categories"] == [{"id": 0, "name": "ball"}, {"id": 1, "name": "bat"}]
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 4
    assert all(image["width"] == 640 and image["height"] == 640 for image in coco["images"])
