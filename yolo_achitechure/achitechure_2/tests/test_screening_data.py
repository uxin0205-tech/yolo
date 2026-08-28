from __future__ import annotations

from pathlib import Path

import pytest

from achitechure_2.screening_data import (
    prepare_screening_data,
    validate_screening_data,
)


def _make_image(root: Path, relative: str, class_id: int = 0) -> str:
    image = root / "images" / relative
    label = root / "labels" / Path(relative).with_suffix(".txt")
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fixture")
    label.write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return str(image.resolve())


def _write_list(path: Path, entries: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return path


def test_screening_manifests_are_fixed_grouped_and_do_not_touch_formal_val(
    tmp_path: Path,
) -> None:
    coco_root = tmp_path / "coco"
    coco = [
        _make_image(coco_root, f"train/{index:04d}.jpg", index % 2)
        for index in range(20)
    ]
    pose_root = tmp_path / "pose"
    detect_root = tmp_path / "detect"
    pose: list[str] = []
    detect: list[str] = []
    for group in range(5):
        for variant in range(2):
            name = f"source{group}.rf.variant{variant}.jpg"
            pose.append(_make_image(pose_root, f"train/{name}", group % 2))
            detect.append(_make_image(detect_root, f"train/{name}", group % 2))
    pose_val = [
        _make_image(pose_root, f"val/heldout.rf.variant{index}.jpg", index % 2)
        for index in range(2)
    ]
    sources = {
        "coco_train_list": _write_list(tmp_path / "coco-train.txt", coco),
        "bbat5_pose_search_train": _write_list(tmp_path / "pose-train.txt", pose),
        "bbat5_detect_search_train": _write_list(
            tmp_path / "detect-train.txt", detect
        ),
        "bbat5_pose_search_val": _write_list(tmp_path / "pose-val.txt", pose_val),
    }
    destination = tmp_path / "screen"

    planned = prepare_screening_data(
        **sources,
        destination=destination,
        fraction=0.2,
        coco_search_val_size=3,
        seed=0,
    )
    assert not planned.executed
    assert not destination.exists()
    assert planned.coco_train_count == 4
    assert planned.bbat5_train_count == 2

    created = prepare_screening_data(
        **sources,
        destination=destination,
        fraction=0.2,
        coco_search_val_size=3,
        seed=0,
        execute=True,
    )
    assert created.executed
    validation = validate_screening_data(destination)
    assert validation["valid"]
    assert validation["coco_train_search_val_overlap"] == 0
    assert validation["bbat5_train_search_val_group_leakage"] == 0
    assert validation["pose_detect_assignment_equal"]
    assert validation["formal_validation_used"] is False
    assert validation["canonical_data_modified"] is False

    pose_names = {
        Path(line).name
        for line in (destination / "bbat5/pose-train.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    detect_names = {
        Path(line).name
        for line in (destination / "bbat5/detect-train.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert pose_names == detect_names
    assert len({name.partition(".rf.")[0] for name in pose_names}) == 1

    with pytest.raises(FileExistsError, match="禁止覆寫"):
        prepare_screening_data(
            **sources,
            destination=destination,
            fraction=0.2,
            coco_search_val_size=3,
            seed=0,
            execute=True,
        )
