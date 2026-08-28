from __future__ import annotations

from pathlib import Path

from achitechure_2.screening_data import prepare_screening_data


def _list(path: Path, entries: list[Path]) -> Path:
    path.write_text(
        "\n".join(str(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    return path


def test_screening_keeps_pose_and_detect_symlink_view_paths(tmp_path: Path) -> None:
    raw_images = tmp_path / "raw/images"
    raw_images.mkdir(parents=True)
    pose_images = tmp_path / "pose/images/train"
    detect_images = tmp_path / "detect/images/train"
    pose_labels = tmp_path / "pose/labels/train"
    detect_labels = tmp_path / "detect/labels/train"
    for directory in (pose_images, detect_images, pose_labels, detect_labels):
        directory.mkdir(parents=True)

    pose_entries: list[Path] = []
    detect_entries: list[Path] = []
    for index in range(4):
        name = f"source{index}.rf.variant.jpg"
        raw = raw_images / name
        raw.write_bytes(b"image")
        pose_image = pose_images / name
        detect_image = detect_images / name
        pose_image.symlink_to(raw)
        detect_image.symlink_to(raw)
        (pose_labels / name.replace(".jpg", ".txt")).write_text(
            "0 0.5 0.5 0.2 0.2 0.4 0.4 2\n",
            encoding="utf-8",
        )
        (detect_labels / name.replace(".jpg", ".txt")).write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
        pose_entries.append(pose_image)
        detect_entries.append(detect_image)

    val_image = pose_images / "heldout.rf.variant.jpg"
    val_raw = raw_images / val_image.name
    val_raw.write_bytes(b"image")
    val_image.symlink_to(val_raw)
    (pose_labels / "heldout.rf.variant.txt").write_text(
        "1 0.5 0.5 0.2 0.2 0.6 0.6 2\n",
        encoding="utf-8",
    )
    coco_root = tmp_path / "coco"
    (coco_root / "images").mkdir(parents=True)
    (coco_root / "labels").mkdir()
    coco_entries: list[Path] = []
    for index in range(10):
        image = coco_root / f"images/{index}.jpg"
        image.write_bytes(b"image")
        (coco_root / f"labels/{index}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
        coco_entries.append(image)

    output = tmp_path / "screen"
    prepare_screening_data(
        coco_train_list=_list(tmp_path / "coco.txt", coco_entries),
        bbat5_pose_search_train=_list(tmp_path / "pose.txt", pose_entries),
        bbat5_detect_search_train=_list(tmp_path / "detect.txt", detect_entries),
        bbat5_pose_search_val=_list(tmp_path / "val.txt", [val_image]),
        destination=output,
        fraction=0.5,
        coco_search_val_size=2,
        execute=True,
    )
    pose_output = (output / "bbat5/pose-train.txt").read_text(encoding="utf-8")
    detect_output = (output / "bbat5/detect-train.txt").read_text(encoding="utf-8")
    assert "/pose/images/" in pose_output
    assert "/detect/images/" in detect_output
    assert pose_output != detect_output
