from __future__ import annotations

import json
from pathlib import Path

import yaml

from achitechure_1.ball_bat_evaluation import (
    convert_pose_label_line,
    prepare_ball_bat_detection_validation,
)


def test_pose_label_conversion_remaps_ball_and_bat_to_coco80() -> None:
    ball = "0 0.5 0.4 0.1 0.2 0.4 0.4 2 0.6 0.4 2"
    bat = "1 0.3 0.2 0.2 0.1 0.2 0.2 2 0.4 0.2 2"

    assert convert_pose_label_line(ball) == "32 0.5 0.4 0.1 0.2"
    assert convert_pose_label_line(bat) == "34 0.3 0.2 0.2 0.1"


def test_prepare_ball_bat_validation_preserves_source_and_records_overlap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "valid/images").mkdir(parents=True)
    (source / "valid/labels").mkdir(parents=True)
    (source / "valid/images/000000000123_jpg.rf.hash.jpg").write_bytes(b"image")
    original = "0 0.5 0.4 0.1 0.2 0.4 0.4 2 0.6 0.4 2\n"
    (source / "valid/labels/000000000123_jpg.rf.hash.txt").write_text(original, encoding="utf-8")
    (source / "data.yaml").write_text(
        yaml.safe_dump({"names": ["ball", "bat"], "kpt_shape": [2, 3]}), encoding="utf-8"
    )
    coco_root = tmp_path / "coco"
    (coco_root / "images/train2017").mkdir(parents=True)
    (coco_root / "images/train2017/000000000123.jpg").write_bytes(b"coco")
    names = [f"class-{index}" for index in range(80)]
    names[32] = "sports ball"
    names[34] = "baseball bat"
    coco_yaml = tmp_path / "coco.yaml"
    coco_yaml.write_text(yaml.safe_dump({"path": str(coco_root), "names": names}), encoding="utf-8")

    output = tmp_path / "derived"
    data = prepare_ball_bat_detection_validation(
        source_root=source, output_root=output, coco_data=coco_yaml
    )

    assert (output / "valid/images").is_dir()
    assert (output / "valid/images/000000000123_jpg.rf.hash.jpg").is_symlink()
    assert (output / "valid/labels/000000000123_jpg.rf.hash.txt").read_text() == "32 0.5 0.4 0.1 0.2\n"
    assert (source / "valid/labels/000000000123_jpg.rf.hash.txt").read_text() == original
    derived_config = yaml.safe_load(data.read_text())
    assert derived_config["train"] == derived_config["val"] == "valid/images"
    assert derived_config["names"][34] == "baseball bat"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["instances"] == {"baseball_bat": 0, "sports_ball": 1}
    assert manifest["independence_warning"]["coco_train2017_id_overlap_images"] == 1
