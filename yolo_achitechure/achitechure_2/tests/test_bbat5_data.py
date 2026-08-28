from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from achitechure_2.pose_data import (
    export_bbat5_github_dataset,
    export_bbat5_metadata,
    prepare_bbat5_dataset,
    source_group,
    validate_bbat5_dataset,
    validate_bbat5_github_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    pose: Path,
    detect: Path,
    split: str,
    name: str,
    label: str,
) -> None:
    for root in (pose, detect):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "images" / name).write_bytes(b"fixture-image")
    pose_label = pose / split / "labels" / f"{Path(name).stem}.txt"
    pose_label.write_text(label, encoding="utf-8")
    detect_lines = [" ".join(line.split()[:5]) for line in label.splitlines()]
    (detect / split / "labels" / pose_label.name).write_text(
        "\n".join(detect_lines) + ("\n" if detect_lines else ""),
        encoding="utf-8",
    )


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    pose = tmp_path / "pose-source"
    detect = tmp_path / "detect-source"
    base = "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.3 0.4 2\n"
    _record(
        pose,
        detect,
        "train",
        "000000000001_jpg.rf.aaa.jpg",
        "0 -0.001 0.5 0.2 0.2 0.1 0.2 2 0.3 0.4 2\n",
    )
    _record(pose, detect, "valid", "000000000001_jpg.rf.bbb.jpg", base)
    _record(pose, detect, "train", "source-B.rf.ccc.jpg", base)
    _record(pose, detect, "valid", "source-C.rf.ddd.jpg", base)
    _record(pose, detect, "train", "source-D.rf.eee.jpg", base)
    (pose / "data.yaml").write_text("names: [ball, bat]\n", encoding="utf-8")
    (detect / "data.yaml").write_text("names: [ball, bat]\n", encoding="utf-8")
    coco = tmp_path / "train2017.txt"
    coco.write_text("./images/train2017/000000000001.jpg\n", encoding="utf-8")
    return pose, detect, coco


def test_prepare_bbat5_builds_two_views_and_zero_leakage_search(tmp_path: Path) -> None:
    pose, detect, coco = _sources(tmp_path)
    destination = tmp_path / "derived" / "bbat5-v1"
    source_label = pose / "train/labels/000000000001_jpg.rf.aaa.txt"
    before = _sha256(source_label)

    report = prepare_bbat5_dataset(
        pose,
        detect,
        destination,
        coco_train_list=coco,
        execute=True,
        expected_patch_count=1,
    )

    assert report.executed
    assert report.formal_ready
    assert report.images == 5
    assert report.groups == 4
    assert report.patched_coordinates == 1
    assert report.coco_train_overlap_groups == 1
    assert _sha256(source_label) == before
    assert "-0.001" in source_label.read_text(encoding="utf-8")
    assert source_group("000000000001_jpg.rf.aaa.jpg") == "000000000001_jpg"

    split = json.loads(
        (destination / "manifests/split-manifest.json").read_text(encoding="utf-8")
    )
    exclusion = json.loads(
        (destination / "manifests/coco-exclusion-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(split["formal"]["train_groups"]).isdisjoint(
        split["formal"]["val_groups"]
    )
    assert set(split["search"]["train_groups"]).isdisjoint(
        split["search"]["val_groups"]
    )
    assert set(split["search"]["train_groups"]) | set(
        split["search"]["val_groups"]
    ) == set(split["formal"]["train_groups"])
    assert "000000000001_jpg" in split["formal"]["train_groups"]
    assert "000000000001_jpg" not in split["formal"]["val_groups"]
    assert "000000000001_jpg" not in split["search"]["val_groups"]
    assert exclusion["status"] == "passed"
    assert exclusion["formal_val_overlap_after_exclusion"] == []

    pose_output = destination / "pose/labels/train/000000000001_jpg.rf.aaa.txt"
    detect_output = destination / "detect/labels/train/000000000001_jpg.rf.aaa.txt"
    assert pose_output.read_text(encoding="utf-8").split()[1] == "0"
    assert detect_output.read_text(encoding="utf-8").split() == [
        "0",
        "0",
        "0.5",
        "0.2",
        "0.2",
    ]
    assert len(detect_output.read_text(encoding="utf-8").split()) == 5
    assert all(
        path.is_symlink()
        for view in ("pose", "detect")
        for split_name in ("train", "val")
        for path in (destination / view / "images" / split_name).iterdir()
    )
    assert not any(destination.glob("*/images/test"))
    assert not any(destination.glob("*/labels/test"))

    pose_yaml = yaml.safe_load(
        (destination / "configs/pose-search.yaml").read_text(encoding="utf-8")
    )
    detect_yaml = yaml.safe_load(
        (destination / "configs/detect.yaml").read_text(encoding="utf-8")
    )
    assert pose_yaml["kpt_shape"] == [2, 3]
    assert Path(pose_yaml["train"]).name == "search-train.txt"
    assert detect_yaml["names"] == {0: "ball", 1: "bat"}
    assert "不會啟動 Pose 訓練" in (destination / "README.md").read_text(
        encoding="utf-8"
    )

    validation = validate_bbat5_dataset(destination)
    assert validation["valid"]
    assert validation["formal_ready"]
    assert validation["pose_images"] == 5
    assert validation["detect_images"] == 5
    assert validation["test_split"] is None


def test_prepare_is_plan_only_by_default_and_fails_closed_without_coco(tmp_path: Path) -> None:
    pose, detect, _ = _sources(tmp_path)
    destination = tmp_path / "bbat5-v1"

    report = prepare_bbat5_dataset(
        pose,
        detect,
        destination,
        coco_train_list=tmp_path / "missing-train2017.txt",
        execute=False,
        expected_patch_count=1,
    )

    assert not report.executed
    assert not report.formal_ready
    assert report.coco_exclusion_status == "blocked"
    assert not destination.exists()


def test_derived_version_is_immutable(tmp_path: Path) -> None:
    pose, detect, coco = _sources(tmp_path)
    destination = tmp_path / "bbat5-v1"
    prepare_bbat5_dataset(
        pose,
        detect,
        destination,
        coco_train_list=coco,
        execute=True,
        expected_patch_count=1,
    )

    with pytest.raises(FileExistsError, match="不可覆寫"):
        prepare_bbat5_dataset(
            pose,
            detect,
            destination,
            coco_train_list=coco,
            execute=True,
            expected_patch_count=1,
        )


def test_detect_source_is_audit_only_and_mismatch_fails_closed(tmp_path: Path) -> None:
    pose, detect, coco = _sources(tmp_path)
    label = detect / "train/labels/source-B.rf.ccc.txt"
    label.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Detect.*一致性"):
        prepare_bbat5_dataset(
            pose,
            detect,
            tmp_path / "bbat5-v1",
            coco_train_list=coco,
            execute=False,
            expected_patch_count=1,
        )


def test_metadata_export_contains_only_readme_configs_and_manifests(tmp_path: Path) -> None:
    pose, detect, coco = _sources(tmp_path)
    source = tmp_path / "derived/bbat5-v1"
    destination = tmp_path / "git-metadata/bbat5-v1"
    prepare_bbat5_dataset(
        pose,
        detect,
        source,
        coco_train_list=coco,
        execute=True,
        expected_patch_count=1,
    )

    report = export_bbat5_metadata(source, destination, execute=True)

    assert report["execute"]
    assert len(report["files"]) == 10
    assert (destination / "README.md").is_file()
    assert not any(
        part in {"images", "labels"}
        for path in destination.rglob("*")
        for part in path.relative_to(destination).parts
    )
    assert not any(path.is_symlink() for path in destination.rglob("*"))
    with pytest.raises(FileExistsError, match="不可覆寫"):
        export_bbat5_metadata(source, destination, execute=True)


def test_github_export_materializes_portable_dataset_without_weights(
    tmp_path: Path,
) -> None:
    pose, detect, coco = _sources(tmp_path)
    source = tmp_path / "derived/bbat5-v1"
    destination = tmp_path / "git/bbat5-v1/github-dataset"
    prepare_bbat5_dataset(
        pose,
        detect,
        source,
        coco_train_list=coco,
        execute=True,
        expected_patch_count=1,
    )

    plan = export_bbat5_github_dataset(source, destination)
    assert not plan["execute"]
    assert not destination.exists()

    report = export_bbat5_github_dataset(source, destination, execute=True)

    assert report["execute"]
    assert report["counts"]["unique_images"] == 5
    assert report["validation"]["destination"] == str(destination.resolve())
    assert not any(path.is_symlink() for path in destination.rglob("*"))
    assert not any(
        path.suffix.lower() in {".pt", ".pth", ".onnx", ".engine"}
        for path in destination.rglob("*")
        if path.is_file()
    )
    pose_yaml = yaml.safe_load(
        (destination / "pose/data.yaml").read_text(encoding="utf-8")
    )
    assert "path" not in pose_yaml
    assert pose_yaml["train"] == "splits/formal-train.txt"
    search_lines = (
        destination / "pose/splits/search-train.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert search_lines
    assert all(line.startswith("./../images/train/") for line in search_lines)
    validation = validate_bbat5_github_dataset(destination)
    assert validation["valid"]
    assert validation["unique_images"] == 5
    assert validation["symlinks"] == 0
    assert validation["forbidden_files"] == 0
    assert validation["test_split"] is None
    with pytest.raises(FileExistsError, match="不可覆寫"):
        export_bbat5_github_dataset(source, destination, execute=True)


def test_github_dataset_validation_rejects_weight_file(tmp_path: Path) -> None:
    pose, detect, coco = _sources(tmp_path)
    source = tmp_path / "derived/bbat5-v1"
    destination = tmp_path / "github-dataset"
    prepare_bbat5_dataset(
        pose,
        detect,
        source,
        coco_train_list=coco,
        execute=True,
        expected_patch_count=1,
    )
    export_bbat5_github_dataset(source, destination, execute=True)
    (destination / "accidental.pt").write_bytes(b"forbidden")

    with pytest.raises(AssertionError, match="禁止檔案"):
        validate_bbat5_github_dataset(destination)


def test_validation_preserves_immutable_v1_historical_spec_lineage(tmp_path: Path) -> None:
    pose, detect, coco = _sources(tmp_path)
    destination = tmp_path / "derived/bbat5-v1"
    prepare_bbat5_dataset(
        pose,
        detect,
        destination,
        coco_train_list=coco,
        execute=True,
        expected_patch_count=1,
    )
    historical_sha = (
        "75db239262de75998171a05a85c8755d"
        "ea10c2bc76920dd2aabe8ff0dabb7a3b"
    )
    for manifest_path in (destination / "manifests").glob("*.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["spec_version"] = "2.0.0"
        payload["spec_sha256"] = historical_sha
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_bbat5_dataset(destination)

    assert validation["valid"]
    assert validation["spec_version"] == "2.0.0"
    assert validation["spec_sha256"] == historical_sha
