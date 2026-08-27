from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from yolo_combine.data import PreparedDataset
from yolo_combine.diagnostic_sampling import (
    DIAGNOSTIC_MARKER_NAME,
    DiagnosticSamplingPolicy,
    checkpoint_is_diagnostic,
    diagnostic_marker_for_checkpoint,
    mark_diagnostic_run,
    prepare_pose_diagnostic_view,
)
from yolo_combine.joint_config import JointExperimentConfig


def _prepared_runtime(tmp_path: Path) -> PreparedDataset:
    root = tmp_path / "runtime"
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    for group in range(4):
        for sibling in range(3):
            stem = f"source-{group}.rf.variant-{sibling}"
            (root / "train" / "images" / f"{stem}.jpg").write_bytes(b"image")
            (root / "train" / "labels" / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.1 0.1 0.5 0.5 2 0 0 0\n",
                encoding="utf-8",
            )
    for index in range(3):
        stem = f"validation-{index}"
        (root / "val" / "images" / f"{stem}.jpg").write_bytes(b"image")
        (root / "val" / "labels" / f"{stem}.txt").write_text(
            "1 0.5 0.5 0.1 0.1 0.4 0.5 2 0.6 0.5 2\n",
            encoding="utf-8",
        )
    data = {
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "ball", 1: "bat"},
        "kpt_shape": [2, 3],
        "flip_idx": [0, 1],
    }
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_id": "bbat5-v1",
                "split_counts": {"train": 12, "val": 3},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedDataset(
        dataset_id="bbat5-v1",
        root=root,
        yaml=data_yaml,
        manifest=manifest,
        source_yaml=data_yaml,
        images=15,
        labels=15,
        source_patched_coordinates=0,
    )


def test_pose_diagnostic_selection_is_seeded_grouped_and_keeps_full_val(
    tmp_path: Path,
) -> None:
    prepared = _prepared_runtime(tmp_path)
    policy = DiagnosticSamplingPolicy(fraction=0.5, seed=7)

    view = prepare_pose_diagnostic_view(
        prepared,
        tmp_path / "diagnostic",
        policy=policy,
    )
    repeated = prepare_pose_diagnostic_view(
        prepared,
        tmp_path / "diagnostic",
        policy=policy,
    )

    assert view == repeated
    assert view.formal_eligible is False
    assert view.full_train_images == 12
    assert view.selected_train_images == 6
    assert view.full_train_groups == 4
    assert view.selected_train_groups == 2
    assert view.full_validation_images == 3
    selected = [Path(line) for line in view.train_list.read_text().splitlines()]
    groups = {path.stem.split(".rf.", maxsplit=1)[0] for path in selected}
    assert len(groups) == 2
    assert all(
        sum(path.stem.startswith(group + ".rf.") for path in selected) == 3
        for group in groups
    )
    payload = yaml.safe_load(view.yaml.read_text())
    assert payload["val"] == "val/images"
    assert payload["diagnostic_only"] is True
    assert payload["formal_eligible"] is False


def test_pose_diagnostic_destination_is_fail_closed_on_policy_drift(
    tmp_path: Path,
) -> None:
    prepared = _prepared_runtime(tmp_path)
    destination = tmp_path / "diagnostic"
    prepare_pose_diagnostic_view(
        prepared,
        destination,
        policy=DiagnosticSamplingPolicy(fraction=0.5, seed=1),
    )

    with pytest.raises(FileExistsError):
        prepare_pose_diagnostic_view(
            prepared,
            destination,
            policy=DiagnosticSamplingPolicy(fraction=0.5, seed=2),
        )


def test_diagnostic_marker_is_discoverable_and_formal_preflight_rejects_it(
    tmp_path: Path,
) -> None:
    prepared = _prepared_runtime(tmp_path)
    view = prepare_pose_diagnostic_view(
        prepared,
        tmp_path / "diagnostic-view",
        policy=DiagnosticSamplingPolicy(fraction=0.5, seed=0),
    )
    run_dir = tmp_path / "pose" / "diagnostic" / "run"
    checkpoint = run_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    marker = mark_diagnostic_run(run_dir, view=view, stage="p1")

    assert marker.name == DIAGNOSTIC_MARKER_NAME
    assert diagnostic_marker_for_checkpoint(checkpoint) == marker
    config = replace(
        JointExperimentConfig.load("variants/full35/configs/joint.yaml"),
        pose_checkpoint=checkpoint,
    )
    report = config.preflight()
    assert report.ready is False
    assert any("fraction 診斷產物" in blocker for blocker in report.blockers)


def test_interrupted_run_in_diagnostic_tree_is_rejected_without_marker(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pose" / "diagnostic" / "run" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"partial checkpoint")

    assert diagnostic_marker_for_checkpoint(checkpoint) is None
    assert checkpoint_is_diagnostic(checkpoint) is True


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.1])
def test_diagnostic_fraction_must_be_strictly_between_zero_and_one(
    fraction: float,
) -> None:
    with pytest.raises(ValueError):
        DiagnosticSamplingPolicy(fraction=fraction)
