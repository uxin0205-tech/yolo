from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn
from ultralytics.engine.trainer import BaseTrainer

from achitechure_2.freezing import (
    StageFrozenStateGuard,
    apply_stage_freeze,
    enforce_stage_eval,
)
from achitechure_2.pose_data import (
    prepare_grouped_pose_dataset,
    recertify_grouped_pose_dataset,
    source_group,
    validate_grouped_pose_dataset,
)
from achitechure_2.training import assert_pose_rle_contract, optimizer_group_report


def _record(root: Path, split: str, name: str, label: str) -> None:
    images = root / split / "images"
    labels = root / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    (images / name).write_bytes(b"image")
    (labels / f"{Path(name).stem}.txt").write_text(label, encoding="utf-8")


def test_grouped_pose_split_has_zero_leakage_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _record(source, "train", "frameA.rf.111.jpg", "1 0.5 -0.001 0.2 0.2 0.1 0.2 2 0.3 0.4 2\n")
    _record(source, "valid", "frameA.rf.222.jpg", "1 0.5 0.5 0.2 0.2 0.1 0.2 2 0.3 0.4 2\n")
    _record(source, "train", "frameB.rf.333.jpg", "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.3 0.4 2\n")
    _record(source, "valid", "frameC.rf.444.jpg", "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.3 0.4 2\n")
    destination = tmp_path / "derived"
    report = prepare_grouped_pose_dataset(
        source,
        destination,
        execute=True,
        expected_patch_count=1,
    )
    assert report.leakage == ()
    assert report.patched_coordinates == 1
    assert source_group("frameA.rf.111.jpg") == "frameA"
    split = json.loads((destination / "split-manifest.json").read_text())
    assert set(split["train_groups"]).isdisjoint(split["val_groups"])
    assert split["assignment"]["frameA"] in {"train", "val"}
    assert not (destination / "images/test").exists()
    assert any(path.is_symlink() for path in (destination / "images").glob("*/*"))
    assert "-0.001" in (source / "train/labels/frameA.rf.111.txt").read_text()
    patch = json.loads((destination / "patch-manifest.json").read_text())
    assert len(patch["patches"]) == 1
    assert split["spec_version"] == "1.2.0"
    assert patch["spec_sha256"] == split["spec_sha256"]
    validation = validate_grouped_pose_dataset(destination)
    assert validation["valid"]
    assert validation["image_symlinks"] == 4
    output = Path(patch["patches"][0]["output"])
    assert " 0 " in output.read_text()


def test_pose_rle_and_stage_freeze_contracts(pose_parent) -> None:
    assert_pose_rle_contract(pose_parent, 1.0)
    indices = apply_stage_freeze(pose_parent, "P1")
    assert indices == tuple(range(23))
    assert all(
        not parameter.requires_grad for layer in pose_parent.model[:23] for parameter in layer.parameters()
    )
    guard = StageFrozenStateGuard.capture(pose_parent, "P1")
    pose_parent.train()
    with pytest.raises(AssertionError, match="entered training mode"):
        guard.assert_unchanged(pose_parent)
    enforce_stage_eval(pose_parent, "P1")
    guard.assert_unchanged(pose_parent)
    del pose_parent.model[23].flow_model
    with pytest.raises(ValueError, match="已接受但未生效"):
        assert_pose_rle_contract(pose_parent, 1.0)


class _Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cv3 = nn.Conv2d(4, 4, 1)


class _MuonToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = [nn.Conv2d(4, 4, 3, padding=1) for _ in range(23)]
        layers.append(_Head())
        self.model = nn.ModuleList(layers)


def test_musgd_effective_groups_include_muon_and_head_3x_lr() -> None:
    owner = SimpleNamespace(
        data={"nc": 2},
        args=SimpleNamespace(warmup_bias_lr=0.1),
    )
    optimizer = BaseTrainer.build_optimizer(
        owner,
        _MuonToy(),
        name="MuSGD",
        lr=0.01,
        momentum=0.9,
        decay=0.0005,
        iterations=100,
    )
    report = optimizer_group_report(optimizer)
    nonempty = [item for item in report if item["parameters"]]
    assert any(item["muon"] for item in nonempty)
    assert {round(item["lr"], 4) for item in nonempty} >= {0.01, 0.03}
    assert all(item["trainable_parameters"] == item["parameters"] for item in nonempty)



def test_pose_dataset_recertification_preserves_previous_lineage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _record(source, "train", "frameA.rf.111.jpg", "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.3 0.4 2\n")
    _record(source, "valid", "frameB.rf.222.jpg", "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.3 0.4 2\n")
    destination = tmp_path / "derived"
    prepare_grouped_pose_dataset(source, destination, execute=True, expected_patch_count=0)
    for filename in ("split-manifest.json", "patch-manifest.json"):
        path = destination / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["spec_version"] = "old"
        payload["spec_sha256"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    report = recertify_grouped_pose_dataset(destination)
    assert report["valid"]
    split = json.loads((destination / "split-manifest.json").read_text(encoding="utf-8"))
    assert split["spec_version"] == "1.2.0"
    assert split["certification_history"][-1]["spec_version"] == "old"
