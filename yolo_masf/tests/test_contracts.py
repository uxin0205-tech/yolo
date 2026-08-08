from __future__ import annotations

from pathlib import Path

import pytest

from masf_yolo.contracts import (
    DatasetManifest,
    Phase1Config,
    canonical_json,
    sha256_file,
    sha256_value,
)


def test_canonical_hash_is_independent_of_mapping_insertion_order() -> None:
    left = {"beta": [2, 3], "alpha": {"path": Path("weights/model.pt")}}
    right = {"alpha": {"path": Path("weights/model.pt")}, "beta": [2, 3]}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_value(left) == sha256_value(right)


def test_file_hash_changes_when_content_changes(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"first")
    first = sha256_file(payload)

    payload.write_bytes(b"second")

    assert sha256_file(payload) != first


def test_dataset_manifest_round_trip_preserves_paths_and_tuples(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        source_root=tmp_path / "source",
        output_root=tmp_path / "artifacts" / "dataset",
        dataset_hash="a" * 64,
        split_ratios=(0.8, 0.1, 0.1),
        split_counts={"train": 80, "val": 10, "test": 10},
        class_names=("ball", "bat"),
        group_count=37,
    )

    restored = DatasetManifest.from_dict(manifest.to_dict())

    assert restored == manifest
    assert isinstance(restored.source_root, Path)
    assert isinstance(restored.split_ratios, tuple)


def test_phase1_config_rejects_non_801010_split() -> None:
    raw = Phase1Config.minimal_mapping()
    raw["dataset"]["split_ratios"] = [0.7, 0.2, 0.1]

    with pytest.raises(ValueError, match="80/10/10"):
        Phase1Config.from_mapping(raw)


def test_phase1_config_rejects_wrong_class_contract() -> None:
    raw = Phase1Config.minimal_mapping()
    raw["dataset"]["class_names"] = ["bat", "ball"]

    with pytest.raises(ValueError, match="ball.*bat"):
        Phase1Config.from_mapping(raw)


def test_phase1_config_rejects_unknown_keys() -> None:
    raw = Phase1Config.minimal_mapping()
    raw["training"]["automatic_magic"] = True

    with pytest.raises(ValueError, match="unknown.*automatic_magic"):
        Phase1Config.from_mapping(raw)


def test_phase1_config_rejects_phase2_variant() -> None:
    raw = Phase1Config.minimal_mapping()
    raw["variants"].append("S0")

    with pytest.raises(ValueError, match="unsupported variant.*S0"):
        Phase1Config.from_mapping(raw)
