from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from masf_yolo.evaluation.reference import (
    B0ReferenceDefinition,
    inspect_b0_reference,
    load_b0_definition,
)


EXPECTED_HASH = "9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d"


def test_b0_definition_is_strict_and_records_pose_provenance(tmp_path: Path) -> None:
    definition = load_b0_definition(Path("configs/b0-reference.yaml"))

    assert isinstance(definition, B0ReferenceDefinition)
    assert definition.reference_id == "B0"
    assert definition.checkpoint_hash == EXPECTED_HASH
    assert definition.class_names == ("ball", "bat")
    assert definition.strides == (8, 16, 32)
    assert definition.task == "detect"
    assert definition.ultralytics == "8.4.90"
    assert definition.data_exposed is True
    assert definition.selection_eligible is False
    assert "pose" in definition.provenance.lower()

    raw = yaml.safe_load(Path("configs/b0-reference.yaml").read_text())
    raw["unknown"] = True
    invalid = tmp_path / "b0.yaml"
    invalid.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown B0 reference keys"):
        load_b0_definition(invalid)


def test_b0_inspection_checks_actual_detect_model_and_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "b0.json"

    manifest = inspect_b0_reference(Path("configs/b0-reference.yaml"), output)

    assert manifest["reference_id"] == "B0"
    assert manifest["checkpoint_hash"] == EXPECTED_HASH
    assert manifest["task"] == "detect"
    assert manifest["class_names"] == ["ball", "bat"]
    assert manifest["strides"] == [8.0, 16.0, 32.0]
    assert manifest["detect_scales"] == 3
    assert manifest["forward_640"] is True
    assert manifest["source_train_task"] == "pose"
    assert manifest["data_exposed"] is True
    assert manifest["selection_eligible"] is False
    assert json.loads(output.read_text()) == manifest


def test_b0_hash_mismatch_fails_before_model_use(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/b0-reference.yaml").read_text())
    raw["checkpoint_sha256"] = "0" * 64
    invalid = tmp_path / "b0.yaml"
    invalid.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="checkpoint hash"):
        inspect_b0_reference(invalid, tmp_path / "output.json")
