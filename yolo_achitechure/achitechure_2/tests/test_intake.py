from __future__ import annotations

import json

import pytest
import torch
import ultralytics

from achitechure_2.intake import (
    HandoffManifest,
    file_sha256,
    require_accepted_intake,
    validate_handoff,
    write_intake,
)


def _manifest(tmp_path, float_path, bittrue_path, variant: str = "full35"):
    selection = tmp_path / "selection.json"
    selection.write_text("{}", encoding="utf-8")
    path = tmp_path / "handoff.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": variant,
                "float_checkpoint": {
                    "path": str(float_path),
                    "sha256": file_sha256(float_path),
                    "map50_95": 0.5,
                },
                "bittrue_checkpoint": {
                    "path": str(bittrue_path),
                    "sha256": file_sha256(bittrue_path),
                    "map50_95": 0.499,
                },
                "environment": {"torch": torch.__version__, "ultralytics": ultralytics.__version__},
                "selection_manifest": {
                    "path": str(selection),
                    "sha256": file_sha256(selection),
                },
                "model_selection": {"selected_variant": variant, "basis": "formal selection"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_requires_selection_and_versions(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 1, "variant": "full35"}), encoding="utf-8")
    with pytest.raises(ValueError, match="environment"):
        HandoffManifest.load(path)


def test_validate_handoff_checks_both_backends_and_sha(tmp_path, toy_parent, bittrue_parent) -> None:
    float_path, bittrue_path = tmp_path / "float.pt", tmp_path / "bittrue.pt"
    float_path.write_bytes(b"float")
    bittrue_path.write_bytes(b"bittrue")
    manifest = _manifest(tmp_path, float_path, bittrue_path)

    def loader(path):
        return toy_parent if path.name == "float.pt" else bittrue_parent

    report = validate_handoff(
        manifest,
        project_root=tmp_path,
        loader=loader,
        require_fresh_process=False,
    )
    assert report.accepted
    assert report.variant == "full35"
    write_intake(report, tmp_path / "artifacts/intake/accepted.json")
    assert require_accepted_intake(tmp_path)["accepted"]
    (tmp_path / "selection.json").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="selection manifest changed"):
        require_accepted_intake(tmp_path)
    float_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_handoff(manifest, project_root=tmp_path, loader=loader, require_fresh_process=False)


def test_formal_intake_rejects_missing_masf(tmp_path, toy_parent, bittrue_parent) -> None:
    float_path, bittrue_path = tmp_path / "float.pt", tmp_path / "bittrue.pt"
    float_path.write_bytes(b"float")
    bittrue_path.write_bytes(b"bittrue")
    manifest = _manifest(tmp_path, float_path, bittrue_path)
    del toy_parent.model[16].p3_masf
    with pytest.raises(ValueError, match="must contain P3 MASF"):
        validate_handoff(
            manifest,
            project_root=tmp_path,
            loader=lambda path: toy_parent if path.name == "float.pt" else bittrue_parent,
            require_fresh_process=False,
        )
