from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from yolo_quantize import (
    DiagnosticManifest,
    DiagnosticManifestBuilder,
    DiagnosticManifestSpec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_builder_uses_fixed_canonical_calibration_and_probe_contract() -> None:
    manifest = DiagnosticManifestBuilder().build(DiagnosticManifestSpec())
    payload = manifest.payload
    calibration = payload["selections"]["calibration"]
    probe = payload["selections"]["probe"]
    calibration_groups = tuple(
        item["source_group"] for item in calibration["bbat_pose"]["samples"]
    )
    probe_groups = tuple(item["source_group"] for item in probe["bbat_pose"]["samples"])

    assert {
        "kind": payload["kind"],
        "diagnostic_only": payload["diagnostic_only"],
        "formal_training": payload["formal_training"],
        "seed": payload["seed"],
        "coco_calibration_split": calibration["coco_detect"]["split"],
        "coco_probe_split": probe["coco_detect"]["split"],
        "bbat_calibration_split": calibration["bbat_pose"]["split"],
        "bbat_probe_split": probe["bbat_pose"]["split"],
        "coco_calibration_count": len(calibration["coco_detect"]["samples"]),
        "coco_probe_count": len(probe["coco_detect"]["samples"]),
        "bbat_calibration_count": len(calibration["bbat_pose"]["samples"]),
        "bbat_probe_count": len(probe["bbat_pose"]["samples"]),
        "coco_calibration_has_person": (
            calibration["coco_detect"]["coverage"]["person_images"] > 0
        ),
        "coco_probe_has_person": (
            probe["coco_detect"]["coverage"]["person_images"] > 0
        ),
        "bbat_calibration_classes": calibration["bbat_pose"]["coverage"]["class_ids"],
        "bbat_probe_classes": probe["bbat_pose"]["coverage"]["class_ids"],
        "bbat_calibration_pose_rows_positive": (
            calibration["bbat_pose"]["coverage"]["pose_rows"] > 0
        ),
        "bbat_probe_pose_rows_positive": (
            probe["bbat_pose"]["coverage"]["pose_rows"] > 0
        ),
        "calibration_groups_unique": len(set(calibration_groups)),
        "probe_groups_unique": len(set(probe_groups)),
        "group_overlap": len(set(calibration_groups) & set(probe_groups)),
        "dataset_id": payload["sources"]["bbat5"]["dataset_id"],
        "assignment_changed": payload["sources"]["bbat5"]["assignment_changed"],
    } == {
        "kind": "full35_quantization_diagnostic_manifest",
        "diagnostic_only": True,
        "formal_training": False,
        "seed": 20260830,
        "coco_calibration_split": "train2017",
        "coco_probe_split": "val2017",
        "bbat_calibration_split": "formal-train",
        "bbat_probe_split": "formal-val",
        "coco_calibration_count": 32,
        "coco_probe_count": 64,
        "bbat_calibration_count": 32,
        "bbat_probe_count": 64,
        "coco_calibration_has_person": True,
        "coco_probe_has_person": True,
        "bbat_calibration_classes": [0, 1],
        "bbat_probe_classes": [0, 1],
        "bbat_calibration_pose_rows_positive": True,
        "bbat_probe_pose_rows_positive": True,
        "calibration_groups_unique": 32,
        "probe_groups_unique": 64,
        "group_overlap": 0,
        "dataset_id": "bbat5-v1",
        "assignment_changed": False,
    }


def test_manifest_loader_verifies_digest_and_exposes_fixed_paths(
    tmp_path: Path,
) -> None:
    path = PROJECT_ROOT / "artifacts/manifests/full35-diagnostic-cal32-probe64-v1.json"

    manifest = DiagnosticManifest.from_json(path, verify_files=True)

    assert len(manifest.paths("calibration", "coco_detect")) == 32
    assert len(manifest.paths("calibration", "bbat_pose")) == 32
    assert len(manifest.paths("probe", "coco_detect")) == 64
    assert len(manifest.paths("probe", "bbat_pose")) == 64
    assert manifest.sha256 == (
        "b0eb2a068aa515caa5ea22f2781dc68225971224cae2324d17a6be1e2d31eef2"
    )

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["seed"] += 1
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        DiagnosticManifest.from_json(tampered_path, verify_files=False)


def test_manifest_loader_rejects_a_malformed_bbat_source_contract(
    tmp_path: Path,
) -> None:
    path = PROJECT_ROOT / "artifacts/manifests/full35-diagnostic-cal32-probe64-v1.json"
    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["sources"]["bbat5"] = None
    malformed.pop("manifest_sha256")
    encoded = json.dumps(
        malformed,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    malformed["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")

    with pytest.raises(ValueError, match="BBAT5 contract"):
        DiagnosticManifest.from_json(malformed_path, verify_files=False)
