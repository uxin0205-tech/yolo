from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import ultralytics
import yaml
from torch import nn
from ultralytics.nn.modules.block import C3k2

from achitechure_2.intake import (
    HandoffManifest,
    file_sha256,
    require_accepted_intake,
    validate_handoff,
    write_intake,
)


class CombinedFixture(nn.Module):
    def __init__(self, contract: dict) -> None:
        super().__init__()
        self.trunk = nn.Module()
        self.trunk.layers = nn.ModuleList(
            [C3k2(8, 8, n=1, c3k=True, e=0.5), nn.Identity()]
        )
        self.detect_head = nn.Identity()
        self.pose_head = nn.Identity()
        self._contract = contract

    def contract(self) -> dict:
        return self._contract


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _contract(kind: str = "shared_dual_head") -> dict:
    coco = yaml.safe_load(Path("configs/data/coco2017.yaml").read_text(encoding="utf-8"))
    return {
        "interface": "model(images, tasks=detect|pose|both)",
        "model_kind": kind,
        "head_inputs": [16, 19, 22],
        "detect_nc": 80,
        "pose_nc": 2,
        "kpt_shape": [2, 3],
        "detect_names": coco["names"],
        "pose_names": {0: "ball", 1: "bat"},
    }


def _manifest(tmp_path: Path, *, kind: str = "shared_dual_head") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    for name in (
        "winner.pt",
        "builder.py",
        "architecture.yaml",
        "training.yaml",
        "selection.json",
        "fresh-process.json",
        "coco.yaml",
        "pose.yaml",
        "detect.yaml",
    ):
        path = tmp_path / name
        path.write_text(f"{name}\n", encoding="utf-8")
        artifacts[name] = path
    artifacts["training.yaml"].write_text(
        yaml.safe_dump(
            {
                "batch": 128,
                "fraction": 1.0,
                "scale": 0.5,
                "cache": False,
                "imgsz": 640,
                "optimizer": "MuSGD",
                "lr0": 0.00038,
                "lrf": 0.5,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "warmup_epochs": 1.0,
                "nbs": 64,
                "losses": {"detect": "upstream", "pose": "upstream"},
                "augmentation": {"source": "upstream"},
                "freeze_policy": {"source": "upstream"},
                "amp": True,
                "deterministic": True,
                "seed": 0,
                "optimizer_steps": 100,
                "validation_interval": 1,
                "task_ratio": "2:1",
                "epochs": 100,
                "patience": 20,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    artifacts["fresh-process.json"].write_text(
        json.dumps(
            {
                "passed": True,
                "state_dict_reload": True,
                "tasks": ["detect", "pose", "both"],
            }
        ),
        encoding="utf-8",
    )
    if kind == "shared_dual_head":
        regions = [
            {
                "region_id": "shared",
                "role": "shared",
                "tasks": ["detect", "pose"],
                "module_paths": ["trunk.layers.0"],
                "head_paths": ["detect_head", "pose_head"],
            }
        ]
    elif kind == "routed_dual":
        regions = [
            {
                "region_id": "detect",
                "role": "detect_specific",
                "tasks": ["detect"],
                "module_paths": ["trunk.layers.0"],
                "head_paths": ["detect_head"],
            },
            {
                "region_id": "pose",
                "role": "pose_specific",
                "tasks": ["pose"],
                "module_paths": ["trunk.layers.2"],
                "head_paths": ["pose_head"],
            },
        ]
    else:
        raise AssertionError(kind)
    payload = {
        "schema_version": 2,
        "producer": "yolo_combine",
        "revision_id": "winner-r1",
        "winner_id": "F1-full35-seed0",
        "fusion_kind": kind,
        "checkpoint_format": "state_dict_only",
        "checkpoint": _artifact(artifacts["winner.pt"]),
        "builder": _artifact(artifacts["builder.py"]),
        "architecture": _artifact(artifacts["architecture.yaml"]),
        "training_recipe": _artifact(artifacts["training.yaml"]),
        "datasets": {
            "coco_detect": _artifact(artifacts["coco.yaml"]),
            "bbat5_pose": _artifact(artifacts["pose.yaml"]),
            "bbat5_detect": _artifact(artifacts["detect.yaml"]),
        },
        "selection": _artifact(artifacts["selection.json"]),
        "fresh_process_report": _artifact(artifacts["fresh-process.json"]),
        "environment": {
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "git_revision": "0123456789abcdef",
        },
        "model_contract": _contract(kind),
        "candidate_regions": regions,
        "protected_module_paths": ["trunk.layers.1", "detect_head", "pose_head"],
        "frozen_module_paths": ["trunk.layers.1"],
    }
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_manifest_accepts_shared_winner_and_requires_main_task_semantics(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = HandoffManifest.load(manifest_path)

    assert manifest.revision_id == "winner-r1"
    assert manifest.fusion_kind == "shared_dual_head"
    assert manifest.model_contract["detect_nc"] == 80
    assert manifest.model_contract["pose_nc"] == 2
    assert manifest.candidate_regions[0].module_paths == ("trunk.layers.0",)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model_contract"]["detect_nc"] = 2
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="detect_nc=80"):
        HandoffManifest.load(manifest_path)


def test_manifest_rejects_region_topology_that_disagrees_with_fusion_kind(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, kind="routed_dual")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["candidate_regions"][0]["role"] = "shared"
    payload["candidate_regions"][0]["tasks"] = ["detect", "pose"]
    payload["candidate_regions"][0]["head_paths"] = ["detect_head", "pose_head"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="routed_dual"):
        HandoffManifest.load(manifest_path)


def test_validate_handoff_checks_hashes_contract_paths_and_immutability(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    contract = _contract()
    model = CombinedFixture(contract)

    report = validate_handoff(
        manifest_path,
        project_root=tmp_path,
        loader=lambda _: model,
    )
    assert report.accepted
    assert report.revision_id == "winner-r1"
    assert report.graph_audit["candidate_module_paths"] == ["trunk.layers.0"]
    accepted = tmp_path / "artifacts/intake/accepted.json"
    write_intake(report, accepted)
    assert require_accepted_intake(tmp_path)["revision_id"] == "winner-r1"

    (tmp_path / "training.yaml").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="training_recipe changed"):
        require_accepted_intake(tmp_path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_handoff(manifest_path, project_root=tmp_path, loader=lambda _: model)


def test_fresh_process_evidence_and_model_contract_are_required(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    fresh = tmp_path / "fresh-process.json"
    fresh.write_text(json.dumps({"passed": False, "state_dict_reload": False, "tasks": []}))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fresh_process_report"] = _artifact(fresh)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fresh-process"):
        validate_handoff(
            manifest_path,
            project_root=tmp_path,
            loader=lambda _: CombinedFixture(_contract()),
        )

    manifest_path = _manifest(tmp_path / "contract-mismatch")
    with pytest.raises(ValueError, match="model contract mismatch"):
        validate_handoff(
            manifest_path,
            project_root=tmp_path,
            loader=lambda _: CombinedFixture(_contract("routed_dual")),
        )


def test_metadata_only_handoff_inspection_cannot_be_written_as_accepted(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest(tmp_path)
    report = validate_handoff(manifest_path, project_root=tmp_path, loader=None)
    assert not report.accepted
    assert report.graph_audit["not_materialized"]
    with pytest.raises(ValueError, match="materialize"):
        write_intake(report, tmp_path / "artifacts/intake/accepted.json")


def test_repository_handoff_example_matches_v2_loader_contract() -> None:
    manifest = HandoffManifest.load("handoff-manifest.example.json")

    assert manifest.revision_id == "replace-with-immutable-revision-id"
    assert manifest.winner_id == "replace-with-selected-winner-id"
    assert manifest.fusion_kind == "shared_dual_head"
    assert manifest.model_contract["detect_nc"] == 80
    assert manifest.model_contract["pose_nc"] == 2
    assert manifest.model_contract["detect_names"][0] == "person"
    assert manifest.model_contract["pose_names"] == {0: "ball", 1: "bat"}
