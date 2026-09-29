from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from achitechure_2.config import SPEC_PATH, SPEC_VERSION, file_sha256
from achitechure_2.result_export import (
    MICRO_F1_METHOD,
    _best_validation,
    _candidate_build_report,
    _candidate_metrics,
    _profile_index,
)


def _ap(*, name0: str = "ball", name1: str = "bat") -> dict:
    return {
        "map50": 0.8,
        "map50_95": 0.7,
        "per_class": {
            "0": {"name": name0, "ap50": 0.75, "ap50_95": 0.65},
            "1": {"name": name1, "ap50": 0.85, "ap50_95": 0.75},
        },
    }


def _validation() -> dict:
    return {
        "detect": {
            "box": {
                "ap": {
                    "map50": 0.6,
                    "map50_95": 0.5,
                    "per_class": {
                        "0": {
                            "name": "person",
                            "ap50": 0.7,
                            "ap50_95": 0.55,
                        }
                    },
                }
            }
        },
        "pose": {
            "status": "measured",
            "official_combined_fitness": 1.2,
            "box": {"ap": _ap()},
            "keypoints": {
                "ap": _ap(),
                "f1": {
                    "confidence_threshold": 0.42,
                    "macro_f1": 0.8,
                    "micro_f1": 0.82,
                    "micro_f1_method": MICRO_F1_METHOD,
                    "per_class": {
                        "0": {
                            "name": "ball",
                            "precision": 0.75,
                            "recall": 0.85,
                            "f1": 0.8,
                        },
                        "1": {
                            "name": "bat",
                            "precision": 0.85,
                            "recall": 0.75,
                            "f1": 0.8,
                        },
                    },
                },
            },
        },
    }


def _profile() -> dict:
    return {
        "params": 123,
        "tasks": {
            "both": {
                "gflops": 4.5,
                "latency_median_ms": 2.25,
                "peak_allocated_mib": 321.0,
            }
        },
    }


def test_candidate_metrics_maps_detect_pose_classes_f1_and_cost() -> None:
    value = _candidate_metrics("C1", _validation(), _profile())
    assert value.candidate_id == "C1"
    assert value.coco_box_map50_95 == pytest.approx(0.5)
    assert value.coco_person_ap50_95 == pytest.approx(0.55)
    assert value.bbat5_pose_box_map50_95 == pytest.approx(0.7)
    assert value.bbat5_keypoint_map50_95 == pytest.approx(0.7)
    assert value.classes["ball"].f1 == pytest.approx(0.8)
    assert value.macro_f1 == pytest.approx(0.8)
    assert value.micro_f1 == pytest.approx(0.82)
    assert value.params == 123
    assert value.gflops == pytest.approx(4.5)


def test_candidate_metrics_rejects_class_or_micro_f1_drift() -> None:
    validation = _validation()
    validation["pose"]["keypoints"]["f1"]["micro_f1_method"] = "unknown"
    with pytest.raises(ValueError, match="Micro F1 method"):
        _candidate_metrics("C2", validation, _profile())

    validation = _validation()
    validation["pose"]["box"]["ap"]["per_class"]["1"]["name"] = "person"
    with pytest.raises(ValueError, match="名稱漂移"):
        _candidate_metrics("C2", validation, _profile())


def test_profile_index_requires_exact_matrix_order_and_hashes(tmp_path: Path) -> None:
    training = tmp_path / "run.yaml"
    training.write_text("training: fixed\n", encoding="utf-8")
    config = SimpleNamespace(path=training, candidates=("C0", "C1", "C2", "C3"))
    payload = {
        "schema_version": 1,
        "status": "completed",
        "spec_version": SPEC_VERSION,
        "spec_sha256": file_sha256(SPEC_PATH),
        "training_yaml_sha256": file_sha256(training),
        "profiles": [{"candidate": value} for value in config.candidates],
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _, indexed = _profile_index(path, config)
    assert tuple(indexed) == config.candidates

    payload["profiles"].reverse()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="順序"):
        _profile_index(path, config)


def test_best_validation_requires_complete_epochs_steps_and_checkpoints(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        run_root=tmp_path,
        seed=0,
        epochs=2,
        payload={"training": {"expected_optimizer_steps_per_candidate": 4}},
    )
    run_dir = tmp_path / "c0-control-seed0"
    complete_path = run_dir / "complete.json"
    complete_path.parent.mkdir(parents=True)
    complete = {
        "status": "completed_screening",
        "epochs_completed": 2,
        "global_macro_steps": 4,
        "best_state": {"joint_screening": {"epoch": 1, "score": 0.5}},
    }
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    for epoch in range(2):
        metrics = run_dir / f"validation/epoch-{epoch:04d}/float/metrics.json"
        metrics.parent.mkdir(parents=True)
        metrics.write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "backend": "float",
                    "formal_split_used": False,
                }
            ),
            encoding="utf-8",
        )
    for label in (
        "best-detect",
        "best-pose-research",
        "best-pose-official",
        "best-joint-screening",
    ):
        for kind in ("checkpoints", "inference"):
            checkpoint = run_dir / kind / f"{label}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.touch()

    _, metrics, path = _best_validation(config, "C0")
    assert metrics["epoch"] == 1
    assert path.name == "metrics.json"

    complete["global_macro_steps"] = 3
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(ValueError, match="steps 不完整"):
        _best_validation(config, "C0")


def test_candidate_build_report_preserves_full_transfer_lists_and_hash(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(run_root=tmp_path, seed=0)
    run_dir = tmp_path / "c1-control-seed0"
    run_dir.mkdir(parents=True)
    lineage = {"parent_checkpoint_sha256": "a" * 64}
    transfer = {
        "loaded": ["kept.weight"],
        "loaded_count": 1,
        "matched": ["kept.weight"],
        "matched_count": 1,
        "missing": [],
        "unexpected": [],
        "shape_mismatch": [
            {
                "name": "changed.weight",
                "source_shape": [4, 4],
                "target_shape": [3, 4],
            }
        ],
    }
    manifest = {
        "lineage": lineage,
        "candidate_build": {
            "candidate_id": "C1",
            "resolved_id": "C1",
            "changed_fields": ["e"],
            "changed_module_paths": ["graph.model.6"],
            "model_contract_unchanged": True,
            "parent_unchanged": True,
            "transfer": transfer,
        },
    }
    path = run_dir / "run-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _candidate_build_report(config, "C1", {"lineage": lineage})
    assert report["summary"] == {
        "matched": 1,
        "missing": 0,
        "unexpected": 0,
        "shape_mismatch": 1,
    }
    assert report["candidate_build"]["transfer"]["shape_mismatch"][0]["name"] == "changed.weight"
    assert report["run_manifest_sha256"] == file_sha256(path)

    manifest["candidate_build"]["transfer"]["matched_count"] = 2
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="transfer counts"):
        _candidate_build_report(config, "C1", {"lineage": lineage})
