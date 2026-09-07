from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from yolo_quantize.qat_continuation import QATPatienceContinuation


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_patience_continuation_reconstructs_existing_stale_epochs(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"source")
    continuation = QATPatienceContinuation(
        config_path=tmp_path / "continuation.yaml",
        config_sha256="continuation-sha",
        authorization_id="user-test",
        base_plan=tmp_path / "base.yaml",
        base_plan_sha256="base-sha",
        resume_checkpoint=checkpoint,
        resume_checkpoint_sha256=_digest(checkpoint),
        from_patience=0,
        to_patience=5,
        boundary_next_epoch=6,
        score_history=(
            (0, 0.80),
            (1, 0.79),
            (2, 0.81),
            (3, 0.80),
            (4, 0.79),
            (5, 0.78),
        ),
    )

    state = continuation.early_stop_state()

    assert state["best_score"] == 0.81
    assert state["stale_epochs"] == 3
    assert state["patience"] == 5


def test_patience_continuation_migrates_only_resume_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    payload = {
        "checkpoint_kind": "full_resume",
        "progress": {
            "stage": "j3",
            "next_epoch": 2,
            "joint_epochs_completed": 2,
        },
        "loader_state": {
            "snapshot_boundary": "formal_epoch_end",
            "epoch": 1,
            "early_stop": None,
        },
        "resolved_config": {
            "qat_experiment": {"plan_sha256": "base-sha"},
            "stage_policies": {"j3": {"patience": 0}},
        },
        "best_state": {"last": {"epoch": 1, "score": 0.79}},
        "provenance": {"source": "fixture"},
        "model_state": {"weight": torch.tensor([1.0])},
    }
    torch.save(payload, source)
    continuation = QATPatienceContinuation(
        config_path=tmp_path / "continuation.yaml",
        config_sha256="continuation-sha",
        authorization_id="user-test",
        base_plan=tmp_path / "base.yaml",
        base_plan_sha256="base-sha",
        resume_checkpoint=source,
        resume_checkpoint_sha256=_digest(source),
        from_patience=0,
        to_patience=5,
        boundary_next_epoch=2,
        score_history=((0, 0.80), (1, 0.79)),
    )
    destination = tmp_path / "migrated.pt"
    expected = {
        "qat_experiment": {
            "plan_sha256": "base-sha",
            "patience_continuation": continuation.to_dict(),
        },
        "stage_policies": {"j3": {"patience": 5}},
    }

    evidence = continuation.prepare_resume_checkpoint(
        destination,
        expected_resolved_config=expected,
    )
    migrated = torch.load(destination, map_location="cpu", weights_only=True)

    assert migrated["model_state"]["weight"].item() == 1.0
    assert migrated["resolved_config"] == expected
    assert migrated["loader_state"]["early_stop"] == {
        "schema_version": 1,
        "stage": "j3",
        "patience": 5,
        "min_delta": 0.0,
        "best_score": 0.8,
        "stale_epochs": 1,
    }
    assert (
        migrated["provenance"]["qat_patience_continuation"]["authorization_id"]
        == "user-test"
    )
    assert evidence["migrated_checkpoint_sha256"] == _digest(destination)
