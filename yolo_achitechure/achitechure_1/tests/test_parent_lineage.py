from __future__ import annotations

import json
from pathlib import Path

import pytest

from achitechure_1.lineage import load_parent_study


def _write_study(path: Path, checkpoint_sha256: str) -> None:
    payload = {
        "schema_version": 1,
        "completed_at": "2026-08-18",
        "queue": {"revision": 95, "jobs_succeeded": 19, "jobs_total": 19},
        "formal_winner": {
            "sha256": checkpoint_sha256,
            "map50_95": 0.5067368995935831,
            "reason": "trained gain did not satisfy the formal gate",
        },
        "best_observed": {
            "sha256": "c70cd1d0315518d61b6ac6b5936173f04dc5b0eadcb86b288699655434dcf9fb",
            "map50_95": 0.5069386684559859,
            "delta_vs_formal_winner": 0.0002017688624028,
            "included_in_final": False,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parent_study_accepts_completed_formal_winner(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"formal parent")

    from achitechure_1.checkpoint import file_sha256

    study = tmp_path / "final-results.json"
    _write_study(study, file_sha256(checkpoint))

    report = load_parent_study(study, checkpoint)

    assert report["completed"] is True
    assert report["queue"] == {"revision": 95, "succeeded": 19, "total": 19}
    assert report["formal_winner"]["selected"] is True
    assert report["best_observed"]["selected"] is False
    assert report["best_observed"]["delta_vs_formal_winner"] < 0.001


def test_parent_study_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"different checkpoint")
    study = tmp_path / "final-results.json"
    _write_study(study, "0" * 64)

    with pytest.raises(ValueError, match="formal winner SHA256"):
        load_parent_study(study, checkpoint)


def test_parent_study_rejects_incomplete_queue(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"formal parent")

    from achitechure_1.checkpoint import file_sha256

    study = tmp_path / "final-results.json"
    _write_study(study, file_sha256(checkpoint))
    payload = json.loads(study.read_text(encoding="utf-8"))
    payload["queue"]["jobs_succeeded"] = 18
    study.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="queue is incomplete"):
        load_parent_study(study, checkpoint)
