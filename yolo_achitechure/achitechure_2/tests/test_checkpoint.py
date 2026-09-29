from __future__ import annotations

import torch

from achitechure_2.candidate import build_candidate, resolve_candidate_matrix
from achitechure_2.checkpoint import load_candidate_checkpoint, save_candidate_checkpoint
from achitechure_2.intake import CandidateRegion


def _region() -> CandidateRegion:
    return CandidateRegion(
        "shared",
        "shared",
        ("detect", "pose"),
        ("trunk.layers.0", "trunk.layers.1"),
        ("detect_head", "pose_head"),
    )


def _lineage() -> dict[str, str]:
    return {
        "spec_version": "2.3.0",
        "spec_sha256": "a" * 64,
        "handoff_revision": "winner-r1",
        "handoff_manifest_sha256": "b" * 64,
        "architecture_yaml_sha256": "c" * 64,
        "training_yaml_sha256": "d" * 64,
        "dataset_yaml_sha256": "e" * 64,
        "parent_checkpoint_sha256": "f" * 64,
        "candidate_id": "C1",
        "resolved_candidate_id": "C1",
    }


def test_state_dict_checkpoint_rebuilds_exact_candidate(tmp_path, combined_parent) -> None:
    resolved = resolve_candidate_matrix("shared_dual_head", (_region(),))[1]
    model, report = build_candidate(combined_parent, resolved, seed=0)
    checkpoint = tmp_path / "candidate.pt"

    save_candidate_checkpoint(checkpoint, model, report, lineage=_lineage())
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "state_dict" in raw
    assert "model" not in raw
    assert raw["builder"] == "achitechure_2"
    assert raw["lineage"]["handoff_revision"] == "winner-r1"

    def builder():
        return build_candidate(combined_parent, resolved, seed=0)[0]

    reloaded, metadata = load_candidate_checkpoint(checkpoint, builder)
    assert metadata["lineage"] == _lineage()
    assert metadata["resolved_candidate_id"] == "C1"
    assert all(
        torch.equal(value, reloaded.state_dict()[name])
        for name, value in model.state_dict().items()
    )
    sample = torch.randn(1, 3, 32, 32)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        expected = model(sample, tasks="both")
        actual = reloaded(sample, tasks="both")
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)


def test_checkpoint_rejects_incomplete_lineage_and_wrong_builder(tmp_path, combined_parent) -> None:
    resolved = resolve_candidate_matrix("shared_dual_head", (_region(),))[1]
    model, report = build_candidate(combined_parent, resolved, seed=0)
    checkpoint = tmp_path / "candidate.pt"
    lineage = _lineage()
    del lineage["dataset_yaml_sha256"]

    try:
        save_candidate_checkpoint(checkpoint, model, report, lineage=lineage)
    except ValueError as error:
        assert "lineage" in str(error)
    else:
        raise AssertionError("不完整 lineage 不得寫入 checkpoint")

    save_candidate_checkpoint(checkpoint, model, report, lineage=_lineage())
    c0 = resolve_candidate_matrix("shared_dual_head", (_region(),))[0]
    try:
        load_candidate_checkpoint(
            checkpoint,
            lambda: build_candidate(combined_parent, c0, seed=0)[0],
        )
    except ValueError as error:
        assert "contract" in str(error)
    else:
        raise AssertionError("錯誤 builder 不得載入候選 checkpoint")
