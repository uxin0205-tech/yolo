from __future__ import annotations

import json
from pathlib import Path

import pytest

import masf_yolo.training.profiles as profile_module
from masf_yolo.pipeline import FormalPipeline
from masf_yolo.training.profiles import b1_a_profile, formal_profile, profile_differences


def test_formal_profiles_differ_only_in_allowed_run_identity_fields() -> None:
    b1 = formal_profile("B1", "models/b1.pt", "artifacts/runs", epochs=90)
    m0 = formal_profile("M0", "models/m0.pt", "artifacts/runs", epochs=100)

    assert profile_differences(b1, m0) == {"model", "name", "epochs"}
    for profile in (b1, m0):
        assert profile["imgsz"] == 640
        assert profile["optimizer"] == "SGD"
        assert profile["momentum"] == 0.937
        assert profile["lr0"] == 0.001
        assert profile["cos_lr"] is True
        assert profile["seed"] == 42
        assert profile["deterministic"] is True
        assert profile["amp"] is True
        assert profile["nbs"] == 64


def test_b1_a_has_exact_freeze_and_warmup_policy() -> None:
    profile = b1_a_profile("models/b1-init.pt", "artifacts/runs")

    assert profile["epochs"] == 10
    assert profile["lr0"] == 0.01
    assert profile["freeze"] == list(range(11))
    assert profile["optimizer"] == "SGD"


@pytest.mark.parametrize("variant,name", [("SP2", "sp2-a"), ("SP2M3", "sp2p-a")])
def test_frozen_stage_profile_matches_required_ten_epoch_policy(
    variant: str, name: str
) -> None:
    assert hasattr(profile_module, "frozen_stage_profile")
    profile = profile_module.frozen_stage_profile(
        variant, "models/parent.pt", "artifacts/runs", name=name
    )

    assert profile["epochs"] == 10
    assert profile["freeze"] == list(range(11))
    assert profile["lr0"] == 0.01
    assert profile["name"] == name


def test_sp2_and_sp2p_b_profiles_are_unfrozen_ninety_epoch_runs(tmp_path: Path) -> None:
    for stage in ("sp2_a", "sp2p_a"):
        path = tmp_path / "training" / stage
        path.mkdir(parents=True)
        (path / "run.json").write_text(
            json.dumps({"best": str(path / "best.pt")}), encoding="utf-8"
        )
    pipeline = object.__new__(FormalPipeline)
    pipeline.artifact_root = tmp_path
    pipeline.data_yaml = tmp_path / "dataset" / "data.yaml"
    pipeline._common_batch = lambda: 4

    for stage, variant, expected_name in (
        ("sp2_b", "SP2", "sp2-b"),
        ("sp2p_b", "SP2M3", "sp2p-b"),
    ):
        profile = pipeline._profile_for_stage(stage, variant)
        assert profile["epochs"] == 90
        assert profile["freeze"] is None
        assert profile["lr0"] == 0.001
        assert profile["name"] == expected_name
