from __future__ import annotations

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
