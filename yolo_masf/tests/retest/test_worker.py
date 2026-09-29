from pathlib import Path

from masf_yolo.retest.worker import RetestWorkerRequest, profile_for


def test_retest_worker_request_round_trip_and_stage_profiles():
    request = RetestWorkerRequest.from_dict({
        "config_path": "configs/retest/b1r_p2_p3_retest.yaml", "stage": "b1r_a", "family": "B1R",
        "variant": None, "source_weights": "weights.pt", "data_yaml": "data.yaml",
        "project": "artifacts/retest", "resume_path": None,
    })
    profile = profile_for(request, "model.pt")
    assert profile["epochs"] == 10
    assert profile["lr0"] == 0.01
    assert profile["freeze"] == list(range(11))


def test_retest_formal_profile_is_direct_unfrozen_one_hundred_epochs():
    request = RetestWorkerRequest(
        Path("config"), "formal", "P2", "PaperFormula-Full", None,
        Path("data"), Path("project"), None,
    )
    profile = profile_for(request, "model.pt")
    assert profile["epochs"] == 100
    assert profile["lr0"] == 0.001
    assert profile["freeze"] is None


def test_b0_fair_profile_matches_p3_formal_training_budget():
    b0_request = RetestWorkerRequest(
        Path("config"), "b0_fair", "B0", None, Path("source.pt"),
        Path("data"), Path("project"), None,
    )
    formal_request = RetestWorkerRequest(
        Path("config"), "formal", "P3", "Partial25-35", Path("source.pt"),
        Path("data"), Path("project"), None,
    )
    b0 = profile_for(b0_request, "model.pt")
    formal = profile_for(formal_request, "model.pt")
    for key in ("epochs", "imgsz", "batch", "optimizer", "lr0", "momentum", "cos_lr", "seed", "deterministic", "amp", "nbs", "freeze"):
        assert b0[key] == formal[key]
    assert b0["name"] == "b0-fair-seed42"


def test_control_head_profile_only_leaves_p2_and_p2_detect_trainable():
    request = RetestWorkerRequest(Path("config"), "control_head", "B1R", None, None, Path("data"), Path("project"), None)
    profile = profile_for(request, "model.pt")
    assert profile["epochs"] == 20
    assert 20 not in profile["freeze"]
    assert "31.cv2.1" in profile["freeze"]
