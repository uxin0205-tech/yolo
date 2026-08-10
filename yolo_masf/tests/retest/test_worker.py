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
