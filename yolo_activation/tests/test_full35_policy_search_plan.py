from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN = PROJECT_ROOT / "training/full35/policy-search-plan.yaml"


def test_policy_search_plan_separates_training_only_regions() -> None:
    payload = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    deployment = payload["deployment_contract"]
    training_only = set(deployment["training_only_regions"])
    recovery = set(payload["region_recovery"]["queue"])

    assert training_only == {
        "detect_one2many",
        "pose_one2many",
        "pose_flow",
    }
    assert training_only.isdisjoint(recovery)
    assert set(deployment["initially_protected_silu_regions"]) <= recovery


def test_mixed_search_is_bounded_and_keeps_protected_regions_out() -> None:
    payload = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    search = payload["mixed_search"]
    trials = search["preregistered_trials"]
    protected = set(payload["deployment_contract"]["initially_protected_silu_regions"])

    assert len(trials) + search["adaptive_slots"] <= search["maximum_trials"]
    assert all(
        len(trial["regions"]) <= search["maximum_changed_regions"] for trial in trials
    )
    assert all(protected.isdisjoint(trial["regions"]) for trial in trials)
