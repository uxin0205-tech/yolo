from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE = PROJECT_ROOT / "training/full35/activation-recipe.yaml"
PLAN = PROJECT_ROOT / "training/full35/policy-search-plan.yaml"
QUEUE = PROJECT_ROOT / "training/full35/experiment-queue.yaml"


def test_full35_accuracy_budget_is_frozen_consistently_after_silu_control() -> None:
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    queue = yaml.safe_load(QUEUE.read_text(encoding="utf-8"))
    recovery = plan["short_recovery"]

    assert recovery["accuracy_budget_status"] == "frozen_after_silu_control"
    assert recovery["matched_control"]["epoch"] == 9
    assert recovery["matched_control"]["run_name"] == (
        "short-recovery-v2-lr01-uniform-silu-seed1"
    )
    assert recipe["comparison"]["maximum_map50_95_drop"] == 0.015
    assert recovery["accuracy_gate"]["maximum_map50_95_drop"] == 0.015
    assert queue["accuracy_gate"]["maximum_map50_95_drop"] == 0.015
