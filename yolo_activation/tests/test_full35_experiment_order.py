from __future__ import annotations

from pathlib import Path

from activation_lab.training import Full35ExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE = PROJECT_ROOT / "training/full35/activation-recipe.yaml"


def test_full35_runs_zero_shot_sensitivity_before_any_recovery() -> None:
    config = Full35ExperimentConfig.load(RECIPE)
    phases = tuple(config.phases)

    assert phases.index("activation_profile") < phases.index(
        "region_zero_shot_sensitivity"
    )
    assert phases.index("region_zero_shot_sensitivity") < phases.index("short_recovery")


def test_exploratory_recovery_uses_preregistered_tenth_scale_learning_rate() -> None:
    config = Full35ExperimentConfig.load(RECIPE)

    assert config.phase("short_recovery").learning_rate_scale == 0.1
    assert config.phase("region_sensitivity").learning_rate_scale == 0.1
    assert config.phase("policy_search").learning_rate_scale == 0.1
