from __future__ import annotations

from pathlib import Path

from torch import nn

from activation_lab.training import (
    Full35ActivationExperiment,
    Full35ExperimentConfig,
    load_full35_manifest,
    uniform_full35_policy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE = PROJECT_ROOT / "training/full35/activation-recipe.yaml"


def test_full35_recipe_uses_all_canonical_data_and_frozen_phase_budgets() -> None:
    config = Full35ExperimentConfig.load(RECIPE)

    assert config.fraction == 1.0
    assert not config.resampling
    assert config.phase("short_recovery").epochs == 10
    assert config.phase("finalist_seed1").epochs == 20
    assert config.phase("finalist_seed2").seed == 2
    assert config.phase("qat_if_needed").learning_rate_scale == 0.5


def test_full35_preflight_and_reviewed_manifest_match_release() -> None:
    experiment = Full35ActivationExperiment.from_yaml(RECIPE)
    report = experiment.preflight(verify_hashes=False)
    manifest = load_full35_manifest(experiment.config)

    assert report.ready, report.blockers
    assert manifest.reviewed
    assert manifest.model_source_sha256 == experiment.config.checkpoint_sha256
    assert len(manifest.sites) == 190
    assert len({site.module_path for site in manifest.sites}) == 190
    assert "unmapped" not in {site.region for site in manifest.sites}


def test_full35_uniform_policy_replaces_all_shared_graph_references() -> None:
    experiment = Full35ActivationExperiment.from_yaml(RECIPE)
    manifest = load_full35_manifest(experiment.config)
    loaded = experiment.load_policy_model(
        manifest,
        uniform_full35_policy("poly_shift"),
    )

    assert len(loaded.applied.changed_paths) == 190
    assert not loaded.applied.unchanged_paths
    assert all(
        not isinstance(loaded.model.get_submodule(site.module_path), nn.SiLU)
        for site in manifest.sites
    )
