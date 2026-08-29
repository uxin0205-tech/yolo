from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from activation_lab.training import (
    ActivationManifest,
    ActivationSite,
    DatasetContract,
    PolicyObservation,
    RegionRule,
    SearchConfig,
    StaticPolicy,
    TrainingArchitecture,
    TrainingConfig,
    load_training_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.early = _Block()
        self.neck = _Block()
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_sigmoid(self.neck(self.early(x)))


def _dataset() -> DatasetContract:
    return DatasetContract(
        dataset_id="bbat5-v1-detect",
        source_kind="canonical_bbat5_v1",
        yaml_path="/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml",
        task="detect",
        num_classes=2,
    )


def _config() -> TrainingConfig:
    return TrainingConfig(
        architecture_id="test-sipa-bcsp",
        datasets=(_dataset(),),
        search=SearchConfig(
            max_deployment_kernels=3,
            beam_width=4,
            max_changed_regions=2,
            max_finalists=2,
        ),
    )


def _manifest() -> ActivationManifest:
    return ActivationManifest(
        model_id="toy",
        reviewed=True,
        sites=(
            ActivationSite("early.act", "early"),
            ActivationSite("neck.act", "neck"),
        ),
    )


def _observe(plan, losses: dict[str, float]) -> tuple[PolicyObservation, ...]:
    return tuple(
        PolicyObservation(
            dataset_id=experiment.dataset_id,
            stage=experiment.stage,
            policy=experiment.policy,
            map_loss=losses.get(experiment.policy.policy_id, 0.0),
        )
        for experiment in plan.experiments
    )


def test_inspection_uses_only_silu_and_requires_explicit_review() -> None:
    architecture = TrainingArchitecture(_config())
    model = _ToyModel()
    rules = (
        RegionRule(r"^early\.act$", "early"),
        RegionRule(r"^neck\.act$", "neck"),
    )
    draft = architecture.inspect(model, model_id="toy", region_rules=rules)
    assert not draft.reviewed
    assert {site.module_path for site in draft.sites} == {"early.act", "neck.act"}
    assert "output_sigmoid" not in {site.module_path for site in draft.sites}
    assert architecture.plan(draft).next_stage == "blocked"
    assert draft.approve().reviewed


def test_static_policy_application_is_transactional_and_preserves_original() -> None:
    architecture = TrainingArchitecture(_config())
    model = _ToyModel().eval()
    policy = StaticPolicy("uniform--poly_shift", default_activation="poly_shift")
    applied = architecture.apply(model, _manifest(), policy)
    assert isinstance(model.early.act, nn.SiLU)
    assert not isinstance(applied.model.early.act, nn.SiLU)
    assert applied.changed_paths == ("early.act", "neck.act")
    sample = torch.randn(1, 2, 4, 4)
    assert torch.isfinite(applied.model(sample)).all()


def test_preflight_rejects_stale_manifest_before_any_mutation() -> None:
    architecture = TrainingArchitecture(_config())
    model = _ToyModel()
    model.early.act = nn.ReLU()
    policy = StaticPolicy("uniform--poly_shift", default_activation="poly_shift")
    with pytest.raises(ValueError, match="non-SiLU source paths"):
        architecture.apply(model, _manifest(), policy, clone_model=False)
    assert isinstance(model.early.act, nn.ReLU)
    assert isinstance(model.neck.act, nn.SiLU)


def test_dataset_contract_rejects_noncanonical_bbat5_path() -> None:
    with pytest.raises(ValueError, match="Canonical BBAT5 v1"):
        DatasetContract(
            dataset_id="bbat5-v1-detect",
            source_kind="canonical_bbat5_v1",
            yaml_path="/tmp/re-split.yaml",
            task="detect",
            num_classes=2,
        )


def test_bcsp_advances_one_gated_stage_at_a_time_and_stops_at_two_finalists() -> None:
    architecture = TrainingArchitecture(_config())
    manifest = _manifest()
    observations: tuple[PolicyObservation, ...] = ()

    baseline = architecture.plan(manifest, observations)
    assert baseline.next_stage == "baseline_reproduction"
    observations += _observe(baseline, {})

    zero = architecture.plan(manifest, observations)
    assert zero.next_stage == "zero_shot"
    assert {item.policy.policy_id for item in zero.experiments} == {
        "uniform--hardswish",
        "uniform--relu",
        "uniform--qsilu_pq",
        "uniform--poly_shift",
        "uniform--poly_quality",
    }
    observations += _observe(
        zero,
        {
            "uniform--hardswish": 0.03,
            "uniform--relu": 0.20,
            "uniform--qsilu_pq": 0.008,
            "uniform--poly_shift": 0.02,
            "uniform--poly_quality": 0.01,
        },
    )

    recovery = architecture.plan(manifest, observations)
    assert recovery.next_stage == "short_recovery"
    assert "uniform--relu" not in {
        item.policy.policy_id for item in recovery.experiments
    }
    observations += _observe(
        recovery,
        {
            "uniform--hardswish": 0.010,
            "uniform--qsilu_pq": 0.012,
            "uniform--poly_shift": 0.020,
            "uniform--poly_quality": 0.015,
        },
    )

    regions = architecture.plan(manifest, observations)
    assert regions.next_stage == "region_sensitivity"
    assert {item.policy.policy_id for item in regions.experiments} == {
        "static--early=poly_shift",
        "static--neck=poly_shift",
    }
    observations += _observe(
        regions,
        {
            "static--early=poly_shift": 0.005,
            "static--neck=poly_shift": 0.007,
        },
    )

    search = architecture.plan(manifest, observations)
    assert search.next_stage == "policy_search"
    assert len(search.experiments) == 1
    assert search.experiments[0].policy.policy_id == (
        "static--early=poly_shift__neck=poly_shift"
    )
    observations += _observe(
        search,
        {"static--early=poly_shift__neck=poly_shift": 0.004},
    )

    full_seed1 = architecture.plan(manifest, observations)
    assert full_seed1.next_stage == "full_recovery_seed1"
    assert 1 <= len(full_seed1.experiments) <= 2
    observations += _observe(full_seed1, {})

    full_seed2 = architecture.plan(manifest, observations)
    assert full_seed2.next_stage == "full_recovery_seed2"
    assert {item.seed for item in full_seed2.experiments} == {2}
    observations += _observe(full_seed2, {})

    complete = architecture.plan(manifest, observations)
    assert complete.next_stage == "complete"
    assert 1 <= len(complete.frontier_policy_ids) <= 2


def test_repository_pipeline_loads_and_keeps_datasets_separate() -> None:
    config = load_training_config(PROJECT_ROOT / "training/configs/pipeline.yaml")
    assert [dataset.dataset_id for dataset in config.datasets] == [
        "bbat5-v1-detect",
        "coco2017-detect",
    ]
    plan = TrainingArchitecture(config).plan(_manifest())
    assert len(plan.experiments) == 2
    assert len({item.dataset_yaml for item in plan.experiments}) == 2


def test_cli_toy_dry_run_crosses_the_public_training_seam() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/activation_training.py"),
            "toy-dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["applied_policy"]["original_model_unchanged"]
    assert payload["applied_policy"]["output_finite"]
    assert payload["initial_plan"]["next_stage"] == "baseline_reproduction"
