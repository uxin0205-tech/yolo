from __future__ import annotations

from pathlib import Path

import yaml

from activation_lab.training import (
    ActivationManifest,
    ActivationSite,
    TrainingArchitecture,
    load_observations,
    load_training_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_plan_policy_round_trips_into_observation_yaml(tmp_path) -> None:
    config = load_training_config(PROJECT_ROOT / "training/configs/pipeline.yaml")
    manifest = ActivationManifest(
        model_id="roundtrip-model",
        reviewed=True,
        sites=(ActivationSite("model.0.act", "early"),),
    )
    plan = TrainingArchitecture(config).plan(manifest)
    serialized = plan.to_dict()
    experiment = serialized["experiments"][0]
    assert experiment["policy"]["region_assignments"] == {}
    assert experiment["policy"]["site_assignments"] == {}

    observation_payload = {
        "schema_version": 1,
        "observations": [
            {
                "dataset_id": experiment["dataset_id"],
                "stage": experiment["stage"],
                "policy": experiment["policy"],
                "map_loss": 0.0,
                "ap_s_loss": None,
                "latency_ms": None,
                "failed": False,
            }
        ],
    }
    observation_path = tmp_path / "observations.yaml"
    observation_path.write_text(
        yaml.safe_dump(observation_payload, sort_keys=False),
        encoding="utf-8",
    )
    loaded = load_observations(observation_path)
    assert loaded[0].policy == plan.experiments[0].policy
