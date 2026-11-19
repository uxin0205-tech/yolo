from __future__ import annotations

from pathlib import Path

from activation_lab.training import Full35ActivationExperiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE = PROJECT_ROOT / "training/full35/activation-recipe.yaml"
COMBINE_JOINT_CONFIG = Path(
    "/home/uxin/yolo/yolo_combine/final/full35/configs/joint.yaml"
)


def test_activation_experiment_uses_combine_joint_data_and_macro_contract() -> None:
    experiment = Full35ActivationExperiment.from_yaml(RECIPE)
    report = experiment.preflight(verify_hashes=False)

    assert report.ready, report.blockers
    assert experiment.config.joint_config == COMBINE_JOINT_CONFIG
    resolved = report.resolved["full35"]
    assert resolved["detect_data"] == "/home/uxin/yolo/coco2017.yaml"
    assert resolved["pose_data"].endswith("bbat5-v1/configs/pose.yaml")
    # COMBINE base uses physical64 (4 forwards/macro); J3 recovery applies
    # the release-reviewed physical32 runtime override (8 forwards/macro).
    assert resolved["macro"]["detect_physical_microbatches"] == 4
    assert resolved["macro"]["pose_batches"] == 1
    assert resolved["macro"]["detect_weight"] == 1.0
    assert resolved["macro"]["pose_weight"] == 0.25
