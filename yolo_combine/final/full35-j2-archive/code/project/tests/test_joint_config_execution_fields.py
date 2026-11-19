from pathlib import Path

from yolo_combine.joint_config import JointExperimentConfig


def test_joint_yaml_augmentation_and_bn_fields_are_executable() -> None:
    config = JointExperimentConfig.load(
        Path("variants/full35/configs/joint.yaml")
    )

    assert config.detect_mosaic == 0.0
    assert config.pose_mosaic == 0.0
    assert config.detect_fliplr == 0.5
    assert config.pose_fliplr == 0.0
    assert config.shared_bn_affine_trainable is True
    assert config.as_dict()["augmentation"]["detect_fliplr"] == 0.5
    assert config.as_dict()["shared_bn"]["running_statistics_frozen"] is True

