from __future__ import annotations

from pathlib import Path

import pytest

from yolo_combine.factory import FusionModelFactory
from yolo_combine.fusion_model import GraphSharedDualHeadModel
from yolo_combine.xnor import XNORExecutionConfig


POSE_CHECKPOINT = Path(
    "/home/uxin/yolo/yolo_combine/artifacts/runs/p0/"
    "p0-full35-p1-seed0/weights/best.pt"
)


@pytest.mark.integration
def test_factory_requires_pose_checkpoint_for_formal_build(source_bundle) -> None:
    factory = FusionModelFactory(
        source_bundle,
        detect_data_yaml="/home/uxin/yolo/coco2017.yaml",
        pose_data_yaml=(
            "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml"
        ),
        xnor=XNORExecutionConfig(token_tile=32),
    )

    with pytest.raises(ValueError, match="Pose checkpoint"):
        factory.build()


@pytest.mark.integration
def test_factory_emits_complete_loading_and_dataset_report(source_bundle) -> None:
    factory = FusionModelFactory(
        source_bundle,
        detect_data_yaml="/home/uxin/yolo/coco2017.yaml",
        pose_data_yaml=(
            "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml"
        ),
        xnor=XNORExecutionConfig(token_tile=32),
    )
    result = factory.build(pose_head_checkpoint=POSE_CHECKPOINT)

    assert isinstance(result.model, GraphSharedDualHeadModel)
    assert result.report.complete
    assert result.report.datasets.detect_nc == 80
    assert result.report.datasets.detect_names[0] == "person"
    assert result.report.datasets.pose_names == {0: "ball", 1: "bat"}
    assert result.report.datasets.kpt_shape == (2, 3)
    assert result.report.datasets.flip_idx == (0, 1)
    assert result.report.weights.loaded_shared_tensors == 587
    assert result.report.weights.loaded_pose_head_tensors == 411
    assert result.report.weights.missing_keys == ()
    assert result.report.weights.unexpected_keys == ()
    assert result.report.weights.shape_mismatches == ()
    assert result.report.xnor.token_tile == 32
    assert result.report.pose_head_checkpoint == POSE_CHECKPOINT.resolve()
    assert len(result.report.pose_head_sha256) == 64
