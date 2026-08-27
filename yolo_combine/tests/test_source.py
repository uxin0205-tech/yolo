import pytest
import torch
from ultralytics.nn.modules.head import Detect, Pose26

from yolo_combine.source import BuiltTaskModels, SourceBundle, transfer_pose_head


@pytest.mark.integration
def test_authoritative_bundle_manifest_and_main_candidate(source_bundle: SourceBundle):
    report = source_bundle.verify_manifest()

    assert report.files == 121
    assert report.bytes > 0
    assert source_bundle.architecture == "full35"
    assert source_bundle.model_id == "full35-a2"
    assert source_bundle.checkpoint().name == "full35-a2.pt"


@pytest.mark.integration
def test_pose_factory_transfers_every_shared_tensor(full35_models: BuiltTaskModels):
    report = full35_models.transfer
    detect_head = full35_models.detect.model[-1]
    pose_head = full35_models.pose.model[-1]

    assert report.complete
    assert report.source_layers == report.target_layers == 23
    assert report.compatible_tensors == 587
    assert isinstance(detect_head, Detect)
    assert not isinstance(detect_head, Pose26)
    assert isinstance(pose_head, Pose26)
    assert int(detect_head.nc) == 80
    assert int(pose_head.nc) == 2
    assert tuple(pose_head.kpt_shape) == (2, 3)
    assert type(getattr(full35_models.detect.model[16], "p3_masf")).__name__ == "P3MASFFull35"


@pytest.mark.integration
def test_pose_head_transfer_is_complete(
    source_bundle: SourceBundle,
    full35_models: BuiltTaskModels,
):
    target, _ = source_bundle.build_pose_model(full35_models.detect)
    report = transfer_pose_head(full35_models.pose, target)

    assert report.complete
    assert report.compatible_tensors == 411
    source_state = full35_models.pose.model[-1].state_dict()
    target_state = target.model[-1].state_dict()
    assert source_state.keys() == target_state.keys()
    for name, tensor in target_state.items():
        assert torch.equal(tensor, source_state[name]), name
