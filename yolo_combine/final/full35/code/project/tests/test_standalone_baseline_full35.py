import pytest

from yolo_combine.source import SourceBundle
from yolo_combine.standalone_baseline import copy_float_state_to_bittrue


@pytest.mark.integration
def test_full35_pose_float_to_bittrue_mapping_is_complete(
    source_bundle: SourceBundle,
) -> None:
    float_pose, _ = source_bundle.build_pose_model(kind="float")
    bittrue_pose, _ = source_bundle.build_pose_model(kind="bittrue")

    report = copy_float_state_to_bittrue(float_pose, bittrue_pose)

    assert report.complete
    assert report.copied_tensors > 900
    assert report.preserved_bittrue_tensors == 2
    assert report.ignored_float_tensors == 4

