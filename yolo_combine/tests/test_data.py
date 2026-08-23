import pytest

from yolo_combine.data import audit_bbt5


@pytest.mark.integration
def test_bbt5_detect_is_an_exact_view_of_pose():
    report = audit_bbt5()

    assert report.derivation_exact
    assert report.derivation_mismatches == 0
    assert report.broken_image_links == 0
    assert report.source_group_overlap == ()
    assert report.train.images == report.train.labels == 5964
    assert report.train.instances == 8134
    assert report.train.empty_labels == 38
    assert report.train.ball_instances == 3220
    assert report.train.bat_instances == 4914
    assert report.train.negative_keypoint_rows == 0
    assert report.valid.images == report.valid.labels == 683
    assert report.valid.instances == 932
    assert report.valid.empty_labels == 0
    assert report.valid.ball_instances == 393
    assert report.valid.bat_instances == 539
    assert report.valid.negative_keypoint_rows == 0
