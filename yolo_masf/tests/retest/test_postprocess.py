from masf_yolo.retest.postprocess import formal_checkpoints


def test_postprocess_declares_b0_and_two_parent_controls():
    checkpoints = formal_checkpoints()
    assert "B0-Original-3Scale" in checkpoints
    assert "P2-Control-Head" in checkpoints
    assert "P2-Base-Direct" in checkpoints
