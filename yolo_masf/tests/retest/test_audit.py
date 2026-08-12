from masf_yolo.retest.audit import audit


def test_postprocess_audit_passes():
    assert audit()["ok"] is True
