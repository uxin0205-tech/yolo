import pytest

from masf_yolo.retest.audit import audit
from masf_yolo.retest.postprocess import ART

pytestmark = pytest.mark.skipif(
    not (ART / "queue_state.json").is_file(),
    reason="legacy data-exposed retest artifacts were intentionally removed",
)


def test_postprocess_audit_passes():
    assert audit()["ok"] is True
