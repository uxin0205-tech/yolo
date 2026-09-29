import pytest

from masf_yolo.retest.postprocess import ART
from masf_yolo.retest.report import build_summary

pytestmark = pytest.mark.skipif(
    not (ART / "queue_state.json").is_file(),
    reason="legacy data-exposed retest artifacts were intentionally removed",
)


def test_summary_builder_has_unified_metrics():
    result = build_summary()
    assert result["models"] >= 28
    assert result["best_test"]["split"] == "test"
