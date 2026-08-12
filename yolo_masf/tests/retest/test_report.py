from masf_yolo.retest.report import build_summary


def test_summary_builder_has_unified_metrics():
    result = build_summary()
    assert result["models"] >= 28
    assert result["best_test"]["split"] == "test"
