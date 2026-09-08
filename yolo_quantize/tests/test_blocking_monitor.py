import json

import pytest

from yolo_quantize.blocking_monitor import snapshot, wait_for_event


def payload(**updates):
    value = {
        "status": "running",
        "current_index": 0,
        "current_candidate": "a",
        "current_arm": "qat",
        "completed_jobs": [],
        "error": None,
    }
    return value | updates


@pytest.mark.parametrize(
    "status",
    ["complete", "error", "paused", "cpu_preflight_complete_pending_integration"],
)
def test_initial_terminal_does_not_sleep(tmp_path, status):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(payload(status=status)))
    assert (
        wait_for_event(path, sleep=lambda _: pytest.fail("must not sleep"))["status"]
        == status
    )


def test_silent_unchanged_then_changed(tmp_path, capsys):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(payload()))
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) == 3:
            path.write_text(json.dumps(payload(completed_jobs=["done"])))

    result = wait_for_event(path, sleep=sleep)
    assert calls == [600, 600, 600]
    assert result["completed_jobs"] == 1
    assert capsys.readouterr().out == ""


def test_missing_file_returns_error(tmp_path):
    assert snapshot(tmp_path / "missing")["error"]["kind"] == "monitor_read_error"


def test_invalid_count_returns_error(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(payload(completed_jobs=True)))
    assert snapshot(path)["status"] == "error"


def test_nonnull_error_returns_without_wait(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps(payload(error="failed")))
    assert (
        wait_for_event(path, sleep=lambda _: pytest.fail("must not sleep"))["error"]
        == "failed"
    )
