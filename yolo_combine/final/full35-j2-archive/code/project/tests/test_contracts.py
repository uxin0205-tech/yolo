import pytest

from yolo_combine.contracts import ALL_TASKS, Task, normalize_tasks


@pytest.mark.parametrize("value", [None, "both", [Task.DETECT, Task.POSE]])
def test_both_selects_every_task(value):
    assert normalize_tasks(value) == ALL_TASKS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("detect", frozenset({Task.DETECT})),
        (Task.POSE, frozenset({Task.POSE})),
        (["detect"], frozenset({Task.DETECT})),
        (["both"], ALL_TASKS),
    ],
)
def test_single_and_iterable_task_selection(value, expected):
    assert normalize_tasks(value) == expected


@pytest.mark.parametrize("value", ["unknown", [], ["detect", "unknown"]])
def test_invalid_task_selection_fails_closed(value):
    with pytest.raises(ValueError):
        normalize_tasks(value)
