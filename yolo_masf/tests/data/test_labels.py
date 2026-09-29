from __future__ import annotations

from pathlib import Path

import pytest

from masf_yolo.data.labels import Box, parse_yolo_label


def test_parse_two_class_detection_labels(tmp_path: Path) -> None:
    label = tmp_path / "frame.txt"
    label.write_text("0 0.5 0.25 0.1 0.05\n1 0.4 0.6 0.2 0.3\n")

    assert parse_yolo_label(label) == (
        Box(0, 0.5, 0.25, 0.1, 0.05),
        Box(1, 0.4, 0.6, 0.2, 0.3),
    )


@pytest.mark.parametrize(
    "row, message",
    [
        ("0 0.5 0.5 0.1", "expected 5"),
        ("2 0.5 0.5 0.1 0.1", "class"),
        ("0 nan 0.5 0.1 0.1", "finite"),
        ("0 1.2 0.5 0.1 0.1", "range"),
        ("0 0.5 0.5 0 0.1", "positive"),
    ],
)
def test_bad_labels_fail_closed(tmp_path: Path, row: str, message: str) -> None:
    label = tmp_path / "bad.txt"
    label.write_text(row + "\n")

    with pytest.raises(ValueError, match=message):
        parse_yolo_label(label)
