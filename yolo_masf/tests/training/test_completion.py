from __future__ import annotations

from pathlib import Path

import pytest
import torch

from masf_yolo.training.completion import (
    classify_training_output,
    ensure_complete_training_output,
)
from masf_yolo.training.resume import PermanentTrainingError, TransientTrainingError


HEADER = "epoch,time,train/box_loss,metrics/mAP50-95(B)\n"


def _checkpoint(path: Path, epochs: int, *, epoch: int = -1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "train_args": {"epochs": epochs}, "model": None, "ema": None}, path)


def _results(path: Path, epochs: int, *, final_value: str = "0.4") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [HEADER]
    for epoch in range(1, epochs + 1):
        value = final_value if epoch == epochs else "0.3"
        rows.append(f"{epoch},{epoch * 10.0},0.5,{value}\n")
    path.write_text("".join(rows), encoding="utf-8")


def test_absent_output_starts_new_training(tmp_path: Path) -> None:
    state = classify_training_output(tmp_path / "run", expected_epochs=3)

    assert state.status == "absent"
    assert state.completed_epochs == 0
    assert state.best is None
    assert state.last is None


def test_incomplete_output_uses_loadable_native_last(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _results(run / "results.csv", 1)
    _checkpoint(run / "weights" / "last.pt", 3, epoch=0)

    state = classify_training_output(run, expected_epochs=3)

    assert state.status == "incomplete"
    assert state.completed_epochs == 1
    assert state.last == (run / "weights" / "last.pt").resolve()
    assert state.last_hash is not None
    assert state.results_hash is not None


def test_complete_stripped_output_skips_resume(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _results(run / "results.csv", 3)
    _checkpoint(run / "weights" / "last.pt", 3, epoch=-1)
    _checkpoint(run / "weights" / "best.pt", 3, epoch=-1)

    state = classify_training_output(run, expected_epochs=3)

    assert state.status == "complete"
    assert state.completed_epochs == 3
    assert state.best == (run / "weights" / "best.pt").resolve()
    assert state.last == (run / "weights" / "last.pt").resolve()
    assert all((state.results_hash, state.best_hash, state.last_hash))


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("too_many", "exceeds"),
        ("nonfinite", "non-finite"),
        ("mismatch", "checkpoint epochs"),
        ("missing_best", "best.pt"),
        ("unreadable", "unreadable"),
        ("nonconsecutive", "consecutive"),
    ],
)
def test_contradictory_outputs_fail_closed(tmp_path: Path, case: str, match: str) -> None:
    run = tmp_path / "run"
    row_count = 4 if case == "too_many" else 3
    _results(run / "results.csv", row_count, final_value="nan" if case == "nonfinite" else "0.4")
    if case == "nonconsecutive":
        text = (run / "results.csv").read_text().replace("2,20.0", "7,20.0")
        (run / "results.csv").write_text(text)
    if case == "unreadable":
        path = run / "weights" / "last.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a checkpoint")
    else:
        _checkpoint(run / "weights" / "last.pt", 4 if case == "mismatch" else 3)
    if case != "missing_best":
        _checkpoint(run / "weights" / "best.pt", 3)

    state = classify_training_output(run, expected_epochs=3)

    assert state.status == "invalid"
    assert match in state.reason


def test_completed_output_never_launches_worker_or_resume(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _results(run / "results.csv", 3)
    _checkpoint(run / "weights" / "last.pt", 3)
    _checkpoint(run / "weights" / "best.pt", 3)
    calls: list[Path | None] = []

    state = ensure_complete_training_output(
        run,
        expected_epochs=3,
        launch_worker=lambda resume: calls.append(resume),
        max_attempts=3,
    )

    assert state.status == "complete"
    assert calls == []


def test_incomplete_output_passes_exact_last_to_native_resume(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _results(run / "results.csv", 1)
    _checkpoint(run / "weights" / "last.pt", 3, epoch=0)
    calls: list[Path | None] = []

    def finish(resume: Path | None) -> None:
        calls.append(resume)
        _results(run / "results.csv", 3)
        _checkpoint(run / "weights" / "last.pt", 3)
        _checkpoint(run / "weights" / "best.pt", 3)

    state = ensure_complete_training_output(
        run,
        expected_epochs=3,
        launch_worker=finish,
        max_attempts=3,
    )

    assert state.status == "complete"
    assert calls == [(run / "weights" / "last.pt").resolve()]


def test_invalid_output_never_launches_worker(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _results(run / "results.csv", 4)
    _checkpoint(run / "weights" / "last.pt", 3)
    calls = 0

    def launch(_resume: Path | None) -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(PermanentTrainingError, match="exceeds"):
        ensure_complete_training_output(run, 3, launch, max_attempts=3)

    assert calls == 0


def test_worker_that_does_not_advance_retries_only_three_times(tmp_path: Path) -> None:
    calls: list[Path | None] = []

    with pytest.raises(TransientTrainingError, match="did not produce"):
        ensure_complete_training_output(
            tmp_path / "run",
            expected_epochs=3,
            launch_worker=lambda resume: calls.append(resume),
            max_attempts=3,
        )

    assert calls == [None, None, None]
