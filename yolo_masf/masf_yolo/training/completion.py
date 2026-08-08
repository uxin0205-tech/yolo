"""Fail-closed classification of native Ultralytics training outputs."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable

from ultralytics.utils.torch_utils import torch_load

from masf_yolo.contracts import sha256_file

from .resume import PermanentTrainingError, TransientTrainingError, execute_with_retries


TrainingOutputStatus = Literal["absent", "incomplete", "complete", "invalid"]


@dataclass(frozen=True, slots=True)
class TrainingOutputState:
    status: TrainingOutputStatus
    completed_epochs: int
    expected_epochs: int
    best: Path | None
    last: Path | None
    results_hash: str | None
    best_hash: str | None
    last_hash: str | None
    reason: str


def _state(
    status: TrainingOutputStatus,
    expected_epochs: int,
    *,
    completed_epochs: int = 0,
    best: Path | None = None,
    last: Path | None = None,
    results_hash: str | None = None,
    best_hash: str | None = None,
    last_hash: str | None = None,
    reason: str,
) -> TrainingOutputState:
    return TrainingOutputState(
        status=status,
        completed_epochs=completed_epochs,
        expected_epochs=expected_epochs,
        best=best,
        last=last,
        results_hash=results_hash,
        best_hash=best_hash,
        last_hash=last_hash,
        reason=reason,
    )


def _checkpoint_epochs(path: Path) -> int:
    try:
        checkpoint: Any = torch_load(path, map_location="cpu")
    except BaseException as error:
        raise ValueError(f"unreadable checkpoint {path.name}: {type(error).__name__}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError(f"unreadable checkpoint {path.name}: root is not a mapping")
    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, dict) or not isinstance(train_args.get("epochs"), int):
        raise ValueError(f"unreadable checkpoint {path.name}: missing train_args.epochs")
    return int(train_args["epochs"])


def _read_results(path: Path) -> tuple[int, str | None]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeError) as error:
        return 0, f"unreadable results.csv: {type(error).__name__}"
    if not rows:
        return 0, "results.csv has no epoch rows"
    expected = list(range(1, len(rows) + 1))
    try:
        epochs = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return len(rows), "results.csv epoch values are invalid"
    if epochs != expected:
        return len(rows), "results.csv epochs must be consecutive from 1"
    for row in rows:
        for key, raw in row.items():
            if key == "epoch":
                continue
            try:
                finite = math.isfinite(float(raw))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                return len(rows), f"results.csv contains non-finite value in {key}"
    return len(rows), None


def classify_training_output(run_dir: Path, expected_epochs: int) -> TrainingOutputState:
    """Classify a run before any GPU model is built or resume is attempted."""
    if expected_epochs < 1:
        raise ValueError("expected_epochs must be positive")
    run_dir = run_dir.resolve()
    results = run_dir / "results.csv"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not any(path.exists() for path in (results, best, last)):
        return _state("absent", expected_epochs, reason="no native training output")
    if not results.is_file():
        return _state("invalid", expected_epochs, reason="missing results.csv beside checkpoint output")
    completed_epochs, results_error = _read_results(results)
    results_hash = sha256_file(results)
    if results_error:
        return _state(
            "invalid",
            expected_epochs,
            completed_epochs=completed_epochs,
            results_hash=results_hash,
            reason=results_error,
        )
    if completed_epochs > expected_epochs:
        return _state(
            "invalid",
            expected_epochs,
            completed_epochs=completed_epochs,
            results_hash=results_hash,
            reason=f"results.csv exceeds expected epochs: {completed_epochs} > {expected_epochs}",
        )
    if not last.is_file():
        return _state(
            "invalid",
            expected_epochs,
            completed_epochs=completed_epochs,
            results_hash=results_hash,
            reason="missing last.pt for existing results.csv",
        )
    try:
        last_epochs = _checkpoint_epochs(last)
    except ValueError as error:
        return _state(
            "invalid",
            expected_epochs,
            completed_epochs=completed_epochs,
            results_hash=results_hash,
            reason=str(error),
        )
    last_hash = sha256_file(last)
    if last_epochs != expected_epochs:
        return _state(
            "invalid",
            expected_epochs,
            completed_epochs=completed_epochs,
            last=last.resolve(),
            results_hash=results_hash,
            last_hash=last_hash,
            reason=f"checkpoint epochs mismatch: {last_epochs} != {expected_epochs}",
        )
    if completed_epochs < expected_epochs:
        return _state(
            "incomplete",
            expected_epochs,
            completed_epochs=completed_epochs,
            best=best.resolve() if best.is_file() else None,
            last=last.resolve(),
            results_hash=results_hash,
            best_hash=sha256_file(best) if best.is_file() else None,
            last_hash=last_hash,
            reason=f"native resume available after {completed_epochs} epochs",
        )
    if not best.is_file():
        return _state(
            "invalid",
            expected_epochs,
            completed_epochs=completed_epochs,
            last=last.resolve(),
            results_hash=results_hash,
            last_hash=last_hash,
            reason="missing best.pt for completed training output",
        )
    try:
        best_epochs = _checkpoint_epochs(best)
    except ValueError as error:
        return _state(
            "invalid",
            expected_epochs,
            completed_epochs=completed_epochs,
            best=best.resolve(),
            last=last.resolve(),
            results_hash=results_hash,
            last_hash=last_hash,
            reason=str(error),
        )
    if best_epochs != expected_epochs:
        return _state(
            "invalid",
            expected_epochs,
            completed_epochs=completed_epochs,
            best=best.resolve(),
            last=last.resolve(),
            results_hash=results_hash,
            best_hash=sha256_file(best),
            last_hash=last_hash,
            reason=f"best checkpoint epochs mismatch: {best_epochs} != {expected_epochs}",
        )
    return _state(
        "complete",
        expected_epochs,
        completed_epochs=completed_epochs,
        best=best.resolve(),
        last=last.resolve(),
        results_hash=results_hash,
        best_hash=sha256_file(best),
        last_hash=last_hash,
        reason=f"completed all {expected_epochs} epochs",
    )


def ensure_complete_training_output(
    run_dir: Path,
    expected_epochs: int,
    launch_worker: Callable[[Path | None], None],
    *,
    max_attempts: int,
) -> TrainingOutputState:
    """Return a completed run, launching only absent or resumable native work."""

    def operation(_attempt: int, _resume: bool) -> TrainingOutputState:
        before = classify_training_output(run_dir, expected_epochs)
        if before.status == "complete":
            return before
        if before.status == "invalid":
            raise PermanentTrainingError(before.reason)
        launch_worker(before.last if before.status == "incomplete" else None)
        after = classify_training_output(run_dir, expected_epochs)
        if after.status == "complete":
            return after
        if after.status == "invalid":
            raise PermanentTrainingError(after.reason)
        raise TransientTrainingError(
            f"training worker did not produce a complete output: {after.reason}"
        )

    return execute_with_retries(operation, max_attempts=max_attempts)
