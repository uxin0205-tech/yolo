"""Bounded transient retry policy for native Ultralytics resume."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


Result = TypeVar("Result")


class TransientTrainingError(RuntimeError):
    pass


class PermanentTrainingError(RuntimeError):
    pass


class NonFiniteLossError(PermanentTrainingError):
    pass


def execute_with_retries(
    operation: Callable[[int, bool], Result],
    *,
    max_attempts: int = 3,
) -> Result:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation(attempt, attempt > 1)
        except TransientTrainingError:
            if attempt == max_attempts:
                raise
    raise AssertionError("retry loop is unreachable")
