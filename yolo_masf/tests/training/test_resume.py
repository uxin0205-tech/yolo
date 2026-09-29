from __future__ import annotations

import pytest

from masf_yolo.training.resume import (
    NonFiniteLossError,
    PermanentTrainingError,
    TransientTrainingError,
    execute_with_retries,
)


def test_transient_failure_resumes_and_stops_after_three_attempts() -> None:
    calls: list[tuple[int, bool]] = []

    def operation(attempt: int, resume: bool) -> str:
        calls.append((attempt, resume))
        raise TransientTrainingError("worker interrupted")

    with pytest.raises(TransientTrainingError):
        execute_with_retries(operation, max_attempts=3)

    assert calls == [(1, False), (2, True), (3, True)]


def test_transient_failure_can_resume_to_success() -> None:
    def operation(attempt: int, resume: bool) -> str:
        if attempt < 3:
            raise TransientTrainingError("temporary")
        assert resume is True
        return "best.pt"

    assert execute_with_retries(operation, max_attempts=3) == "best.pt"


@pytest.mark.parametrize("error", [PermanentTrainingError("hash"), NonFiniteLossError("nan")])
def test_permanent_failures_are_not_retried(error: Exception) -> None:
    attempts = 0

    def operation(attempt: int, resume: bool) -> str:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(type(error)):
        execute_with_retries(operation, max_attempts=3)

    assert attempts == 1
