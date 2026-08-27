"""Small public task vocabulary shared by routed and fused models."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class Task(StrEnum):
    """A model responsibility, not a dataset class name."""

    DETECT = "detect"
    POSE = "pose"


ALL_TASKS = frozenset(Task)


def normalize_tasks(tasks: Task | str | Iterable[Task | str] | None) -> frozenset[Task]:
    """Normalize the compact public task selector and fail closed on unknown tasks."""

    if tasks is None or tasks == "both":
        return ALL_TASKS
    if isinstance(tasks, (Task, str)):
        raw_tasks: Iterable[Task | str] = (tasks,)
    else:
        raw_tasks = tasks
    normalized: set[Task] = set()
    for task in raw_tasks:
        if task == "both":
            normalized.update(ALL_TASKS)
            continue
        try:
            normalized.add(Task(task))
        except ValueError as exc:
            raise ValueError(f"unknown task {task!r}; expected detect, pose, or both") from exc
    if not normalized:
        raise ValueError("at least one task is required")
    return frozenset(normalized)
