"""Atomic persistence and single-worker locking for experiment queues."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .queue_model import QueueState


class QueueLockedError(RuntimeError):
    pass


class QueueStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.state_path = self.root / "queue.json"
        self.temporary_path = self.root / "queue.json.tmp"
        self.events_path = self.root / "events.jsonl"
        self.lock_path = self.root / "worker.lock"
        self.generated_root = self.root / "generated"

    def initialize(self, state: QueueState) -> None:
        if self.state_path.exists():
            raise FileExistsError(f"queue already exists: {self.state_path}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.generated_root.mkdir(exist_ok=True)
        self.save(state)

    def load(self) -> QueueState:
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("queue.json must contain an object")
        return QueueState.from_dict(data)

    def save(self, state: QueueState) -> None:
        state.validate()
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with self.temporary_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(self.temporary_path, self.state_path)

    def append_event(
        self,
        event: str,
        *,
        job_id: str | None,
        details: dict[str, Any],
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "job_id": job_id,
            "details": details,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def worker_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise QueueLockedError(f"queue worker lock is held: {self.lock_path}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
