"""Crash-safe JSON writes and process-level pipeline locking."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Any


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class PipelineLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = None

    def __enter__(self) -> "PipelineLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.close()
            raise RuntimeError(f"pipeline already locked: {self.path}") from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())
        self._stream = stream
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None
