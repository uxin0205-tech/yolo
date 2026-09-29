"""Structured JSONL/CSV/TensorBoard logging and deterministic PNG curves."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

TensorBoardMode = Literal["off", "auto", "required"]


class ExperimentLogger:
    """One append-only logging interface for macro and epoch events."""

    def __init__(
        self,
        directory: str | Path,
        *,
        tensorboard: TensorBoardMode = "auto",
    ) -> None:
        if tensorboard not in {"off", "auto", "required"}:
            raise ValueError("tensorboard must be off, auto, or required")
        self.root = Path(directory).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.capability_path = self.root / "logging-capabilities.json"
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._closed = False
        self._writer: Any | None = None
        tensorboard_reason: str | None = None
        if tensorboard != "off":
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ModuleNotFoundError as error:
                tensorboard_reason = f"{type(error).__name__}: {error}"
                if tensorboard == "required":
                    raise RuntimeError(
                        "TensorBoard logging was required but tensorboard is not installed"
                    ) from error
            else:
                self._writer = SummaryWriter(log_dir=str(self.root / "tensorboard"))
        else:
            tensorboard_reason = "disabled by configuration"
        self.capabilities = {
            "jsonl": True,
            "csv": True,
            "png": True,
            "tensorboard": {
                "mode": tensorboard,
                "enabled": self._writer is not None,
                "reason": tensorboard_reason,
            },
        }
        self.capability_path.write_text(
            json.dumps(self.capabilities, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._load_history()

    def _load_history(self) -> None:
        if not self.events_path.is_file():
            return
        for line_number, line in enumerate(
            self.events_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or "kind" not in payload:
                raise ValueError(
                    f"malformed log record at {self.events_path}:{line_number}"
                )
            self._history.setdefault(str(payload["kind"]), []).append(payload)

    @staticmethod
    def _values(values: Mapping[str, float | int]) -> dict[str, float]:
        resolved = {str(name): float(value) for name, value in values.items()}
        invalid = {
            name: value
            for name, value in resolved.items()
            if not math.isfinite(value)
        }
        if invalid:
            raise ValueError(f"log values must be finite: {invalid}")
        return resolved

    def log(
        self,
        kind: str,
        *,
        step: int,
        values: Mapping[str, float | int],
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("logger is closed")
        if not kind or "/" in kind or "\\" in kind:
            raise ValueError("kind must be a simple non-empty label")
        if step < 0:
            raise ValueError("step cannot be negative")
        resolved = self._values(values)
        record = {
            "kind": kind,
            "step": int(step),
            "time_unix": time.time(),
            "values": resolved,
            "context": dict(context or {}),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        csv_path = self.root / f"{kind}.csv"
        new_file = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(("step", "time_unix", "metric", "value"))
            for name, value in sorted(resolved.items()):
                writer.writerow((step, record["time_unix"], name, value))
        self._history.setdefault(kind, []).append(record)
        if self._writer is not None:
            for name, value in resolved.items():
                self._writer.add_scalar(f"{kind}/{name}", value, step)
            self._writer.flush()

    def plot(
        self,
        kind: str,
        *,
        metrics: Sequence[str] | None = None,
    ) -> Path:
        records = self._history.get(kind, [])
        if not records:
            raise ValueError(f"there are no {kind!r} records to plot")
        available = sorted(
            {
                name
                for record in records
                for name in record["values"]
            }
        )
        selected = list(metrics) if metrics is not None else available
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"plot metrics were never logged: {missing}")
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for name in selected:
            points = [
                (int(record["step"]), float(record["values"][name]))
                for record in records
                if name in record["values"]
            ]
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=1.5,
                label=name,
            )
        axis.set_xlabel("step")
        axis.set_ylabel("value")
        axis.set_title(f"{kind} metrics")
        axis.grid(True, alpha=0.3)
        if selected:
            axis.legend(fontsize="small")
        plot_dir = self.root / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        target = plot_dir / f"{kind}-curves.png"
        figure.savefig(target, dpi=160)
        plt.close(figure)
        return target

    def close(self) -> None:
        if self._closed:
            return
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        self._closed = True

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
