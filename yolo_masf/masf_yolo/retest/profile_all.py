"""Collect identical static hardware profiles for all retest checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from ..evaluation.profiling import profile_module
from .postprocess import formal_checkpoints, ART


def profile_all() -> None:
    from ultralytics import YOLO

    out = ART / "profiles"
    out.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, 640, 640)
    rows = []
    for name, checkpoint in formal_checkpoints().items():
        path = out / (name.lower().replace(" ", "_") + ".json")
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        model = YOLO(str(checkpoint), task="detect").model
        profile = profile_module(model, sample)
        row = {"name": name, **asdict(profile), "checkpoint": str(checkpoint)}
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    profile_all()
