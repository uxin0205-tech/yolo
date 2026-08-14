"""Materialize a trainer view that cannot address the historical test split."""

from __future__ import annotations

from pathlib import Path

import yaml


def write_train_val_view(locked_data_yaml: Path, output: Path) -> dict:
    raw = yaml.safe_load(locked_data_yaml.read_text(encoding="utf-8"))
    required = {"path", "train", "val", "nc", "names"}
    if not required.issubset(raw):
        raise ValueError(f"locked data YAML is missing keys: {sorted(required - set(raw))}")
    view = {key: raw[key] for key in ("path", "train", "val", "nc", "names")}
    view["path"] = str(Path(str(raw["path"])).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(view, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return view
