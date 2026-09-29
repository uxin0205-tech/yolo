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
    # The locked evidence directory may be relocated without rewriting its
    # historical manifest.  train.txt and val.txt live beside this YAML, so
    # the operational trainer root must follow the YAML's current location
    # instead of trusting a stale absolute path recorded at audit time.
    view["path"] = str(locked_data_yaml.resolve().parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(view, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return view


def write_evaluation_view(locked_data_yaml: Path, output: Path) -> dict:
    """Relocate all locked split lists without changing their membership."""
    raw = yaml.safe_load(locked_data_yaml.read_text(encoding="utf-8"))
    required = {"path", "train", "val", "test", "nc", "names"}
    if not required.issubset(raw):
        raise ValueError(f"locked data YAML is missing keys: {sorted(required - set(raw))}")
    view = {
        key: raw[key] for key in ("path", "train", "val", "test", "nc", "names")
    }
    view["path"] = str(locked_data_yaml.resolve().parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(view, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return view
