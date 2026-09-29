"""Repair a completed pre-EMA formal artifact from Ultralytics ``last.pt``."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .model import build_student
from .trainer import freeze_to_attention_only
from .variants.definitions import get_variant, variant_from_resolved_config
from .workflow import PLAN_NAME, _git_revision, finalize_training_artifact


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _variant_from_run(run: Path):
    resolved = json.loads((run / "resolved_config.json").read_text())
    return variant_from_resolved_config(resolved)


def _ema_checkpoint_epoch(run: Path, payload: dict) -> tuple[int, str]:
    """Recover the completed epoch when Ultralytics stripped it from last.pt."""

    checkpoint_epoch = payload.get("epoch", -1)
    if isinstance(checkpoint_epoch, int) and checkpoint_epoch >= 0:
        return checkpoint_epoch + 1, "last.pt epoch metadata"
    curves = run / "training_curves.csv"
    if not curves.exists():
        curves = run / "ultralytics" / "train" / "results.csv"
    try:
        with curves.open(newline="") as handle:
            completed_epochs = sum(1 for _row in csv.DictReader(handle))
    except OSError:
        completed_epochs = 0
    if completed_epochs <= 0:
        raise ValueError("cannot recover EMA checkpoint epoch from stripped last.pt or training curves")
    return completed_epochs, f"{curves.name} completed-row count; last.pt epoch metadata stripped"


def repair_run(run: Path, source_weights: Path) -> Path:
    """Replace a raw-model strict checkpoint with its matching epoch EMA."""

    run = run.resolve()
    status_path = run / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("completed") is not True or status.get("valid_for_research") is not True:
        raise ValueError(f"run is not a completed research artifact: {run}")
    last = run / "ultralytics" / "train" / "weights" / "last.pt"
    if not last.exists():
        raise FileNotFoundError(f"Ultralytics last.pt is missing: {last}")
    payload = torch.load(last, map_location="cpu", weights_only=False)
    serialized = payload.get("model") or payload.get("ema") if isinstance(payload, dict) else None
    if not hasattr(serialized, "state_dict"):
        raise TypeError("last.pt does not contain a serialized EMA/model module")

    variant = _variant_from_run(run)
    model = build_student(run / "model.yaml", variant)
    model.load_state_dict(serialized.float().state_dict(), strict=True)
    freeze_to_attention_only(model)
    finalize_training_artifact(run, model, variant, source_weights=source_weights)
    ema_checkpoint_epoch, epoch_evidence = _ema_checkpoint_epoch(run, payload)
    provenance = {
        "checkpoint_weight_source": "epoch_ema",
        "metrics_weight_source": "epoch_ema",
        "ema_repair_source": str(last.resolve()),
        "ema_checkpoint_epoch": ema_checkpoint_epoch,
        "ema_checkpoint_epoch_evidence": epoch_evidence,
    }
    for name in ("status.json", "training_args.json"):
        path = run / name
        value = json.loads(path.read_text())
        value.update(provenance)
        _write_json(path, value)
    environment_path = run / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment.update({
        "ema_repair_revision": _git_revision(),
        "ema_repair_source": str(last.resolve()),
    })
    _write_json(environment_path, environment)
    return run / "checkpoints" / "strict-last.pt"


def latest_repairable(root: Path, variant_id: str) -> Path | None:
    artifact = get_variant(variant_id).artifact_name
    candidates = []
    for run in (root / "artifacts" / "runs" / artifact / "full").glob("*"):
        try:
            status = json.loads((run / "status.json").read_text())
            resolved = json.loads((run / "resolved_config.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            status.get("completed") is True
            and status.get("valid_for_research") is True
            and status.get("checkpoint_weight_source") != "epoch_ema"
            and resolved.get("plan_name") == PLAN_NAME
            and (run / "ultralytics" / "train" / "weights" / "last.pt").exists()
        ):
            candidates.append((run.stat().st_mtime_ns, run))
    return max(candidates)[1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--latest-variant")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-weights", type=Path, default=Path("../original/weight/yolo11m.pt"))
    args = parser.parse_args()
    run = args.run or (latest_repairable(args.root, args.latest_variant) if args.latest_variant else None)
    if run is None:
        print("no repairable artifact")
        return
    print(repair_run(run, args.source_weights.resolve()))


if __name__ == "__main__":
    main()
