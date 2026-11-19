from __future__ import annotations

import csv
from pathlib import Path

from yolo_attention.queue_backend import completed_training_checkpoint
from yolo_attention.run_config import TrainingRecipe


def _partial_run(root: Path) -> Path:
    weights = root / "ultralytics/weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best")
    (weights / "last.pt").write_bytes(b"last")
    results = root / "ultralytics/results.csv"
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch"])
        writer.writeheader()
        writer.writerow({"epoch": 3})
    return root


def test_early_stop_requires_normal_completion_marker(tmp_path: Path) -> None:
    run = _partial_run(tmp_path / "run")
    assert completed_training_checkpoint(run, expected_epochs=8) is None
    (run / "training-complete.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    assert (
        completed_training_checkpoint(run, expected_epochs=8)
        == (run / "ultralytics/weights/best.pt").resolve()
    )


def test_long_phase_patience_is_valid() -> None:
    recipe = TrainingRecipe(
        phase="phase-c",
        parent="phase-b-gate",
        trainable_scope="attention_refinement",
        weights="parent.pt",
        data="data.yaml",
        epochs=24,
        batch=16,
        imgsz=640,
        device="0",
        workers=8,
        seed=2,
        patience=8,
        optimizer="AdamW",
        lr0=2.5e-6,
        scheduler="constant",
        lrf=1.0,
        warmup_epochs=0.0,
        warmup_bias_lr=0.0,
        weight_decay=0.0005,
        selection_metric="metrics/mAP50-95(B)",
        amp=True,
        deterministic=True,
    )
    assert recipe.patience == 8
