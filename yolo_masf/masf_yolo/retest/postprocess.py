"""Unified validation/test, lineage, profiling and report input generation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from ..evaluation.runner import run_variant_evaluation
from ..contracts import sha256_file

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "b1r-p2-p3-retest"
DATA = ROOT / "artifacts" / "static-phase1" / "dataset"
SOURCE = ROOT / "bbt5-detect-baseline" / "weights" / "yolo11m_bat_detect_init.pt"
RUN_ROOT = Path("/home/uxin/single_view_pipeline/runs/detect/artifacts/b1r-p2-p3-retest/training")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def formal_checkpoints() -> dict[str, Path]:
    result: dict[str, Path] = {
        "B0-Original-3Scale": SOURCE,
        "P2-Base-Direct": RUN_ROOT / "direct/weights/best.pt",
        "P2-Control-Head": RUN_ROOT / "bbt5-p2-control-head/weights/best.pt",
        "P2-Control-Full": RUN_ROOT / "bbt5-p2-control-full/weights/best.pt",
    }
    for family in ("p2", "p3"):
        for path in sorted((ART / "worker").glob(f"queue_{family}_formal_*.json")):
            record = _json(path)
            variant = record["variant"]
            result[f"{family.upper()}-{variant}"] = Path(record["best"])
    return result


def write_lineage(checkpoints: dict[str, Path]) -> None:
    root = ART / "lineage"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, checkpoint in checkpoints.items():
        rows.append({"name": name, "checkpoint": str(checkpoint), "exists": checkpoint.is_file(), "sha256": sha256_file(checkpoint) if checkpoint.is_file() else None})
    (root / "checkpoints.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_all() -> None:
    checkpoints = formal_checkpoints()
    write_lineage(checkpoints)
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint for {name}: {checkpoint}")
        family_dir = name.lower().replace(" ", "_")
        for split in ("val", "test"):
            coco = DATA / f"{split}.coco.json"
            output = ART / "evaluation" / split / family_dir
            metrics_path = output / "metrics.json"
            if metrics_path.is_file():
                continue
            print(f"[{split}] {name}", flush=True)
            run_variant_evaluation(checkpoint, DATA / "data.yaml", coco, split=split, output_dir=output, device=0)


if __name__ == "__main__":
    evaluate_all()
