"""Model construction and explicit YOLO11 Detect weight transfer."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from p2_study import ROOT
from ultralytics import YOLO

MODEL_YAMLS = {
    "A0": ROOT / "ultralytics/cfg/models/11/yolo11m.yaml",
    "A1": ROOT / "ultralytics/cfg/models/11/yolo11m-p2.yaml",
    "A2": ROOT / "ultralytics/cfg/models/11/yolo11m-p2.yaml",
}

# Freeze the official backbone/neck and copied P3/P4/P5 Detect towers during A2 stage 1.
A2_HEAD_FREEZE = (
    *range(23),
    "26.cv2.1",
    "26.cv2.2",
    "26.cv2.3",
    "26.cv3.1",
    "26.cv3.2",
    "26.cv3.3",
)


def build_model(experiment: str) -> YOLO:
    """Build one study model from its immutable YAML definition."""
    return YOLO(MODEL_YAMLS[experiment])


def transfer_pretrained(source: YOLO, target: YOLO, manifest_path: str | Path | None = None) -> dict:
    """Transfer shared layers and explicitly shift P3/P4/P5 Detect tensors by one branch.

    Args:
        source (YOLO): Official three-scale pretrained model.
        target (YOLO): New four-scale P2 model.
        manifest_path (str | Path, optional): JSON destination for tensor-level evidence.

    Returns:
        (dict): Auditable transfer summary and every source/destination tensor mapping.
    """
    source_state = source.model.state_dict()
    target_state = target.model.state_dict()
    source_detect = len(source.model.model) - 1
    target_detect = len(target.model.model) - 1
    copied = {}

    for key, value in source_state.items():
        match = re.match(rf"model\.{source_detect}\.(cv[23])\.(\d+)\.(.+)", key)
        if match:
            old_index = int(match.group(2))
            if old_index < 3:
                destination = f"model.{target_detect}.{match.group(1)}.{old_index + 1}.{match.group(3)}"
                if destination not in target_state or target_state[destination].shape != value.shape:
                    raise ValueError(f"Detect mapping shape mismatch: {key} -> {destination}")
                copied[key] = destination
            continue
        if key.startswith(f"model.{source_detect}."):
            continue
        if key in target_state and target_state[key].shape == value.shape:
            copied[key] = key

    updated = dict(target_state)
    for source_key, target_key in copied.items():
        updated[target_key] = source_state[source_key].detach().clone()
    missing, unexpected = target.model.load_state_dict(updated, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Strict transfer failed: missing={missing}, unexpected={unexpected}")

    verification = []
    loaded_state = target.model.state_dict()
    for source_key, target_key in copied.items():
        source_tensor, target_tensor = source_state[source_key], loaded_state[target_key]
        equal = source_tensor.shape == target_tensor.shape and torch.equal(source_tensor, target_tensor)
        if not equal:
            raise RuntimeError(f"Transferred tensor differs: {source_key} -> {target_key}")
        verification.append(
            {
                "source": source_key,
                "target": target_key,
                "shape": list(source_tensor.shape),
                "equal": True,
            }
        )

    detect_count = sum(source.startswith(f"model.{source_detect}.cv") for source in copied)
    manifest = {
        "source_detect_index": source_detect,
        "target_detect_index": target_detect,
        "loaded_tensors": len(copied),
        "detect_tensors": detect_count,
        "randomly_initialized_tensors": sorted(set(target_state) - set(copied.values())),
        "verified": True,
        "mappings": verification,
    }
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_initial_weights(pretrained: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Create the immutable initial checkpoints used by every gate and formal run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = YOLO(pretrained)
    checkpoints = {"A0": Path(pretrained).resolve()}
    for experiment in ("A1",):
        target = build_model(experiment)
        transfer_pretrained(source, target, output / f"{experiment.lower()}_weight_transfer.json")
        checkpoint = output / f"{experiment.lower()}_initial.pt"
        target.save(checkpoint)
        checkpoints[experiment] = checkpoint
    return checkpoints
