"""Strict definition and inspection of the pose-derived B0 detection reference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.utils.torch_utils import torch_load

from masf_yolo.artifacts.io import atomic_write_json
from masf_yolo.contracts import sha256_file, sha256_value


@dataclass(frozen=True, slots=True)
class B0ReferenceDefinition:
    reference_id: str
    checkpoint_path: Path
    checkpoint_hash: str
    task: str
    class_names: tuple[str, str]
    strides: tuple[int, int, int]
    ultralytics: str
    provenance: str
    data_exposed: bool
    selection_eligible: bool

    @property
    def definition_hash(self) -> str:
        values = asdict(self)
        values["checkpoint_path"] = str(self.checkpoint_path)
        return sha256_value(values)


_DEFINITION_KEYS = frozenset(
    {
        "reference_id",
        "checkpoint",
        "checkpoint_sha256",
        "task",
        "class_names",
        "strides",
        "ultralytics",
        "provenance",
        "data_exposed",
        "selection_eligible",
    }
)


def load_b0_definition(path: Path) -> B0ReferenceDefinition:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("B0 reference definition must be a mapping")
    unknown = set(raw) - _DEFINITION_KEYS
    if unknown:
        raise ValueError(f"unknown B0 reference keys: {sorted(unknown)}")
    missing = _DEFINITION_KEYS - set(raw)
    if missing:
        raise ValueError(f"missing B0 reference keys: {sorted(missing)}")
    definition = B0ReferenceDefinition(
        reference_id=str(raw["reference_id"]),
        checkpoint_path=Path(str(raw["checkpoint"])).absolute(),
        checkpoint_hash=str(raw["checkpoint_sha256"]),
        task=str(raw["task"]),
        class_names=tuple(str(name) for name in raw["class_names"]),  # type: ignore[arg-type]
        strides=tuple(int(value) for value in raw["strides"]),  # type: ignore[arg-type]
        ultralytics=str(raw["ultralytics"]),
        provenance=str(raw["provenance"]),
        data_exposed=raw["data_exposed"] is True,
        selection_eligible=raw["selection_eligible"] is True,
    )
    locked = (
        definition.reference_id == "B0"
        and definition.task == "detect"
        and definition.class_names == ("ball", "bat")
        and definition.strides == (8, 16, 32)
        and definition.ultralytics == "8.4.90"
        and definition.data_exposed is True
        and definition.selection_eligible is False
        and "pose" in definition.provenance.lower()
    )
    if not locked:
        raise ValueError("B0 reference definition violates the locked research contract")
    return definition


def inspect_b0_reference(definition_path: Path, output_path: Path) -> dict[str, Any]:
    definition = load_b0_definition(definition_path)
    actual_hash = sha256_file(definition.checkpoint_path)
    if actual_hash != definition.checkpoint_hash:
        raise ValueError("B0 checkpoint hash does not match the locked reference")
    if ultralytics.__version__ != definition.ultralytics:
        raise ValueError("B0 Ultralytics version does not match the locked reference")
    checkpoint = torch_load(definition.checkpoint_path, map_location="cpu")
    train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
    wrapper = YOLO(str(definition.checkpoint_path), task="detect")
    model = wrapper.model.float().cpu().eval()
    detect = model.model[-1]
    if not isinstance(detect, Detect):
        raise ValueError("B0 checkpoint does not end in a Detect head")
    names = tuple(model.names[index] for index in range(len(model.names)))
    strides = tuple(float(value) for value in model.stride.tolist())
    if wrapper.task != definition.task:
        raise ValueError(f"B0 task mismatch: {wrapper.task}")
    if names != definition.class_names:
        raise ValueError(f"B0 class order mismatch: {names}")
    if strides != tuple(float(value) for value in definition.strides) or detect.nl != 3:
        raise ValueError(f"B0 detection scale mismatch: strides={strides}, scales={detect.nl}")
    with torch.no_grad():
        model(torch.zeros(1, 3, 640, 640))
    manifest = {
        "reference_id": definition.reference_id,
        "definition_hash": definition.definition_hash,
        "checkpoint": str(definition.checkpoint_path),
        "checkpoint_hash": actual_hash,
        "task": wrapper.task,
        "class_names": list(names),
        "strides": list(strides),
        "detect_scales": detect.nl,
        "ultralytics": ultralytics.__version__,
        "forward_640": True,
        "source_train_task": train_args.get("task"),
        "provenance": definition.provenance,
        "data_exposed": definition.data_exposed,
        "selection_eligible": definition.selection_eligible,
    }
    atomic_write_json(output_path.resolve(), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = inspect_b0_reference(args.definition, args.output)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
