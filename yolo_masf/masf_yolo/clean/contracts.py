"""Locked contracts for the clean-initializer comparison."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..contracts import sha256_file, sha256_value


@dataclass(frozen=True, slots=True)
class CleanExperimentSpec:
    name: str
    family: str
    variant: str | None
    schedule: str
    comparison_tier: str
    parent_required: bool = False


CLEAN_EXPERIMENTS = {
    "B0-Clean": CleanExperimentSpec("B0-Clean", "B0", None, "direct100", "strict_fair"),
    "P3-PaperFormula-Clean": CleanExperimentSpec(
        "P3-PaperFormula-Clean", "P3", "PaperFormula-Full", "direct100", "strict_fair"
    ),
    "P3-Partial25-Clean": CleanExperimentSpec(
        "P3-Partial25-Clean", "P3", "Partial25-35", "direct100", "strict_fair"
    ),
    "P2-Direct-Clean": CleanExperimentSpec(
        "P2-Direct-Clean", "P2", None, "direct100", "strict_fair"
    ),
    "P2-Control-Clean-Head": CleanExperimentSpec(
        "P2-Control-Clean-Head", "P2", None, "head20", "optimization_control"
    ),
    "P2-Control-Clean-Full": CleanExperimentSpec(
        "P2-Control-Clean-Full", "P2", None, "full80", "optimization_control", True
    ),
}


@dataclass(frozen=True, slots=True)
class CleanStudyConfig:
    values: dict[str, Any]
    root: Path

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, root: Path) -> "CleanStudyConfig":
        values = deepcopy(dict(raw))
        if values.get("schema_version") != 1 or values.get("study_name") != "clean-bbt5-ablation":
            raise ValueError("invalid clean study identity")
        if values.get("environment", {}).get("ultralytics") != "8.4.90":
            raise ValueError("clean study requires the pinned Ultralytics 8.4.90 environment")
        initializer = values.get("initializer", {})
        if initializer.get("sha256") != "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95":
            raise ValueError("clean initializer hash is not the pinned Ultralytics YOLO11m asset")
        if initializer.get("expected_nc") != 80:
            raise ValueError("clean initializer must retain the 80-class COCO head")
        dataset = values.get("dataset", {})
        if dataset.get("dataset_hash") != "6e16c975941dbae2af174a2eb4b5424bffd4736c74aad56d424805da019b8fbc":
            raise ValueError("clean study must use the audited BBT5 dataset")
        visibility = dataset.get("visibility", {})
        if visibility.get("test") != "historical_already_observed":
            raise ValueError("the existing test split must be labelled historical")
        if visibility.get("final_holdout") is not None:
            raise ValueError("final_holdout must remain null until genuinely new data is registered")
        if tuple(values.get("seeds", ())) != (42, 43, 44):
            raise ValueError("clean study requires seeds 42, 43, and 44")
        if tuple(values.get("experiments", ())) != tuple(CLEAN_EXPERIMENTS):
            raise ValueError("clean experiment order does not match the locked matrix")
        training = values.get("training", {})
        expected = {
            "imgsz": 640, "batch": 16, "optimizer": "SGD", "momentum": 0.937,
            "cos_lr": True, "deterministic": True, "amp": True, "nbs": 64,
            "direct_epochs": 100, "direct_lr0": 0.001,
            "control_head_epochs": 20, "control_head_lr0": 0.01,
            "control_full_epochs": 80, "control_full_lr0": 0.001,
        }
        if any(training.get(key) != value for key, value in expected.items()):
            raise ValueError("clean training settings differ from the locked fair schedule")
        return cls(values, root.resolve())

    @property
    def config_hash(self) -> str:
        return sha256_value(self.values)

    @property
    def initializer_path(self) -> Path:
        return (self.root / self.values["initializer"]["path"]).resolve()

    @property
    def locked_data_yaml(self) -> Path:
        return (self.root / self.values["dataset"]["locked_data_yaml"]).resolve()

    def verify_initializer(self) -> None:
        if sha256_file(self.initializer_path) != self.values["initializer"]["sha256"]:
            raise RuntimeError("clean initializer file hash mismatch")

    def experiment(self, name: str) -> CleanExperimentSpec:
        try:
            return CLEAN_EXPERIMENTS[name]
        except KeyError as error:
            raise ValueError(f"unsupported clean experiment: {name}") from error

    def assert_split_use(self, *, split: str, purpose: str) -> None:
        allowed = {
            ("train", "fit"),
            ("val", "selection"),
            ("test", "historical_report"),
            ("final_holdout", "unseen_claim"),
        }
        if (split, purpose) not in allowed:
            raise ValueError(f"forbidden split use: {split} for {purpose}")
        if split == "final_holdout" and self.values["dataset"]["visibility"]["final_holdout"] is None:
            raise RuntimeError("unseen claims are blocked until a new final holdout is registered")


def load_clean_config(path: Path | str) -> CleanStudyConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return CleanStudyConfig.from_mapping(raw, root=config_path.parents[2])
