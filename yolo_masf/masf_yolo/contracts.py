"""Immutable, JSON-safe contracts shared by the Phase 1 pipeline."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping, Self


PIPELINE_SCHEMA_VERSION = 1
PHASE1_VARIANTS = ("B1", "M7", "M0", "M1", "M2", "M3", "P3M", "SP2")
CLASS_NAMES = ("ball", "bat")
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def canonical_json(value: object) -> bytes:
    """Return stable UTF-8 JSON bytes suitable for hashing."""
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_value(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class JsonManifest:
    """Dataclass serialization that restores typed Paths and tuples."""

    _path_fields: ClassVar[frozenset[str]] = frozenset()
    _tuple_fields: ClassVar[frozenset[str]] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Self:
        allowed = {field.name for field in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
        values = dict(raw)
        for name in cls._path_fields:
            if name in values:
                values[name] = Path(values[name])
        for name in cls._tuple_fields:
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    @property
    def manifest_hash(self) -> str:
        return sha256_value(self.to_dict())


@dataclass(frozen=True, slots=True)
class EnvironmentManifest(JsonManifest):
    python: str
    torch: str
    cuda: str | None
    ultralytics: str
    faster_coco_eval: str
    onnx: str
    ml_dtypes: str
    protobuf: str
    device_name: str
    official_model_yaml_hash: str
    official_default_yaml_hash: str
    source_weights_hash: str


@dataclass(frozen=True, slots=True)
class DatasetManifest(JsonManifest):
    source_root: Path
    output_root: Path
    dataset_hash: str
    split_ratios: tuple[float, float, float]
    split_counts: dict[str, int]
    class_names: tuple[str, str]
    group_count: int

    _path_fields: ClassVar[frozenset[str]] = frozenset({"source_root", "output_root"})
    _tuple_fields: ClassVar[frozenset[str]] = frozenset({"split_ratios", "class_names"})


@dataclass(frozen=True, slots=True)
class CheckpointManifest(JsonManifest):
    variant_id: str
    variant_hash: str
    checkpoint_path: Path
    checkpoint_hash: str
    state_dict_hash: str
    data_hash: str
    config_hash: str
    environment_hash: str

    _path_fields: ClassVar[frozenset[str]] = frozenset({"checkpoint_path"})


@dataclass(frozen=True, slots=True)
class PipelineState(JsonManifest):
    pipeline_id: str
    stage: str
    status: str
    attempt: int
    epoch: int | None
    input_hashes: dict[str, str]
    output_hashes: dict[str, str]
    error: str | None = None


_ALLOWED: dict[str, frozenset[str]] = {
    "root": frozenset({"schema_version", "pipeline_name", "artifacts_root", "environment", "dataset", "model", "training", "profiling", "pipeline", "variants"}),
    "environment": frozenset({"python", "torch", "cuda", "ultralytics", "faster_coco_eval", "onnx", "ml_dtypes", "protobuf"}),
    "dataset": frozenset({"source", "split_ratios", "class_names", "seed", "minimum_ball_count"}),
    "model": frozenset({"source_weights", "imgsz", "nc", "scale", "source_weights_sha256", "official_model_yaml_sha256", "official_default_yaml_sha256", "sp2_hidden_channels", "sp2_auxiliary_loss_weight"}),
    "training": frozenset({"optimizer", "momentum", "cos_lr", "deterministic", "amp", "nbs", "batch_candidates", "b1_a_epochs", "b1_b_epochs", "smoke_epochs", "variant_epochs", "b1_a_lr0", "formal_lr0", "freeze"}),
    "profiling": frozenset({"precision", "batch", "warmup", "iterations"}),
    "pipeline": frozenset({"max_attempts", "device", "systemd_unit_prefix", "wait_for_units"}),
}


def _reject_unknown(section: str, raw: Mapping[str, Any]) -> None:
    unknown = set(raw) - _ALLOWED[section]
    if unknown:
        raise ValueError(f"unknown {section} keys: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class Phase1Config:
    """Validated raw configuration with an immutable canonical hash."""

    values: dict[str, Any]

    @classmethod
    def minimal_mapping(cls) -> dict[str, Any]:
        return {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "pipeline_name": "static-phase1",
            "artifacts_root": "artifacts/static-phase1",
            "variants": list(PHASE1_VARIANTS),
            "environment": {
                "python": "3.12",
                "torch": "2.11.0+cu128",
                "cuda": "12.8",
                "ultralytics": "8.4.90",
                "faster_coco_eval": "1.7.2",
                "onnx": "1.22.0",
                "ml_dtypes": "0.5.4",
                "protobuf": "7.35.1",
            },
            "dataset": {
                "source": "bbt5-detect-baseline/dataset",
                "split_ratios": list(SPLIT_RATIOS),
                "class_names": list(CLASS_NAMES),
                "seed": 42,
                "minimum_ball_count": 50,
            },
            "model": {
                "source_weights": "bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt",
                "source_weights_sha256": "9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d",
                "official_model_yaml_sha256": "43d8a7c86acc77282ddf4966d6526e091e5b064c140ed4a6e4c0b68ecbc3784c",
                "official_default_yaml_sha256": "f6e19ab7228826341cc6370bcc8c78d10478d2bd604e25a13d9975b5efc9b410",
                "imgsz": 640,
                "nc": 2,
                "scale": "m",
                "sp2_hidden_channels": 32,
                "sp2_auxiliary_loss_weight": 1.0,
            },
            "training": {
                "optimizer": "SGD",
                "momentum": 0.937,
                "cos_lr": True,
                "deterministic": True,
                "amp": True,
                "nbs": 64,
                "batch_candidates": [16, 8, 4, 2, 1],
                "b1_a_epochs": 10,
                "b1_b_epochs": 90,
                "smoke_epochs": 3,
                "variant_epochs": 100,
                "b1_a_lr0": 0.01,
                "formal_lr0": 0.001,
                "freeze": list(range(11)),
            },
            "profiling": {
                "precision": "fp16",
                "batch": 1,
                "warmup": 100,
                "iterations": 1000,
            },
            "pipeline": {
                "max_attempts": 3,
                "device": 0,
                "systemd_unit_prefix": "masf-yolo-phase1",
                "wait_for_units": ["yolo-p2-study.service"],
            },
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        _reject_unknown("root", raw)
        for section in ("environment", "dataset", "model", "training", "profiling", "pipeline"):
            value = raw.get(section)
            if not isinstance(value, Mapping):
                raise ValueError(f"{section} must be a mapping")
            _reject_unknown(section, value)

        values = deepcopy(dict(raw))
        if values.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PIPELINE_SCHEMA_VERSION}")
        dataset = values["dataset"]
        if tuple(dataset.get("split_ratios", ())) != SPLIT_RATIOS:
            raise ValueError("dataset split must be exactly 80/10/10")
        if tuple(dataset.get("class_names", ())) != CLASS_NAMES:
            raise ValueError("class contract must be ball=0 and bat=1")
        variants = tuple(values.get("variants", ()))
        unsupported = [item for item in variants if item not in PHASE1_VARIANTS]
        if unsupported:
            raise ValueError(f"unsupported variant for Phase 1: {unsupported[0]}")
        if variants != PHASE1_VARIANTS:
            raise ValueError(f"variants must be {list(PHASE1_VARIANTS)} in order")
        model = values["model"]
        if model.get("imgsz") != 640 or model.get("nc") != 2 or model.get("scale") != "m":
            raise ValueError("model contract requires YOLO11m, nc=2, imgsz=640")
        if model.get("sp2_hidden_channels") != 32 or model.get("sp2_auxiliary_loss_weight") != 1.0:
            raise ValueError("SP2 contract requires hidden channels 32 and auxiliary loss weight 1.0")
        training = values["training"]
        expected = {
            "optimizer": "SGD", "momentum": 0.937, "cos_lr": True,
            "deterministic": True, "amp": True, "nbs": 64,
        }
        for key, expected_value in expected.items():
            if training.get(key) != expected_value:
                raise ValueError(f"training.{key} must be {expected_value!r}")
        profiling = values["profiling"]
        if profiling != {"precision": "fp16", "batch": 1, "warmup": 100, "iterations": 1000}:
            raise ValueError("profiling contract requires FP16 batch=1 warmup=100 iterations=1000")
        pipeline = values["pipeline"]
        if pipeline.get("systemd_unit_prefix") != "masf-yolo-phase1":
            raise ValueError("pipeline.systemd_unit_prefix must be masf-yolo-phase1")
        if tuple(pipeline.get("wait_for_units", ())) != ("yolo-p2-study.service",):
            raise ValueError("pipeline must wait for yolo-p2-study.service")
        return cls(values=values)

    @property
    def config_hash(self) -> str:
        return sha256_value(self.values)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.values)
