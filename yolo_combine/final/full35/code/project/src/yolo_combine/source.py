"""Deep adapter for the immutable Full35/Partial75 source bundle."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import ultralytics
from torch import nn
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.nn.modules.head import Detect, Pose26
from ultralytics.nn.tasks import DetectionModel, PoseModel

from .models import RoutedDualModel, SharedDualHeadModel

Architecture = Literal["full35", "partial75"]
CheckpointKind = Literal["float", "bittrue"]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestReport:
    files: int
    bytes: int


@dataclass(frozen=True)
class TrunkTransferReport:
    source_layers: int
    target_layers: int
    compatible_tensors: int
    missing_tensors: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_tensors and not self.shape_mismatches


@dataclass(frozen=True)
class HeadTransferReport:
    compatible_tensors: int
    missing_tensors: tuple[str, ...]
    unexpected_tensors: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not (
            self.missing_tensors or self.unexpected_tensors or self.shape_mismatches
        )


@dataclass(frozen=True)
class BuiltTaskModels:
    detect: DetectionModel
    pose: PoseModel
    transfer: TrunkTransferReport
    pose_head_transfer: HeadTransferReport | None = None
    pose_head_checkpoint: Path | None = None


@dataclass(frozen=True)
class SourceBundle:
    """Validate and materialize models from one immutable experiment bundle."""

    root: Path
    architecture: Architecture = "full35"
    stage: str = "a2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        if self.architecture not in {"full35", "partial75"}:
            raise ValueError("architecture must be full35 or partial75")
        if self.stage != "a2":
            raise ValueError("only retained A2 checkpoints are accepted as initialization sources")

    @property
    def model_id(self) -> str:
        return f"{self.architecture}-{self.stage}"

    @property
    def code_dir(self) -> Path:
        return self.root / "code"

    def checkpoint(self, kind: CheckpointKind = "float") -> Path:
        path = self.root / "weights" / kind / f"{self.model_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def attention_config(self, kind: CheckpointKind = "float") -> Path:
        name = "float-pwl-final.yaml" if kind == "float" else "bittrue-pwl-final.yaml"
        path = self.root / "configs" / "attention" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def verify_manifest(self) -> ManifestReport:
        manifest_path = self.root / "MANIFEST.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        total_bytes = 0
        for record in payload.get("files", []):
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe manifest path: {relative}")
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise ValueError(f"manifest path escapes bundle: {relative}")
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            if size != int(record["bytes"]):
                raise ValueError(f"size mismatch for {relative}: {size} != {record['bytes']}")
            digest = file_sha256(path)
            if digest != record["sha256"]:
                raise ValueError(f"SHA256 mismatch for {relative}")
            total_bytes += size
        records = payload.get("files", [])
        if not records:
            raise ValueError("bundle manifest contains no files")
        return ManifestReport(files=len(records), bytes=total_bytes)

    def verify_environment(self) -> None:
        if torch.__version__ != "2.11.0+cu128":
            raise RuntimeError(f"expected torch 2.11.0+cu128, got {torch.__version__}")
        if ultralytics.__version__ != "8.4.90":
            raise RuntimeError(f"expected ultralytics 8.4.90, got {ultralytics.__version__}")

    def activate_code(self) -> None:
        """Make the exact pickle modules importable and reject shadowed packages."""

        if not self.code_dir.is_dir():
            raise FileNotFoundError(self.code_dir)
        code = str(self.code_dir)
        if code not in sys.path:
            sys.path.insert(0, code)
        for package in ("achitechure_1", "yolo_attention"):
            module = importlib.import_module(package)
            module_path = Path(module.__file__).resolve()
            if self.code_dir not in module_path.parents:
                raise RuntimeError(f"{package} was imported from shadowing path {module_path}")

    def _graph_functions(self):
        self.activate_code()
        model_module = importlib.import_module("achitechure_1.model")
        attention_integration = importlib.import_module("yolo_attention.integration")
        attention_config = importlib.import_module("yolo_attention.config")
        return (
            model_module.graft_p3_masf,
            model_module.inspect_yolo26_graph,
            attention_integration.convert_yolo26_model,
            attention_config.VariantConfig,
        )

    def load_detect_model(self, kind: CheckpointKind = "float") -> DetectionModel:
        self.verify_environment()
        _, inspect_graph, _, _ = self._graph_functions()
        model = YOLO(str(self.checkpoint(kind))).model
        if not isinstance(model, DetectionModel):
            raise TypeError(f"expected DetectionModel, got {type(model).__name__}")
        head = model.model[-1]
        if not isinstance(head, Detect) or isinstance(head, Pose26):
            raise TypeError(f"expected Detect head, got {type(head).__name__}")
        graph = inspect_graph(model)
        p3 = model.model[graph.p3_index]
        masf = getattr(p3, "p3_masf", None)
        expected_name = "P3MASFFull35" if self.architecture == "full35" else "P3MASFPartial75"
        if type(masf).__name__ != expected_name:
            raise ValueError(f"expected {expected_name}, got {type(masf).__name__}")
        if kind == "float":
            # Ultralytics strips inference checkpoints by disabling gradients.
            model.requires_grad_(True)
        return model

    def load_pose_checkpoint(self, checkpoint: str | Path) -> PoseModel:
        """Load a trained P0 model and reject a task, schema, or architecture mismatch."""

        self.verify_environment()
        _, inspect_graph, _, _ = self._graph_functions()
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        model = YOLO(str(path)).model
        if not isinstance(model, PoseModel):
            raise TypeError(f"expected PoseModel, got {type(model).__name__}")
        head = model.model[-1]
        if not isinstance(head, Pose26):
            raise TypeError(f"expected Pose26 head, got {type(head).__name__}")
        if int(head.nc) != 2 or tuple(int(value) for value in head.kpt_shape) != (2, 3):
            raise ValueError(f"unexpected Pose head schema nc={head.nc}, kpt_shape={head.kpt_shape}")
        if dict(model.names) != {0: "ball", 1: "bat"}:
            raise ValueError(f"unexpected Pose class names: {model.names}")
        graph = inspect_graph(model)
        masf = getattr(model.model[graph.p3_index], "p3_masf", None)
        expected_name = "P3MASFFull35" if self.architecture == "full35" else "P3MASFPartial75"
        if type(masf).__name__ != expected_name:
            raise ValueError(f"expected {expected_name}, got {type(masf).__name__}")
        model.requires_grad_(True)
        return model

    def build_pose_model(
        self,
        source: DetectionModel | None = None,
        *,
        kind: CheckpointKind = "float",
    ) -> tuple[PoseModel, TrunkTransferReport]:
        """Build Pose26 with the same graph and copy every shared tensor exactly."""

        if source is None:
            source = self.load_detect_model(kind)
        graft_masf, inspect_graph, convert_attention, variant_config = self._graph_functions()
        pose = PoseModel("yolo26m-pose.yaml", ch=3, nc=2, data_kpt_shape=(2, 3), verbose=False)
        convert_attention(pose, variant_config.from_yaml(self.attention_config(kind)))
        graft_masf(pose, self.architecture)
        report = transfer_shared_trunk(source, pose)
        if not report.complete:
            raise RuntimeError(f"incomplete shared trunk transfer: {report}")
        inspect_graph(pose)
        pose.names = {0: "ball", 1: "bat"}
        pose.nc = 2
        pose.args = get_cfg(DEFAULT_CFG, overrides={"task": "pose", "imgsz": 640})
        pose.requires_grad_(kind == "float")
        return pose, report

    def build_task_models(
        self,
        kind: CheckpointKind = "float",
        *,
        pose_head_checkpoint: str | Path | None = None,
    ) -> BuiltTaskModels:
        detect = self.load_detect_model(kind)
        pose, transfer = self.build_pose_model(detect, kind=kind)
        head_transfer = None
        head_path = None
        if pose_head_checkpoint is not None:
            head_path = Path(pose_head_checkpoint).expanduser().resolve()
            trained_pose = self.load_pose_checkpoint(head_path)
            head_transfer = transfer_pose_head(trained_pose, pose)
            if not head_transfer.complete:
                raise RuntimeError(f"incomplete Pose head transfer: {head_transfer}")
        return BuiltTaskModels(
            detect=detect,
            pose=pose,
            transfer=transfer,
            pose_head_transfer=head_transfer,
            pose_head_checkpoint=head_path,
        )

    def build_routed(
        self,
        kind: CheckpointKind = "float",
        *,
        pose_head_checkpoint: str | Path | None = None,
    ) -> RoutedDualModel:
        models = self.build_task_models(kind, pose_head_checkpoint=pose_head_checkpoint)
        return RoutedDualModel(models.detect, models.pose)

    def build_shared(
        self,
        kind: CheckpointKind = "float",
        *,
        pose_head_checkpoint: str | Path | None = None,
    ) -> SharedDualHeadModel:
        models = self.build_task_models(kind, pose_head_checkpoint=pose_head_checkpoint)
        return SharedDualHeadModel.from_task_models(models.detect, models.pose)

    def provenance(self, kind: CheckpointKind = "float") -> dict[str, str]:
        checkpoint = self.checkpoint(kind)
        return {
            "bundle_root": str(self.root),
            "architecture": self.architecture,
            "stage": self.stage,
            "checkpoint_kind": kind,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
        }


def transfer_pose_head(source: PoseModel, target: PoseModel) -> HeadTransferReport:
    """Copy only a trained ball/bat Pose26 head into a fresh target graph."""

    source_head = source.model[-1]
    target_head = target.model[-1]
    if not isinstance(source_head, Pose26) or not isinstance(target_head, Pose26):
        raise TypeError("Pose head transfer requires Pose26 source and target heads")
    source_inputs = tuple(int(index) for index in source_head.f)
    target_inputs = tuple(int(index) for index in target_head.f)
    if source_inputs != target_inputs:
        raise ValueError(f"Pose head input mismatch: {source_inputs} != {target_inputs}")
    if int(source_head.nc) != int(target_head.nc):
        raise ValueError(f"Pose head class mismatch: {source_head.nc} != {target_head.nc}")
    if tuple(source_head.kpt_shape) != tuple(target_head.kpt_shape):
        raise ValueError(
            f"Pose keypoint shape mismatch: {source_head.kpt_shape} != {target_head.kpt_shape}"
        )
    source_state = source_head.state_dict()
    target_state = target_head.state_dict()
    source_names = set(source_state)
    target_names = set(target_state)
    missing = tuple(sorted(target_names - source_names))
    unexpected = tuple(sorted(source_names - target_names))
    mismatched = tuple(
        sorted(
            name
            for name in source_names & target_names
            if source_state[name].shape != target_state[name].shape
        )
    )
    compatible = tuple(
        name
        for name in target_state
        if name in source_state and source_state[name].shape == target_state[name].shape
    )
    report = HeadTransferReport(
        compatible_tensors=len(compatible),
        missing_tensors=missing,
        unexpected_tensors=unexpected,
        shape_mismatches=mismatched,
    )
    if not report.complete or len(compatible) != len(target_state):
        return report
    target_head.load_state_dict(source_state, strict=True)
    return report


def _shared_tensor_names(model: nn.Module, head_index: int) -> tuple[str, ...]:
    names: list[str] = []
    for name in model.state_dict():
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1].isdigit() and int(parts[1]) < head_index:
            names.append(name)
    return tuple(names)


def transfer_shared_trunk(source: DetectionModel, target: PoseModel) -> TrunkTransferReport:
    """Copy all pre-head tensors and fail if graph names or shapes diverge."""

    if len(source.model) != len(target.model):
        raise ValueError(f"graph layer mismatch: {len(source.model)} != {len(target.model)}")
    source_head_index = len(source.model) - 1
    target_head_index = len(target.model) - 1
    source_inputs = tuple(int(index) for index in source.model[-1].f)
    target_inputs = tuple(int(index) for index in target.model[-1].f)
    if source_inputs != target_inputs:
        raise ValueError(f"head input mismatch: {source_inputs} != {target_inputs}")
    source_state = source.state_dict()
    target_state = target.state_dict()
    target_names = _shared_tensor_names(target, target_head_index)
    source_names = set(_shared_tensor_names(source, source_head_index))
    missing = tuple(name for name in target_names if name not in source_names)
    mismatched = tuple(
        name
        for name in target_names
        if name in source_state and source_state[name].shape != target_state[name].shape
    )
    compatible = tuple(
        name
        for name in target_names
        if name in source_state and source_state[name].shape == target_state[name].shape
    )
    if missing or mismatched or len(compatible) != len(target_names):
        return TrunkTransferReport(
            source_layers=source_head_index,
            target_layers=target_head_index,
            compatible_tensors=len(compatible),
            missing_tensors=missing,
            shape_mismatches=mismatched,
        )
    with torch.no_grad():
        for name in compatible:
            target_state[name].copy_(source_state[name])
    target.load_state_dict(target_state, strict=True)
    return TrunkTransferReport(
        source_layers=source_head_index,
        target_layers=target_head_index,
        compatible_tensors=len(compatible),
        missing_tensors=(),
        shape_mismatches=(),
    )
