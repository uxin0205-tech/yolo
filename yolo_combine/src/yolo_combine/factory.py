"""Fail-fast construction of the formal graph-shared fusion model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics.data.utils import check_det_dataset

from .fusion_model import (
    AssemblyReport,
    GraphSharedDualHeadModel,
    assemble_graph_shared_model,
)
from .joint_data import validate_canonical_pose_source
from .source import (
    CheckpointKind,
    ManifestReport,
    SourceBundle,
    file_sha256,
)
from .xnor import (
    XNORExecutionConfig,
    XNORInstallationReport,
    install_xnor_backend,
)


def _names(data: dict[str, Any]) -> dict[int, str]:
    raw = data.get("names")
    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(index): str(name) for index, name in raw.items()}
    raise TypeError("dataset names must be a list or mapping")


@dataclass(frozen=True)
class DatasetPairReport:
    detect_yaml: Path
    pose_yaml: Path
    detect_nc: int
    pose_nc: int
    detect_names: dict[int, str]
    pose_names: dict[int, str]
    kpt_shape: tuple[int, int]
    flip_idx: tuple[int, ...]
    pose_dataset_id: str


@dataclass(frozen=True)
class WeightLoadingReport:
    loaded_shared_tensors: int
    loaded_detect_head_tensors: int
    loaded_pose_head_tensors: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not (
            self.missing_keys or self.unexpected_keys or self.shape_mismatches
        )


@dataclass(frozen=True)
class FusionFactoryReport:
    source_manifest: ManifestReport
    source_provenance: dict[str, str]
    xnor: XNORInstallationReport
    datasets: DatasetPairReport
    weights: WeightLoadingReport
    assembly: AssemblyReport
    pose_head_checkpoint: Path | None
    pose_head_sha256: str | None

    @property
    def complete(self) -> bool:
        return (
            self.assembly.audit.compatible
            and self.assembly.complete
            and self.weights.complete
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_manifest": {
                "files": self.source_manifest.files,
                "bytes": self.source_manifest.bytes,
            },
            "source_provenance": self.source_provenance,
            "xnor": {
                "module": self.xnor.module,
                "backend": self.xnor.backend,
                "token_tile": self.xnor.token_tile,
                "newly_installed": self.xnor.newly_installed,
            },
            "datasets": {
                "detect_yaml": str(self.datasets.detect_yaml),
                "pose_yaml": str(self.datasets.pose_yaml),
                "detect_nc": self.datasets.detect_nc,
                "pose_nc": self.datasets.pose_nc,
                "detect_names": self.datasets.detect_names,
                "pose_names": self.datasets.pose_names,
                "kpt_shape": list(self.datasets.kpt_shape),
                "flip_idx": list(self.datasets.flip_idx),
                "pose_dataset_id": self.datasets.pose_dataset_id,
            },
            "weights": {
                "loaded_shared_tensors": self.weights.loaded_shared_tensors,
                "loaded_detect_head_tensors": self.weights.loaded_detect_head_tensors,
                "loaded_pose_head_tensors": self.weights.loaded_pose_head_tensors,
                "missing_keys": list(self.weights.missing_keys),
                "unexpected_keys": list(self.weights.unexpected_keys),
                "shape_mismatches": list(self.weights.shape_mismatches),
            },
            "assembly": self.assembly.as_dict(),
            "pose_head_checkpoint": (
                str(self.pose_head_checkpoint)
                if self.pose_head_checkpoint is not None
                else None
            ),
            "pose_head_sha256": self.pose_head_sha256,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class FusionBuildResult:
    model: GraphSharedDualHeadModel
    report: FusionFactoryReport


class FusionModelFactory:
    """Deep construction module for dependencies, datasets, weights, and graph."""

    def __init__(
        self,
        source: SourceBundle,
        *,
        detect_data_yaml: str | Path,
        pose_data_yaml: str | Path,
        xnor: XNORExecutionConfig = XNORExecutionConfig(),
    ) -> None:
        self.source = source
        self.detect_data_yaml = Path(detect_data_yaml).expanduser().resolve()
        self.pose_data_yaml = Path(pose_data_yaml).expanduser().resolve()
        self.xnor = xnor

    def _datasets(
        self,
        model: GraphSharedDualHeadModel,
    ) -> DatasetPairReport:
        pose_source = validate_canonical_pose_source(self.pose_data_yaml)
        detect_data = check_det_dataset(
            str(self.detect_data_yaml),
            autodownload=False,
        )
        pose_data = check_det_dataset(
            str(self.pose_data_yaml),
            autodownload=False,
        )
        detect_names = _names(detect_data)
        pose_names = _names(pose_data)
        detect_nc = int(detect_data["nc"])
        pose_nc = int(pose_data["nc"])
        kpt_shape = tuple(int(value) for value in pose_data.get("kpt_shape", ()))
        flip_idx = tuple(int(value) for value in pose_data.get("flip_idx", ()))
        mismatches: list[str] = []
        if detect_nc != int(model.detect_head.nc):
            mismatches.append(
                f"Detect nc dataset={detect_nc}, head={model.detect_head.nc}"
            )
        if detect_names != model.detect_names:
            mismatches.append(
                f"Detect names dataset={detect_names}, head={model.detect_names}"
            )
        if pose_nc != int(model.pose_head.nc):
            mismatches.append(
                f"Pose nc dataset={pose_nc}, head={model.pose_head.nc}"
            )
        if pose_names != model.pose_names:
            mismatches.append(
                f"Pose names dataset={pose_names}, head={model.pose_names}"
            )
        if kpt_shape != tuple(model.pose_head.kpt_shape):
            mismatches.append(
                f"Pose kpt_shape dataset={kpt_shape}, head={model.pose_head.kpt_shape}"
            )
        if flip_idx != (0, 1):
            mismatches.append(f"Pose flip_idx must be (0, 1), got {flip_idx}")
        if mismatches:
            raise ValueError(
                "dataset/checkpoint contract mismatch:\n- "
                + "\n- ".join(mismatches)
            )
        return DatasetPairReport(
            detect_yaml=self.detect_data_yaml,
            pose_yaml=self.pose_data_yaml,
            detect_nc=detect_nc,
            pose_nc=pose_nc,
            detect_names=detect_names,
            pose_names=pose_names,
            kpt_shape=(int(kpt_shape[0]), int(kpt_shape[1])),
            flip_idx=flip_idx,
            pose_dataset_id=pose_source.dataset_id,
        )

    def build(
        self,
        *,
        pose_head_checkpoint: str | Path | None = None,
        checkpoint_kind: CheckpointKind = "float",
        allow_untrained_pose_head: bool = False,
    ) -> FusionBuildResult:
        """Build once; formal mode refuses a randomly initialized Pose26 head."""

        if pose_head_checkpoint is None and not allow_untrained_pose_head:
            raise ValueError(
                "a trained Pose checkpoint is required for formal fusion; "
                "set allow_untrained_pose_head only for CPU interface tests"
            )
        pose_path = (
            Path(pose_head_checkpoint).expanduser().resolve()
            if pose_head_checkpoint is not None
            else None
        )
        if pose_path is not None and not pose_path.is_file():
            raise FileNotFoundError(pose_path)
        manifest = self.source.verify_manifest()
        self.source.verify_environment()
        self.source.activate_code()
        installation = install_xnor_backend(self.xnor)
        built = self.source.build_task_models(
            checkpoint_kind,
            pose_head_checkpoint=pose_path,
        )
        model, assembly = assemble_graph_shared_model(built.detect, built.pose)
        datasets = self._datasets(model)
        pose_transfer = built.pose_head_transfer
        missing = tuple(
            f"shared.{name}" for name in built.transfer.missing_tensors
        )
        shape_mismatches = tuple(
            f"shared.{name}" for name in built.transfer.shape_mismatches
        )
        unexpected: tuple[str, ...] = ()
        if pose_transfer is not None:
            missing += tuple(
                f"pose_head.{name}" for name in pose_transfer.missing_tensors
            )
            unexpected += tuple(
                f"pose_head.{name}" for name in pose_transfer.unexpected_tensors
            )
            shape_mismatches += tuple(
                f"pose_head.{name}" for name in pose_transfer.shape_mismatches
            )
        weights = WeightLoadingReport(
            loaded_shared_tensors=built.transfer.compatible_tensors,
            loaded_detect_head_tensors=assembly.loaded_detect_head_tensors,
            loaded_pose_head_tensors=(
                pose_transfer.compatible_tensors
                if pose_transfer is not None
                else assembly.loaded_pose_head_tensors
            ),
            missing_keys=missing,
            unexpected_keys=unexpected,
            shape_mismatches=shape_mismatches,
        )
        report = FusionFactoryReport(
            source_manifest=manifest,
            source_provenance=self.source.provenance(checkpoint_kind),
            xnor=installation,
            datasets=datasets,
            weights=weights,
            assembly=assembly,
            pose_head_checkpoint=pose_path,
            pose_head_sha256=file_sha256(pose_path) if pose_path else None,
        )
        if not report.complete:
            raise RuntimeError(f"incomplete fusion build: {report.as_dict()}")
        return FusionBuildResult(model=model, report=report)
