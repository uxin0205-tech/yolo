"""Task-separated Ultralytics loaders and a detect-primary joint epoch."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
import yaml
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from .contracts import Task
from .data import CanonicalBBAT5, DEFAULT_BBAT5_REGISTRY
from .fusion_model import GraphSharedDualHeadModel

Mode = Literal["train", "val"]

_COMMON_AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}


@dataclass(frozen=True)
class TaskLoaderSettings:
    """All task-specific loader facts that must not leak across datasets."""

    task: Task
    batch_size: int
    workers: int
    imgsz: int = 640
    mode: Mode = "train"
    fraction: float = 1.0
    seed: int = 0
    augmentation: Mapping[str, float] = MappingProxyType({})

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.workers < 0:
            raise ValueError("workers cannot be negative")
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("imgsz must be a multiple of 32")
        if self.mode not in {"train", "val"}:
            raise ValueError("mode must be train or val")
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")

    @classmethod
    def for_detect(
        cls,
        *,
        batch_size: int = 128,
        workers: int = 4,
        imgsz: int = 640,
        mode: Mode = "train",
        fraction: float = 1.0,
        seed: int = 0,
    ) -> "TaskLoaderSettings":
        return cls(
            task=Task.DETECT,
            batch_size=batch_size,
            workers=workers,
            imgsz=imgsz,
            mode=mode,
            fraction=fraction,
            seed=seed,
            augmentation=MappingProxyType(
                {**_COMMON_AUGMENTATION, "fliplr": 0.5}
            ),
        )

    @classmethod
    def for_pose(
        cls,
        *,
        batch_size: int = 16,
        workers: int = 8,
        imgsz: int = 640,
        mode: Mode = "train",
        fraction: float = 1.0,
        seed: int = 0,
    ) -> "TaskLoaderSettings":
        return cls(
            task=Task.POSE,
            batch_size=batch_size,
            workers=workers,
            imgsz=imgsz,
            mode=mode,
            fraction=fraction,
            seed=seed,
            augmentation=MappingProxyType(
                {**_COMMON_AUGMENTATION, "fliplr": 0.0}
            ),
        )

    def overrides(self, data_yaml: Path) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "data": str(data_yaml),
            "batch": self.batch_size,
            "imgsz": self.imgsz,
            "workers": self.workers,
            "fraction": self.fraction,
            "seed": self.seed,
            "cache": False,
            "rect": self.mode == "val",
            **dict(self.augmentation),
        }


@dataclass(frozen=True)
class CanonicalPoseSourceReport:
    dataset_id: str
    source_kind: Literal["canonical", "runtime_view"]
    yaml: Path
    registry: Path
    train_images: int
    val_images: int
    kpt_shape: tuple[int, int]
    flip_idx: tuple[int, ...]


@dataclass(frozen=True)
class PreparedTaskLoader:
    task: Task
    data_yaml: Path
    data: dict[str, Any]
    dataset: Any
    loader: Any
    settings: TaskLoaderSettings
    device: torch.device

    def preprocess(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Move only the current microbatch and retain native label tensors."""

        processed = {
            key: (
                value.to(
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }
        image = processed.get("img")
        if not isinstance(image, torch.Tensor):
            raise TypeError("Ultralytics batch contains no img tensor")
        processed["img"] = image.float() / 255.0
        return processed


@dataclass(frozen=True)
class JointRawMacro:
    detect_batches: tuple[Mapping[str, Any], ...]
    pose_batches: tuple[Mapping[str, Any], ...]

    @staticmethod
    def _images(batches: Sequence[Mapping[str, Any]]) -> int:
        return sum(int(batch["img"].shape[0]) for batch in batches)

    @property
    def detect_images(self) -> int:
        return self._images(self.detect_batches)

    @property
    def pose_images(self) -> int:
        return self._images(self.pose_batches)


@dataclass(frozen=True)
class JointEpochReport:
    macros: int
    detect_batches: int
    pose_batches: int
    detect_images: int
    pose_images: int
    detect_dataset_images: int
    pose_dataset_images: int
    detect_dataset_passes: float
    pose_dataset_passes: float
    pose_wraps: int


def _loader_dataset_images(loader: Iterable[Mapping[str, Any]]) -> int:
    dataset = getattr(loader, "dataset", None)
    if isinstance(dataset, Sized):
        return len(dataset)
    if isinstance(loader, Sequence):
        return sum(int(batch["img"].shape[0]) for batch in loader)
    raise TypeError(
        "loader must expose a sized dataset or be a finite Sequence"
    )


class JointEpochScheduler:
    """Yield raw CPU batches in 2:1 macros with Detect defining the epoch."""

    def __init__(
        self,
        *,
        detect_loader: Iterable[Mapping[str, Any]],
        pose_loader: Iterable[Mapping[str, Any]],
        detect_batches_per_macro: int = 2,
    ) -> None:
        if detect_batches_per_macro < 1:
            raise ValueError("detect_batches_per_macro must be positive")
        self.detect_loader = detect_loader
        self.pose_loader = pose_loader
        self.detect_batches_per_macro = detect_batches_per_macro
        self.detect_dataset_images = _loader_dataset_images(detect_loader)
        self.pose_dataset_images = _loader_dataset_images(pose_loader)
        if self.detect_dataset_images < 1 or self.pose_dataset_images < 1:
            raise ValueError("task loaders cannot be empty")
        self._report: JointEpochReport | None = None

    def __iter__(self) -> Iterator[JointRawMacro]:
        if self._report is not None:
            raise RuntimeError("JointEpochScheduler instances are single-use")
        detect_iterator = iter(self.detect_loader)
        pose_iterator = iter(self.pose_loader)
        pose_wraps = 0
        detect_batches = 0
        pose_batches = 0
        detect_images = 0
        pose_images = 0
        macros = 0
        while True:
            selected_detect: list[Mapping[str, Any]] = []
            for _ in range(self.detect_batches_per_macro):
                try:
                    selected_detect.append(next(detect_iterator))
                except StopIteration:
                    break
            if not selected_detect:
                break
            try:
                selected_pose = next(pose_iterator)
            except StopIteration:
                pose_wraps += 1
                pose_iterator = iter(self.pose_loader)
                try:
                    selected_pose = next(pose_iterator)
                except StopIteration as error:
                    raise ValueError("Pose loader produced no batches") from error
            macro = JointRawMacro(
                detect_batches=tuple(selected_detect),
                pose_batches=(selected_pose,),
            )
            macros += 1
            detect_batches += len(macro.detect_batches)
            pose_batches += 1
            detect_images += macro.detect_images
            pose_images += macro.pose_images
            yield macro
        self._report = JointEpochReport(
            macros=macros,
            detect_batches=detect_batches,
            pose_batches=pose_batches,
            detect_images=detect_images,
            pose_images=pose_images,
            detect_dataset_images=self.detect_dataset_images,
            pose_dataset_images=self.pose_dataset_images,
            detect_dataset_passes=detect_images / self.detect_dataset_images,
            pose_dataset_passes=pose_images / self.pose_dataset_images,
            pose_wraps=pose_wraps,
        )

    def report(self) -> JointEpochReport:
        if self._report is None:
            raise RuntimeError("joint epoch has not completed")
        return self._report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_canonical_pose_source(
    data_yaml: str | Path,
    *,
    registry: str | Path = DEFAULT_BBAT5_REGISTRY,
) -> CanonicalPoseSourceReport:
    """Reject legacy BBAT5 inputs before dataset caches can be created."""

    contract = CanonicalBBAT5.load(registry)
    path = Path(data_yaml).expanduser().resolve()
    if path == contract.pose.yaml:
        return CanonicalPoseSourceReport(
            dataset_id=contract.dataset_id,
            source_kind="canonical",
            yaml=path,
            registry=contract.registry,
            train_images=contract.train_images,
            val_images=contract.val_images,
            kpt_shape=contract.pose.kpt_shape or (-1, -1),
            flip_idx=contract.pose.flip_idx or (),
        )
    manifest_path = path.parent / "manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(
            f"Pose training source must be canonical {contract.pose.yaml} "
            "or a verified bbat5-v1 runtime View"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(manifest, dict):
        raise ValueError("runtime View data.yaml and manifest must be mappings")
    expected_manifest = {
        "dataset_id": contract.dataset_id,
        "task": Task.POSE.value,
        "source_registry": str(contract.registry),
        "source_yaml": str(contract.pose.yaml),
        "source_yaml_sha256": _sha256(contract.pose.yaml),
        "images": contract.images,
        "labels": contract.images,
        "split_counts": {
            "train": contract.train_images,
            "val": contract.val_images,
        },
        "storage": "symlink-only-runtime-view",
        "source_group_overlap": [],
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"runtime View is not canonical bbat5-v1: {mismatches}")
    if (
        data.get("dataset_id") != contract.dataset_id
        or data.get("source_registry") != str(contract.registry)
        or data.get("source_yaml") != str(contract.pose.yaml)
        or tuple(data.get("kpt_shape", ())) != contract.pose.kpt_shape
        or tuple(data.get("flip_idx", ())) != contract.pose.flip_idx
    ):
        raise ValueError("runtime View data.yaml canonical metadata drifted")
    return CanonicalPoseSourceReport(
        dataset_id=contract.dataset_id,
        source_kind="runtime_view",
        yaml=path,
        registry=contract.registry,
        train_images=contract.train_images,
        val_images=contract.val_images,
        kpt_shape=contract.pose.kpt_shape or (-1, -1),
        flip_idx=contract.pose.flip_idx or (),
    )


def _normalize_names(value: object) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(index): str(name) for index, name in value.items()}
    raise TypeError("dataset names must be a list or mapping")


def build_task_loader(
    model: GraphSharedDualHeadModel,
    *,
    data_yaml: str | Path,
    settings: TaskLoaderSettings,
    device: torch.device,
    registry: str | Path = DEFAULT_BBAT5_REGISTRY,
) -> PreparedTaskLoader:
    """Build one native loader and fail closed on task/schema drift."""

    yaml_path = Path(data_yaml).expanduser().resolve()
    if settings.task is Task.POSE:
        validate_canonical_pose_source(yaml_path, registry=registry)
    data = check_det_dataset(str(yaml_path), autodownload=False)
    head = model.head_for(settings.task)
    names = _normalize_names(data.get("names"))
    expected_names = (
        model.detect_names
        if settings.task is Task.DETECT
        else model.pose_names
    )
    if int(data["nc"]) != int(head.nc) or names != expected_names:
        raise ValueError(
            f"{settings.task.value} dataset schema nc={data['nc']}, names={names} "
            f"does not match head nc={head.nc}, names={expected_names}"
        )
    if settings.task is Task.POSE:
        actual_shape = tuple(int(value) for value in data.get("kpt_shape", ()))
        expected_shape = tuple(int(value) for value in model.pose_head.kpt_shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Pose kpt_shape={actual_shape}, expected={expected_shape}"
            )
        if tuple(int(value) for value in data.get("flip_idx", ())) != (0, 1):
            raise ValueError("canonical BBAT5 flip_idx must remain [0, 1]")
    args = get_cfg(DEFAULT_CFG, overrides=settings.overrides(yaml_path))
    split = "train" if settings.mode == "train" else "val"
    stride = max(int(head.stride.max().item()), 32)
    dataset = build_yolo_dataset(
        args,
        data[split],
        settings.batch_size,
        data,
        mode=settings.mode,
        rect=settings.mode == "val",
        stride=stride,
        fraction=settings.fraction if settings.mode == "train" else 1.0,
    )
    loader = build_dataloader(
        dataset,
        batch=settings.batch_size,
        workers=settings.workers,
        shuffle=settings.mode == "train",
        rank=-1,
        drop_last=False,
    )
    return PreparedTaskLoader(
        task=settings.task,
        data_yaml=yaml_path,
        data=data,
        dataset=dataset,
        loader=loader,
        settings=settings,
        device=device,
    )
