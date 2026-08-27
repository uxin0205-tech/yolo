"""Fail-closed configuration for formal shared-trunk training."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
import ultralytics
import yaml

from .metrics import GATE_METRICS
from .stage_policy import JOINT_STAGES, OptimizerName

Architecture = Literal["full35", "partial75"]


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"joint config section {key!r} must be a mapping")
    return value


def _path(value: object, *, base: Path, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("configured path must be a non-empty string")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _float(value: object, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


def _int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if resolved != value:
        raise ValueError(f"{name} must be an integer")
    return resolved


@dataclass(frozen=True)
class FormalPreflightReport:
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    baseline: dict[str, float] | None


@dataclass(frozen=True)
class JointExperimentConfig:
    """Every consequential setting needed to reproduce one fusion run."""

    path: Path
    architecture: Architecture
    enabled: bool
    source_bundle: Path
    pose_checkpoint: Path | None
    provisional_pose_checkpoint: Path | None
    baseline_metrics_path: Path | None
    registry: Path
    detect_data: Path
    pose_data: Path
    run_root: Path
    imgsz: int
    detect_batch_size: int
    detect_microbatch_size: int
    pose_batch_size: int
    detect_val_batch_size: int
    pose_val_batch_size: int
    detect_workers: int
    pose_workers: int
    xnor_token_tile: int
    qk_ste: bool
    stages: tuple[str, ...]
    enable_j3: bool
    optimizer: OptimizerName
    challenger_optimizer: OptimizerName
    seed: int
    optional_second_seed: int
    amp: bool
    amp_max_overflow_retries: int
    detect_batches_per_macro: int
    pose_batches_per_macro: int
    reference_batch_size: int
    detect_weight: float
    pose_weight: float
    gradient_clip_norm: float
    gradient_statistics_interval: int
    weight_decay: float
    beta1: float
    beta2: float
    warmup_epochs: int
    warmup_start_factor: float
    cosine_final_lr_factor: float
    validation_backends: tuple[str, ...]
    selection_backend: str
    maximum_map_drop: float
    validation_plots: bool
    save_coco_json: bool
    tensorboard: str
    required_ultralytics: str
    required_torch: str

    @classmethod
    def load(cls, path: str | Path) -> "JointExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("joint config schema_version must be 1")
        architecture = payload.get("architecture")
        if architecture not in {"full35", "partial75"}:
            raise ValueError("architecture must be full35 or partial75")
        if config_path.parents[1].name != architecture:
            raise ValueError("joint config architecture differs from variant folder")
        dependencies = _mapping(payload, "dependencies")
        source = _mapping(payload, "source")
        data = _mapping(payload, "data")
        xnor = _mapping(payload, "xnor")
        training = _mapping(payload, "training")
        validation = _mapping(payload, "validation")
        logging = _mapping(payload, "logging")
        runs = _mapping(payload, "runs")
        stages = tuple(str(value) for value in training.get("stages", ()))
        if not stages or any(value not in JOINT_STAGES for value in stages):
            raise ValueError(f"training stages must come from {tuple(JOINT_STAGES)}")
        if "j3" in stages:
            raise ValueError("J3 must be enabled explicitly, not placed in default stages")
        optimizer = str(training.get("optimizer"))
        challenger = str(training.get("challenger_optimizer"))
        if optimizer not in {"AdamW", "MuSGD"} or challenger not in {"AdamW", "MuSGD"}:
            raise ValueError("optimizer values must be AdamW or MuSGD")
        backends = tuple(str(value) for value in validation.get("backends", ()))
        if backends != ("float", "bittrue"):
            raise ValueError("validation.backends must preserve float then bittrue")
        if xnor.get("backend") != "tiled_exact":
            raise ValueError("formal fusion requires tiled_exact XNOR")
        if bool(xnor.get("qk_ste")):
            raise ValueError("Q/K STE challenger is excluded from the baseline")
        if bool(dependencies.get("compile")) or bool(dependencies.get("ddp")):
            raise ValueError("v1 requires compile=false and ddp=false")
        resolved = cls(
            path=config_path,
            architecture=architecture,
            enabled=bool(payload.get("enabled")),
            source_bundle=_path(source.get("bundle"), base=config_path.parent),  # type: ignore[arg-type]
            pose_checkpoint=_path(source.get("pose_checkpoint"), base=config_path.parent, optional=True),
            provisional_pose_checkpoint=_path(source.get("provisional_pose_checkpoint"), base=config_path.parent, optional=True),
            baseline_metrics_path=_path(source.get("baseline_metrics"), base=config_path.parent, optional=True),
            registry=_path(data.get("registry"), base=config_path.parent),  # type: ignore[arg-type]
            detect_data=_path(data.get("detect"), base=config_path.parent),  # type: ignore[arg-type]
            pose_data=_path(data.get("pose"), base=config_path.parent),  # type: ignore[arg-type]
            run_root=_path(runs.get("root"), base=config_path.parent),  # type: ignore[arg-type]
            imgsz=_int(data.get("imgsz"), "data.imgsz"),
            detect_batch_size=_int(data.get("detect_train_batch"), "data.detect_train_batch"),
            detect_microbatch_size=_int(data.get("detect_train_microbatch"), "data.detect_train_microbatch"),
            pose_batch_size=_int(data.get("pose_train_batch"), "data.pose_train_batch"),
            detect_val_batch_size=_int(data.get("detect_val_batch"), "data.detect_val_batch"),
            pose_val_batch_size=_int(data.get("pose_val_batch"), "data.pose_val_batch"),
            detect_workers=_int(data.get("detect_workers"), "data.detect_workers"),
            pose_workers=_int(data.get("pose_workers"), "data.pose_workers"),
            xnor_token_tile=_int(xnor.get("token_tile"), "xnor.token_tile"),
            qk_ste=bool(xnor.get("qk_ste")),
            stages=stages,
            enable_j3=bool(training.get("enable_j3")),
            optimizer=optimizer,  # type: ignore[arg-type]
            challenger_optimizer=challenger,  # type: ignore[arg-type]
            seed=_int(training.get("seed"), "training.seed"),
            optional_second_seed=_int(training.get("optional_second_seed"), "training.optional_second_seed"),
            amp=bool(training.get("amp")),
            amp_max_overflow_retries=_int(training.get("amp_max_overflow_retries"), "training.amp_max_overflow_retries"),
            detect_batches_per_macro=_int(training.get("detect_batches_per_macro"), "training.detect_batches_per_macro"),
            pose_batches_per_macro=_int(training.get("pose_batches_per_macro"), "training.pose_batches_per_macro"),
            reference_batch_size=_int(training.get("reference_batch_size"), "training.reference_batch_size"),
            detect_weight=_float(training.get("detect_weight"), "training.detect_weight"),
            pose_weight=_float(training.get("pose_weight"), "training.pose_weight"),
            gradient_clip_norm=_float(training.get("gradient_clip_norm"), "training.gradient_clip_norm"),
            gradient_statistics_interval=_int(training.get("gradient_statistics_interval"), "training.gradient_statistics_interval"),
            weight_decay=_float(training.get("weight_decay"), "training.weight_decay"),
            beta1=_float(training.get("beta1"), "training.beta1"),
            beta2=_float(training.get("beta2"), "training.beta2"),
            warmup_epochs=_int(training.get("warmup_epochs"), "training.warmup_epochs"),
            warmup_start_factor=_float(training.get("warmup_start_factor"), "training.warmup_start_factor"),
            cosine_final_lr_factor=_float(training.get("cosine_final_lr_factor"), "training.cosine_final_lr_factor"),
            validation_backends=backends,
            selection_backend=str(validation.get("selection_backend")),
            maximum_map_drop=_float(validation.get("maximum_map50_95_drop"), "validation.maximum_map50_95_drop"),
            validation_plots=bool(validation.get("plots")),
            save_coco_json=bool(validation.get("save_coco_json")),
            tensorboard=str(logging.get("tensorboard")),
            required_ultralytics=str(dependencies.get("ultralytics")),
            required_torch=str(dependencies.get("torch")),
        )
        resolved._validate()
        return resolved

    def _validate(self) -> None:
        positive_ints = {
            "imgsz": self.imgsz,
            "detect batch": self.detect_batch_size,
            "detect microbatch": self.detect_microbatch_size,
            "pose batch": self.pose_batch_size,
            "detect validation batch": self.detect_val_batch_size,
            "pose validation batch": self.pose_val_batch_size,
            "detect batches per macro": self.detect_batches_per_macro,
            "pose batches per macro": self.pose_batches_per_macro,
            "reference batch": self.reference_batch_size,
            "XNOR token tile": self.xnor_token_tile,
            "gradient statistics interval": self.gradient_statistics_interval,
        }
        invalid = {name: value for name, value in positive_ints.items() if value < 1}
        if invalid:
            raise ValueError(f"positive integer settings invalid: {invalid}")
        if self.amp_max_overflow_retries < 0:
            raise ValueError("training.amp_max_overflow_retries cannot be negative")
        if self.detect_batch_size % self.detect_microbatch_size:
            raise ValueError(
                "data.detect_train_batch must be divisible by "
                "data.detect_train_microbatch"
            )
        if self.pose_batches_per_macro != 1:
            raise ValueError("v1 scheduler requires one Pose batch per macro")
        if self.selection_backend != "bittrue":
            raise ValueError("formal checkpoint selection must use bittrue metrics")
        if not 0 <= self.maximum_map_drop <= 1:
            raise ValueError("maximum mAP drop must be in [0,1]")
        if self.tensorboard not in {"off", "auto", "required"}:
            raise ValueError("logging.tensorboard must be off, auto, or required")

    @property
    def detect_microbatches_per_logical_batch(self) -> int:
        """Number of physical forwards representing one Detect batch."""

        return self.detect_batch_size // self.detect_microbatch_size

    @property
    def detect_microbatches_per_macro(self) -> int:
        """Physical Detect forwards accumulated before one optimizer step."""

        return self.detect_batches_per_macro * self.detect_microbatches_per_logical_batch

    def load_baseline(self) -> dict[str, float] | None:
        if self.baseline_metrics_path is None or not self.baseline_metrics_path.is_file():
            return None
        payload = yaml.safe_load(self.baseline_metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("baseline metrics file must be a mapping")
        raw = payload.get("metrics", payload)
        if not isinstance(raw, dict):
            raise ValueError("baseline metrics must be a mapping")
        values = {str(name): float(value) for name, value in raw.items()}
        missing = tuple(name for name in GATE_METRICS if name not in values)
        if missing:
            raise ValueError(f"baseline metrics are missing {missing}")
        return values

    def preflight(self) -> FormalPreflightReport:
        blockers: list[str] = []
        warnings: list[str] = []
        if not self.enabled:
            blockers.append(f"{self.architecture} 尚未由使用者啟用")
        required_paths = (self.source_bundle, self.registry, self.detect_data, self.pose_data)
        for path in required_paths:
            if not path.exists():
                blockers.append(f"必要路徑不存在：{path}")
        if self.pose_checkpoint is None:
            blockers.append("尚未指定 canonical P1→P2→P3 完成後的 Pose26 checkpoint")
        elif not self.pose_checkpoint.is_file():
            blockers.append(f"Pose26 checkpoint 不存在：{self.pose_checkpoint}")
        try:
            baseline = self.load_baseline()
        except (OSError, TypeError, ValueError) as error:
            baseline = None
            blockers.append(f"八項獨立 baseline 不合法：{error}")
        if baseline is None:
            blockers.append("尚未提供同口徑的八項 mAP50-95 獨立 baseline")
        if ultralytics.__version__ != self.required_ultralytics:
            blockers.append(
                f"Ultralytics 版本漂移：{ultralytics.__version__} != {self.required_ultralytics}"
            )
        installed_torch = str(torch.__version__)
        if installed_torch != self.required_torch:
            blockers.append(f"PyTorch 版本漂移：{installed_torch} != {self.required_torch}")
        try:
            importlib.metadata.version("tensorboard")
        except importlib.metadata.PackageNotFoundError:
            warnings.append("TensorBoard 未安裝；auto 模式仍會保留 JSONL、CSV、PNG")
        warnings.append(
            f"logical Detect batch={self.detect_batch_size}、physical microbatch="
            f"{self.detect_microbatch_size}；正式 run 前仍須在空閒 RTX 5090 執行 VRAM gate"
        )
        return FormalPreflightReport(
            ready=not blockers,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
            baseline=baseline,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "architecture": self.architecture,
            "config": str(self.path),
            "source_bundle": str(self.source_bundle),
            "pose_checkpoint": str(self.pose_checkpoint) if self.pose_checkpoint else None,
            "baseline_metrics": str(self.baseline_metrics_path) if self.baseline_metrics_path else None,
            "registry": str(self.registry),
            "detect_data": str(self.detect_data),
            "pose_data": str(self.pose_data),
            "run_root": str(self.run_root),
            "imgsz": self.imgsz,
            "batches": {
                "detect_train_logical": self.detect_batch_size,
                "detect_train_physical_microbatch": self.detect_microbatch_size,
                "pose_train_physical": self.pose_batch_size,
                "detect_validation": self.detect_val_batch_size,
                "pose_validation": self.pose_val_batch_size,
                "loss_reference": self.reference_batch_size,
            },
            "macro": {
                "detect_logical_batches": self.detect_batches_per_macro,
                "detect_physical_microbatches": self.detect_microbatches_per_macro,
                "pose_batches": self.pose_batches_per_macro,
                "detect_weight": self.detect_weight,
                "pose_weight": self.pose_weight,
            },
            "amp": {
                "enabled": self.amp,
                "max_overflow_retries": self.amp_max_overflow_retries,
            },
            "optimizer": self.optimizer,
            "stages": list(self.stages),
            "enable_j3": self.enable_j3,
            "xnor": {"backend": "tiled_exact", "token_tile": self.xnor_token_tile, "qk_ste": self.qk_ste},
            "maximum_map_drop": self.maximum_map_drop,
        }

