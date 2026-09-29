"""Production adapter for activation experiments on the accepted Full35 release."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
import yaml
from torch import nn

from ..activations import ActivationName
from .domain import ActivationManifest, ActivationSite, StaticPolicy
from .io import load_manifest, load_region_rules
from .model import AppliedPolicy, apply_static_policy, inspect_silu_sites

CANONICAL_BBAT5_REGISTRY = Path("/home/uxin/yolo/configs/datasets/bbat5-v1.yaml")
CANONICAL_BBAT5_POSE = Path(
    "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml"
)
CANONICAL_BBAT5_DETECT = Path(
    "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml"
)
CANONICAL_COCO_DETECT = Path("/home/uxin/yolo/coco2017.yaml")
FULL35_GATE_METRICS = (
    "coco/box/map50_95",
    "coco/person/box/map50_95",
    "bbat/box/map50_95",
    "bbat/pose/map50_95",
    "bbat/ball/box/map50_95",
    "bbat/bat/box/map50_95",
    "bbat/ball/pose/map50_95",
    "bbat/bat/pose/map50_95",
)


def _fp32_bbox_iou(original: Any) -> Any:
    """Keep CIoU ratio arithmetic out of FP16 while preserving its formula."""

    @wraps(original)
    def stable(
        box1: torch.Tensor, box2: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        with torch.autocast(device_type=box1.device.type, enabled=False):
            return original(box1.float(), box2.float(), *args, **kwargs)

    return stable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _path(value: Any, *, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError("configured paths must be non-empty strings")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


@dataclass(frozen=True)
class Full35Phase:
    name: str
    mode: str
    epochs: int
    seed: int
    patience: int = 0
    warmup_epochs: int = 0
    learning_rate_scale: float = 1.0
    full_training_split: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.mode:
            raise ValueError("phase name and mode cannot be empty")
        if self.epochs < 0 or self.seed < 0:
            raise ValueError("phase epochs and seed must be non-negative")
        if self.patience < 0 or self.warmup_epochs < 0:
            raise ValueError("phase patience and warmup must be non-negative")
        if self.learning_rate_scale <= 0:
            raise ValueError("phase learning_rate_scale must be positive")
        if self.mode in {"recovery", "qat"} and self.epochs < 1:
            raise ValueError(f"{self.mode} phase requires positive epochs")


@dataclass(frozen=True)
class Full35ExperimentConfig:
    path: Path
    experiment_id: str
    release_root: Path
    joint_config: Path
    region_rules: Path
    manifest: Path
    run_root: Path
    checkpoint: Path
    checkpoint_sha256: str
    baseline_metrics: Path
    baseline_metrics_sha256: str
    full_resume_checkpoint: Path
    full_resume_sha256: str
    release_manifest_sha256: str
    fraction: float
    resampling: bool
    coco_detect: Path
    bbat5_registry: Path
    bbat5_pose: Path
    bbat5_detect: Path
    maximum_map50_95_drop: float
    learning_rates: Mapping[str, float]
    phases: Mapping[str, Full35Phase]

    @classmethod
    def load(cls, path: str | Path) -> Full35ExperimentConfig:
        recipe = Path(path).expanduser().resolve()
        payload = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError("Full35 activation recipe requires schema_version: 1")
        baseline = _mapping(payload, "baseline")
        data = _mapping(payload, "data")
        optimizer = _mapping(payload, "optimizer")
        comparison = _mapping(payload, "comparison")
        raw_learning_rates = _mapping(optimizer, "learning_rates")
        raw_phases = _mapping(payload, "phases")
        phases: dict[str, Full35Phase] = {}
        for name, raw in raw_phases.items():
            if not isinstance(raw, Mapping):
                raise TypeError(f"phase {name!r} must be a mapping")
            phases[str(name)] = Full35Phase(
                name=str(name),
                mode=str(raw["mode"]),
                epochs=int(raw.get("epochs", 0)),
                seed=int(raw.get("seed", 0)),
                patience=int(raw.get("patience", 0)),
                warmup_epochs=int(raw.get("warmup_epochs", 0)),
                learning_rate_scale=float(raw.get("learning_rate_scale", 1.0)),
                full_training_split=bool(raw.get("full_training_split", False)),
            )
        config = cls(
            path=recipe,
            experiment_id=str(payload["experiment_id"]),
            release_root=_path(payload["release_root"], base=recipe.parent),
            joint_config=_path(payload["joint_config"], base=recipe.parent),
            region_rules=_path(payload["region_rules"], base=recipe.parent),
            manifest=_path(payload["manifest"], base=recipe.parent),
            run_root=_path(payload["run_root"], base=recipe.parent),
            checkpoint=_path(baseline["checkpoint"], base=recipe.parent),
            checkpoint_sha256=str(baseline["checkpoint_sha256"]),
            baseline_metrics=_path(baseline["metrics_contract"], base=recipe.parent),
            baseline_metrics_sha256=str(baseline["metrics_contract_sha256"]),
            full_resume_checkpoint=_path(
                baseline["full_resume_checkpoint"], base=recipe.parent
            ),
            full_resume_sha256=str(baseline["full_resume_sha256"]),
            release_manifest_sha256=str(baseline["release_manifest_sha256"]),
            fraction=float(data["fraction"]),
            resampling=bool(data["resampling"]),
            coco_detect=_path(data["coco_detect"], base=recipe.parent),
            bbat5_registry=_path(data["bbat5_registry"], base=recipe.parent),
            bbat5_pose=_path(data["bbat5_pose"], base=recipe.parent),
            bbat5_detect=_path(data["bbat5_detect"], base=recipe.parent),
            maximum_map50_95_drop=float(comparison["maximum_map50_95_drop"]),
            learning_rates=MappingProxyType(
                {str(name): float(value) for name, value in raw_learning_rates.items()}
            ),
            phases=MappingProxyType(phases),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id cannot be empty")
        if self.fraction != 1.0 or self.resampling:
            raise ValueError(
                "Full35 formal activation work requires fraction=1.0 and no resampling"
            )
        if not 0.0 < self.maximum_map50_95_drop <= 0.08:
            raise ValueError("activation maximum_map50_95_drop must be in (0, 0.08]")
        expected = {
            "coco_detect": (self.coco_detect, CANONICAL_COCO_DETECT),
            "bbat5_registry": (self.bbat5_registry, CANONICAL_BBAT5_REGISTRY),
            "bbat5_pose": (self.bbat5_pose, CANONICAL_BBAT5_POSE),
            "bbat5_detect": (self.bbat5_detect, CANONICAL_BBAT5_DETECT),
        }
        drift = [
            name
            for name, (actual, canonical) in expected.items()
            if actual != canonical
        ]
        if drift:
            raise ValueError(f"canonical dataset paths drifted: {drift}")
        required_roles = {
            "backbone",
            "neck",
            "masf",
            "attention",
            "detect_head",
            "pose_head",
        }
        if set(self.learning_rates) != required_roles:
            raise ValueError("Full35 recovery learning-rate roles are incomplete")
        required_phases = {
            "baseline_reproduction",
            "activation_profile",
            "region_zero_shot_sensitivity",
            "uniform_zero_shot",
            "short_recovery",
            "region_sensitivity",
            "policy_search",
            "finalist_seed1",
            "finalist_seed2",
            "claim_ablation",
            "ptq",
            "qat_if_needed",
            "board_validation",
        }
        if set(self.phases) != required_phases:
            raise ValueError(
                "Full35 phases differ: "
                f"missing={sorted(required_phases - set(self.phases))}, "
                f"unexpected={sorted(set(self.phases) - required_phases)}"
            )

    def phase(self, name: str) -> Full35Phase:
        try:
            return self.phases[name]
        except KeyError as error:
            raise ValueError(f"unknown Full35 activation phase: {name}") from error

    def load_baseline_metrics(self) -> dict[str, float]:
        payload = yaml.safe_load(self.baseline_metrics.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError("accepted Full35 baseline contract is malformed")
        if Path(str(payload.get("checkpoint"))).resolve() != self.checkpoint:
            raise ValueError("accepted baseline checkpoint path differs from recipe")
        if payload.get("checkpoint_sha256") != self.checkpoint_sha256:
            raise ValueError("accepted baseline checkpoint hash differs from recipe")
        raw = payload.get("metrics")
        if not isinstance(raw, Mapping):
            raise TypeError("accepted Full35 baseline metrics are missing")
        missing = tuple(name for name in FULL35_GATE_METRICS if name not in raw)
        if missing:
            raise ValueError(f"accepted Full35 baseline metrics are missing {missing}")
        values = {name: float(raw[name]) for name in FULL35_GATE_METRICS}
        invalid = {
            name: value
            for name, value in values.items()
            if not math.isfinite(value) or not 0.0 <= value <= 1.0
        }
        if invalid:
            raise ValueError(f"accepted Full35 baseline metrics are invalid: {invalid}")
        return values


@dataclass(frozen=True)
class Full35PreflightReport:
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    resolved: Mapping[str, Any]


@dataclass(frozen=True)
class Full35LoadedPolicy:
    model: nn.Module
    source: Any
    joint_config: Any
    factory_report: Any
    loaded_checkpoint: Any
    applied: AppliedPolicy


class _PolicyAwareSource:
    """Materialize validation graphs with the same parameterless activation policy."""

    def __init__(
        self,
        source: Any,
        manifest: ActivationManifest,
        policy: StaticPolicy,
    ) -> None:
        self._source = source
        self._manifest = manifest
        self._policy = policy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    @staticmethod
    def _target_path(path: str, task: str) -> str | None:
        detect_prefix = "graph.model.23.detect_head."
        pose_prefix = "graph.model.23.pose_head."
        if path.startswith(detect_prefix):
            return (
                "model.23." + path.removeprefix(detect_prefix)
                if task == "detect"
                else None
            )
        if path.startswith(pose_prefix):
            return (
                "model.23." + path.removeprefix(pose_prefix) if task == "pose" else None
            )
        if path.startswith("graph.model."):
            return path.removeprefix("graph.")
        raise ValueError(f"Full35 manifest path is outside the graph: {path}")

    def _apply(self, model: nn.Module, task: str) -> None:
        translated: list[ActivationSite] = []
        path_map: dict[str, str] = {}
        for site in self._manifest.sites:
            target = self._target_path(site.module_path, task)
            if target is None:
                continue
            path_map[site.module_path] = target
            translated.append(replace(site, module_path=target))
        manifest = ActivationManifest(
            model_id=f"{self._manifest.model_id}--{task}",
            sites=tuple(translated),
            reviewed=True,
            model_source_sha256=self._manifest.model_source_sha256,
        )
        policy = StaticPolicy(
            policy_id=self._policy.policy_id,
            default_activation=self._policy.default_activation,
            region_assignments=self._policy.region_assignments,
            site_assignments=tuple(
                (path_map[path], activation)
                for path, activation in self._policy.site_assignments
                if path in path_map
            ),
        )
        apply_static_policy(model, manifest, policy, clone_model=False)

    def build_task_models(self, *args: Any, **kwargs: Any) -> Any:
        built = self._source.build_task_models(*args, **kwargs)
        self._apply(built.detect, "detect")
        self._apply(built.pose, "pose")
        return built


class _RecoveryConfigProxy:
    def __init__(
        self,
        base: Any,
        *,
        recipe: Full35ExperimentConfig,
        phase: Full35Phase,
        manifest: ActivationManifest,
        policy: StaticPolicy,
    ) -> None:
        self._base = base
        self._recipe = recipe
        self._phase = phase
        self._manifest = manifest
        self._policy = policy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    @property
    def maximum_map_drop(self) -> float:
        return self._recipe.maximum_map50_95_drop

    def preflight(self) -> Any:
        return replace(
            self._base.preflight(),
            baseline=self._recipe.load_baseline_metrics(),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = self._base.as_dict()
        payload["baseline_metrics"] = str(self._recipe.baseline_metrics)
        payload["maximum_map_drop"] = self.maximum_map_drop
        payload["activation_experiment"] = {
            "experiment_id": self._recipe.experiment_id,
            "recipe": str(self._recipe.path),
            "recipe_sha256": _sha256(self._recipe.path),
            "phase": self._phase.name,
            "phase_config": {
                "mode": self._phase.mode,
                "epochs": self._phase.epochs,
                "patience": self._phase.patience,
                "warmup_epochs": self._phase.warmup_epochs,
                "seed": self._phase.seed,
                "learning_rate_scale": self._phase.learning_rate_scale,
            },
            "manifest": str(self._recipe.manifest),
            "manifest_sha256": _sha256(self._recipe.manifest),
            "manifest_model_id": self._manifest.model_id,
            "policy": {
                "policy_id": self._policy.policy_id,
                "default_activation": self._policy.default_activation,
                "region_assignments": dict(self._policy.region_assignments),
                "site_assignments": dict(self._policy.site_assignments),
            },
            "initial_checkpoint": str(self._recipe.checkpoint),
            "initial_checkpoint_sha256": self._recipe.checkpoint_sha256,
            "initialization": {
                "model": "accepted_inference_ema_state_dict",
                "optimizer": "fresh",
                "scheduler": "fresh",
                "ema": "fresh_from_loaded_model",
                "rng": "reseeded_per_candidate",
            },
            "full_resume_checkpoint": str(self._recipe.full_resume_checkpoint),
            "full_resume_checkpoint_role": ("lineage_and_emergency_exact_resume_only"),
            "accuracy_gate": {
                "baseline_metrics": str(self._recipe.baseline_metrics),
                "maximum_map50_95_drop": self.maximum_map_drop,
            },
            "numerical_stability": {
                "model_amp": True,
                "ciou_loss_precision": "fp32",
                "ciou_formula_changed": False,
            },
            "data_fraction": self._recipe.fraction,
            "resampling": self._recipe.resampling,
        }
        return payload


class Full35ActivationExperiment:
    """Load, inspect, validate, and recover Full35 through one production seam."""

    def __init__(self, config: Full35ExperimentConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> Full35ActivationExperiment:
        return cls(Full35ExperimentConfig.load(path))

    @property
    def _project_source(self) -> Path:
        return self.config.release_root / "code/project/src"

    def _imports(self) -> dict[str, Any]:
        source = str(self._project_source)
        if source not in sys.path:
            sys.path.insert(0, source)
        package = importlib.import_module("yolo_combine")
        package_file = Path(package.__file__).resolve()
        if self._project_source not in package_file.parents:
            raise RuntimeError(f"yolo_combine is shadowed by {package_file}")
        names = {
            "joint_config": "yolo_combine.joint_config",
            "source": "yolo_combine.source",
            "factory": "yolo_combine.factory",
            "inference": "yolo_combine.inference",
            "xnor": "yolo_combine.xnor",
            "validation": "yolo_combine.validation",
            "data": "yolo_combine.data",
            "formal_training": "yolo_combine.formal_training",
            "formal_impl": "yolo_combine._formal_training_impl",
            "stage_policy": "yolo_combine.stage_policy",
        }
        return {key: importlib.import_module(value) for key, value in names.items()}

    def _joint_config(self, modules: Mapping[str, Any]) -> Any:
        return modules["joint_config"].JointExperimentConfig.load(
            self.config.joint_config
        )

    def preflight(self, *, verify_hashes: bool = True) -> Full35PreflightReport:
        blockers: list[str] = []
        warnings: list[str] = []
        paths = (
            self.config.release_root,
            self.config.joint_config,
            self.config.region_rules,
            self.config.checkpoint,
            self.config.baseline_metrics,
            self.config.full_resume_checkpoint,
            self.config.coco_detect,
            self.config.bbat5_registry,
            self.config.bbat5_pose,
            self.config.bbat5_detect,
        )
        for path in paths:
            if not path.exists():
                blockers.append(f"required path does not exist: {path}")
        release_manifest = self.config.release_root / "MANIFEST.json"
        if not release_manifest.is_file():
            blockers.append(f"release manifest does not exist: {release_manifest}")
        if verify_hashes and not blockers:
            checks = (
                (self.config.checkpoint, self.config.checkpoint_sha256),
                (
                    self.config.baseline_metrics,
                    self.config.baseline_metrics_sha256,
                ),
                (
                    self.config.full_resume_checkpoint,
                    self.config.full_resume_sha256,
                ),
                (release_manifest, self.config.release_manifest_sha256),
            )
            for path, expected in checks:
                actual = _sha256(path)
                if actual != expected:
                    blockers.append(f"SHA-256 mismatch: {path}: {actual} != {expected}")
        resolved: dict[str, Any] = {
            "experiment_id": self.config.experiment_id,
            "release_root": str(self.config.release_root),
            "joint_config": str(self.config.joint_config),
            "checkpoint": str(self.config.checkpoint),
            "data_fraction": self.config.fraction,
            "resampling": self.config.resampling,
            "run_root": str(self.config.run_root),
            "recovery_initialization": {
                "model": "accepted_inference_ema_state_dict",
                "optimizer": "fresh",
                "scheduler": "fresh",
                "ema": "fresh_from_loaded_model",
                "rng": "reseeded_per_candidate",
                "full_resume_checkpoint_role": (
                    "lineage_and_emergency_exact_resume_only"
                ),
            },
            "activation_accuracy_gate": {
                "baseline_metrics": str(self.config.baseline_metrics),
                "maximum_map50_95_drop": self.config.maximum_map50_95_drop,
            },
            "phases": {
                name: {
                    "mode": phase.mode,
                    "epochs": phase.epochs,
                    "patience": phase.patience,
                    "warmup_epochs": phase.warmup_epochs,
                    "seed": phase.seed,
                    "learning_rate_scale": phase.learning_rate_scale,
                }
                for name, phase in self.config.phases.items()
            },
        }
        if not blockers:
            try:
                self.config.load_baseline_metrics()
            except (OSError, TypeError, ValueError) as error:
                blockers.append(f"accepted Full35 baseline is invalid: {error}")
        if not blockers:
            try:
                modules = self._imports()
                joint = self._joint_config(modules)
                report = joint.preflight()
                blockers.extend(report.blockers)
                warnings.extend(report.warnings)
                resolved["full35"] = joint.as_dict()
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                blockers.append(f"Full35 preflight failed: {error}")
        return Full35PreflightReport(
            ready=not blockers,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
            resolved=resolved,
        )

    def _build_clean(self) -> Full35LoadedPolicy:
        modules = self._imports()
        joint = self._joint_config(modules)
        source = modules["source"].SourceBundle(
            joint.source_bundle,
            architecture=joint.architecture,
        )
        factory = modules["factory"].FusionModelFactory(
            source,
            detect_data_yaml=joint.detect_data,
            pose_data_yaml=joint.pose_data,
            xnor=modules["xnor"].XNORExecutionConfig(token_tile=joint.xnor_token_tile),
        )
        built = factory.build(
            checkpoint_kind="float",
            allow_untrained_pose_head=True,
        )
        loaded = modules["inference"].load_combined_weights(
            built.model,
            self.config.checkpoint,
            prefer_ema=True,
        )
        identity = StaticPolicy("uniform--silu")
        placeholder = AppliedPolicy(
            model=built.model,
            policy=identity,
            changed_paths=(),
            unchanged_paths=(),
        )
        return Full35LoadedPolicy(
            model=built.model,
            source=source,
            joint_config=joint,
            factory_report=built.report,
            loaded_checkpoint=loaded,
            applied=placeholder,
        )

    def build_manifest(self, *, approve: bool = False) -> ActivationManifest:
        loaded = self._build_clean()
        rules = load_region_rules(self.config.region_rules)
        manifest = inspect_silu_sites(
            loaded.model,
            model_id="full35-j3-best-joint",
            region_rules=rules,
        )
        manifest = replace(
            manifest,
            model_source_sha256=self.config.checkpoint_sha256,
        )
        return manifest.approve() if approve else manifest

    def save_manifest(
        self,
        manifest: ActivationManifest,
        path: str | Path | None = None,
    ) -> Path:
        target = (
            self.config.manifest if path is None else Path(path).expanduser().resolve()
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(
                manifest.to_dict(),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return target

    def load_policy_model(
        self,
        manifest: ActivationManifest,
        policy: StaticPolicy,
    ) -> Full35LoadedPolicy:
        if manifest.model_source_sha256 != self.config.checkpoint_sha256:
            raise ValueError(
                "activation manifest checkpoint SHA-256 does not match recipe"
            )
        loaded = self._build_clean()
        applied = apply_static_policy(
            loaded.model,
            manifest,
            policy,
            clone_model=False,
        )
        return replace(loaded, model=applied.model, applied=applied)

    def validate(
        self,
        manifest: ActivationManifest,
        policy: StaticPolicy,
        *,
        run_name: str,
        device: str = "0",
        backends: Sequence[str] = ("float", "bittrue"),
    ) -> dict[str, Any]:
        if not run_name or "/" in run_name or "\\" in run_name:
            raise ValueError("run_name must be one non-empty path component")
        if not backends or any(kind not in {"float", "bittrue"} for kind in backends):
            raise ValueError("backends must contain float and/or bittrue")
        loaded = self.load_policy_model(manifest, policy)
        modules = self._imports()
        output = self.config.run_root / "evaluations" / run_name
        pose_view = modules["data"].prepare_bbt5_view(
            self.config.bbat5_registry,
            output / "datasets" / "bbat5-v1-runtime",
        )
        model = loaded.model.to(
            torch.device("cpu" if device == "cpu" else f"cuda:{device}")
        )
        aware_source = _PolicyAwareSource(loaded.source, manifest, policy)
        validator = modules["validation"].JointValidator(
            aware_source,
            detect_data_yaml=self.config.coco_detect,
            pose_data_yaml=pose_view.yaml,
            output_root=output,
            settings=modules["validation"].ValidationSettings(
                imgsz=loaded.joint_config.imgsz,
                detect_batch_size=loaded.joint_config.detect_val_batch_size,
                pose_batch_size=loaded.joint_config.pose_val_batch_size,
                detect_workers=loaded.joint_config.detect_workers,
                pose_workers=loaded.joint_config.pose_workers,
                device=str(next(model.parameters()).device),
                plots=loaded.joint_config.validation_plots,
                save_coco_json=loaded.joint_config.save_coco_json,
            ),
        )
        reports = validator.validate_backends(model, epoch=0, kinds=tuple(backends))
        payload = {
            "schema_version": 1,
            "experiment_id": self.config.experiment_id,
            "run_name": run_name,
            "checkpoint": str(self.config.checkpoint),
            "checkpoint_sha256": self.config.checkpoint_sha256,
            "data_fraction": self.config.fraction,
            "resampling": self.config.resampling,
            "policy": {
                "policy_id": policy.policy_id,
                "default_activation": policy.default_activation,
                "region_assignments": dict(policy.region_assignments),
                "site_assignments": dict(policy.site_assignments),
            },
            "changed_paths": list(loaded.applied.changed_paths),
            "metrics": {name: report.metrics for name, report in reports.items()},
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "activation-summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return payload

    @contextmanager
    def _patched_recovery(
        self,
        modules: Mapping[str, Any],
        manifest: ActivationManifest,
        policy: StaticPolicy,
        phase: Full35Phase,
    ) -> Iterator[None]:
        impl = modules["formal_impl"]
        joint_config_module = modules["joint_config"]
        stage_policy_module = modules["stage_policy"]
        original_stages = impl.JOINT_STAGES
        original_joint_config_stages = joint_config_module.JOINT_STAGES
        original_stage_policy_stages = stage_policy_module.JOINT_STAGES
        original_factory = impl.FusionModelFactory
        original_validator = impl.JointValidator
        loss_module = importlib.import_module("ultralytics.utils.loss")
        original_bbox_iou = loss_module.bbox_iou
        original_j3 = original_stages["j3"]
        learning_rates = MappingProxyType(
            {
                role: float(rate) * phase.learning_rate_scale
                for role, rate in self.config.learning_rates.items()
            }
        )
        recovery_j3 = replace(
            original_j3,
            epochs=phase.epochs,
            patience=phase.patience,
            warmup_epochs=phase.warmup_epochs,
            learning_rates=learning_rates,
        )
        stages = MappingProxyType({**dict(original_stages), "j3": recovery_j3})
        adapter = self

        class PolicyFactory:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._inner = original_factory(*args, **kwargs)

            def build(self, *args: Any, **kwargs: Any) -> Any:
                built = self._inner.build(*args, **kwargs)
                adapter._imports()["inference"].load_combined_weights(
                    built.model,
                    adapter.config.checkpoint,
                    prefer_ema=True,
                )
                apply_static_policy(
                    built.model,
                    manifest,
                    policy,
                    clone_model=False,
                )
                return built

        class PolicyValidator(original_validator):
            def __init__(self, source: Any, *args: Any, **kwargs: Any) -> None:
                super().__init__(
                    _PolicyAwareSource(source, manifest, policy),
                    *args,
                    **kwargs,
                )

        impl.JOINT_STAGES = stages
        joint_config_module.JOINT_STAGES = stages
        stage_policy_module.JOINT_STAGES = stages
        impl.FusionModelFactory = PolicyFactory
        impl.JointValidator = PolicyValidator
        loss_module.bbox_iou = _fp32_bbox_iou(original_bbox_iou)
        try:
            yield
        finally:
            impl.JOINT_STAGES = original_stages
            joint_config_module.JOINT_STAGES = original_joint_config_stages
            stage_policy_module.JOINT_STAGES = original_stage_policy_stages
            impl.FusionModelFactory = original_factory
            impl.JointValidator = original_validator
            loss_module.bbox_iou = original_bbox_iou

    def run_recovery(
        self,
        manifest: ActivationManifest,
        policy: StaticPolicy,
        *,
        phase_name: str,
        run_name: str,
        device: str = "0",
        detect_microbatch_size: int = 32,
    ) -> Any:
        phase = self.config.phase(phase_name)
        if phase.mode not in {"recovery", "qat"}:
            raise ValueError(f"phase {phase_name} is not a trainable recovery phase")
        if manifest.model_source_sha256 != self.config.checkpoint_sha256:
            raise ValueError(
                "activation manifest checkpoint SHA-256 does not match recipe"
            )
        modules = self._imports()
        base = self._joint_config(modules)
        derived = replace(
            base,
            stages=(),
            enable_j3=True,
            seed=phase.seed,
            run_root=self.config.run_root,
        )
        proxy = _RecoveryConfigProxy(
            derived,
            recipe=self.config,
            phase=phase,
            manifest=manifest,
            policy=policy,
        )
        with self._patched_recovery(modules, manifest, policy, phase):
            report = (
                modules["formal_training"]
                .FormalJointTrainingSession(
                    proxy,
                    device=device,
                    run_name=run_name,
                    detect_microbatch_size=detect_microbatch_size,
                )
                .run(enable_j3=True)
            )
        run_dir = self.config.run_root / run_name
        payload = proxy.as_dict()["activation_experiment"]
        (run_dir / "activation-experiment.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report


def uniform_full35_policy(activation: ActivationName) -> StaticPolicy:
    return StaticPolicy(
        policy_id=f"uniform--{activation}",
        default_activation=activation,
    )


def load_full35_manifest(config: Full35ExperimentConfig) -> ActivationManifest:
    return load_manifest(config.manifest)


__all__ = (
    "Full35ActivationExperiment",
    "Full35ExperimentConfig",
    "Full35LoadedPolicy",
    "Full35Phase",
    "Full35PreflightReport",
    "load_full35_manifest",
    "uniform_full35_policy",
)
