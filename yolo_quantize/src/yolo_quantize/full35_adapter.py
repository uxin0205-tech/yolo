"""Full35 adapter for activation-output quantization experiments."""

from __future__ import annotations

import hashlib
import importlib
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from torch import nn

from .activation_adapter import (
    ActivationOutputAdapter,
    ActivationOutputPolicy,
    AppliedActivationQuantization,
)
from .quantizers import LSQPlusSpec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Full35ActivationPolicy:
    """One immutable activation placement and activation-output bit width."""

    activation: str
    bits: int
    signed_codes: bool = False
    region_assignments: tuple[tuple[str, str], ...] = ()

    _SUPPORTED: ClassVar[tuple[str, ...]] = (
        "silu",
        "qsilu_pq",
        "poly_quality",
        "poly_shift",
        "hardswish",
        "relu",
    )

    def __post_init__(self) -> None:
        if self.activation not in self._SUPPORTED:
            raise ValueError(
                f"unsupported Full35 activation {self.activation!r}; "
                f"expected one of {self._SUPPORTED}"
            )
        regions = tuple(region for region, _ in self.region_assignments)
        if any(not region for region in regions):
            raise ValueError("activation region names must not be empty")
        if len(regions) != len(set(regions)):
            raise ValueError("duplicate activation region assignments are not allowed")
        for _, activation in self.region_assignments:
            if activation not in self._SUPPORTED:
                raise ValueError(
                    f"unsupported Full35 regional activation {activation!r}; "
                    f"expected one of {self._SUPPORTED}"
                )
        LSQPlusSpec(bits=self.bits, signed_codes=self.signed_codes)

    @property
    def activation_policy_id(self) -> str:
        if not self.region_assignments:
            return self.activation
        assignments = "__".join(
            f"{region}={activation}"
            for region, activation in sorted(self.region_assignments)
        )
        return f"{self.activation}--regional--{assignments}"

    @property
    def policy_id(self) -> str:
        return f"{self.activation_policy_id}--lsq-plus-a{self.bits}"

    def output_policy(self) -> ActivationOutputPolicy:
        return ActivationOutputPolicy(
            policy_id=self.policy_id,
            bits=self.bits,
            signed_codes=self.signed_codes,
        )


@dataclass(frozen=True)
class Full35ActivationBuild:
    """Loaded Full35 model plus its quantization and provenance handles."""

    policy: Full35ActivationPolicy
    applied: AppliedActivationQuantization
    checkpoint_path: Path
    checkpoint_sha256: str
    activation_counts: dict[str, int]
    protected_before: dict[str, int]
    protected_after: dict[str, int]
    training_only_paths: tuple[str, ...]
    training_only_regions: tuple[str, ...]
    source: Any
    joint_config: Any
    factory_report: Any
    loaded_checkpoint: Any

    @property
    def model(self) -> nn.Module:
        return self.applied.model


class Full35ActivationAdapter:
    """Build quantized activation policies from the accepted Full35 release."""

    _TRAINING_ONLY_REGIONS: ClassVar[frozenset[str]] = frozenset(
        {"detect_one2many", "pose_one2many", "pose_flow"}
    )
    _PROTECTED_CLASSES: ClassVar[tuple[str, ...]] = (
        "achitechure_1.masf.P3MASFFull35",
        "ultralytics.nn.modules.block.RealNVP",
        "ultralytics.nn.modules.head.Detect",
        "ultralytics.nn.modules.head.Pose26",
        "yolo_attention.attention.HardwareFriendlyAttention",
        "yolo_attention.binary_basis.BinaryScore",
        "yolo_attention.normalization.PiecewiseLinearSoftmax",
        "yolo_attention.projection.ModularQKVProjection",
        "yolo_attention.relative_bias.RelativePositionBias",
    )

    def __init__(
        self,
        *,
        activation_root: str | Path = "/home/uxin/yolo/yolo_activation",
        recipe: str | Path | None = None,
    ) -> None:
        self.activation_root = Path(activation_root).expanduser().resolve()
        self.activation_source = self.activation_root / "src"
        self.recipe = (
            Path(recipe).expanduser().resolve()
            if recipe is not None
            else self.activation_root / "training/full35/activation-recipe.yaml"
        )

    def _upstream(self) -> Any:
        source = str(self.activation_source)
        if source not in sys.path:
            sys.path.insert(0, source)
        package = importlib.import_module("activation_lab")
        package_file = Path(package.__file__).resolve()
        if self.activation_source not in package_file.parents:
            raise RuntimeError(f"activation_lab is shadowed by {package_file}")
        return importlib.import_module("activation_lab.training.full35")

    def _experiment(self) -> Any:
        upstream = self._upstream()
        return upstream.Full35ActivationExperiment.from_yaml(self.recipe)

    def preflight(self) -> Any:
        return self._experiment().preflight()

    def _protected_signature(self, model: nn.Module) -> dict[str, int]:
        counts = {name: 0 for name in self._PROTECTED_CLASSES}
        for module in model.modules():
            class_name = f"{type(module).__module__}.{type(module).__name__}"
            if class_name in counts:
                counts[class_name] += 1
        return counts

    def build(
        self,
        policy: Full35ActivationPolicy,
        *,
        checkpoint: str | Path | None = None,
        checkpoint_sha256: str | None = None,
    ) -> Full35ActivationBuild:
        upstream = self._upstream()
        experiment = self._experiment()
        if checkpoint is None:
            checkpoint_path = Path(experiment.config.checkpoint).resolve()
            verified_checkpoint_sha256 = experiment.config.checkpoint_sha256
        else:
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            if checkpoint_sha256 is None:
                raise ValueError(
                    "checkpoint_sha256 is required for an explicit checkpoint"
                )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            verified_checkpoint_sha256 = _sha256(checkpoint_path)
            if verified_checkpoint_sha256 != checkpoint_sha256:
                raise ValueError(
                    "explicit checkpoint SHA-256 mismatch: "
                    f"{verified_checkpoint_sha256} != {checkpoint_sha256}"
                )
        preflight = experiment.preflight()
        if not preflight.ready:
            raise RuntimeError(
                "Full35 activation preflight failed: " + "; ".join(preflight.blockers)
            )
        manifest = upstream.load_full35_manifest(experiment.config)
        known_regions = set(manifest.regions)
        unknown_regions = sorted(set(dict(policy.region_assignments)) - known_regions)
        if unknown_regions:
            raise ValueError(
                "unknown Full35 activation regions: " + ", ".join(unknown_regions)
            )
        if policy.region_assignments:
            static_policy = upstream.StaticPolicy(
                policy_id=policy.activation_policy_id,
                default_activation=policy.activation,
                region_assignments=policy.region_assignments,
            )
        else:
            static_policy = upstream.uniform_full35_policy(policy.activation)
        loaded = experiment.load_policy_model(
            manifest,
            static_policy,
        )
        loaded_checkpoint = loaded.loaded_checkpoint
        if checkpoint_path != Path(experiment.config.checkpoint).resolve():
            loaded_checkpoint = experiment._imports()[
                "inference"
            ].load_combined_weights(
                loaded.model,
                checkpoint_path,
                prefer_ema=True,
            )
        eligible_paths = tuple(
            site.module_path
            for site in manifest.sites
            if site.eligible and site.region not in self._TRAINING_ONLY_REGIONS
        )
        training_only_paths = tuple(
            site.module_path
            for site in manifest.sites
            if site.eligible and site.region in self._TRAINING_ONLY_REGIONS
        )
        training_only_regions = tuple(
            sorted(
                {
                    site.region
                    for site in manifest.sites
                    if site.module_path in training_only_paths
                }
            )
        )
        activation_counts = dict(
            sorted(
                Counter(static_policy.resolve(site) for site in manifest.sites).items()
            )
        )
        protected_before = self._protected_signature(loaded.model)
        applied = ActivationOutputAdapter().apply(
            loaded.model,
            module_paths=eligible_paths,
            policy=policy.output_policy(),
            clone_model=False,
        )
        protected_after = self._protected_signature(applied.model)
        if protected_after != protected_before:
            raise RuntimeError(
                "protected Full35 module signature changed during activation wrapping"
            )
        if any(count == 0 for count in protected_after.values()):
            raise RuntimeError(
                "one or more protected Full35 module classes were not found"
            )
        return Full35ActivationBuild(
            policy=policy,
            applied=applied,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=verified_checkpoint_sha256,
            activation_counts=activation_counts,
            protected_before=protected_before,
            protected_after=protected_after,
            training_only_paths=training_only_paths,
            training_only_regions=training_only_regions,
            source=loaded.source,
            joint_config=loaded.joint_config,
            factory_report=loaded.factory_report,
            loaded_checkpoint=loaded_checkpoint,
        )
