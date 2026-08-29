"""Immutable domain types for SIPA-BCSP training architecture."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from ..activations import ActivationName, available_activations

CANONICAL_BBAT5_DETECT_YAML = (
    "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml"
)
COCO2017_DETECT_YAML = "/home/uxin/yolo/coco2017.yaml"

_ACTIVATIONS = frozenset(available_activations())


def _validate_activation(name: str) -> None:
    if name not in _ACTIVATIONS:
        choices = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"unsupported activation {name!r}; expected one of: {choices}")


@dataclass(frozen=True)
class RegionRule:
    pattern: str
    region: str
    eligible: bool = True

    def __post_init__(self) -> None:
        if not self.region or self.region == "unmapped":
            raise ValueError("region rule must use a non-empty, mapped region name")
        re.compile(self.pattern)


@dataclass(frozen=True)
class ActivationSite:
    module_path: str
    region: str
    eligible: bool = True
    cost_weight: float = 1.0
    original_activation: str = "silu"

    def __post_init__(self) -> None:
        if not self.module_path:
            raise ValueError("activation module_path cannot be empty")
        if not self.region:
            raise ValueError("activation region cannot be empty")
        if self.original_activation != "silu":
            raise ValueError("Phase 0 replacement sites must originate from SiLU")
        if not math.isfinite(self.cost_weight) or self.cost_weight <= 0:
            raise ValueError("cost_weight must be finite and positive")


@dataclass(frozen=True)
class ActivationManifest:
    model_id: str
    sites: tuple[ActivationSite, ...]
    reviewed: bool = False
    model_source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        paths = [site.module_path for site in self.sites]
        if len(paths) != len(set(paths)):
            raise ValueError("activation manifest contains duplicate module paths")

    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(sorted({site.region for site in self.sites if site.eligible}))

    def audit(self, *, require_review: bool = True) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.sites:
            errors.append("manifest contains no SiLU activation sites")
        unmapped = [
            site.module_path for site in self.sites if site.region == "unmapped"
        ]
        if unmapped:
            errors.append("unmapped activation sites: " + ", ".join(unmapped))
        if require_review and not self.reviewed:
            errors.append("manifest has not been explicitly reviewed")
        return tuple(errors)

    def approve(self) -> ActivationManifest:
        errors = self.audit(require_review=False)
        if errors:
            raise ValueError("cannot approve manifest: " + "; ".join(errors))
        return replace(self, reviewed=True)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = 1
        return payload


@dataclass(frozen=True)
class StaticPolicy:
    policy_id: str
    default_activation: ActivationName = "silu"
    region_assignments: tuple[tuple[str, ActivationName], ...] = ()
    site_assignments: tuple[tuple[str, ActivationName], ...] = ()

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id cannot be empty")
        _validate_activation(self.default_activation)
        self._validate_assignments(self.region_assignments, "region")
        self._validate_assignments(self.site_assignments, "site")

    @staticmethod
    def _validate_assignments(
        assignments: tuple[tuple[str, ActivationName], ...], kind: str
    ) -> None:
        keys = [key for key, _ in assignments]
        if any(not key for key in keys):
            raise ValueError(f"{kind} assignment keys cannot be empty")
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate {kind} assignments are not allowed")
        for _, activation in assignments:
            _validate_activation(activation)

    def resolve(self, site: ActivationSite) -> ActivationName:
        if not site.eligible:
            return "silu"
        site_map = dict(self.site_assignments)
        if site.module_path in site_map:
            return site_map[site.module_path]
        region_map = dict(self.region_assignments)
        return region_map.get(site.region, self.default_activation)

    def used_activations(
        self, manifest: ActivationManifest
    ) -> tuple[ActivationName, ...]:
        return tuple(sorted({self.resolve(site) for site in manifest.sites}))

    def changed_regions(self) -> tuple[str, ...]:
        if self.default_activation != "silu":
            return ("*",)
        return tuple(
            sorted(region for region, act in self.region_assignments if act != "silu")
        )


@dataclass(frozen=True)
class DatasetContract:
    dataset_id: str
    source_kind: str
    yaml_path: str
    task: str
    num_classes: int
    immutable: bool = True

    def __post_init__(self) -> None:
        if self.task != "detect":
            raise ValueError(
                "this activation architecture currently supports detect only"
            )
        if not self.immutable:
            raise ValueError("training datasets must be declared immutable")
        if self.source_kind == "canonical_bbat5_v1":
            if self.yaml_path != CANONICAL_BBAT5_DETECT_YAML or self.num_classes != 2:
                raise ValueError(
                    "Canonical BBAT5 v1 Detect must use its nc=2 formal YAML"
                )
        elif self.source_kind == "coco80_detect":
            if self.yaml_path != COCO2017_DETECT_YAML or self.num_classes != 80:
                raise ValueError(
                    "COCO80 Detect must use /home/uxin/yolo/coco2017.yaml with nc=80"
                )
        else:
            raise ValueError(f"unsupported dataset source_kind: {self.source_kind}")


@dataclass(frozen=True)
class SearchConfig:
    max_deployment_kernels: int = 3
    beam_width: int = 4
    max_changed_regions: int = 3
    max_finalists: int = 2
    max_map_loss: float | None = None
    max_ap_s_loss: float | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_deployment_kernels <= 3:
            raise ValueError("max_deployment_kernels must be between 1 and 3")
        if self.beam_width < 1:
            raise ValueError("beam_width must be positive")
        if self.max_changed_regions < 1:
            raise ValueError("max_changed_regions must be positive")
        if not 1 <= self.max_finalists <= 2:
            raise ValueError("max_finalists must be one or two")


@dataclass(frozen=True)
class TrainingConfig:
    architecture_id: str
    datasets: tuple[DatasetContract, ...]
    seed: int = 1
    optional_finalist_seed: int | None = 2
    accuracy_reference: ActivationName = "silu"
    hardware_neighbor_baseline: ActivationName = "hardswish"
    cheap_control: ActivationName = "relu"
    proposed_candidates: tuple[ActivationName, ...] = (
        "qsilu_pq",
        "poly_shift",
        "poly_quality",
    )
    region_search_candidates: tuple[ActivationName, ...] = ("poly_shift",)
    search: SearchConfig = SearchConfig()

    def __post_init__(self) -> None:
        if not self.architecture_id:
            raise ValueError("architecture_id cannot be empty")
        dataset_ids = [dataset.dataset_id for dataset in self.datasets]
        if not dataset_ids or len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError(
                "datasets must be non-empty and have unique dataset_id values"
            )
        if self.seed < 0 or (
            self.optional_finalist_seed is not None and self.optional_finalist_seed < 0
        ):
            raise ValueError("seeds must be non-negative")
        for activation in (
            self.accuracy_reference,
            self.hardware_neighbor_baseline,
            self.cheap_control,
            *self.proposed_candidates,
            *self.region_search_candidates,
        ):
            _validate_activation(activation)
        if self.accuracy_reference != "silu":
            raise ValueError(
                "the accuracy reference must remain the delivered SiLU baseline"
            )
        if not set(self.region_search_candidates).issubset(self.proposed_candidates):
            raise ValueError("region_search_candidates must be proposed candidates")


@dataclass(frozen=True)
class PolicyCost:
    variable_multiplications: float
    constant_multiplications: float
    range_operations: float
    transcendental_operations: float
    coefficient_count: int
    kernel_count: int

    def objective_tuple(self) -> tuple[float, ...]:
        return (
            self.transcendental_operations,
            self.variable_multiplications,
            self.constant_multiplications,
            self.range_operations,
            float(self.coefficient_count),
            float(self.kernel_count),
        )


@dataclass(frozen=True)
class PolicyObservation:
    dataset_id: str
    stage: str
    policy: StaticPolicy
    map_loss: float | None
    ap_s_loss: float | None = None
    latency_ms: float | None = None
    failed: bool = False

    def __post_init__(self) -> None:
        for field_name, value in (
            ("map_loss", self.map_loss),
            ("ap_s_loss", self.ap_s_loss),
            ("latency_ms", self.latency_ms),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite when provided")
        if not self.failed and self.map_loss is None:
            raise ValueError("successful observations require map_loss")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset_id: str
    dataset_yaml: str
    stage: str
    mode: str
    seed: int
    policy: StaticPolicy
    policy_cost: PolicyCost
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingPlan:
    architecture_id: str
    manifest_model_id: str
    next_stage: str
    experiments: tuple[ExperimentSpec, ...]
    frontier_policy_ids: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for experiment in payload["experiments"]:
            policy = experiment["policy"]
            policy["region_assignments"] = dict(policy["region_assignments"])
            policy["site_assignments"] = dict(policy["site_assignments"])
        return payload


@dataclass(frozen=True)
class DeliveryAudit:
    delivery_id: str | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    normalized: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        return not self.errors
