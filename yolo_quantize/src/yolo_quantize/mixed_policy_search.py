"""Compile and validate reviewed mixed weight-format policies for Full35."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .activation_smoke import (
    _atomic_json,
    _compare_outputs,
    _deployment_comparison,
    _letterbox,
)
from .balance_selection import (
    BalanceCandidate,
    QuantizationBalanceSelector,
)
from .full35_adapter import Full35ActivationPolicy
from .metric_gate import (
    Full35MetricCandidate,
    Full35MetricGate,
    Full35MetricGateSpec,
    Full35MetricSnapshot,
)
from .metric_regate import regate_search_report
from .search_validation import (
    Full35SearchValidationPlan,
    _build_deployment_policy,
    _calibrate_activation_a8,
    _prepare_contract,
    _validate_role,
)
from .weight_quantization import (
    ExactTernaryWeightSpec,
    FilterwiseTWNWeightSpec,
    FixedSD4WeightSpec,
    Full35WeightRegionCatalog,
    PaperTWNWeightSpec,
    UniformWeightSpec,
    WeightFormatSpec,
    WeightQuantizationAdapter,
    WeightRegionAssignment,
)
from .weight_sensitivity import WeightSensitivityStudy

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _verified_file(record: object, label: str) -> Path:
    payload = _mapping(record, label)
    if set(payload) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = _resolve(payload["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    expected = str(payload["sha256"])
    if actual != expected:
        raise ValueError(f"{label} SHA-256 drifted: {actual} != {expected}")
    return path


def parse_weight_format_spec(record: object) -> WeightFormatSpec:
    """Parse one closed-schema PTQ format declaration."""

    payload = _mapping(record, "mixed-policy weight format")
    family = str(payload.get("family", ""))
    if family == "uniform":
        if set(payload) != {"family", "bits", "scale_method"}:
            raise ValueError(
                "uniform mixed-policy format must contain exactly family, bits, "
                "and scale_method"
            )
        return UniformWeightSpec(
            bits=int(payload["bits"]),
            scale_method=str(payload["scale_method"]),  # type: ignore[arg-type]
        )
    if family == "fixed_sd4":
        if set(payload) != {"family", "scale_method"}:
            raise ValueError(
                "Fixed SD4 mixed-policy format must contain exactly family and "
                "scale_method"
            )
        return FixedSD4WeightSpec(
            scale_method=str(payload["scale_method"]),  # type: ignore[arg-type]
        )
    if family == "paper_twn":
        if set(payload) != {"family", "threshold_multiplier"}:
            raise ValueError(
                "Paper-TWN mixed-policy format must contain exactly family and "
                "threshold_multiplier"
            )
        return PaperTWNWeightSpec(
            threshold_multiplier=float(payload["threshold_multiplier"])
        )
    if family == "exact_scaled_ternary":
        if set(payload) != {"family", "scale_method"}:
            raise ValueError(
                "exact ternary mixed-policy format must contain exactly family "
                "and scale_method"
            )
        return ExactTernaryWeightSpec(scale_method=str(payload["scale_method"]))
    if family == "twn_filterwise":
        if set(payload) != {"family", "threshold_multiplier"}:
            raise ValueError(
                "filter-wise TWN mixed-policy format must contain exactly family "
                "and threshold_multiplier"
            )
        return FilterwiseTWNWeightSpec(
            threshold_multiplier=float(payload["threshold_multiplier"])
        )
    raise ValueError(f"unsupported mixed-policy weight family: {family!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PinnedPathRoute:
    """A hash-verified path list imported as evidence, not implicit authorization."""

    route_id: str
    source_path: Path
    source_sha256: str
    list_key: str
    paths: tuple[str, ...]
    source_execution_authorized: bool
    map_validation_run: bool

    @classmethod
    def from_yaml(
        cls,
        *,
        route_id: str,
        path: str | Path,
        expected_sha256: str,
        list_key: str,
    ) -> PinnedPathRoute:
        source_path = Path(path).expanduser().resolve()
        if not route_id.strip() or not list_key.strip():
            raise ValueError("pinned route identity and list key must not be empty")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        actual_sha256 = _sha256(source_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"pinned route SHA-256 drifted: {actual_sha256} != {expected_sha256}"
            )
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("pinned route manifest must be a mapping")
        values = payload.get(list_key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"pinned route list is empty or invalid: {list_key}")
        paths = tuple(str(value) for value in values)
        if any(not value.strip() for value in paths) or len(set(paths)) != len(paths):
            raise ValueError("pinned route paths must be non-empty and unique")
        return cls(
            route_id=route_id,
            source_path=source_path,
            source_sha256=actual_sha256,
            list_key=list_key,
            paths=paths,
            source_execution_authorized=payload.get("execution_authorized") is True,
            map_validation_run=payload.get("map_validation_run") is True,
        )


@dataclass(frozen=True)
class RegionFormatDefault:
    """Default format for every unoverridden deployment layer in one region."""

    region: str
    spec: WeightFormatSpec

    def __post_init__(self) -> None:
        if not self.region.strip():
            raise ValueError("mixed-policy default region must not be empty")


@dataclass(frozen=True)
class PathFormatRoute:
    """One immutable path list that overrides region defaults with another format."""

    route_id: str
    paths: tuple[str, ...]
    spec: WeightFormatSpec

    def __post_init__(self) -> None:
        if not self.route_id.strip() or not self.paths:
            raise ValueError("mixed-policy path route identity must not be empty")
        if any(not path.strip() for path in self.paths):
            raise ValueError("mixed-policy route paths must not be empty")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("mixed-policy route paths must be unique")


@dataclass(frozen=True)
class MixedWeightPolicyCandidate:
    """A small reviewed policy interface compiled against real deployment paths."""

    candidate_id: str
    region_defaults: tuple[RegionFormatDefault, ...] = ()
    path_routes: tuple[PathFormatRoute, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("mixed weight candidate ID must not be empty")
        if not self.region_defaults and not self.path_routes:
            raise ValueError("mixed weight candidate must contain at least one rule")
        regions = tuple(item.region for item in self.region_defaults)
        if len(set(regions)) != len(regions):
            raise ValueError("mixed-policy default regions must be unique")
        route_ids = tuple(item.route_id for item in self.path_routes)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("mixed-policy route IDs must be unique")

    def compile(
        self,
        catalog: Full35WeightRegionCatalog,
    ) -> tuple[WeightRegionAssignment, ...]:
        """Resolve route paths to regions and reject non-deployment/overlap drift."""

        deployment_by_path = {site.path: site for site in catalog.deployment_sites}
        known_regions = {site.region for site in catalog.deployment_sites}
        unknown_defaults = tuple(
            item.region
            for item in self.region_defaults
            if item.region not in known_regions
        )
        if unknown_defaults:
            raise ValueError(
                "mixed-policy defaults reference unknown deployment regions: "
                + ",".join(unknown_defaults)
            )
        assignments = [
            WeightRegionAssignment(item.region, item.spec)
            for item in self.region_defaults
        ]
        claimed_paths: set[str] = set()
        for route in self.path_routes:
            missing = tuple(
                path for path in route.paths if path not in deployment_by_path
            )
            if missing:
                raise ValueError(
                    f"mixed-policy route {route.route_id} contains non-deployment paths: "
                    + ",".join(missing)
                )
            overlap = claimed_paths & set(route.paths)
            if overlap:
                raise ValueError(
                    "mixed-policy path routes overlap: " + ",".join(sorted(overlap))
                )
            claimed_paths.update(route.paths)
            paths_by_region: dict[str, list[str]] = {}
            for path in route.paths:
                region = deployment_by_path[path].region
                paths_by_region.setdefault(region, []).append(path)
            assignments.extend(
                WeightRegionAssignment(
                    region=region,
                    spec=route.spec,
                    paths=tuple(paths),
                )
                for region, paths in paths_by_region.items()
            )
        return tuple(assignments)

    @property
    def format_id(self) -> str:
        """Return a deterministic family summary for metric-gate semantics."""

        ordered: list[str] = []
        for item in (*self.region_defaults, *self.path_routes):
            format_id = item.spec.format_id
            if format_id not in ordered:
                ordered.append(format_id)
        return "mixed-" + "-".join(ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "format_id": self.format_id,
            "region_defaults": [
                {"region": item.region, "format": _format_record(item.spec)}
                for item in self.region_defaults
            ],
            "path_routes": [
                {
                    "route_id": item.route_id,
                    "path_count": len(item.paths),
                    "paths": list(item.paths),
                    "format": _format_record(item.spec),
                }
                for item in self.path_routes
            ],
        }


def _format_record(spec: WeightFormatSpec) -> dict[str, object]:
    if isinstance(spec, UniformWeightSpec):
        return {
            "family": "uniform",
            "bits": spec.bits,
            "scale_method": spec.scale_method,
        }
    if isinstance(spec, FixedSD4WeightSpec):
        return {
            "family": "fixed_sd4",
            "bits": spec.bits,
            "scale_method": spec.scale_method,
        }
    if isinstance(spec, ExactTernaryWeightSpec):
        return {
            "family": "exact_scaled_ternary",
            "bits": spec.bits,
            "scale_method": spec.scale_method,
        }
    if isinstance(spec, FilterwiseTWNWeightSpec):
        return {
            "family": "twn_filterwise",
            "bits": spec.bits,
            "threshold_multiplier": spec.threshold_multiplier,
        }
    return {
        "family": "paper_twn",
        "bits": spec.bits,
        "threshold_multiplier": spec.threshold_multiplier,
    }


@dataclass(frozen=True)
class PinnedComparisonSource:
    """One hash-pinned, already validated candidate used only for Pareto context."""

    candidate: BalanceCandidate
    source_path: Path
    source_sha256: str

    @classmethod
    def from_report(
        cls,
        *,
        record: object,
        expected_policy_id: str,
        expected_reference_bytes: int,
    ) -> PinnedComparisonSource:
        payload = _mapping(record, "mixed-policy comparison source")
        if set(payload) != {"path", "sha256", "candidate_id"}:
            raise ValueError(
                "comparison source must contain exactly path, sha256, and candidate_id"
            )
        path = _resolve(payload["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = _sha256(path)
        if actual_sha256 != str(payload["sha256"]):
            raise ValueError("mixed-policy comparison report SHA-256 drifted")
        report = _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            "mixed-policy comparison report",
        )
        if report.get("status") != "completed":
            raise ValueError("mixed-policy comparison report is not completed")
        candidate_id = str(payload["candidate_id"])
        raw_candidates = report.get("candidates")
        if not isinstance(raw_candidates, list):
            raise TypeError("mixed-policy comparison report has no candidate list")
        matches = tuple(
            item
            for item in raw_candidates
            if isinstance(item, dict) and item.get("candidate_id") == candidate_id
        )
        if len(matches) != 1:
            raise ValueError(
                f"comparison candidate must resolve exactly once: {candidate_id}"
            )
        item = matches[0]
        if item.get("policy_id") != expected_policy_id:
            raise ValueError("comparison candidate activation policy differs")
        if int(item["reference_fp32_bytes"]) != expected_reference_bytes:
            raise ValueError("comparison candidate reference byte contract differs")
        candidate = BalanceCandidate(
            candidate_id=candidate_id,
            policy_id=str(item["policy_id"]),
            format_id=str(item["format_id"]),
            worst_total_delta=float(item["worst_total_delta"]),
            packed_weight_bytes=int(item["packed_weight_bytes"]),
            reference_fp32_bytes=int(item["reference_fp32_bytes"]),
            hard_gate_passed=bool(item["hard_gate_passed"]),
        )
        return cls(candidate, path, actual_sha256)


@dataclass(frozen=True)
class Full35MixedPolicySearchPlan:
    """Reviewed, candidate-only mixed-format PTQ search contract."""

    config_path: Path
    config_sha256: str
    plan_id: str
    execution_authorization_id: str
    metric_contract_id: str
    base_plan: Full35SearchValidationPlan
    weight_study: WeightSensitivityStudy
    activation_policy: Full35ActivationPolicy
    activation_checkpoint: Path
    activation_checkpoint_sha256: str
    pinned_routes: dict[str, PinnedPathRoute]
    candidates: tuple[MixedWeightPolicyCandidate, ...]
    comparison_sources: tuple[PinnedComparisonSource, ...]
    reference_report: Path
    reference_report_sha256: str
    reference_regate: Path
    gate_spec: Full35MetricGateSpec
    accuracy_tolerance: float
    formal_training: bool = False
    formal_validation: bool = False

    @property
    def comparison_candidates(self) -> tuple[BalanceCandidate, ...]:
        return tuple(item.candidate for item in self.comparison_sources)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Full35MixedPolicySearchPlan:
        config_path = Path(path).expanduser().resolve()
        payload = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "mixed-policy search plan",
        )
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported mixed-policy search plan schema")
        if payload.get("execution_authorized") is not True:
            raise ValueError("mixed-policy search execution is not authorized")
        if payload.get("formal_training") is not False:
            raise ValueError("mixed-policy search cannot authorize training")
        if payload.get("formal_validation") is not False:
            raise ValueError("mixed-policy search cannot use formal validation")

        authorization = _mapping(
            payload.get("execution_authorization"), "mixed-policy authorization"
        )
        authorization_id = str(authorization.get("authorization_id", "")).strip()
        if not authorization_id or authorization.get("validation_roles") != [
            "candidate"
        ]:
            raise ValueError("mixed-policy execution authorization is invalid")

        upstream = _mapping(payload.get("upstream"), "mixed-policy upstream")
        base_path = _verified_file(
            upstream.get("base_search_plan"), "mixed-policy base search plan"
        )
        base_plan = Full35SearchValidationPlan.from_yaml(base_path)
        weight_path = _verified_file(
            upstream.get("weight_plan"), "mixed-policy weight plan"
        )
        weight_study = WeightSensitivityStudy.from_yaml(weight_path)
        reference_regate = _verified_file(
            upstream.get("reference_map50_regate"),
            "mixed-policy reference map50 re-gate",
        )
        reference_record = _mapping(
            upstream.get("reference_search_report"),
            "mixed-policy reference search report",
        )
        if set(reference_record) != {"path", "sha256", "reuse_roles"}:
            raise ValueError("mixed-policy reference report fields differ")
        if reference_record.get("reuse_roles") != ["accepted", "matched"]:
            raise ValueError(
                "mixed-policy reference reuse must be accepted plus matched"
            )
        reference_report = _resolve(reference_record["path"])
        reference_sha256 = str(reference_record["sha256"])
        if _sha256(reference_report) != reference_sha256:
            raise ValueError("mixed-policy reference report SHA-256 drifted")

        activation = _mapping(payload.get("activation"), "mixed-policy activation")
        if set(activation) != {
            "default",
            "bits",
            "signed_codes",
            "region_assignments",
            "policy_id",
            "checkpoint",
        }:
            raise ValueError("mixed-policy activation fields differ from schema")
        raw_assignments = activation["region_assignments"]
        if not isinstance(raw_assignments, dict):
            raise TypeError("activation region_assignments must be a mapping")
        activation_policy = Full35ActivationPolicy(
            activation=str(activation["default"]),
            bits=int(activation["bits"]),
            signed_codes=bool(activation["signed_codes"]),
            region_assignments=tuple(
                (str(region), str(value)) for region, value in raw_assignments.items()
            ),
        )
        if activation_policy.policy_id != activation["policy_id"]:
            raise ValueError("mixed-policy activation policy_id is not canonical")
        if activation_policy.policy_id != base_plan.matched_policy_id:
            raise ValueError(
                "mixed-policy activation must match the pinned matched reference"
            )
        activation_checkpoint = _verified_file(
            activation.get("checkpoint"), "mixed-policy activation checkpoint"
        )
        activation_checkpoint_sha256 = str(
            _mapping(activation["checkpoint"], "activation checkpoint")["sha256"]
        )

        raw_routes = payload.get("routes")
        if not isinstance(raw_routes, dict):
            raise TypeError("mixed-policy routes must be a mapping")
        pinned_routes: dict[str, PinnedPathRoute] = {}
        for route_id, raw_route in raw_routes.items():
            route = _mapping(raw_route, f"route {route_id}")
            if set(route) != {"manifest", "list_key"}:
                raise ValueError(f"mixed-policy route fields differ: {route_id}")
            manifest = _mapping(route["manifest"], f"route manifest {route_id}")
            if set(manifest) != {"path", "sha256"}:
                raise ValueError(f"mixed-policy route manifest differs: {route_id}")
            pinned_routes[str(route_id)] = PinnedPathRoute.from_yaml(
                route_id=str(route_id),
                path=_resolve(manifest["path"]),
                expected_sha256=str(manifest["sha256"]),
                list_key=str(route["list_key"]),
            )

        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("mixed-policy candidates must be a non-empty list")
        candidates: list[MixedWeightPolicyCandidate] = []
        used_routes: set[str] = set()
        for index, raw_candidate in enumerate(raw_candidates):
            item = _mapping(raw_candidate, f"mixed-policy candidate {index}")
            if set(item) != {"candidate_id", "region_defaults", "path_routes"}:
                raise ValueError(f"mixed-policy candidate {index} fields differ")
            raw_defaults = item["region_defaults"]
            raw_path_routes = item["path_routes"]
            if not isinstance(raw_defaults, list) or not isinstance(
                raw_path_routes, list
            ):
                raise TypeError("mixed-policy rules must be lists")
            defaults = tuple(
                RegionFormatDefault(
                    region=str(_mapping(value, "region default")["region"]),
                    spec=parse_weight_format_spec(
                        _mapping(value, "region default")["format"]
                    ),
                )
                for value in raw_defaults
            )
            routes: list[PathFormatRoute] = []
            for value in raw_path_routes:
                route_record = _mapping(value, "candidate path route")
                if set(route_record) != {"route_id", "format"}:
                    raise ValueError("candidate path route fields differ")
                route_id = str(route_record["route_id"])
                if route_id not in pinned_routes:
                    raise ValueError(f"candidate references unknown route: {route_id}")
                used_routes.add(route_id)
                routes.append(
                    PathFormatRoute(
                        route_id=route_id,
                        paths=pinned_routes[route_id].paths,
                        spec=parse_weight_format_spec(route_record["format"]),
                    )
                )
            candidates.append(
                MixedWeightPolicyCandidate(
                    candidate_id=str(item["candidate_id"]),
                    region_defaults=defaults,
                    path_routes=tuple(routes),
                )
            )

        authorized_candidates = tuple(
            str(value) for value in authorization.get("candidate_ids", ())
        )
        if authorized_candidates != tuple(item.candidate_id for item in candidates):
            raise ValueError("mixed-policy candidate order differs from authorization")
        authorized_routes = {str(value) for value in authorization.get("route_ids", ())}
        if used_routes != authorized_routes:
            raise ValueError("mixed-policy route use differs from authorization")

        gates = _mapping(payload.get("gates"), "mixed-policy gates")
        if gates.get("all_eight_metrics_required") is not True:
            raise ValueError("mixed-policy gate must require all eight metrics")
        if gates.get("activation_replacement_included") is not True:
            raise ValueError("mixed-policy gate must include activation replacement")
        gate_spec = Full35MetricGateSpec(
            metric_family=str(gates["metric_family"]),  # type: ignore[arg-type]
            total_max_drop=float(gates["total_max_drop"]),
            w8_incremental_max_drop=float(gates["w8_incremental_max_drop"]),
            sham_max_absolute_drift=float(gates["sham_max_absolute_drift"]),
            recovery_floor=float(gates["recovery_floor"]),
        )
        if gate_spec.metric_family != "map50" or gate_spec.total_max_drop != 0.015:
            raise ValueError("mixed-policy search must use total mAP50 drop 0.015")

        reference_bytes = (
            int(
                weight_study.expected_catalog["totals"][  # type: ignore[index]
                    "deployment_weight_elements"
                ]
            )
            * 4
        )
        raw_comparisons = upstream.get("comparison_candidates")
        if not isinstance(raw_comparisons, list) or not raw_comparisons:
            raise ValueError("mixed-policy plan requires comparison candidates")
        comparison_sources = tuple(
            PinnedComparisonSource.from_report(
                record=record,
                expected_policy_id=activation_policy.policy_id,
                expected_reference_bytes=reference_bytes,
            )
            for record in raw_comparisons
        )

        plan_id = str(payload.get("plan_id", "")).strip()
        metric_contract_id = str(payload.get("metric_contract_id", "")).strip()
        if not plan_id or not metric_contract_id:
            raise ValueError("mixed-policy plan and metric contract IDs are required")
        selection = _mapping(payload.get("selection"), "mixed-policy selection")
        return cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            plan_id=plan_id,
            execution_authorization_id=authorization_id,
            metric_contract_id=metric_contract_id,
            base_plan=base_plan,
            weight_study=weight_study,
            activation_policy=activation_policy,
            activation_checkpoint=activation_checkpoint,
            activation_checkpoint_sha256=activation_checkpoint_sha256,
            pinned_routes=pinned_routes,
            candidates=tuple(candidates),
            comparison_sources=comparison_sources,
            reference_report=reference_report,
            reference_report_sha256=reference_sha256,
            reference_regate=reference_regate,
            gate_spec=gate_spec,
            accuracy_tolerance=float(selection["accuracy_tolerance"]),
        )

    @property
    def coco_yaml(self) -> Path:
        return self.base_plan.coco_yaml

    @property
    def pose_search_yaml(self) -> Path:
        return self.base_plan.pose_search_yaml

    @property
    def bbat5_registry(self) -> Path:
        return self.base_plan.bbat5_registry

    @property
    def runtime_view(self) -> Path:
        return self.base_plan.runtime_view

    @property
    def detect_batch_size(self) -> int:
        return self.base_plan.detect_batch_size

    @property
    def pose_batch_size(self) -> int:
        return self.base_plan.pose_batch_size

    @property
    def detect_workers(self) -> int:
        return self.base_plan.detect_workers

    @property
    def pose_workers(self) -> int:
        return self.base_plan.pose_workers

    @property
    def image_size(self) -> int:
        return self.base_plan.image_size

    @property
    def plots(self) -> bool:
        return False

    @property
    def save_coco_json(self) -> bool:
        return False

    @property
    def matched_policy_id(self) -> str:
        return self.activation_policy.policy_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan": str(self.config_path),
            "plan_sha256": self.config_sha256,
            "execution_authorization_id": self.execution_authorization_id,
            "metric_contract_id": self.metric_contract_id,
            "base_plan": str(self.base_plan.config_path),
            "base_plan_sha256": self.base_plan.config_sha256,
            "weight_plan": str(self.weight_study.config_path),
            "weight_plan_sha256": self.weight_study.config_sha256,
            "activation_policy_id": self.activation_policy.policy_id,
            "activation_checkpoint": str(self.activation_checkpoint),
            "activation_checkpoint_sha256": self.activation_checkpoint_sha256,
            "routes": {
                route_id: {
                    "path": str(route.source_path),
                    "sha256": route.source_sha256,
                    "list_key": route.list_key,
                    "path_count": len(route.paths),
                    "source_execution_authorized": route.source_execution_authorized,
                    "source_map_validation_run": route.map_validation_run,
                }
                for route_id, route in self.pinned_routes.items()
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "comparison_candidates": [
                {
                    **item.candidate.to_dict(),
                    "source_path": str(item.source_path),
                    "source_sha256": item.source_sha256,
                }
                for item in self.comparison_sources
            ],
            "datasets": {
                "coco": str(self.coco_yaml),
                "bbat5_pose_search": str(self.pose_search_yaml),
                "bbat5_registry": str(self.bbat5_registry),
                "runtime_view": str(self.runtime_view),
                "assignment_changed": False,
            },
            "batch": {
                "detect": self.detect_batch_size,
                "pose": self.pose_batch_size,
            },
            "workers": {
                "detect": self.detect_workers,
                "pose": self.pose_workers,
            },
            "image_size": self.image_size,
            "gates": self.gate_spec.to_dict(),
            "accuracy_tolerance": self.accuracy_tolerance,
            "formal_training": False,
            "formal_validation": False,
        }


def _calibration_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "manifest_sha256",
            "samples",
            "quantizers",
            "valid_observers",
            "invalid_observers",
            "observed_minimum",
            "observed_maximum",
        )
    }


def _applied_policy_record(applied: Any) -> dict[str, Any]:
    return {
        "assignments": list(applied.assignment_records),
        "quantized_modules": applied.quantized_modules,
        "weight_elements": applied.weight_elements,
        "weight_code_bytes": applied.weight_code_bytes,
        "scale_bytes": applied.scale_bytes,
        "formats": [
            {
                "region": item.region,
                "format_id": item.format_id,
                "quantized_modules": item.quantized_modules,
                "weight_elements": item.weight_elements,
                "numeric": item.numeric,
                "sites": list(item.site_metrics),
            }
            for item in applied.applied
        ],
    }


def _packed_cost(
    *, deployment_weight_elements: int, policy_record: Mapping[str, Any]
) -> tuple[int, int]:
    reference = deployment_weight_elements * 4
    packed = (
        (deployment_weight_elements - int(policy_record["weight_elements"])) * 4
        + int(policy_record["weight_code_bytes"])
        + int(policy_record["scale_bytes"])
    )
    return packed, reference


def run_mixed_policy_search(
    *,
    plan: Full35MixedPolicySearchPlan,
    output: Path,
    device_index: int,
    resume: bool,
) -> int:
    """Run fixed-probe diagnostics then candidate-only eight-metric validation."""

    prepared, base_contract = _prepare_contract(plan, device_index=device_index)  # type: ignore[arg-type]
    contract = {
        **base_contract,
        "kind": "full35_mixed_weight_policy_map50_search",
        "reference_reuse": {
            "report": str(plan.reference_report),
            "sha256": plan.reference_report_sha256,
            "roles": ["accepted", "matched"],
            "validation_rerun": False,
        },
        "diagnostic": {
            "manifest": str(plan.weight_study.diagnostic_manifest),
            "manifest_sha256": plan.weight_study.diagnostic_manifest_sha256,
            "fixed_probe_per_task": 1,
            "all_finite_required": True,
            "same_structure_required": True,
            "selection_claim": False,
        },
    }
    regated = regate_search_report(
        source_report=plan.reference_report,
        expected_source_sha256=plan.reference_report_sha256,
        metric_contract_id=plan.metric_contract_id,
        spec=plan.gate_spec,
    )
    pinned_regate = _mapping(
        json.loads(plan.reference_regate.read_text(encoding="utf-8")),
        "pinned reference re-gate",
    )
    if pinned_regate.get("gate") != regated.get("gate"):
        raise RuntimeError("fresh mixed-policy reference re-gate differs from pin")

    source_payload = _mapping(
        json.loads(plan.reference_report.read_text(encoding="utf-8")),
        "mixed-policy reference report",
    )
    source_roles = _mapping(source_payload.get("roles"), "reference roles")
    expected_calibration = _mapping(
        _mapping(source_roles.get("matched"), "reference matched role").get(
            "calibration"
        ),
        "reference matched calibration",
    )

    output_path = output.expanduser().resolve()
    validation_root = PROJECT_ROOT / "artifacts/runs" / plan.plan_id / "validation"
    diagnostics: dict[str, Any] = {}
    results: dict[str, Any] = {}
    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"output already exists; use --resume or a new path: {output_path}"
            )
        existing = _mapping(
            json.loads(output_path.read_text(encoding="utf-8")),
            "existing mixed-policy report",
        )
        if existing.get("contract") != contract:
            raise RuntimeError(
                "existing mixed-policy contract differs; refusing resume"
            )
        if isinstance(existing.get("diagnostics"), dict):
            diagnostics.update(existing["diagnostics"])
        if isinstance(existing.get("results"), dict):
            results.update(existing["results"])
    elif validation_root.exists():
        raise FileExistsError(
            "mixed-policy validation artifacts exist without report; refusing overwrite"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "full35_mixed_weight_policy_map50_search",
        "status": "running",
        "contract": contract,
        "reference": regated,
        "diagnostics": diagnostics,
        "results": results,
        "selection": None,
        "formal_training": False,
        "formal_validation": False,
    }
    _atomic_json(output_path, payload)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(device_index)
    model, source, activation_applied, build = _build_deployment_policy(
        activation=plan.activation_policy.activation,
        checkpoint=plan.activation_checkpoint,
        checkpoint_sha256=plan.activation_checkpoint_sha256,
        expected_catalog=plan.weight_study.expected_catalog,
    )
    calibration = _calibrate_activation_a8(
        plan=plan,  # type: ignore[arg-type]
        model=model,
        applied=activation_applied,
        device_index=device_index,
    )
    if _calibration_identity(calibration) != _calibration_identity(
        expected_calibration
    ):
        raise RuntimeError("mixed-policy activation calibration differs from reference")
    catalog = Full35WeightRegionCatalog.inspect(model)
    if catalog.summary() != plan.weight_study.expected_catalog:
        raise RuntimeError("mixed-policy catalog differs from reviewed contract")

    manifest = plan.weight_study.load_diagnostic_manifest(verify_files=True)
    probe_paths = {
        "detect": manifest.paths("probe", "coco_detect")[0],
        "pose": manifest.paths("probe", "bbat_pose")[0],
    }
    probes = {
        task: _letterbox(path, plan.image_size).cuda(device_index)
        for task, path in probe_paths.items()
    }
    with torch.inference_mode():
        probe_references = {
            task: model(image, task=task) for task, image in probes.items()
        }

    adapter = WeightQuantizationAdapter()
    try:
        for index, candidate in enumerate(plan.candidates, start=1):
            if diagnostics.get(candidate.candidate_id, {}).get("status") in {
                "diagnostic_pass",
                "diagnostic_failed",
            }:
                continue
            print(
                f"[diagnostic {index}/{len(plan.candidates)}] {candidate.candidate_id}",
                flush=True,
            )
            payload["current_candidate"] = candidate.candidate_id
            payload["current_phase"] = "diagnostic"
            _atomic_json(output_path, payload)
            started = time.perf_counter()
            assignments = candidate.compile(catalog)
            with adapter.quantized_policy(
                model,
                catalog=catalog,
                assignments=assignments,
            ) as applied:
                with torch.inference_mode():
                    probe_candidates = {
                        task: model(image, task=task) for task, image in probes.items()
                    }
                comparisons = {
                    task: {
                        **_compare_outputs(
                            probe_references[task], probe_candidates[task]
                        ),
                        "deployment": _deployment_comparison(
                            probe_references[task],
                            probe_candidates[task],
                            task=task,
                        ),
                    }
                    for task in probes
                }
                all_finite = all(item["all_finite"] for item in comparisons.values())
                same_structure = all(
                    item["same_structure"] for item in comparisons.values()
                )
                policy_record = _applied_policy_record(applied)
            diagnostics[candidate.candidate_id] = {
                "status": (
                    "diagnostic_pass"
                    if all_finite and same_structure
                    else "diagnostic_failed"
                ),
                "selection_claim": False,
                "candidate": candidate.to_dict(),
                "weight_quantization": policy_record,
                "tasks": comparisons,
                "all_finite": all_finite,
                "same_structure": same_structure,
                "seconds": time.perf_counter() - started,
            }
            payload["diagnostics"] = diagnostics
            _atomic_json(output_path, payload)

        accepted_metrics = _mapping(
            _mapping(regated["roles"], "re-gated roles")["accepted"],
            "re-gated accepted role",
        )["metrics"]
        matched_metrics = _mapping(
            _mapping(regated["roles"], "re-gated roles")["matched"],
            "re-gated matched role",
        )["metrics"]
        accepted = Full35MetricSnapshot(
            run_id=f"{plan.plan_id}:accepted-reused",
            policy_id=str(regated["roles"]["accepted"]["policy_id"]),
            metric_contract_id=plan.metric_contract_id,
            metrics=accepted_metrics,
        )
        matched = Full35MetricSnapshot(
            run_id=f"{plan.plan_id}:matched-reused",
            policy_id=plan.matched_policy_id,
            metric_contract_id=plan.metric_contract_id,
            metrics=matched_metrics,
        )

        for index, candidate in enumerate(plan.candidates, start=1):
            if results.get(candidate.candidate_id, {}).get("status") == "completed":
                continue
            diagnostic = _mapping(
                diagnostics.get(candidate.candidate_id),
                f"diagnostic {candidate.candidate_id}",
            )
            if diagnostic.get("status") != "diagnostic_pass":
                results[candidate.candidate_id] = {
                    "status": "skipped_diagnostic_failed",
                    "candidate": candidate.to_dict(),
                }
                payload["results"] = results
                _atomic_json(output_path, payload)
                continue
            print(
                f"[validation {index}/{len(plan.candidates)}] {candidate.candidate_id}",
                flush=True,
            )
            payload["current_candidate"] = candidate.candidate_id
            payload["current_phase"] = "search_validation"
            _atomic_json(output_path, payload)
            assignments = candidate.compile(catalog)
            with adapter.quantized_policy(
                model,
                catalog=catalog,
                assignments=assignments,
            ) as applied:
                record = _validate_role(
                    role=candidate.candidate_id,
                    model=model,
                    source=source,
                    plan=plan,  # type: ignore[arg-type]
                    pose_yaml=prepared.yaml,
                    output_root=validation_root,
                    device_index=device_index,
                )
                policy_record = _applied_policy_record(applied)
            metric_candidate = Full35MetricCandidate(
                run_id=f"{plan.plan_id}:{candidate.candidate_id}",
                format_id=candidate.format_id,
                stage="ptq",
                policy_id=plan.matched_policy_id,
                metric_contract_id=plan.metric_contract_id,
                metrics=record["metrics"],
            )
            gate = Full35MetricGate(plan.gate_spec).evaluate(
                metric_candidate, accepted, matched
            )
            record.update(
                {
                    "status": "completed",
                    "policy_id": plan.matched_policy_id,
                    "candidate": candidate.to_dict(),
                    "activation_output_quantization": "calibrated_lsq_plus_a8",
                    "weight_quantization": policy_record,
                    "gate": gate.to_dict(),
                    "build": build,
                    "calibration": calibration,
                }
            )
            results[candidate.candidate_id] = record
            payload["results"] = results
            _atomic_json(output_path, payload)

        deployment_elements = int(
            plan.weight_study.expected_catalog["totals"][  # type: ignore[index]
                "deployment_weight_elements"
            ]
        )
        all_candidates = list(plan.comparison_candidates)
        for candidate in plan.candidates:
            record = _mapping(results[candidate.candidate_id], "mixed-policy result")
            if record.get("status") != "completed":
                continue
            weight = _mapping(
                record.get("weight_quantization"), "mixed-policy weight result"
            )
            packed, reference_bytes = _packed_cost(
                deployment_weight_elements=deployment_elements,
                policy_record=weight,
            )
            gate = _mapping(record.get("gate"), "mixed-policy metric gate")
            all_candidates.append(
                BalanceCandidate(
                    candidate_id=candidate.candidate_id,
                    policy_id=plan.matched_policy_id,
                    format_id=candidate.format_id,
                    worst_total_delta=float(gate["worst_total_delta"]),
                    packed_weight_bytes=packed,
                    reference_fp32_bytes=reference_bytes,
                    hard_gate_passed=bool(gate["passed"]),
                )
            )
        selection = QuantizationBalanceSelector(
            total_max_drop=plan.gate_spec.total_max_drop,
            accuracy_tolerance=plan.accuracy_tolerance,
        ).select(tuple(all_candidates))
        payload.pop("current_candidate", None)
        payload.pop("current_phase", None)
        payload["candidates"] = [item.to_dict() for item in all_candidates]
        payload["selection"] = selection.to_dict()
        payload["status"] = "completed"
        _atomic_json(output_path, payload)
        print(json.dumps(payload["selection"], ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        _atomic_json(output_path, payload)
        raise
    finally:
        del model, source, activation_applied, build, calibration, catalog
        gc.collect()
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full35 reviewed mixed weight-format mAP50 search"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs/experiments/v10-qsilu-mixed-weight-policy-search-v1.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/reports/v10-qsilu-mixed-weight-policy-search-v1.json"
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--execute-reviewed-plan", action="store_true")
    args = parser.parse_args(argv)
    plan = Full35MixedPolicySearchPlan.from_yaml(args.plan)
    if args.list_only:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.execute_reviewed_plan:
        parser.error("execution requires --execute-reviewed-plan")
    if (
        not torch.cuda.is_available()
        or not 0 <= args.device < torch.cuda.device_count()
    ):
        parser.error(f"CUDA device {args.device} is unavailable")
    return run_mixed_policy_search(
        plan=plan,
        output=args.output,
        device_index=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
