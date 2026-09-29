"""Persistent progressive PTQ-to-QAT queue contracts and selection logic."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import yaml

from .metric_gate import FULL35_MAP50_95_KEYS, FULL35_MAP50_KEYS
from .mixed_policy_search import parse_weight_format_spec
from .progressive_preparation import LockedQATParentSpec
from .weight_quantization import WeightFormatSpec

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_COCO = Path("/home/uxin/yolo/coco2017.yaml")
_CANONICAL_REGISTRY = Path("/home/uxin/yolo/configs/datasets/bbat5-v1.yaml")
_CANONICAL_POSE_SEARCH = Path(
    "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose-search.yaml"
)
_PROGRESSIVE_METRIC_KEYS = FULL35_MAP50_KEYS + FULL35_MAP50_95_KEYS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _verified_file(record: object, label: str) -> tuple[Path, str]:
    payload = _mapping(record, label)
    if set(payload) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = _resolve(payload["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(payload["sha256"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 drifted: {actual} != {expected}")
    return path, actual


def _metrics(path: Path) -> dict[str, float]:
    payload = _mapping(
        json.loads(path.read_text(encoding="utf-8")),
        f"metrics {path}",
    )
    values = _mapping(payload.get("metrics"), f"metrics mapping {path}")
    missing = tuple(key for key in _PROGRESSIVE_METRIC_KEYS if key not in values)
    if missing:
        raise ValueError("metrics artifact is missing: " + ", ".join(missing))
    result = {key: float(values[key]) for key in _PROGRESSIVE_METRIC_KEYS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("metrics artifact contains non-finite values")
    return result


@dataclass(frozen=True)
class ProgressiveNamedFormat:
    format_id: str
    spec: WeightFormatSpec
    cpu_format_id: str


@dataclass(frozen=True)
class ProgressiveStage:
    stage_id: str
    region: str
    path: str
    formats: tuple[ProgressiveNamedFormat, ...]
    promotion_min_savings_bytes: int
    role: str
    promotion_allowed: bool = True


@dataclass(frozen=True)
class ProgressiveValidationSpec:
    image_size: int
    detect_batch: int
    pose_batch: int
    detect_workers: int
    pose_workers: int
    plots: bool
    save_coco_json: bool


@dataclass(frozen=True)
class ProgressiveShortQATSpec:
    execution_authorized: bool
    maximum_candidates: int
    main_candidates: int
    sentinel_candidates: int
    epochs: int
    patience: int
    fp32_epochs: int
    progressive_ramp_epochs: int
    full_quant_epochs: int
    detect_logical_batch: int
    detect_microbatch: int
    pose_batch: int
    optimizer: str
    added_noise: bool


def _cpu_format_id(spec: WeightFormatSpec) -> str:
    if spec.format_id in {"w4", "w5", "w6", "w7", "w8"}:
        return f"uniform-{spec.format_id}-per_output_channel-optimal_scaled_codebook"
    if spec.format_id == "fixed-sd4":
        return "fixed-sd4-per_output_channel-optimal_scaled_codebook"
    if spec.format_id == "paper-twn":
        return "paper-twn-layerwise"
    if spec.format_id == "twn-v3-0.75-filterwise":
        return "twn-v3-0.75-filterwise"
    if spec.format_id == "exact-scaled-ternary":
        return "exact-scaled-ternary-per_tensor"
    raise ValueError(f"format is outside the progressive CPU matrix: {spec.format_id}")


@dataclass(frozen=True)
class ProgressiveQueuePlan:
    config_path: Path
    config_sha256: str
    queue_id: str
    authorization_id: str
    parent: LockedQATParentSpec
    cpu_profile_path: Path
    cpu_profile_sha256: str
    accepted_metrics_path: Path
    accepted_metrics: dict[str, float]
    locked_parent_metrics_path: Path
    locked_parent_metrics: dict[str, float]
    coco_yaml: Path
    bbat5_registry: Path
    pose_search_yaml: Path
    runtime_view: Path
    validation: ProgressiveValidationSpec
    device_index: int
    poll_seconds: int
    maximum_retries_per_candidate: int
    map50_max_drop: float
    map50_95_max_drop: float
    recover_map50_floor: float
    recover_map50_95_floor: float
    map50_accuracy_tolerance: float
    map50_95_accuracy_tolerance: float
    skip_ternary_when_both_4bit_controls_reject: bool
    format_catalog: dict[str, ProgressiveNamedFormat]
    stages: tuple[ProgressiveStage, ...]
    short_qat: ProgressiveShortQATSpec
    run_root: Path
    formal_validation: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProgressiveQueuePlan:
        config_path = Path(path).expanduser().resolve()
        payload = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "progressive queue plan",
        )
        if payload.get("schema_version") != 1:
            raise ValueError("progressive queue schema_version must be 1")
        if payload.get("execution_authorized") is not True:
            raise ValueError("progressive queue execution is not authorized")
        if payload.get("formal_training") is not False:
            raise ValueError("PTQ queue cannot authorize formal training")
        if payload.get("formal_validation") is not False:
            raise ValueError("progressive queue cannot use formal validation")
        authorization = _mapping(
            payload.get("execution_authorization"), "queue authorization"
        )
        authorization_id = str(authorization.get("authorization_id", "")).strip()
        if not authorization_id:
            raise ValueError("progressive queue authorization_id is empty")

        sources = _mapping(payload.get("sources"), "queue sources")
        parent_path, parent_sha256 = _verified_file(
            sources.get("locked_parent"), "locked parent"
        )
        parent = LockedQATParentSpec.from_yaml(parent_path)
        if parent.config_sha256 != parent_sha256:
            raise ValueError("locked parent digest differs after parsing")
        profile_path, profile_sha256 = _verified_file(
            sources.get("cpu_profile"), "CPU profile"
        )
        accepted_path, _ = _verified_file(
            sources.get("accepted_metrics"), "accepted metrics"
        )
        locked_metrics_path, _ = _verified_file(
            sources.get("locked_parent_metrics"), "locked parent metrics"
        )
        if locked_metrics_path != parent.metrics_path:
            raise ValueError("queue locked metrics differ from parent manifest")
        profile = _mapping(
            json.loads(profile_path.read_text(encoding="utf-8")),
            "progressive CPU profile",
        )
        if (
            profile.get("status") != "completed"
            or profile.get("formal_validation") is not False
        ):
            raise ValueError("progressive CPU profile is incomplete")
        profile_parent = _mapping(profile.get("parent"), "CPU profile parent")
        if (
            profile_parent.get("manifest_sha256") != parent.config_sha256
            or profile_parent.get("parent_id") != parent.parent_id
        ):
            raise ValueError("CPU profile parent differs from queue parent")
        summary = _mapping(profile.get("summary"), "CPU profile summary")
        rows = summary.get("path_rankings")
        if not isinstance(rows, list) or len(rows) != parent.deployment_modules:
            raise ValueError("CPU profile path coverage differs from parent")
        cpu_rows = {
            str(_mapping(row, "CPU path row")["path"]): _mapping(row, "CPU path row")
            for row in rows
        }

        datasets = _mapping(payload.get("datasets"), "queue datasets")
        coco, _ = _verified_file(datasets.get("coco"), "COCO YAML")
        registry, _ = _verified_file(datasets.get("bbat5_registry"), "BBAT5 registry")
        pose_search, _ = _verified_file(
            datasets.get("bbat5_pose_search"), "BBAT5 pose-search YAML"
        )
        if (coco, registry, pose_search) != (
            _CANONICAL_COCO,
            _CANONICAL_REGISTRY,
            _CANONICAL_POSE_SEARCH,
        ):
            raise ValueError("queue datasets differ from canonical contracts")
        if datasets.get("assignment_changed") is not False:
            raise ValueError("queue cannot change BBAT5 assignment")
        runtime_view = _resolve(datasets.get("runtime_view"))

        raw_validation = _mapping(payload.get("validation"), "queue validation")
        batch = _mapping(raw_validation.get("batch"), "validation batch")
        workers = _mapping(raw_validation.get("workers"), "validation workers")
        validation = ProgressiveValidationSpec(
            image_size=int(raw_validation["image_size"]),
            detect_batch=int(batch["detect"]),
            pose_batch=int(batch["pose"]),
            detect_workers=int(workers["detect"]),
            pose_workers=int(workers["pose"]),
            plots=bool(raw_validation["plots"]),
            save_coco_json=bool(raw_validation["save_coco_json"]),
        )
        if (
            raw_validation.get("backend") != "bittrue"
            or min(
                validation.image_size,
                validation.detect_batch,
                validation.pose_batch,
                validation.detect_workers,
                validation.pose_workers,
            )
            < 1
            or validation.plots
            or validation.save_coco_json
        ):
            raise ValueError("queue validation settings differ from search contract")

        gpu = _mapping(payload.get("gpu_queue"), "GPU queue")
        device_index = int(gpu["device"])
        poll_seconds = int(gpu["poll_seconds"])
        retries = int(gpu["maximum_retries_per_candidate"])
        if (
            gpu.get("mode") != "low_token_event_only"
            or gpu.get("wait_when_foreign_compute_process_exists") is not True
            or device_index < 0
            or poll_seconds < 30
            or retries < 0
        ):
            raise ValueError("GPU queue settings are invalid")

        gates = _mapping(payload.get("gates"), "queue gates")
        if (
            gates.get("activation_replacement_included") is not True
            or gates.get("reference") != "accepted_full35"
            or gates.get("incremental_reference") != "previous_locked_stage"
        ):
            raise ValueError("queue gate references are invalid")
        map50_max_drop = -float(gates["all_eight_map50_each_at_least"])
        map50_95_max_drop = -float(gates["all_eight_map50_95_each_at_least"])
        recover_map50_floor = -float(gates["recover_map50_floor"])
        recover_map50_95_floor = -float(gates["recover_map50_95_floor"])
        if (
            map50_max_drop,
            map50_95_max_drop,
            recover_map50_floor,
            recover_map50_95_floor,
        ) != (0.015, 0.04, 0.04, 0.08):
            raise ValueError("queue active dual gate differs from user contract")

        selection = _mapping(payload.get("selection"), "queue selection")
        if (
            selection.get("prefer_smaller_within_tolerance") is not True
            or selection.get("retain_previous_policy_when_no_green_candidate")
            is not True
        ):
            raise ValueError("queue selection contract is invalid")
        map50_tolerance = float(selection["map50_accuracy_tolerance"])
        map50_95_tolerance = float(selection["map50_95_accuracy_tolerance"])

        pruning = _mapping(payload.get("pruning"), "queue pruning")
        if (
            pruning.get("run_4bit_controls_first") is not True
            or pruning.get("cpu_metrics_are_ranking_only") is not True
        ):
            raise ValueError("queue pruning contract is invalid")
        skip_ternary = bool(pruning.get("skip_ternary_when_both_4bit_controls_reject"))

        raw_formats = _mapping(payload.get("formats"), "queue formats")
        formats: dict[str, ProgressiveNamedFormat] = {}
        for format_id, encoded in raw_formats.items():
            spec = parse_weight_format_spec(encoded)
            formats[str(format_id)] = ProgressiveNamedFormat(
                format_id=str(format_id),
                spec=spec,
                cpu_format_id=_cpu_format_id(spec),
            )
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("queue stages must be a non-empty list")
        stages: list[ProgressiveStage] = []
        seen_stage_ids: set[str] = set()
        seen_paths: set[str] = set()
        for raw in raw_stages:
            entry = _mapping(raw, "queue stage")
            stage_id = str(entry["stage_id"])
            region = str(entry["region"])
            weight_path = str(entry["path"])
            format_ids = tuple(str(value) for value in entry["format_ids"])
            if (
                not stage_id
                or stage_id in seen_stage_ids
                or weight_path in seen_paths
                or not format_ids
                or len(set(format_ids)) != len(format_ids)
            ):
                raise ValueError("queue stage identity, path or formats are invalid")
            seen_stage_ids.add(stage_id)
            seen_paths.add(weight_path)
            row = cpu_rows.get(weight_path)
            if row is None or row.get("region") != region:
                raise ValueError(
                    f"queue stage path is absent from CPU profile: {stage_id}"
                )
            row_formats = _mapping(row.get("formats"), "CPU path formats")
            resolved_formats: list[ProgressiveNamedFormat] = []
            for format_id in format_ids:
                if format_id not in formats:
                    raise ValueError(
                        f"queue stage references unknown format: {format_id}"
                    )
                candidate = formats[format_id]
                if candidate.cpu_format_id not in row_formats:
                    raise ValueError(
                        f"queue stage CPU format evidence is absent: {stage_id}"
                    )
                resolved_formats.append(candidate)
            stages.append(
                ProgressiveStage(
                    stage_id=stage_id,
                    region=region,
                    path=weight_path,
                    formats=tuple(resolved_formats),
                    promotion_min_savings_bytes=int(
                        entry["promotion_min_savings_bytes"]
                    ),
                    role=str(entry["role"]),
                    promotion_allowed=bool(entry.get("promotion_allowed", True)),
                )
            )

        raw_qat = _mapping(payload.get("short_qat_queue"), "short QAT queue")
        roles = _mapping(raw_qat.get("roles"), "short QAT roles")
        schedule = _mapping(raw_qat.get("schedule"), "short QAT schedule")
        qat_batch = _mapping(raw_qat.get("batch"), "short QAT batch")
        short_qat = ProgressiveShortQATSpec(
            execution_authorized=bool(raw_qat["execution_authorized"]),
            maximum_candidates=int(raw_qat["maximum_candidates"]),
            main_candidates=int(roles["main"]),
            sentinel_candidates=int(roles["sentinel"]),
            epochs=int(raw_qat["epochs"]),
            patience=int(raw_qat["patience"]),
            fp32_epochs=int(schedule["fp32_epochs"]),
            progressive_ramp_epochs=int(schedule["progressive_ramp_epochs"]),
            full_quant_epochs=int(schedule["full_quant_epochs"]),
            detect_logical_batch=int(qat_batch["detect_logical"]),
            detect_microbatch=int(qat_batch["detect_microbatch"]),
            pose_batch=int(qat_batch["pose"]),
            optimizer=str(raw_qat["optimizer"]),
            added_noise=bool(raw_qat["added_noise"]),
        )
        if (
            not short_qat.execution_authorized
            or raw_qat.get("matched_sham_required") is not True
            or raw_qat.get("from_scratch") is not False
            or short_qat.maximum_candidates
            != short_qat.main_candidates + short_qat.sentinel_candidates
            or short_qat.epochs
            != short_qat.fp32_epochs
            + short_qat.progressive_ramp_epochs
            + short_qat.full_quant_epochs
            or short_qat.optimizer != "AdamW"
            or short_qat.added_noise
        ):
            raise ValueError("short QAT queue contract is invalid")
        deferred = _mapping(payload.get("deferred"), "deferred work")
        if any(
            deferred.get(key) is not False
            for key in (
                "long_qat_execution_authorized",
                "formal_validation_authorized",
                "multi_seed_authorized",
                "final_hardware_optimization_authorized",
            )
        ):
            raise ValueError("deferred final work must remain unauthorized")
        run_root = _resolve(payload.get("run_root"))
        if _PROJECT_ROOT not in run_root.parents:
            raise ValueError("queue run_root must stay inside the project")

        return cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            queue_id=str(payload["queue_id"]),
            authorization_id=authorization_id,
            parent=parent,
            cpu_profile_path=profile_path,
            cpu_profile_sha256=profile_sha256,
            accepted_metrics_path=accepted_path,
            accepted_metrics=_metrics(accepted_path),
            locked_parent_metrics_path=locked_metrics_path,
            locked_parent_metrics=_metrics(locked_metrics_path),
            coco_yaml=coco,
            bbat5_registry=registry,
            pose_search_yaml=pose_search,
            runtime_view=runtime_view,
            validation=validation,
            device_index=device_index,
            poll_seconds=poll_seconds,
            maximum_retries_per_candidate=retries,
            map50_max_drop=map50_max_drop,
            map50_95_max_drop=map50_95_max_drop,
            recover_map50_floor=recover_map50_floor,
            recover_map50_95_floor=recover_map50_95_floor,
            map50_accuracy_tolerance=map50_tolerance,
            map50_95_accuracy_tolerance=map50_95_tolerance,
            skip_ternary_when_both_4bit_controls_reject=skip_ternary,
            format_catalog=formats,
            stages=tuple(stages),
            short_qat=short_qat,
            run_root=run_root,
        )


@dataclass(frozen=True)
class ProgressiveGateResult:
    decision: str
    total_deltas: dict[str, float]
    incremental_deltas: dict[str, float]
    worst_total_map50_delta: float
    worst_total_map50_95_delta: float
    worst_incremental_map50_delta: float
    worst_incremental_map50_95_delta: float


@dataclass(frozen=True)
class ProgressiveDualMetricGate:
    map50_max_drop: float
    map50_95_max_drop: float
    recover_map50_floor: float
    recover_map50_95_floor: float

    def evaluate(
        self,
        candidate: dict[str, float],
        accepted: dict[str, float],
        current: dict[str, float],
    ) -> ProgressiveGateResult:
        for label, values in (
            ("candidate", candidate),
            ("accepted", accepted),
            ("current", current),
        ):
            missing = tuple(
                key for key in _PROGRESSIVE_METRIC_KEYS if key not in values
            )
            if missing:
                raise ValueError(f"{label} metrics are missing: {missing}")
        total = {
            key: float(candidate[key]) - float(accepted[key])
            for key in _PROGRESSIVE_METRIC_KEYS
        }
        incremental = {
            key: float(candidate[key]) - float(current[key])
            for key in _PROGRESSIVE_METRIC_KEYS
        }
        worst_m50 = round(min(total[key] for key in FULL35_MAP50_KEYS), 12)
        worst_m50_95 = round(
            min(total[key] for key in FULL35_MAP50_95_KEYS),
            12,
        )
        worst_incremental_m50 = round(
            min(incremental[key] for key in FULL35_MAP50_KEYS),
            12,
        )
        worst_incremental_m50_95 = round(
            min(incremental[key] for key in FULL35_MAP50_95_KEYS),
            12,
        )
        if (
            worst_m50 >= -self.map50_max_drop
            and worst_m50_95 >= -self.map50_95_max_drop
        ):
            decision = "green"
        elif (
            worst_m50 >= -self.recover_map50_floor
            and worst_m50_95 >= -self.recover_map50_95_floor
        ):
            decision = "recover"
        else:
            decision = "reject"
        return ProgressiveGateResult(
            decision=decision,
            total_deltas=total,
            incremental_deltas=incremental,
            worst_total_map50_delta=worst_m50,
            worst_total_map50_95_delta=worst_m50_95,
            worst_incremental_map50_delta=worst_incremental_m50,
            worst_incremental_map50_95_delta=worst_incremental_m50_95,
        )


@dataclass(frozen=True)
class ProgressivePolicyOutcome:
    candidate_id: str
    format_id: str
    metrics: dict[str, float]
    packed_bytes: int
    gate_decision: str


@dataclass(frozen=True)
class ProgressiveStageSelector:
    map50_tolerance: float
    map50_95_tolerance: float

    def select(
        self,
        *,
        baseline: ProgressivePolicyOutcome,
        candidates: tuple[ProgressivePolicyOutcome, ...],
        promotion_min_savings_bytes: int,
        promotion_allowed: bool,
    ) -> ProgressivePolicyOutcome:
        if not promotion_allowed:
            return baseline
        green = tuple(
            item
            for item in candidates
            if item.gate_decision == "green"
            and baseline.packed_bytes - item.packed_bytes >= promotion_min_savings_bytes
        )
        if not green:
            return baseline
        pool = (baseline, *green)
        best_m50 = max(
            min(item.metrics[key] for key in FULL35_MAP50_KEYS) for item in pool
        )
        near_m50 = tuple(
            item
            for item in pool
            if min(item.metrics[key] for key in FULL35_MAP50_KEYS)
            >= best_m50 - self.map50_tolerance
        )
        best_m50_95 = max(
            min(item.metrics[key] for key in FULL35_MAP50_95_KEYS) for item in near_m50
        )
        finalists = tuple(
            item
            for item in near_m50
            if min(item.metrics[key] for key in FULL35_MAP50_95_KEYS)
            >= best_m50_95 - self.map50_95_tolerance
        )
        return min(
            finalists,
            key=lambda item: (
                item.packed_bytes,
                -min(item.metrics[key] for key in FULL35_MAP50_KEYS),
                -min(item.metrics[key] for key in FULL35_MAP50_95_KEYS),
                item.candidate_id,
            ),
        )


@dataclass(frozen=True)
class ProgressiveAssignment:
    stage_id: str
    region: str
    path: str
    named_format: ProgressiveNamedFormat

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "region": self.region,
            "path": self.path,
            "format_id": self.named_format.format_id,
            "weight_format_id": self.named_format.spec.format_id,
            "cpu_format_id": self.named_format.cpu_format_id,
        }


@dataclass(frozen=True)
class ProgressiveCandidateEvaluation:
    candidate_id: str
    metrics: dict[str, float]
    output_dir: Path
    metrics_path: Path
    metrics_sha256: str
    weight_quantization: tuple[dict[str, object], ...]
    seconds: float


class ProgressiveCandidateEvaluator(Protocol):
    requires_gpu: bool

    def evaluate(
        self,
        *,
        candidate_id: str,
        assignments: tuple[ProgressiveAssignment, ...],
        output_dir: Path,
    ) -> ProgressiveCandidateEvaluation: ...


class QueueRecorder:
    def __init__(self, root: Path, queue_id: str) -> None:
        self.root = root
        self.queue_id = queue_id
        self.last_signature: str | None = None

    def transition(self, kind: str, **values: object) -> None:
        payload = {
            "schema_version": 1,
            "queue_id": self.queue_id,
            "kind": kind,
            "time_unix": time.time(),
            "values": values,
        }
        signature = json.dumps(
            {"kind": kind, "values": values},
            ensure_ascii=False,
            sort_keys=True,
        )
        if signature == self.last_signature:
            return
        self.last_signature = signature
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        _atomic_json(self.root / "status.json", payload)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_complete_evaluation(path: Path) -> ProgressiveCandidateEvaluation | None:
    if not path.is_file():
        return None
    payload = _mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "candidate result",
    )
    if payload.get("status") != "completed":
        return None
    metrics = _mapping(payload.get("metrics"), "candidate metrics")
    missing = tuple(key for key in _PROGRESSIVE_METRIC_KEYS if key not in metrics)
    if missing:
        raise ValueError("completed candidate metrics are incomplete")
    output_dir = Path(str(payload["output_dir"])).resolve()
    metrics_path = Path(str(payload["metrics_path"])).resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics_sha256 = str(payload["metrics_sha256"])
    if _sha256(metrics_path) != metrics_sha256:
        raise ValueError("completed candidate metrics SHA-256 drifted")
    raw_weight = payload.get("weight_quantization")
    if not isinstance(raw_weight, list):
        raise TypeError("completed candidate weight report must be a list")
    return ProgressiveCandidateEvaluation(
        candidate_id=str(payload["candidate_id"]),
        metrics={key: float(metrics[key]) for key in _PROGRESSIVE_METRIC_KEYS},
        output_dir=output_dir,
        metrics_path=metrics_path,
        metrics_sha256=metrics_sha256,
        weight_quantization=tuple(
            dict(_mapping(item, "weight quantization item")) for item in raw_weight
        ),
        seconds=float(payload["seconds"]),
    )


class Full35ProgressiveEvaluator:
    """Production adapter: one locked graph, official COCO/BBAT5 validator."""

    requires_gpu = True

    def __init__(self, plan: ProgressiveQueuePlan) -> None:
        self.plan = plan
        self._loaded: Any | None = None
        self._prepared: Any | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._loaded is not None and self._prepared is not None:
            return self._loaded, self._prepared
        from .qat_plan import Full35QATPlan
        from .qat_runtime import Full35QATRuntime
        from .search_data import prepare_bbat5_search_view

        qat_plan = Full35QATPlan.from_yaml(self.plan.parent.plan_path)
        self._loaded = Full35QATRuntime(qat_plan).load_deployment_parent(
            self.plan.parent.inference_checkpoint,
            checkpoint_sha256=self.plan.parent.inference_sha256,
            full_resume_sha256=self.plan.parent.full_resume_sha256,
            epoch=self.plan.parent.selected_epoch,
        )
        self._prepared = prepare_bbat5_search_view(
            self.plan.pose_search_yaml,
            self.plan.runtime_view,
        )
        return self._loaded, self._prepared

    @staticmethod
    def _official_validation_module() -> Any:
        from .search_validation import _official_validation_module

        return _official_validation_module()

    def evaluate(
        self,
        *,
        candidate_id: str,
        assignments: tuple[ProgressiveAssignment, ...],
        output_dir: Path,
    ) -> ProgressiveCandidateEvaluation:
        from .weight_quantization import (
            WeightQuantizationAdapter,
            WeightRegionAssignment,
        )

        if not assignments:
            raise ValueError("progressive candidate requires at least one assignment")
        if len({item.path for item in assignments}) != len(assignments):
            raise ValueError("progressive candidate paths must be unique")
        root = output_dir.expanduser().resolve()
        result_path = root / "result.json"
        existing = _load_complete_evaluation(result_path)
        if existing is not None:
            if existing.candidate_id != candidate_id:
                raise ValueError("candidate result identity drifted")
            return existing
        loaded, prepared = self._ensure_loaded()
        loaded.model.cpu().eval()
        policies = tuple(
            WeightRegionAssignment(
                region=item.region,
                paths=(item.path,),
                spec=item.named_format.spec,
            )
            for item in assignments
        )
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            result_path,
            {
                "schema_version": 1,
                "status": "running",
                "candidate_id": candidate_id,
                "queue_plan_sha256": self.plan.config_sha256,
                "assignments": [item.to_dict() for item in assignments],
                "formal_validation": False,
            },
        )
        started = time.perf_counter()
        try:
            with WeightQuantizationAdapter().quantized_policy(
                loaded.model,
                catalog=loaded.catalog,
                assignments=policies,
            ) as applied:
                validation = self._official_validation_module()
                settings = validation.ValidationSettings(
                    imgsz=self.plan.validation.image_size,
                    detect_batch_size=self.plan.validation.detect_batch,
                    pose_batch_size=self.plan.validation.pose_batch,
                    detect_workers=self.plan.validation.detect_workers,
                    pose_workers=self.plan.validation.pose_workers,
                    device=str(self.plan.device_index),
                    plots=self.plan.validation.plots,
                    save_coco_json=self.plan.validation.save_coco_json,
                )
                validator = validation.JointValidator(
                    loaded.source,
                    detect_data_yaml=self.plan.coco_yaml,
                    pose_data_yaml=prepared.yaml,
                    output_root=root / "validation",
                    settings=settings,
                )
                log_path = root / "validator.log"
                with (
                    log_path.open("a", encoding="utf-8") as log,
                    contextlib.redirect_stdout(log),
                    contextlib.redirect_stderr(log),
                ):
                    result = validator.validate(
                        loaded.model,
                        epoch=0,
                        kind="bittrue",
                    )
                metrics = {
                    key: float(result.metrics[key]) for key in _PROGRESSIVE_METRIC_KEYS
                }
                metrics_path = result.output_dir / "metrics.json"
                if not metrics_path.is_file():
                    raise FileNotFoundError(metrics_path)
                weight_report = tuple(
                    {
                        "region": item.region,
                        "format_id": item.format_id,
                        "quantized_modules": item.quantized_modules,
                        "weight_elements": item.weight_elements,
                        "numeric": item.numeric,
                        "sites": list(item.site_metrics),
                    }
                    for item in applied.applied
                )
                elapsed = time.perf_counter() - started
                payload = {
                    "schema_version": 1,
                    "status": "completed",
                    "candidate_id": candidate_id,
                    "queue_plan": str(self.plan.config_path),
                    "queue_plan_sha256": self.plan.config_sha256,
                    "parent_id": self.plan.parent.parent_id,
                    "parent_manifest_sha256": self.plan.parent.config_sha256,
                    "assignments": [item.to_dict() for item in assignments],
                    "metrics": metrics,
                    "output_dir": str(result.output_dir),
                    "metrics_path": str(metrics_path),
                    "metrics_sha256": _sha256(metrics_path),
                    "weight_quantization": list(weight_report),
                    "materialization": {
                        "detect_complete": result.materialized.detect_report.complete,
                        "pose_complete": result.materialized.pose_report.complete,
                    },
                    "seconds": elapsed,
                    "formal_training": False,
                    "formal_validation": False,
                }
                _atomic_json(result_path, payload)
                evaluation = ProgressiveCandidateEvaluation(
                    candidate_id=candidate_id,
                    metrics=metrics,
                    output_dir=result.output_dir,
                    metrics_path=metrics_path,
                    metrics_sha256=str(payload["metrics_sha256"]),
                    weight_quantization=weight_report,
                    seconds=elapsed,
                )
                del result, validator
                return evaluation
        except Exception as error:
            _atomic_json(
                result_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "candidate_id": candidate_id,
                    "queue_plan_sha256": self.plan.config_sha256,
                    "assignments": [item.to_dict() for item in assignments],
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "formal_validation": False,
                },
            )
            raise
        finally:
            loaded.model.cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _foreign_gpu_pids(device_index: int) -> tuple[int, ...]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--id={device_index}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    own_pid = os.getpid()
    return tuple(
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit() and int(line.strip()) != own_pid
    )


class ProgressiveExperimentQueue:
    """Deep queue module: resume, wait, evaluate, gate, lock and shortlist."""

    def __init__(
        self,
        plan: ProgressiveQueuePlan,
        *,
        evaluator: ProgressiveCandidateEvaluator | None = None,
    ) -> None:
        self.plan = plan
        self.evaluator = (
            Full35ProgressiveEvaluator(plan) if evaluator is None else evaluator
        )
        self.recorder = QueueRecorder(plan.run_root, plan.queue_id)
        profile = _mapping(
            json.loads(plan.cpu_profile_path.read_text(encoding="utf-8")),
            "queue CPU profile",
        )
        summary = _mapping(profile.get("summary"), "queue CPU profile summary")
        rows = summary.get("path_rankings")
        if not isinstance(rows, list):
            raise TypeError("queue CPU profile has no path rankings")
        self.cpu_rows = {
            str(_mapping(row, "queue CPU path row")["path"]): _mapping(
                row, "queue CPU path row"
            )
            for row in rows
        }
        self.base_packed_bytes = sum(
            int(row["parent_w8_estimated_bytes"]) for row in self.cpu_rows.values()
        )
        self.gate = ProgressiveDualMetricGate(
            map50_max_drop=plan.map50_max_drop,
            map50_95_max_drop=plan.map50_95_max_drop,
            recover_map50_floor=plan.recover_map50_floor,
            recover_map50_95_floor=plan.recover_map50_95_floor,
        )
        self.selector = ProgressiveStageSelector(
            map50_tolerance=plan.map50_accuracy_tolerance,
            map50_95_tolerance=plan.map50_95_accuracy_tolerance,
        )

    def _new_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "queue_id": self.plan.queue_id,
            "queue_plan": str(self.plan.config_path),
            "queue_plan_sha256": self.plan.config_sha256,
            "status": "running",
            "current_stage_index": 0,
            "current_metrics": dict(self.plan.locked_parent_metrics),
            "current_metrics_source": str(self.plan.locked_parent_metrics_path),
            "current_packed_bytes": self.base_packed_bytes,
            "locked_assignments": [],
            "stages": {},
            "recovery_pool": [],
            "formal_validation": False,
        }

    def _load_state(self) -> dict[str, object]:
        path = self.plan.run_root / "queue-state.json"
        if not path.is_file():
            state = self._new_state()
            _atomic_json(path, state)
            return state
        state = _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            "progressive queue state",
        )
        if (
            state.get("queue_id") != self.plan.queue_id
            or state.get("queue_plan_sha256") != self.plan.config_sha256
        ):
            raise ValueError("existing queue state differs from the reviewed plan")
        return state

    def _save_state(self, state: dict[str, object]) -> None:
        _atomic_json(self.plan.run_root / "queue-state.json", state)

    def _assignments(
        self,
        records: object,
    ) -> tuple[ProgressiveAssignment, ...]:
        if not isinstance(records, list):
            raise TypeError("queue locked assignments must be a list")
        result: list[ProgressiveAssignment] = []
        for raw in records:
            item = _mapping(raw, "locked assignment")
            format_id = str(item["format_id"])
            named = self.plan.format_catalog.get(format_id)
            if named is None:
                raise ValueError(f"locked assignment format is unknown: {format_id}")
            result.append(
                ProgressiveAssignment(
                    stage_id=str(item["stage_id"]),
                    region=str(item["region"]),
                    path=str(item["path"]),
                    named_format=named,
                )
            )
        return tuple(result)

    def _packed_bytes(
        self,
        assignments: tuple[ProgressiveAssignment, ...],
    ) -> int:
        packed = self.base_packed_bytes
        for assignment in assignments:
            row = self.cpu_rows[assignment.path]
            formats = _mapping(row["formats"], "queue CPU path formats")
            candidate = _mapping(
                formats[assignment.named_format.cpu_format_id],
                "queue CPU candidate format",
            )
            packed += int(candidate["packed_bytes"]) - int(
                row["parent_w8_estimated_bytes"]
            )
        return packed

    def _wait_for_gpu(self, stage_id: str, candidate_id: str) -> None:
        if not self.evaluator.requires_gpu:
            return
        while True:
            foreign = _foreign_gpu_pids(self.plan.device_index)
            if not foreign:
                self.recorder.transition(
                    "gpu_acquired",
                    stage_id=stage_id,
                    candidate_id=candidate_id,
                    device=self.plan.device_index,
                )
                return
            self.recorder.transition(
                "waiting_for_gpu",
                stage_id=stage_id,
                candidate_id=candidate_id,
                device=self.plan.device_index,
                foreign_pids=list(foreign),
                poll_seconds=self.plan.poll_seconds,
            )
            time.sleep(self.plan.poll_seconds)

    def _candidate_result(
        self,
        *,
        stage: ProgressiveStage,
        named_format: ProgressiveNamedFormat,
        current_assignments: tuple[ProgressiveAssignment, ...],
        current_metrics: dict[str, float],
        attempt_root: Path,
    ) -> tuple[ProgressivePolicyOutcome, dict[str, object]]:
        assignment = ProgressiveAssignment(
            stage_id=stage.stage_id,
            region=stage.region,
            path=stage.path,
            named_format=named_format,
        )
        assignments = (*current_assignments, assignment)
        candidate_id = f"{stage.stage_id}--{named_format.format_id}"
        candidate_root = attempt_root / candidate_id
        legacy_result_path = candidate_root / "result.json"
        evaluation = _load_complete_evaluation(legacy_result_path)
        retries = self.plan.maximum_retries_per_candidate
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            if evaluation is not None:
                break
            attempt_dir = candidate_root / f"attempt-{attempt}"
            result_path = attempt_dir / "result.json"
            evaluation = _load_complete_evaluation(result_path)
            if evaluation is not None:
                break
            if result_path.exists():
                self.recorder.transition(
                    "candidate_attempt_preserved",
                    stage_id=stage.stage_id,
                    candidate_id=candidate_id,
                    attempt=attempt,
                    result_path=str(result_path),
                )
                continue
            self._wait_for_gpu(stage.stage_id, candidate_id)
            self.recorder.transition(
                "candidate_started",
                stage_id=stage.stage_id,
                candidate_id=candidate_id,
                attempt=attempt,
            )
            try:
                evaluation = self.evaluator.evaluate(
                    candidate_id=candidate_id,
                    assignments=assignments,
                    output_dir=attempt_dir,
                )
            except Exception as error:
                last_error = error
                self.recorder.transition(
                    "candidate_failed",
                    stage_id=stage.stage_id,
                    candidate_id=candidate_id,
                    attempt=attempt,
                    failure_type=type(error).__name__,
                    message=str(error),
                )
                if attempt >= retries:
                    raise
                continue
        if evaluation is None and last_error is not None:
            raise last_error
        if evaluation is None:
            raise RuntimeError("candidate evaluation ended without a result")
        gate = self.gate.evaluate(
            evaluation.metrics,
            self.plan.accepted_metrics,
            current_metrics,
        )
        packed = self._packed_bytes(assignments)
        outcome = ProgressivePolicyOutcome(
            candidate_id=candidate_id,
            format_id=named_format.format_id,
            metrics=evaluation.metrics,
            packed_bytes=packed,
            gate_decision=gate.decision,
        )
        record = {
            "candidate_id": candidate_id,
            "format_id": named_format.format_id,
            "weight_format_id": named_format.spec.format_id,
            "cpu_format_id": named_format.cpu_format_id,
            "assignments": [item.to_dict() for item in assignments],
            "metrics": evaluation.metrics,
            "metrics_path": str(evaluation.metrics_path),
            "metrics_sha256": evaluation.metrics_sha256,
            "packed_bytes": packed,
            "savings_from_locked_parent_bytes": self.base_packed_bytes - packed,
            "gate": {
                "decision": gate.decision,
                "worst_total_map50_delta": gate.worst_total_map50_delta,
                "worst_total_map50_95_delta": gate.worst_total_map50_95_delta,
                "worst_incremental_map50_delta": (gate.worst_incremental_map50_delta),
                "worst_incremental_map50_95_delta": (
                    gate.worst_incremental_map50_95_delta
                ),
                "total_deltas": gate.total_deltas,
                "incremental_deltas": gate.incremental_deltas,
            },
            "seconds": evaluation.seconds,
            "status": "completed",
        }
        self.recorder.transition(
            "candidate_completed",
            stage_id=stage.stage_id,
            candidate_id=candidate_id,
            decision=gate.decision,
            worst_map50=gate.worst_total_map50_delta,
            worst_map50_95=gate.worst_total_map50_95_delta,
            packed_bytes=packed,
        )
        return outcome, record

    @staticmethod
    def _outcome_from_record(
        record: Mapping[str, object],
    ) -> ProgressivePolicyOutcome:
        metrics = _mapping(record.get("metrics"), "completed candidate metrics")
        gate = _mapping(record.get("gate"), "completed candidate gate")
        missing = tuple(key for key in _PROGRESSIVE_METRIC_KEYS if key not in metrics)
        if missing or record.get("status") != "completed":
            raise ValueError("saved candidate record is incomplete")
        return ProgressivePolicyOutcome(
            candidate_id=str(record["candidate_id"]),
            format_id=str(record["format_id"]),
            metrics={key: float(metrics[key]) for key in _PROGRESSIVE_METRIC_KEYS},
            packed_bytes=int(record["packed_bytes"]),
            gate_decision=str(gate["decision"]),
        )

    @staticmethod
    def _maybe_add_recovery(
        *,
        record: Mapping[str, object],
        stage: ProgressiveStage,
        recovery_pool: list[object],
        recovery_ids: set[str],
    ) -> None:
        gate = _mapping(record.get("gate"), "recovery candidate gate")
        candidate_id = str(record["candidate_id"])
        if (
            gate.get("decision") == "recover"
            and int(record["savings_from_locked_parent_bytes"])
            >= stage.promotion_min_savings_bytes
            and candidate_id not in recovery_ids
        ):
            recovery_pool.append(dict(record))
            recovery_ids.add(candidate_id)

    def _write_short_qat_queue(self, state: dict[str, object]) -> Path:
        pool = state.get("recovery_pool")
        if not isinstance(pool, list):
            raise TypeError("queue recovery_pool must be a list")
        from .progressive_regate import (
            build_short_qat_queue_payload,
            select_recovery_candidates,
        )

        eligible = select_recovery_candidates(
            self.plan,
            tuple(dict(_mapping(item, "recovery candidate")) for item in pool),
        )
        payload = build_short_qat_queue_payload(self.plan, eligible)
        path = self.plan.run_root / "short-qat-queue.json"
        _atomic_json(path, payload)
        return path

    def run(self) -> dict[str, object]:
        state = self._load_state()
        if state.get("status") in {
            "ptq_complete_short_qat_ready",
            "complete_no_short_qat",
        }:
            return state
        current_assignments = self._assignments(state["locked_assignments"])
        current_metrics = {
            key: float(value)
            for key, value in _mapping(
                state["current_metrics"], "current metrics"
            ).items()
        }
        stages_state = _mapping(state["stages"], "queue stages")
        recovery_pool = state["recovery_pool"]
        if not isinstance(recovery_pool, list):
            raise TypeError("queue recovery_pool must be a list")
        recovery_ids = {
            str(_mapping(item, "saved recovery candidate")["candidate_id"])
            for item in recovery_pool
        }
        start_index = int(state["current_stage_index"])

        for index, stage in enumerate(
            self.plan.stages[start_index:], start=start_index
        ):
            stage_state = stages_state.get(stage.stage_id)
            if (
                isinstance(stage_state, dict)
                and stage_state.get("status") == "completed"
            ):
                raise RuntimeError("queue stage index lags a completed stage")
            self.recorder.transition(
                "stage_started",
                stage_index=index,
                stage_id=stage.stage_id,
                region=stage.region,
                path=stage.path,
            )
            expected_parent_assignments = [
                item.to_dict() for item in current_assignments
            ]
            if stage_state is None:
                stage_record: dict[str, object] = {
                    "status": "running",
                    "stage_index": index,
                    "stage_id": stage.stage_id,
                    "region": stage.region,
                    "path": stage.path,
                    "role": stage.role,
                    "parent_assignments": expected_parent_assignments,
                    "parent_metrics": dict(current_metrics),
                    "parent_packed_bytes": self._packed_bytes(current_assignments),
                    "candidates": {},
                    "pruned": [],
                }
                stages_state[stage.stage_id] = stage_record
                state["stages"] = stages_state
                self._save_state(state)
            else:
                stage_record = _mapping(stage_state, "running queue stage")
                expected = {
                    "status": "running",
                    "stage_index": index,
                    "stage_id": stage.stage_id,
                    "region": stage.region,
                    "path": stage.path,
                    "role": stage.role,
                    "parent_assignments": expected_parent_assignments,
                    "parent_metrics": dict(current_metrics),
                    "parent_packed_bytes": self._packed_bytes(current_assignments),
                }
                drift = {
                    key: {"expected": value, "actual": stage_record.get(key)}
                    for key, value in expected.items()
                    if stage_record.get(key) != value
                }
                if drift:
                    raise ValueError(
                        "running queue stage contract drifted: "
                        + json.dumps(drift, ensure_ascii=False, sort_keys=True)
                    )

            outcomes: list[ProgressivePolicyOutcome] = []
            candidate_records = _mapping(stage_record["candidates"], "stage candidates")
            raw_pruned = stage_record["pruned"]
            if not isinstance(raw_pruned, list):
                raise TypeError("stage pruned record must be a list")
            pruned_formats = {
                str(_mapping(item, "saved pruned candidate")["format_id"])
                for item in raw_pruned
            }
            controls: dict[str, str] = {}
            for named_format in stage.formats:
                candidate_id = f"{stage.stage_id}--{named_format.format_id}"
                saved = candidate_records.get(candidate_id)
                if saved is not None:
                    record = _mapping(saved, "saved completed candidate")
                    outcome = self._outcome_from_record(record)
                    if (
                        outcome.candidate_id != candidate_id
                        or outcome.format_id != named_format.format_id
                    ):
                        raise ValueError("saved candidate identity drifted")
                    outcomes.append(outcome)
                    if named_format.format_id in {"exact_w4", "fixed_sd4"}:
                        controls[named_format.format_id] = outcome.gate_decision
                    self._maybe_add_recovery(
                        record=record,
                        stage=stage,
                        recovery_pool=recovery_pool,
                        recovery_ids=recovery_ids,
                    )
                    continue
                if named_format.format_id in pruned_formats:
                    continue
                if (
                    self.plan.skip_ternary_when_both_4bit_controls_reject
                    and named_format.spec.bits == 2
                    and controls.get("exact_w4") == "reject"
                    and controls.get("fixed_sd4") == "reject"
                ):
                    raw_pruned.append(
                        {
                            "format_id": named_format.format_id,
                            "reason": "both_4bit_controls_rejected",
                        }
                    )
                    pruned_formats.add(named_format.format_id)
                    state["stages"] = stages_state
                    self._save_state(state)
                    self.recorder.transition(
                        "candidate_pruned",
                        stage_id=stage.stage_id,
                        format_id=named_format.format_id,
                        reason="both_4bit_controls_rejected",
                    )
                    continue
                outcome, record = self._candidate_result(
                    stage=stage,
                    named_format=named_format,
                    current_assignments=current_assignments,
                    current_metrics=current_metrics,
                    attempt_root=self.plan.run_root / "candidates",
                )
                outcomes.append(outcome)
                candidate_records[outcome.candidate_id] = record
                if named_format.format_id in {"exact_w4", "fixed_sd4"}:
                    controls[named_format.format_id] = outcome.gate_decision
                self._maybe_add_recovery(
                    record=record,
                    stage=stage,
                    recovery_pool=recovery_pool,
                    recovery_ids=recovery_ids,
                )
                state["recovery_pool"] = recovery_pool
                self._save_state(state)

            baseline = ProgressivePolicyOutcome(
                candidate_id=f"{stage.stage_id}--unchanged",
                format_id="w8_previous_locked_policy",
                metrics=dict(current_metrics),
                packed_bytes=self._packed_bytes(current_assignments),
                gate_decision="green",
            )
            selected = self.selector.select(
                baseline=baseline,
                candidates=tuple(outcomes),
                promotion_min_savings_bytes=stage.promotion_min_savings_bytes,
                promotion_allowed=stage.promotion_allowed,
            )
            if selected.candidate_id != baseline.candidate_id:
                chosen_record = _mapping(
                    candidate_records[selected.candidate_id],
                    "selected candidate",
                )
                selected_format = self.plan.format_catalog[selected.format_id]
                current_assignments = (
                    *current_assignments,
                    ProgressiveAssignment(
                        stage_id=stage.stage_id,
                        region=stage.region,
                        path=stage.path,
                        named_format=selected_format,
                    ),
                )
                current_metrics = dict(selected.metrics)
                metrics_source = str(chosen_record["metrics_path"])
            else:
                metrics_source = str(state["current_metrics_source"])
            stage_record.update(
                {
                    "status": "completed",
                    "selected_candidate_id": selected.candidate_id,
                    "selected_format_id": selected.format_id,
                    "selected_packed_bytes": selected.packed_bytes,
                    "promotion_allowed": stage.promotion_allowed,
                }
            )
            state.update(
                {
                    "current_stage_index": index + 1,
                    "current_metrics": current_metrics,
                    "current_metrics_source": metrics_source,
                    "current_packed_bytes": self._packed_bytes(current_assignments),
                    "locked_assignments": [
                        item.to_dict() for item in current_assignments
                    ],
                    "stages": stages_state,
                    "recovery_pool": recovery_pool,
                }
            )
            self._save_state(state)
            self.recorder.transition(
                "stage_locked",
                stage_index=index,
                stage_id=stage.stage_id,
                selected_candidate_id=selected.candidate_id,
                selected_format_id=selected.format_id,
                packed_bytes=selected.packed_bytes,
            )

        short_qat_queue = self._write_short_qat_queue(state)
        jobs = _mapping(
            json.loads(short_qat_queue.read_text(encoding="utf-8")),
            "short QAT queue",
        ).get("jobs")
        if not isinstance(jobs, list):
            raise TypeError("short QAT jobs must be a list")
        state.update(
            {
                "status": (
                    "ptq_complete_short_qat_ready" if jobs else "complete_no_short_qat"
                ),
                "short_qat_queue": str(short_qat_queue),
                "short_qat_jobs": len(jobs),
            }
        )
        self._save_state(state)
        self.recorder.transition(
            str(state["status"]),
            locked_assignments=len(current_assignments),
            packed_bytes=state["current_packed_bytes"],
            short_qat_jobs=len(jobs),
        )
        return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the persistent V19 progressive PTQ-to-short-QAT queue",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=(
            _PROJECT_ROOT / "configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml"
        ),
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--execute-reviewed-queue", action="store_true")
    args = parser.parse_args(argv)
    plan = ProgressiveQueuePlan.from_yaml(args.plan)
    if args.list_only:
        print(
            json.dumps(
                {
                    "queue_id": plan.queue_id,
                    "plan_sha256": plan.config_sha256,
                    "parent_id": plan.parent.parent_id,
                    "stages": [
                        {
                            "stage_id": stage.stage_id,
                            "region": stage.region,
                            "path": stage.path,
                            "formats": [item.format_id for item in stage.formats],
                        }
                        for stage in plan.stages
                    ],
                    "short_qat": {
                        "authorized": plan.short_qat.execution_authorized,
                        "epochs": plan.short_qat.epochs,
                        "patience": plan.short_qat.patience,
                    },
                    "formal_validation": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.execute_reviewed_queue:
        parser.error("execution requires --execute-reviewed-queue acknowledgement")
    if (
        not torch.cuda.is_available()
        or not 0 <= plan.device_index < torch.cuda.device_count()
    ):
        parser.error(f"CUDA device {plan.device_index} is unavailable")
    state = ProgressiveExperimentQueue(plan).run()
    print(json.dumps(state, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
