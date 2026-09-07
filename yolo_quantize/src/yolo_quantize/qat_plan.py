"""Strict, hash-pinned configuration contract for paired Full35 QAT pilots."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import yaml

from .full35_adapter import Full35ActivationPolicy
from .qat_weights import ProgressiveQuantizationSchedule
from .weight_quantization import (
    ExactTernaryWeightSpec,
    FilterwiseTWNWeightSpec,
    FixedSD4WeightSpec,
    PaperTWNWeightSpec,
    UniformWeightSpec,
    WeightRegionAssignment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QATArm = Literal["sham", "qat"]

_CANONICAL_COCO = Path("/home/uxin/yolo/coco2017.yaml")
_CANONICAL_REGISTRY = Path("/home/uxin/yolo/configs/datasets/bbat5-v1.yaml")
_CANONICAL_POSE_SEARCH = Path(
    "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose-search.yaml"
)
_LR_ROLES = (
    "backbone",
    "neck",
    "masf",
    "attention",
    "detect_head",
    "pose_head",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _verified_file(record: object, label: str) -> Path:
    payload = _mapping(record, label)
    if set(payload) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    configured = Path(str(payload["path"])).expanduser()
    path = (
        configured.resolve()
        if configured.is_absolute()
        else (PROJECT_ROOT / configured).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    expected = str(payload["sha256"])
    if actual != expected:
        raise ValueError(f"{label} SHA-256 drifted: {actual} != {expected}")
    return path


@dataclass(frozen=True)
class QATDataContract:
    coco: Path
    registry: Path
    pose_search: Path
    diagnostic_manifest: Path
    diagnostic_manifest_identity: str


@dataclass(frozen=True)
class QATOptimizerSpec:
    name: Literal["AdamW", "MuSGD"]
    weight_decay: float
    beta1: float
    beta2: float
    qparam_lr_ratio: float
    learning_rates: Mapping[str, float]


@dataclass(frozen=True)
class QATTrainingSpec:
    epochs: int
    seed: int
    patience: int
    warmup_epochs: int
    scale_only_epochs: int
    progressive_start_epoch: int
    progressive_full_epoch: int
    detect_logical_batch: int
    detect_microbatch: int
    pose_batch: int
    gradient_clip_norm: float
    added_noise: bool

    @property
    def progressive_schedule(self) -> ProgressiveQuantizationSchedule:
        return ProgressiveQuantizationSchedule(
            start_epoch=self.progressive_start_epoch,
            full_epoch=self.progressive_full_epoch,
        )


@dataclass(frozen=True)
class QATGateSpec:
    map50_max_drop: float
    map50_95_max_drop: float
    matched_sham_max_absolute_drift: float


@dataclass(frozen=True)
class QATMonitoringSpec:
    mode: Literal["low_token"]
    events: tuple[str, ...]
    full_artifact_logging: bool


@dataclass(frozen=True)
class QATWarmStartSpec:
    """Hash-pinned FP32-shadow fork; optimizer state is intentionally fresh."""

    parent_id: str
    selected_epoch: int
    parent_manifest: Path
    parent_manifest_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    state_key: Literal["ema_state"]
    reset_path_override_quantizers: bool
    load_optimizer_state: bool = False


@dataclass(frozen=True)
class Full35QATPlan:
    config_path: Path
    config_sha256: str
    plan_id: str
    candidate_id: str
    execution_authorization_id: str
    formal_validation: bool
    arms: tuple[QATArm, ...]
    activation: Full35ActivationPolicy
    parent_checkpoint: Path
    parent_checkpoint_sha256: str
    accepted_metrics: Path
    matched_metrics: Path
    joint_config: Path
    activation_recipe: Path
    weight_plan: Path
    warm_start: QATWarmStartSpec | None
    data: QATDataContract
    assignments: tuple[WeightRegionAssignment, ...]
    float_regions: tuple[str, ...]
    expected_master_catalog: Mapping[str, object]
    expected_deployment_catalog: Mapping[str, object]
    optimizer: QATOptimizerSpec
    training: QATTrainingSpec
    gates: QATGateSpec
    monitoring: QATMonitoringSpec
    run_root: Path
    external_control: Mapping[str, object] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> Full35QATPlan:
        config_path = Path(path).expanduser().resolve()
        payload = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "QAT plan",
        )
        if payload.get("schema_version") != 1:
            raise ValueError("QAT plan schema_version must be 1")
        if payload.get("execution_authorized") is not True:
            raise ValueError("QAT training is not authorized")
        if payload.get("formal_validation") is not False:
            raise ValueError("QAT pilot cannot use formal validation")
        authorization = _mapping(
            payload.get("execution_authorization"), "execution authorization"
        )
        authorization_id = str(authorization.get("authorization_id", "")).strip()
        if not authorization_id:
            raise ValueError("QAT execution authorization_id is empty")
        arms = tuple(str(value) for value in authorization.get("arms", ()))
        external_control: Mapping[str, object] | None = None
        if arms == ("qat",):
            raw_external = authorization.get("external_control")
            if not isinstance(raw_external, dict):
                raise ValueError(
                    "qat-only continuation requires external_control reference"
                )
            if set(raw_external) != {"mode", "plan", "completion", "metrics"}:
                raise ValueError("external_control fields differ from schema")
            if raw_external.get("mode") != "v35_external_sham_reference":
                raise ValueError("unsupported external_control mode")

            def verified_external(value: object, label: str) -> tuple[Path, str]:
                record = _mapping(value, label)
                path = _verified_file(record, label)
                digest = str(record["sha256"])
                return path, digest

            external_plan, external_plan_sha = verified_external(
                raw_external["plan"], "external control plan"
            )
            external_completion, external_completion_sha = verified_external(
                raw_external["completion"], "external control completion"
            )
            external_metrics, external_metrics_sha = verified_external(
                raw_external["metrics"], "external control metrics"
            )
            completion_payload = _mapping(
                json.loads(external_completion.read_text(encoding="utf-8")),
                "external control completion payload",
            )
            if (
                completion_payload.get("arm") != "sham"
                or completion_payload.get("formal_validation") is not False
                or completion_payload.get("plan_sha256") != external_plan_sha
            ):
                raise ValueError("external control completion lineage is invalid")
            external_control = MappingProxyType(
                {
                    "mode": str(raw_external["mode"]),
                    "plan": str(external_plan),
                    "plan_sha256": external_plan_sha,
                    "completion": str(external_completion),
                    "completion_sha256": external_completion_sha,
                    "metrics": str(external_metrics),
                    "metrics_sha256": external_metrics_sha,
                }
            )
        elif arms != ("sham", "qat"):
            raise ValueError("QAT pilot must declare matched sham then QAT arms")

        sources = _mapping(payload.get("sources"), "sources")
        parent_checkpoint = _verified_file(
            sources.get("parent_checkpoint"), "parent checkpoint"
        )
        accepted_metrics = _verified_file(
            sources.get("accepted_metrics"), "accepted metrics"
        )
        matched_metrics = _verified_file(
            sources.get("matched_metrics"), "matched metrics"
        )
        joint_config = _verified_file(sources.get("joint_config"), "joint config")
        activation_recipe = _verified_file(
            sources.get("activation_recipe"), "activation recipe"
        )
        weight_plan = _verified_file(sources.get("weight_plan"), "weight plan")
        candidate_evidence = _verified_file(
            sources.get("candidate_evidence"), "candidate evidence"
        )
        candidate_dual_regate = _verified_file(
            sources.get("candidate_dual_regate"), "candidate dual re-gate"
        )

        raw_warm_start = payload.get("warm_start")
        warm_start: QATWarmStartSpec | None = None
        if raw_warm_start is not None:
            warm = _mapping(raw_warm_start, "QAT warm start")
            if set(warm) != {
                "locked_parent",
                "checkpoint_role",
                "state_key",
                "reset_path_override_quantizers",
                "optimizer_state",
            }:
                raise ValueError(
                    "QAT warm-start keys differ from the reviewed contract"
                )
            parent_manifest = _verified_file(
                warm.get("locked_parent"), "warm-start locked parent"
            )
            from .progressive_preparation import LockedQATParentSpec

            parent = LockedQATParentSpec.from_yaml(parent_manifest)
            if (
                warm.get("checkpoint_role") != "full_resume"
                or warm.get("state_key") != "ema_state"
                or warm.get("reset_path_override_quantizers") is not True
                or warm.get("optimizer_state") != "fresh"
            ):
                raise ValueError("QAT warm-start policy is unsafe")
            warm_start = QATWarmStartSpec(
                parent_id=parent.parent_id,
                selected_epoch=parent.selected_epoch,
                parent_manifest=parent_manifest,
                parent_manifest_sha256=str(
                    _mapping(warm["locked_parent"], "warm-start locked parent")[
                        "sha256"
                    ]
                ),
                checkpoint=parent.full_resume_checkpoint,
                checkpoint_sha256=parent.full_resume_sha256,
                state_key="ema_state",
                reset_path_override_quantizers=True,
                load_optimizer_state=False,
            )

        raw_activation = _mapping(payload.get("activation"), "activation")
        activation = Full35ActivationPolicy(
            activation=str(raw_activation["name"]),
            bits=int(raw_activation["bits"]),
            signed_codes=bool(raw_activation.get("signed_codes", False)),
            region_assignments=tuple(
                (str(item["region"]), str(item["activation"]))
                for item in raw_activation.get("region_assignments", ())  # type: ignore[union-attr]
            ),
        )
        if activation.activation not in {"qsilu_pq", "hardswish", "poly_shift"}:
            raise ValueError("QAT plan activation must be an active non-SiLU parent")

        raw_data = _mapping(payload.get("data"), "data")
        coco = _verified_file(raw_data.get("coco"), "COCO dataset YAML")
        registry = _verified_file(raw_data.get("registry"), "BBAT5 registry")
        pose_search = _verified_file(
            raw_data.get("pose_search"), "BBAT5 pose-search YAML"
        )
        if (coco, registry, pose_search) != (
            _CANONICAL_COCO,
            _CANONICAL_REGISTRY,
            _CANONICAL_POSE_SEARCH,
        ):
            raise ValueError("QAT data paths differ from the canonical contracts")
        diagnostic = _mapping(
            raw_data.get("diagnostic_manifest"), "diagnostic manifest"
        )
        diagnostic_path = _verified_file(diagnostic, "diagnostic manifest")
        diagnostic_identity = str(raw_data.get("diagnostic_manifest_identity", ""))
        if len(diagnostic_identity) != 64:
            raise ValueError("diagnostic manifest identity must be SHA-256 length")
        data = QATDataContract(
            coco=coco,
            registry=registry,
            pose_search=pose_search,
            diagnostic_manifest=diagnostic_path,
            diagnostic_manifest_identity=diagnostic_identity,
        )

        raw_policy = _mapping(payload.get("weight_policy"), "weight policy")
        candidate_id = str(raw_policy.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ValueError("QAT weight policy candidate_id is empty")
        evidence_payload = _mapping(
            yaml.safe_load(candidate_evidence.read_text(encoding="utf-8")),
            "candidate evidence report",
        )
        evidence_results = _mapping(
            evidence_payload.get("results"), "candidate evidence results"
        )
        evidence_record = _mapping(
            evidence_results.get(candidate_id), f"candidate evidence {candidate_id}"
        )
        evidence_gate = _mapping(
            evidence_record.get("gate"), f"candidate evidence gate {candidate_id}"
        )
        if (
            evidence_payload.get("status") != "completed"
            or evidence_record.get("status") != "completed"
            or evidence_gate.get("decision") not in {"green", "recover"}
        ):
            raise ValueError("QAT candidate is not a completed recoverable PTQ result")
        dual_payload = _mapping(
            yaml.safe_load(candidate_dual_regate.read_text(encoding="utf-8")),
            "candidate dual re-gate report",
        )
        dual_source = _mapping(
            dual_payload.get("source"), "candidate dual re-gate source"
        )
        if dual_source.get("report_sha256") != _sha256(candidate_evidence):
            raise ValueError(
                "QAT candidate dual re-gate does not pin candidate evidence"
            )
        dual_candidates = _mapping(
            dual_payload.get("candidates"), "candidate dual re-gate candidates"
        )
        dual_record = _mapping(
            dual_candidates.get(candidate_id),
            f"candidate dual re-gate {candidate_id}",
        )
        if dual_payload.get("status") != "completed" or dual_record.get(
            "decision"
        ) not in {"green", "recover"}:
            raise ValueError("QAT candidate did not pass the dual recovery floor")
        raw_float_regions = raw_policy.get("float_regions", ())
        if not isinstance(raw_float_regions, (tuple, list)):
            raise TypeError("QAT float_regions must be a sequence")
        float_regions = tuple(str(value) for value in raw_float_regions)
        if any(not value.strip() for value in float_regions) or len(
            set(float_regions)
        ) != len(float_regions):
            raise ValueError("QAT float_regions must be non-empty unique names")
        raw_assignments = raw_policy.get("assignments")
        if not isinstance(raw_assignments, list) or not raw_assignments:
            raise ValueError("QAT weight policy requires assignments")
        assignments: list[WeightRegionAssignment] = []
        for item in raw_assignments:
            entry = _mapping(item, "weight assignment")
            encoded = _mapping(entry.get("format"), "weight format")
            family = str(encoded.get("family"))
            if family == "uniform":
                spec = UniformWeightSpec(
                    bits=int(encoded["bits"]),
                    scale_method=str(
                        encoded.get("scale_method", "optimal_scaled_codebook")
                    ),  # type: ignore[arg-type]
                )
            elif family == "ls_sd4":
                spec = FixedSD4WeightSpec(
                    scale_method=str(
                        encoded.get("scale_method", "optimal_scaled_codebook")
                    ),  # type: ignore[arg-type]
                )
            elif family == "paper_twn":
                spec = PaperTWNWeightSpec(
                    threshold_multiplier=float(encoded.get("threshold_multiplier", 0.7))
                )
            elif family == "exact_scaled_ternary":
                spec = ExactTernaryWeightSpec(
                    scale_method=str(
                        encoded.get("scale_method", "optimal_scaled_codebook")
                    )
                )
            elif family == "twn_filterwise":
                spec = FilterwiseTWNWeightSpec(
                    threshold_multiplier=float(
                        encoded.get("threshold_multiplier", 0.75)
                    )
                )
            else:
                raise ValueError(f"unsupported QAT weight family: {family}")
            assignments.append(
                WeightRegionAssignment(
                    region=str(entry["region"]),
                    paths=tuple(str(value) for value in entry.get("paths", ())),
                    spec=spec,
                )
            )
        inventory = _mapping(payload.get("graph_inventory"), "graph inventory")
        expected_master = MappingProxyType(
            dict(_mapping(inventory.get("master"), "master catalog"))
        )
        expected_deployment = MappingProxyType(
            dict(_mapping(inventory.get("deployment"), "deployment catalog"))
        )
        if set(expected_master) != {"totals", "deployment_regions"} or set(
            expected_deployment
        ) != {"totals", "deployment_regions"}:
            raise ValueError("QAT graph catalogs require totals and deployment_regions")
        master_regions = tuple(
            _mapping(
                expected_master["deployment_regions"],
                "master deployment regions",
            )
        )
        deployment_regions = tuple(
            _mapping(
                expected_deployment["deployment_regions"],
                "deployment regions",
            )
        )
        defaults = tuple(
            assignment for assignment in assignments if not assignment.paths
        )
        routed = tuple(assignment for assignment in assignments if assignment.paths)
        if assignments != [*defaults, *routed]:
            raise ValueError("QAT path overrides must follow all region defaults")
        default_regions = tuple(assignment.region for assignment in defaults)
        if len(set(default_regions)) != len(default_regions):
            raise ValueError("QAT pilot region defaults must be unique")
        unknown_float_regions = tuple(
            region for region in float_regions if region not in master_regions
        )
        if unknown_float_regions:
            raise ValueError(
                "QAT float_regions reference unknown deployment regions: "
                f"{unknown_float_regions}"
            )
        expected_float_regions = tuple(
            region for region in master_regions if region in set(float_regions)
        )
        expected_default_regions = tuple(
            region for region in master_regions if region not in set(float_regions)
        )
        if (
            master_regions != deployment_regions
            or float_regions != expected_float_regions
            or default_regions != expected_default_regions
        ):
            raise ValueError(
                "QAT weight assignments must exactly cover deployment regions in order"
            )
        unknown_routed_regions = tuple(
            assignment.region
            for assignment in routed
            if assignment.region not in master_regions
        )
        if unknown_routed_regions:
            raise ValueError(
                "QAT path overrides reference unknown deployment regions: "
                f"{unknown_routed_regions}"
            )
        routed_paths = tuple(path for assignment in routed for path in assignment.paths)
        if len(set(routed_paths)) != len(routed_paths):
            raise ValueError("QAT path override sites must be globally unique")
        raw_optimizer = _mapping(payload.get("optimizer"), "optimizer")
        name = str(raw_optimizer.get("name"))
        if name not in {"AdamW", "MuSGD"}:
            raise ValueError("QAT optimizer must be AdamW or MuSGD")
        raw_rates = _mapping(raw_optimizer.get("learning_rates"), "learning rates")
        rates = {role: float(raw_rates[role]) for role in _LR_ROLES}
        if set(raw_rates) != set(_LR_ROLES) or any(
            value <= 0 for value in rates.values()
        ):
            raise ValueError("QAT learning-rate roles must be exact and positive")
        optimizer = QATOptimizerSpec(
            name=name,  # type: ignore[arg-type]
            weight_decay=float(raw_optimizer["weight_decay"]),
            beta1=float(raw_optimizer["beta1"]),
            beta2=float(raw_optimizer["beta2"]),
            qparam_lr_ratio=float(raw_optimizer["qparam_lr_ratio"]),
            learning_rates=MappingProxyType(rates),
        )
        finite_optimizer = (
            optimizer.weight_decay,
            optimizer.beta1,
            optimizer.beta2,
            optimizer.qparam_lr_ratio,
            *optimizer.learning_rates.values(),
        )
        if not all(math.isfinite(value) for value in finite_optimizer):
            raise ValueError("QAT optimizer values must be finite")
        if optimizer.weight_decay < 0 or optimizer.qparam_lr_ratio <= 0:
            raise ValueError("QAT decay must be non-negative and qparam ratio positive")

        raw_training = _mapping(payload.get("training"), "training")
        training = QATTrainingSpec(
            epochs=int(raw_training["epochs"]),
            seed=int(raw_training["seed"]),
            patience=int(raw_training["patience"]),
            warmup_epochs=int(raw_training["warmup_epochs"]),
            scale_only_epochs=int(raw_training["scale_only_epochs"]),
            progressive_start_epoch=int(raw_training["progressive_start_epoch"]),
            progressive_full_epoch=int(raw_training["progressive_full_epoch"]),
            detect_logical_batch=int(raw_training["detect_logical_batch"]),
            detect_microbatch=int(raw_training["detect_microbatch"]),
            pose_batch=int(raw_training["pose_batch"]),
            gradient_clip_norm=float(raw_training["gradient_clip_norm"]),
            added_noise=bool(raw_training["added_noise"]),
        )
        _ = training.progressive_schedule
        if (
            min(
                training.epochs,
                training.warmup_epochs,
                training.detect_logical_batch,
                training.detect_microbatch,
                training.pose_batch,
            )
            < 1
        ):
            raise ValueError("QAT epochs, warmup and batches must be positive")
        if training.detect_logical_batch % training.detect_microbatch:
            raise ValueError("Detect physical microbatch must divide logical batch")
        if training.scale_only_epochs != training.progressive_full_epoch:
            raise ValueError("first joint epoch must begin at full weight fake quant")
        if training.scale_only_epochs >= training.epochs or training.added_noise:
            raise ValueError("QAT pilot requires joint epochs and no added noise")

        raw_gates = _mapping(payload.get("gates"), "gates")
        gates = QATGateSpec(
            map50_max_drop=float(raw_gates["map50_max_drop"]),
            map50_95_max_drop=float(raw_gates["map50_95_max_drop"]),
            matched_sham_max_absolute_drift=float(
                raw_gates["matched_sham_max_absolute_drift"]
            ),
        )
        if gates.map50_max_drop != 0.015 or gates.map50_95_max_drop != 0.04:
            raise ValueError("active QAT pilot must use mAP50 0.015 and mAP50-95 0.04")
        raw_monitoring = _mapping(payload.get("monitoring"), "monitoring")
        monitoring = QATMonitoringSpec(
            mode=str(raw_monitoring["mode"]),  # type: ignore[arg-type]
            events=tuple(str(value) for value in raw_monitoring["events"]),  # type: ignore[index]
            full_artifact_logging=bool(raw_monitoring["full_artifact_logging"]),
        )
        if monitoring.mode != "low_token" or not monitoring.full_artifact_logging:
            raise ValueError("QAT monitoring must be low-token with complete artifacts")
        run_root = Path(str(payload["run_root"])).expanduser()
        run_root = (
            run_root.resolve()
            if run_root.is_absolute()
            else (PROJECT_ROOT / run_root).resolve()
        )
        if PROJECT_ROOT not in run_root.parents:
            raise ValueError("QAT run root must remain inside yolo_quantize")
        plan_id = str(payload.get("plan_id", "")).strip()
        if not plan_id:
            raise ValueError("QAT plan_id is empty")
        return cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            plan_id=plan_id,
            candidate_id=candidate_id,
            execution_authorization_id=authorization_id,
            formal_validation=False,
            arms=arms,  # type: ignore[arg-type]
            activation=activation,
            parent_checkpoint=parent_checkpoint,
            parent_checkpoint_sha256=str(
                _mapping(sources["parent_checkpoint"], "parent checkpoint")["sha256"]
            ),
            accepted_metrics=accepted_metrics,
            matched_metrics=matched_metrics,
            joint_config=joint_config,
            activation_recipe=activation_recipe,
            weight_plan=weight_plan,
            warm_start=warm_start,
            data=data,
            assignments=tuple(assignments),
            float_regions=float_regions,
            expected_master_catalog=expected_master,
            expected_deployment_catalog=expected_deployment,
            optimizer=optimizer,
            training=training,
            gates=gates,
            monitoring=monitoring,
            run_root=run_root,
            external_control=external_control,
        )


__all__ = (
    "Full35QATPlan",
    "QATArm",
    "QATDataContract",
    "QATGateSpec",
    "QATMonitoringSpec",
    "QATOptimizerSpec",
    "QATTrainingSpec",
    "QATWarmStartSpec",
)
