"""Reviewed Full35 PTQ search-validation plan and runner."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .activation_smoke import _atomic_json, _letterbox
from .full35_adapter import Full35ActivationAdapter, Full35ActivationPolicy
from .metric_gate import (
    FULL35_METRIC_KEYS,
    Full35MetricCandidate,
    Full35MetricGate,
    Full35MetricGateSpec,
    Full35MetricSnapshot,
)
from .search_data import PreparedBBAT5SearchView, prepare_bbat5_search_view
from .validation_source import Full35DeploymentValidationSource
from .weight_quantization import (
    Full35WeightRegionCatalog,
    UniformWeightSpec,
    WeightQuantizationAdapter,
)
from .weight_sensitivity import WeightSensitivityCell, WeightSensitivityStudy
from .weight_views import Full35WeightViewAdapter

DEFAULT_OUTPUT = Path("artifacts/reports/v4-qsilu-backbone-early-w8-search-v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"search validation {key} must be a mapping")
    return value


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (config_path.parents[2] / path).resolve()
    )


def _verified_path(
    config_path: Path,
    payload: dict[str, Any],
    *,
    label: str,
) -> Path:
    if set(payload) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = _resolve(config_path, payload["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(payload["sha256"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 drifted: {actual} != {expected}")
    return path


@dataclass(frozen=True)
class Full35SearchValidationPlan:
    """One immutable accepted/matched/candidate search comparison."""

    config_path: Path
    config_sha256: str
    plan_id: str
    execution_authorization_id: str
    metric_contract_id: str
    weight_study: WeightSensitivityStudy
    cell: WeightSensitivityCell
    diagnostic_report: Path
    accepted_policy_id: str
    accepted_checkpoint: Path
    accepted_checkpoint_sha256: str
    matched_policy_id: str
    coco_yaml: Path
    pose_search_yaml: Path
    bbat5_registry: Path
    runtime_view: Path
    detect_batch_size: int
    pose_batch_size: int
    detect_workers: int
    pose_workers: int
    image_size: int
    plots: bool
    save_coco_json: bool
    gate_spec: Full35MetricGateSpec
    formal_training: bool
    formal_validation: bool

    @classmethod
    def from_yaml(cls, path: str | Path) -> Full35SearchValidationPlan:
        config_path = Path(path).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("search validation plan must be a mapping")
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported search validation plan schema")
        if payload.get("execution_authorized") is not True:
            raise ValueError("search validation execution is not authorized")
        if payload.get("formal_training") is not False:
            raise ValueError("search validation cannot authorize training")
        if payload.get("formal_validation") is not False:
            raise ValueError("search validation cannot use formal BBAT5 validation")

        authorization = _mapping(payload, "execution_authorization")
        authorization_id = str(authorization.get("authorization_id", "")).strip()
        if not authorization_id:
            raise ValueError("search validation authorization_id is empty")
        if authorization.get("roles") != ["accepted", "matched", "candidate"]:
            raise ValueError(
                "search validation must authorize accepted, matched and candidate"
            )

        upstream = _mapping(payload, "upstream")
        weight_plan = _mapping(upstream, "weight_plan")
        weight_plan_path = _verified_path(
            config_path,
            weight_plan,
            label="upstream weight plan",
        )
        study = WeightSensitivityStudy.from_yaml(weight_plan_path)
        candidate = _mapping(payload, "candidate")
        cells = study.cells(
            activations=(str(candidate["activation"]),),
            regions=(str(candidate["region"]),),
            bits=(int(candidate["weight_bits"]),),
            scale_method=str(candidate["scale_method"]),  # type: ignore[arg-type]
        )
        if len(cells) != 1 or cells[0].cell_id != candidate.get("cell_id"):
            raise ValueError("search candidate does not resolve to one reviewed cell")
        study.require_execution_authorized(cells)
        cell = cells[0]

        diagnostic_report = _verified_path(
            config_path,
            _mapping(upstream, "diagnostic_report"),
            label="upstream diagnostic report",
        )
        diagnostic = json.loads(diagnostic_report.read_text(encoding="utf-8"))
        result = diagnostic.get("results", {}).get(cell.cell_id)
        diagnostic_contract = diagnostic.get("contract", {})
        if (
            diagnostic.get("status") != "diagnostic_completed"
            or not isinstance(result, dict)
            or result.get("status") != "diagnostic_pass"
            or result.get("selection_claim") is not False
            or diagnostic_contract.get("plan_sha256") != study.config_sha256
        ):
            raise ValueError(
                "upstream diagnostic did not pass as non-selection evidence"
            )

        accepted = _mapping(payload, "accepted")
        if (
            accepted.get("activation") != "silu"
            or accepted.get("activation_output_quantization") != "disabled"
        ):
            raise ValueError("accepted role must be unquantized SiLU")
        accepted_policy_id = str(accepted.get("policy_id", "")).strip()
        if not accepted_policy_id:
            raise ValueError("accepted policy_id is empty")
        accepted_record = _mapping(accepted, "checkpoint")
        accepted_checkpoint = _verified_path(
            config_path,
            accepted_record,
            label="accepted checkpoint",
        )

        matched = _mapping(payload, "matched")
        matched_policy_id = str(matched.get("policy_id", "")).strip()
        if (
            matched_policy_id != cell.parent.policy_id
            or matched.get("activation_output_quantization") != "calibrated_lsq_plus_a8"
        ):
            raise ValueError("matched role must equal the candidate A8 parent policy")
        if (
            candidate.get("granularity") != "per_output_channel"
            or candidate.get("rounding") != "nearest_even"
        ):
            raise ValueError("candidate weight quantization contract changed")

        datasets = _mapping(payload, "datasets")
        if datasets.get("bbat5_dataset_id") != "bbat5-v1":
            raise ValueError("search validation must use BBAT5 v1")
        if datasets.get("assignment_changed") is not False:
            raise ValueError("search validation may not change BBAT5 assignment")
        coco_yaml = _verified_path(
            config_path,
            _mapping(datasets, "coco"),
            label="COCO YAML",
        )
        pose_search_yaml = _verified_path(
            config_path,
            _mapping(datasets, "bbat5_pose_search"),
            label="BBAT5 pose search YAML",
        )
        bbat5_registry = _verified_path(
            config_path,
            _mapping(datasets, "bbat5_registry"),
            label="BBAT5 registry",
        )
        runtime_view = _resolve(config_path, datasets["runtime_view"])

        validation = _mapping(payload, "validation")
        if validation.get("backend") != "bittrue":
            raise ValueError("search validation backend must be bittrue")
        image_size = int(validation.get("image_size", 0))
        if image_size < 32 or image_size % 32:
            raise ValueError("search validation image_size must be a multiple of 32")
        if validation.get("plots") is not False:
            raise ValueError("search validation plots must remain disabled")
        if validation.get("save_coco_json") is not False:
            raise ValueError("search validation COCO JSON export must remain disabled")
        batch = _mapping(validation, "batch")
        workers = _mapping(validation, "workers")
        detect_batch = int(batch["detect"])
        pose_batch = int(batch["pose"])
        detect_workers = int(workers["detect"])
        pose_workers = int(workers["pose"])
        if min(detect_batch, pose_batch) < 1 or min(detect_workers, pose_workers) < 0:
            raise ValueError("search validation batch/workers are invalid")

        gates = _mapping(payload, "gates")
        if gates.get("all_eight_metrics_required") is not True:
            raise ValueError("search gate must require all eight Full35 metrics")
        gate_spec = Full35MetricGateSpec(
            metric_family=str(gates.get("metric_family", "map50_95")),
            total_max_drop=float(gates["total_max_drop"]),
            w8_incremental_max_drop=float(gates["w8_incremental_max_drop"]),
            sham_max_absolute_drift=float(gates["sham_max_absolute_drift"]),
            recovery_floor=float(gates["recovery_floor"]),
        )
        metric_contract_id = str(payload.get("metric_contract_id", "")).strip()
        if not metric_contract_id:
            raise ValueError("search metric_contract_id is empty")
        plan_id = str(payload.get("plan_id", "")).strip()
        if not plan_id:
            raise ValueError("search plan_id is empty")
        return cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            plan_id=plan_id,
            execution_authorization_id=authorization_id,
            metric_contract_id=metric_contract_id,
            weight_study=study,
            cell=cell,
            diagnostic_report=diagnostic_report,
            accepted_policy_id=accepted_policy_id,
            accepted_checkpoint=accepted_checkpoint,
            accepted_checkpoint_sha256=str(accepted_record["sha256"]),
            matched_policy_id=matched_policy_id,
            coco_yaml=coco_yaml,
            pose_search_yaml=pose_search_yaml,
            bbat5_registry=bbat5_registry,
            runtime_view=runtime_view,
            detect_batch_size=detect_batch,
            pose_batch_size=pose_batch,
            detect_workers=detect_workers,
            pose_workers=pose_workers,
            image_size=image_size,
            plots=False,
            save_coco_json=False,
            gate_spec=gate_spec,
            formal_training=False,
            formal_validation=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan": str(self.config_path),
            "plan_sha256": self.config_sha256,
            "execution_authorization_id": self.execution_authorization_id,
            "metric_contract_id": self.metric_contract_id,
            "cell": self.cell.to_dict(),
            "diagnostic_report": str(self.diagnostic_report),
            "accepted_policy_id": self.accepted_policy_id,
            "accepted_checkpoint": str(self.accepted_checkpoint),
            "accepted_checkpoint_sha256": self.accepted_checkpoint_sha256,
            "matched_policy_id": self.matched_policy_id,
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
            "plots": self.plots,
            "save_coco_json": self.save_coco_json,
            "gates": self.gate_spec.to_dict(),
            "formal_training": self.formal_training,
            "formal_validation": self.formal_validation,
        }


def _full35_metrics(
    values: Mapping[str, Any],
    metric_keys: tuple[str, ...] = FULL35_METRIC_KEYS,
) -> dict[str, float]:
    missing = tuple(key for key in metric_keys if key not in values)
    if missing:
        raise RuntimeError(
            "official validation did not return all eight metrics: "
            + ", ".join(missing)
        )
    return {key: float(values[key]) for key in metric_keys}


def _completed_metrics(
    roles: Mapping[str, Any],
    role: str,
    metric_keys: tuple[str, ...] = FULL35_METRIC_KEYS,
) -> dict[str, float]:
    record = roles.get(role)
    if not isinstance(record, dict) or record.get("status") != "completed":
        raise RuntimeError(f"search validation role is incomplete: {role}")
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError(f"search validation role has no metrics: {role}")
    return _full35_metrics(metrics, metric_keys)


def _prepare_contract(
    plan: Full35SearchValidationPlan,
    *,
    device_index: int,
) -> tuple[PreparedBBAT5SearchView, dict[str, Any]]:
    prepared = prepare_bbat5_search_view(
        plan.pose_search_yaml,
        plan.runtime_view,
    )
    validation_source = Path(
        "/home/uxin/yolo/yolo_combine/final/full35/code/project/"
        "src/yolo_combine/validation.py"
    ).resolve()
    if not validation_source.is_file():
        raise FileNotFoundError(validation_source)
    contract = {
        "schema_version": 1,
        "kind": "full35_ptq_eight_metric_search_validation",
        "reviewed_plan": plan.to_dict(),
        "runtime_view": {
            "root": str(prepared.root),
            "yaml": str(prepared.yaml),
            "yaml_sha256": _sha256(prepared.yaml),
            "manifest": str(prepared.manifest),
            "manifest_sha256": _sha256(prepared.manifest),
            "train_images": prepared.train_images,
            "val_images": prepared.val_images,
            "assignment_changed": False,
            "storage": "symlink-only-runtime-view",
        },
        "calibration_manifest": {
            "path": str(plan.weight_study.diagnostic_manifest),
            "sha256": plan.weight_study.diagnostic_manifest_sha256,
            "samples_per_task": 32,
            "split": "canonical_train_exemplars_only",
        },
        "validator": {
            "source": str(validation_source),
            "source_sha256": _sha256(validation_source),
            "backend": "bittrue",
            "split": "val",
            "rect": True,
            "augmentation": False,
        },
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "device_index": device_index,
            "device_name": torch.cuda.get_device_name(device_index),
        },
        "formal_training": False,
        "formal_validation": False,
        "selection_claim": False,
    }
    return prepared, contract


def _build_deployment_policy(
    *,
    activation: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    expected_catalog: Mapping[str, object],
    activation_region_assignments: tuple[tuple[str, str], ...] = (),
) -> tuple[Any, Full35DeploymentValidationSource, Any, dict[str, Any]]:
    started = time.perf_counter()
    built = Full35ActivationAdapter().build(
        Full35ActivationPolicy(
            activation=activation,
            bits=8,
            region_assignments=activation_region_assignments,
        ),
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    views = Full35WeightViewAdapter().build(built.model)
    if views.manifest.deployment_catalog != expected_catalog:
        raise RuntimeError("deployment weight catalog differs from reviewed plan")
    applied = built.applied.rebind(views.deployment)
    model = applied.model.eval()
    source = Full35DeploymentValidationSource(built.source, applied)
    evidence = {
        "activation": activation,
        "activation_region_assignments": [
            {"region": region, "activation": value}
            for region, value in activation_region_assignments
        ],
        "activation_policy_id": built.policy.policy_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_state_source": built.loaded_checkpoint.state_source,
        "activation_quantizers": applied.quantizer_count,
        "training_only_unquantized_sites": len(built.training_only_paths),
        "protected_modules": built.protected_after,
        "weight_view_parity": views.manifest.to_dict(),
        "build_seconds": time.perf_counter() - started,
    }
    del views, built
    gc.collect()
    return model, source, applied, evidence


def _calibrate_activation_a8(
    *,
    plan: Full35SearchValidationPlan,
    model: Any,
    applied: Any,
    device_index: int,
) -> dict[str, Any]:
    manifest = plan.weight_study.load_diagnostic_manifest(verify_files=True)
    paths = {
        "detect": manifest.paths("calibration", "coco_detect"),
        "pose": manifest.paths("calibration", "bbat_pose"),
    }
    if applied.mode != "observe":
        raise RuntimeError(
            f"activation calibration must begin in observe mode: {applied.mode}"
        )
    model.cuda(device_index).eval()
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    with torch.inference_mode():
        for task, task_paths in paths.items():
            for path in task_paths:
                image = _letterbox(path, plan.image_size).cuda(device_index)
                model(image, task=task)
                del image
    torch.cuda.synchronize(device_index)
    ranges = applied.observer_ranges()
    invalid = tuple(
        path
        for path, observed in ranges.items()
        if observed is None or observed[1] <= observed[0]
    )
    if invalid:
        raise RuntimeError(
            "activation observers are missing or degenerate: " + ", ".join(invalid)
        )
    applied.freeze_observers()
    return {
        "manifest": str(manifest.source_path),
        "manifest_sha256": manifest.sha256,
        "samples": {task: len(task_paths) for task, task_paths in paths.items()},
        "quantizers": applied.quantizer_count,
        "valid_observers": len(ranges),
        "invalid_observers": [],
        "observed_minimum": min(
            observed[0] for observed in ranges.values() if observed is not None
        ),
        "observed_maximum": max(
            observed[1] for observed in ranges.values() if observed is not None
        ),
        "seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device_index) / 2**20,
    }


def _official_validation_module() -> Any:
    module = importlib.import_module("yolo_combine.validation")
    expected = Path(
        "/home/uxin/yolo/yolo_combine/final/full35/code/project/"
        "src/yolo_combine/validation.py"
    ).resolve()
    actual = Path(module.__file__).resolve()
    if actual != expected:
        raise RuntimeError(f"yolo_combine.validation is shadowed by {actual}")
    return module


def _validate_role(
    *,
    role: str,
    model: Any,
    source: Full35DeploymentValidationSource,
    plan: Full35SearchValidationPlan,
    pose_yaml: Path,
    output_root: Path,
    device_index: int,
) -> dict[str, Any]:
    validation = _official_validation_module()
    settings = validation.ValidationSettings(
        imgsz=plan.image_size,
        detect_batch_size=plan.detect_batch_size,
        pose_batch_size=plan.pose_batch_size,
        detect_workers=plan.detect_workers,
        pose_workers=plan.pose_workers,
        device=str(device_index),
        plots=plan.plots,
        save_coco_json=plan.save_coco_json,
    )
    validator = validation.JointValidator(
        source,
        detect_data_yaml=plan.coco_yaml,
        pose_data_yaml=pose_yaml,
        output_root=output_root / role,
        settings=settings,
    )
    print(f"[{role}] official COCO + BBAT5 search validation", flush=True)
    started = time.perf_counter()
    result = validator.validate(model, epoch=0, kind="bittrue")
    metrics = _full35_metrics(result.metrics, plan.gate_spec.metric_keys)
    metrics_path = result.output_dir / "metrics.json"
    record = {
        "status": "completed",
        "role": role,
        "metrics": metrics,
        "all_metric_count": len(result.metrics),
        "output_dir": str(result.output_dir),
        "metrics_report": str(metrics_path),
        "metrics_report_sha256": _sha256(metrics_path),
        "materialization": {
            "detect_complete": result.materialized.detect_report.complete,
            "detect_compatible_tensors": (
                result.materialized.detect_report.compatible_tensors
            ),
            "pose_complete": result.materialized.pose_report.complete,
            "pose_compatible_tensors": (
                result.materialized.pose_report.compatible_tensors
            ),
        },
        "seconds": time.perf_counter() - started,
    }
    print(f"[{role}] completed: {json.dumps(metrics, sort_keys=True)}", flush=True)
    del result, validator
    gc.collect()
    torch.cuda.empty_cache()
    return record


def run_search_validation(
    *,
    plan: Full35SearchValidationPlan,
    output: Path,
    device_index: int,
    resume: bool,
) -> int:
    """Run exactly one reviewed accepted/matched/candidate validation triplet."""

    prepared, contract = _prepare_contract(plan, device_index=device_index)
    output_path = output.expanduser().resolve()
    validation_root = (
        Path(__file__).resolve().parents[2]
        / "artifacts/runs"
        / plan.plan_id
        / "validation"
    )
    roles: dict[str, Any] = {}
    if not output_path.exists() and validation_root.exists():
        raise FileExistsError(
            "validation artifacts already exist without the reviewed output report; "
            f"refusing a fresh run that could overwrite evidence: {validation_root}"
        )
    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"output already exists; use --resume or a new path: {output_path}"
            )
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("contract") != contract:
            raise RuntimeError(
                "existing search-validation contract differs; refusing resume"
            )
        raw_roles = existing.get("roles", {})
        if isinstance(raw_roles, dict):
            roles.update(raw_roles)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "full35_ptq_eight_metric_search_validation",
        "status": "running",
        "contract": contract,
        "roles": roles,
        "gate": None,
        "formal_training": False,
        "formal_validation": False,
        "selection_claim": False,
        "promotion_authorized": False,
    }
    _atomic_json(output_path, payload)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(device_index)
    try:
        if roles.get("accepted", {}).get("status") != "completed":
            payload["current_role"] = "accepted"
            _atomic_json(output_path, payload)
            model, source, applied, build = _build_deployment_policy(
                activation="silu",
                checkpoint=plan.accepted_checkpoint,
                checkpoint_sha256=plan.accepted_checkpoint_sha256,
                expected_catalog=plan.weight_study.expected_catalog,
            )
            applied.disable_quantization()
            record = _validate_role(
                role="accepted",
                model=model,
                source=source,
                plan=plan,
                pose_yaml=prepared.yaml,
                output_root=validation_root,
                device_index=device_index,
            )
            record.update(
                {
                    "policy_id": plan.accepted_policy_id,
                    "activation_output_quantization": "disabled",
                    "build": build,
                }
            )
            roles["accepted"] = record
            payload["roles"] = roles
            _atomic_json(output_path, payload)
            del model, source, applied, build
            gc.collect()
            torch.cuda.empty_cache()

        matched_pending = roles.get("matched", {}).get("status") != "completed"
        candidate_pending = roles.get("candidate", {}).get("status") != "completed"
        if matched_pending or candidate_pending:
            model, source, applied, build = _build_deployment_policy(
                activation=plan.cell.parent.activation,
                checkpoint=plan.cell.parent.checkpoint,
                checkpoint_sha256=plan.cell.parent.sha256,
                expected_catalog=plan.weight_study.expected_catalog,
            )
            calibration = _calibrate_activation_a8(
                plan=plan,
                model=model,
                applied=applied,
                device_index=device_index,
            )
            catalog = Full35WeightRegionCatalog.inspect(model)
            if catalog.summary() != plan.weight_study.expected_catalog:
                raise RuntimeError("calibrated deployment catalog differs from plan")

            if matched_pending:
                payload["current_role"] = "matched"
                _atomic_json(output_path, payload)
                record = _validate_role(
                    role="matched",
                    model=model,
                    source=source,
                    plan=plan,
                    pose_yaml=prepared.yaml,
                    output_root=validation_root,
                    device_index=device_index,
                )
                record.update(
                    {
                        "policy_id": plan.matched_policy_id,
                        "activation_output_quantization": "calibrated_lsq_plus_a8",
                        "build": build,
                        "calibration": calibration,
                    }
                )
                roles["matched"] = record
                payload["roles"] = roles
                _atomic_json(output_path, payload)

            if candidate_pending:
                payload["current_role"] = "candidate"
                _atomic_json(output_path, payload)
                with WeightQuantizationAdapter().quantized(
                    model,
                    catalog=catalog,
                    region=plan.cell.region,
                    spec=UniformWeightSpec(
                        bits=plan.cell.bits,
                        scale_method=plan.cell.scale_method,
                    ),
                ) as quantized:
                    record = _validate_role(
                        role="candidate",
                        model=model,
                        source=source,
                        plan=plan,
                        pose_yaml=prepared.yaml,
                        output_root=validation_root,
                        device_index=device_index,
                    )
                    record.update(
                        {
                            "policy_id": plan.matched_policy_id,
                            "cell": plan.cell.to_dict(),
                            "activation_output_quantization": (
                                "calibrated_lsq_plus_a8"
                            ),
                            "weight_quantization": {
                                "format_id": quantized.format_id,
                                "region": quantized.region,
                                "quantized_modules": quantized.quantized_modules,
                                "weight_elements": quantized.weight_elements,
                                "numeric": quantized.numeric,
                                "sites": list(quantized.site_metrics),
                            },
                            "build": build,
                            "calibration": calibration,
                        }
                    )
                roles["candidate"] = record
                payload["roles"] = roles
                _atomic_json(output_path, payload)
            del model, source, applied, build, calibration, catalog
            gc.collect()
            torch.cuda.empty_cache()

        metric_keys = plan.gate_spec.metric_keys
        accepted_metrics = _completed_metrics(roles, "accepted", metric_keys)
        matched_metrics = _completed_metrics(roles, "matched", metric_keys)
        candidate_metrics = _completed_metrics(roles, "candidate", metric_keys)
        accepted = Full35MetricSnapshot(
            run_id=f"{plan.plan_id}:accepted",
            policy_id=plan.accepted_policy_id,
            metric_contract_id=plan.metric_contract_id,
            metrics=accepted_metrics,
        )
        matched = Full35MetricSnapshot(
            run_id=f"{plan.plan_id}:matched",
            policy_id=plan.matched_policy_id,
            metric_contract_id=plan.metric_contract_id,
            metrics=matched_metrics,
        )
        candidate = Full35MetricCandidate(
            run_id=f"{plan.plan_id}:candidate",
            format_id=(
                f"{plan.cell.region}-{plan.cell.format_id}-{plan.cell.scale_method}"
            ),
            stage="ptq",
            policy_id=plan.matched_policy_id,
            metric_contract_id=plan.metric_contract_id,
            metrics=candidate_metrics,
        )
        gate = Full35MetricGate(plan.gate_spec).evaluate(
            candidate,
            accepted,
            matched,
        )
        payload.pop("current_role", None)
        payload["gate"] = gate.to_dict()
        payload["status"] = "completed"
        payload["next_step"] = "stop_and_review_before_any_matrix_expansion_or_training"
        _atomic_json(output_path, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "decision": gate.decision,
                    "worst_total_metric": gate.worst_total_metric,
                    "worst_total_delta": gate.worst_total_delta,
                    "worst_incremental_metric": gate.worst_incremental_metric,
                    "worst_incremental_delta": gate.worst_incremental_delta,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        payload["roles"] = roles
        _atomic_json(output_path, payload)
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full35 reviewed PTQ eight-metric search validation",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "configs/experiments/v4-qsilu-backbone-early-w8-search-v1.yaml"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-reviewed-plan", action="store_true")
    args = parser.parse_args(argv)
    plan = Full35SearchValidationPlan.from_yaml(args.plan)
    if args.list_only:
        print(
            json.dumps(
                plan.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.execute_reviewed_plan:
        parser.error("execution requires --execute-reviewed-plan acknowledgement")
    if (
        not torch.cuda.is_available()
        or not 0 <= args.device < torch.cuda.device_count()
    ):
        parser.error(f"CUDA device {args.device} is unavailable")
    return run_search_validation(
        plan=plan,
        output=args.output,
        device_index=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
