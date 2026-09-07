"""Full35 paired-QAT runtime with a CPU-only fail-closed preflight."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from .activation_smoke import _letterbox
from .diagnostic_manifest import DiagnosticManifest
from .full35_adapter import Full35ActivationAdapter
from .qat_calibration import (
    QATActivationCalibrationReport,
    calibrate_activation_outputs,
)
from .qat_continuation import QATPatienceContinuation
from .qat_graph import (
    BNFoldedHardwareContractGuard,
    materialize_qat_deployment_graph,
    prepare_folded_qat_training_graph,
)
from .qat_metrics import (
    QAT_GATE_METRICS,
    Map50AccuracyGate,
    Map50CheckpointSelectors,
)
from .qat_optimizer import is_quantizer_parameter, split_quantizer_parameter_groups
from .qat_plan import Full35QATPlan, QATArm, QATWarmStartSpec
from .qat_schedule import QATTrainabilityController, QuantizationEpochController
from .qat_validation import QATJointValidatorAdapter, build_qat_deployment_view
from .search_data import prepare_bbat5_search_view
from .validation_source import Full35DeploymentValidationSource
from .weight_quantization import Full35WeightRegionCatalog

_FULL35_SOURCE = Path(
    "/home/uxin/yolo/yolo_combine/final/full35/code/project/src"
).resolve()

_QAT_DEPLOYMENT_GRAPH_SCHEMA = "full35-bn-folded-a8-weight-materialized-v1"


class MatchedShamEvidenceError(RuntimeError):
    """Non-retryable failure in completed matched-sham evidence."""

    retryable = False


def _metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise TypeError(f"metrics artifact has no metrics mapping: {path}")
    return {str(name): float(value) for name, value in payload["metrics"].items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _apply_qat_full_resume_warm_start(
    model: nn.Module,
    weight_policy: Any,
    warm_start: QATWarmStartSpec,
) -> dict[str, Any]:
    """Load only a locked EMA shadow, then refit changed-format qparams."""

    actual_sha256 = _sha256(warm_start.checkpoint)
    if actual_sha256 != warm_start.checkpoint_sha256:
        raise ValueError("QAT warm-start checkpoint SHA-256 drifted")
    if warm_start.load_optimizer_state:
        raise ValueError("QAT warm start must use a fresh optimizer")
    payload = torch.load(
        warm_start.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or payload.get("checkpoint_kind") != "full_resume":
        raise ValueError("QAT warm-start source is not a full-resume checkpoint")
    progress = payload.get("progress")
    if (
        not isinstance(progress, dict)
        or int(progress.get("next_epoch", -1)) != warm_start.selected_epoch + 1
    ):
        raise ValueError("QAT warm-start checkpoint epoch differs from locked parent")
    source_state = payload.get(warm_start.state_key)
    if not isinstance(source_state, dict):
        raise TypeError("QAT warm-start state is not a mapping")

    override_paths = tuple(
        path
        for assignment in weight_policy.assignments
        if assignment.paths
        for path in assignment.paths
    )
    if not override_paths or len(set(override_paths)) != len(override_paths):
        raise ValueError("QAT warm start requires unique path-format overrides")
    target_state = model.state_dict()
    skipped = {
        name
        for name in target_state
        if any(name.startswith(f"{path}.weight_quantizer.") for path in override_paths)
    }
    loadable: dict[str, Tensor] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for name, target in target_state.items():
        if name in skipped:
            continue
        source = source_state.get(name)
        if not torch.is_tensor(source):
            missing.append(name)
            continue
        if source.shape != target.shape:
            mismatched.append(name)
            continue
        loadable[name] = source.detach()
    if missing or mismatched:
        raise ValueError(
            "QAT warm-start graph differs outside path qparams: "
            + json.dumps(
                {"missing": missing[:20], "shape_mismatches": mismatched[:20]},
                sort_keys=True,
            )
        )
    loaded = model.load_state_dict(loadable, strict=False)
    if set(loaded.missing_keys) != skipped or loaded.unexpected_keys:
        raise RuntimeError("QAT warm-start partial state load differed from contract")

    reset_paths: list[str] = []
    for path in override_paths:
        module = model.get_submodule(path)
        quantizer = getattr(module, "weight_quantizer", None)
        reset = getattr(quantizer, "reinitialize_from_weight", None)
        if not callable(reset):
            raise TypeError(f"QAT warm-start override has no quantizer: {path}")
        reset(module.weight)
        reset_paths.append(path)
    return {
        "parent_id": warm_start.parent_id,
        "selected_epoch": warm_start.selected_epoch,
        "parent_manifest": str(warm_start.parent_manifest),
        "parent_manifest_sha256": warm_start.parent_manifest_sha256,
        "checkpoint": str(warm_start.checkpoint),
        "checkpoint_sha256": actual_sha256,
        "state_key": warm_start.state_key,
        "loaded_tensors": len(loadable),
        "skipped_override_qparam_tensors": len(skipped),
        "reset_override_quantizers": len(reset_paths),
        "reset_paths": reset_paths,
        "optimizer_state_loaded": False,
    }


def _latest_log_event(path: Path, kind: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    latest: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"training log event is not a mapping: {path}")
        if payload.get("kind") == kind:
            latest = payload
    return latest


def _checkpoint_paths(run_dir: Path, reported: Mapping[str, Path]) -> dict[str, Path]:
    paths = {name: path for name, path in reported.items() if path.is_file()}
    for name in (
        "best_detect",
        "best_pose",
        "best_joint",
        "best_matched_sham",
        "last",
    ):
        existing = run_dir / "checkpoints" / f"{name}.pt"
        if name not in paths and existing.is_file():
            paths[name] = existing
    return paths


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_qat_deployment_inference_weights(
    resume_module: Any,
    destination: Path,
    *,
    deployment_model: nn.Module,
    contract_source: nn.Module,
    metadata: Mapping[str, Any],
) -> Path:
    """Save one materialized QAT graph using the Full35 inference schema."""

    contract = getattr(contract_source, "contract", None)
    if not callable(contract):
        raise TypeError("QAT contract source must expose contract()")
    frozen_contract = contract()
    if not isinstance(frozen_contract, dict) or "model_kind" not in frozen_contract:
        raise TypeError("QAT contract source must return a model_kind mapping")

    class _CheckpointView(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            object.__setattr__(self, "_deployment_model", deployment_model)
            self._frozen_contract = copy.deepcopy(frozen_contract)

        def contract(self) -> dict[str, Any]:
            return copy.deepcopy(self._frozen_contract)

        def state_dict(self, *args: Any, **kwargs: Any) -> Any:
            return self._deployment_model.state_dict(*args, **kwargs)

    checkpoint_view = _CheckpointView()
    resolved_metadata = dict(metadata)
    configured_schema = resolved_metadata.setdefault(
        "deployment_graph_schema", _QAT_DEPLOYMENT_GRAPH_SCHEMA
    )
    if configured_schema != _QAT_DEPLOYMENT_GRAPH_SCHEMA:
        raise ValueError("QAT deployment graph schema differs from the runtime")
    return resume_module.save_inference_weights(
        destination,
        model=checkpoint_view,
        use_ema=False,
        metadata=resolved_metadata,
    )


def _tensor_objective(value: object) -> tuple[Tensor, int]:
    """Reduce nested Full35 outputs to a finite smoke-only autograd scalar."""

    tensors: list[Tensor] = []

    def collect(item: object) -> None:
        if isinstance(item, Tensor):
            if item.requires_grad and item.numel():
                tensors.append(item)
            return
        if isinstance(item, Mapping):
            for child in item.values():
                collect(child)
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                collect(child)

    collect(value)
    if not tensors:
        raise ValueError("QAT smoke output contains no differentiable tensor")
    terms = tuple(tensor.float().square().mean() for tensor in tensors)
    objective = terms[0]
    for term in terms[1:]:
        objective = objective + term
    return objective, len(tensors)


class _CalibrationImages(Sequence[Tensor]):
    """Load fixed calibration images lazily to avoid a 300+ MiB CPU tensor list."""

    def __init__(self, paths: tuple[Path, ...], *, image_size: int) -> None:
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int | slice) -> Tensor | tuple[Tensor, ...]:
        if isinstance(index, slice):
            return tuple(
                _letterbox(path, self.image_size) for path in self.paths[index]
            )
        return _letterbox(self.paths[index], self.image_size)


@dataclass(frozen=True)
class _QATGraphHandles:
    model: nn.Module
    source: Any
    activation: Any
    weights: Any
    catalog: Any
    fold_report: Any
    calibration: Any | None
    checkpoint_state_source: str
    warm_start: dict[str, Any] | None


@dataclass(frozen=True)
class LoadedQATDeploymentParent:
    """Strictly reconstructed deployment parent for downstream weight studies."""

    model: nn.Module
    source: Full35DeploymentValidationSource
    activation: Any
    catalog: Full35WeightRegionCatalog
    checkpoint_path: Path
    checkpoint_sha256: str
    full_resume_sha256: str
    epoch: int
    metadata: dict[str, Any]
    legacy_schema_inferred: bool


@dataclass(frozen=True)
class QATRuntimePreflightReport:
    """CPU-checkable readiness evidence before a GPU graph is allocated."""

    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    resolved: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "resolved": self.resolved,
        }


class _QATConfigProxy:
    def __init__(
        self,
        base: Any,
        *,
        plan: Full35QATPlan,
        arm: QATArm,
        patience_continuation: QATPatienceContinuation | None = None,
    ) -> None:
        self._base = base
        self._plan = plan
        self._arm = arm
        self._patience_continuation = patience_continuation

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    @property
    def maximum_map_drop(self) -> float:
        return self._plan.gates.map50_max_drop

    def preflight(self) -> Any:
        return replace(
            self._base.preflight(),
            baseline=_metrics(self._plan.accepted_metrics),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = self._base.as_dict()
        payload["maximum_map_drop"] = self.maximum_map_drop
        payload["baseline_metrics"] = str(self._plan.accepted_metrics)
        payload["pose_data"] = str(self._plan.data.pose_search)
        payload["qat_experiment"] = {
            "plan_id": self._plan.plan_id,
            "candidate_id": self._plan.candidate_id,
            "plan": str(self._plan.config_path),
            "plan_sha256": self._plan.config_sha256,
            "authorization_id": self._plan.execution_authorization_id,
            "arm": self._arm,
            "activation": self._plan.activation.policy_id,
            "weight_blend_ratio": Full35QATRuntime.weight_blend_ratio(self._arm),
            "weight_assignments": [
                {
                    "region": assignment.region,
                    "paths": list(assignment.paths),
                    "format": assignment.spec.format_id,
                    "bits": assignment.spec.bits,
                }
                for assignment in self._plan.assignments
            ],
            "float_regions": list(self._plan.float_regions),
            "data": {
                "coco": str(self._plan.data.coco),
                "bbat5_registry": str(self._plan.data.registry),
                "bbat5_pose_search": str(self._plan.data.pose_search),
                "bbat5_pose_runtime_view_role": "canonical-search-runtime-view",
                "assignment_changed": False,
            },
            "gates": {
                "map50_max_drop": self._plan.gates.map50_max_drop,
                "map50_95_max_drop": self._plan.gates.map50_95_max_drop,
            },
            "formal_validation": False,
        }
        if self._patience_continuation is not None:
            payload["qat_experiment"]["patience_continuation"] = (
                self._patience_continuation.to_dict()
            )
        return payload


class Full35QATRuntime:
    """Load one reviewed plan and expose matched Full35 QAT execution seams."""

    def __init__(
        self,
        plan: Full35QATPlan,
        *,
        patience_continuation: QATPatienceContinuation | None = None,
    ) -> None:
        self.plan = plan
        self.patience_continuation = patience_continuation

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        patience_continuation: str | Path | None = None,
    ) -> Full35QATRuntime:
        plan = Full35QATPlan.from_yaml(path)
        continuation = (
            QATPatienceContinuation.from_yaml(
                patience_continuation,
                base_plan=plan.config_path,
                base_plan_sha256=plan.config_sha256,
            )
            if patience_continuation is not None
            else None
        )
        if continuation is not None and (
            continuation.from_patience != plan.training.patience
        ):
            raise ValueError(
                "continuation from_patience differs from the base QAT plan"
            )
        return cls(plan, patience_continuation=continuation)

    @property
    def effective_patience(self) -> int:
        if self.patience_continuation is not None:
            return self.patience_continuation.to_patience
        return self.plan.training.patience

    @staticmethod
    def weight_blend_ratio(arm: QATArm) -> float:
        if arm == "sham":
            return 0.0
        if arm == "qat":
            return 1.0
        raise ValueError(f"unsupported QAT arm: {arm}")

    def run_name(self, arm: QATArm) -> str:
        self.weight_blend_ratio(arm)
        return f"{self.plan.plan_id}-{arm}-seed{self.plan.training.seed}"

    def _read_sham_completion(
        self,
    ) -> tuple[Path, Path, dict[str, Any], int]:
        run_name = self.run_name("sham")
        run_dir = (self.plan.run_root / run_name).resolve()
        manifest_path = run_dir / "qat-experiment.json"
        if not manifest_path.is_file():
            raise MatchedShamEvidenceError(
                f"matched sham completion evidence is missing: {manifest_path}"
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MatchedShamEvidenceError(
                f"matched sham completion evidence is unreadable: {manifest_path}"
            ) from error
        if not isinstance(payload, dict):
            raise MatchedShamEvidenceError(
                "matched sham completion evidence must be a JSON object"
            )
        expected = {
            "schema_version": 1,
            "plan_sha256": self.plan.config_sha256,
            "candidate_id": self.plan.candidate_id,
            "arm": "sham",
            "run_name": run_name,
            "formal_validation": False,
        }
        drift = {
            name: {"expected": value, "actual": payload.get(name)}
            for name, value in expected.items()
            if payload.get(name) != value
        }
        completed_stages = payload.get("completed_stages")
        if completed_stages != ["j3"]:
            drift["completed_stages"] = {
                "expected": ["j3"],
                "actual": completed_stages,
            }
        epochs_completed = payload.get("epochs_completed")
        early_stop = payload.get("early_stop")
        early_values = (
            early_stop.get("values") if isinstance(early_stop, dict) else None
        )
        legitimate_early_stop = (
            isinstance(epochs_completed, int)
            and not isinstance(epochs_completed, bool)
            and 0 < epochs_completed < self.plan.training.epochs
            and payload.get("epochs_planned") == self.plan.training.epochs
            and payload.get("effective_patience") == self.effective_patience
            and isinstance(early_stop, dict)
            and early_stop.get("kind") == "early_stop"
            and isinstance(early_values, dict)
            and float(early_values.get("patience", -1.0))
            == float(self.effective_patience)
            and float(early_values.get("stale_epochs", -1.0))
            >= float(self.effective_patience)
            and float(early_values.get("should_stop", 0.0)) == 1.0
        )
        if (
            not isinstance(epochs_completed, int)
            or isinstance(epochs_completed, bool)
            or (
                epochs_completed < self.plan.training.epochs
                and not legitimate_early_stop
            )
        ):
            drift["epochs_completed"] = {
                "expected_full_or_legitimate_early_stop": (self.plan.training.epochs),
                "actual": epochs_completed,
            }
        if drift:
            raise MatchedShamEvidenceError(
                "matched sham completion evidence differs from the active QAT plan: "
                + json.dumps(drift, ensure_ascii=False, sort_keys=True)
            )
        return run_dir, manifest_path, payload, epochs_completed

    @staticmethod
    def _validated_sham_checkpoint(
        run_dir: Path,
        *,
        role: str,
        checkpoint_value: object,
        expected_digest: object,
    ) -> tuple[Path, str]:
        if not isinstance(checkpoint_value, str) or not isinstance(
            expected_digest, str
        ):
            raise MatchedShamEvidenceError(
                f"matched sham {role} checkpoint is not hash-pinned"
            )
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        try:
            checkpoint.relative_to(run_dir)
        except ValueError as error:
            raise MatchedShamEvidenceError(
                f"matched sham {role} checkpoint is outside its run directory"
            ) from error
        if not checkpoint.is_file():
            raise MatchedShamEvidenceError(
                f"matched sham {role} checkpoint is missing: {checkpoint}"
            )
        actual_digest = _sha256(checkpoint)
        if actual_digest != expected_digest:
            raise MatchedShamEvidenceError(
                f"matched sham {role} checkpoint SHA-256 drifted: "
                f"{actual_digest} != {expected_digest}"
            )
        return checkpoint, actual_digest

    def _assert_sham_ready(self) -> dict[str, Any]:
        """Require a complete matched sham with a drift-feasible checkpoint."""

        run_dir, manifest_path, payload, epochs_completed = self._read_sham_completion()
        paths = payload.get("checkpoint_paths")
        digests = payload.get("checkpoint_sha256")
        if not isinstance(paths, dict) or not isinstance(digests, dict):
            raise MatchedShamEvidenceError(
                "matched sham completion evidence has no hash-pinned checkpoints"
            )
        for role in ("best_matched_sham", "best_joint"):
            if role not in paths and role not in digests:
                continue
            checkpoint, actual_digest = self._validated_sham_checkpoint(
                run_dir,
                role=role,
                checkpoint_value=paths.get(role),
                expected_digest=digests.get(role),
            )
            return {
                "evidence_kind": "manifest",
                "manifest": str(manifest_path),
                "checkpoint_role": role,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": actual_digest,
                "epochs_completed": epochs_completed,
            }

        evidence_path = run_dir / "matched-sham-evidence.json"
        if not evidence_path.is_file():
            raise MatchedShamEvidenceError(
                "matched sham did not produce a drift-feasible checkpoint; "
                "run --reconcile-matched-sham for a completed historical sham"
            )
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MatchedShamEvidenceError(
                f"matched sham reconciliation is unreadable: {evidence_path}"
            ) from error
        if not isinstance(evidence, dict):
            raise MatchedShamEvidenceError(
                "matched sham reconciliation must be a JSON object"
            )
        expected = {
            "schema_version": 1,
            "status": "completed",
            "plan_sha256": self.plan.config_sha256,
            "candidate_id": self.plan.candidate_id,
            "run_name": self.run_name("sham"),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256(manifest_path),
            "matched_sham_passed": True,
            "formal_validation": False,
        }
        drift = {
            name: {"expected": value, "actual": evidence.get(name)}
            for name, value in expected.items()
            if evidence.get(name) != value
        }
        if drift:
            raise MatchedShamEvidenceError(
                "matched sham reconciliation drifted: "
                + json.dumps(drift, ensure_ascii=False, sort_keys=True)
            )
        role = evidence.get("checkpoint_role")
        if not isinstance(role, str) or paths.get(role) != evidence.get("checkpoint"):
            raise MatchedShamEvidenceError(
                "matched sham reconciliation checkpoint differs from completion"
            )
        checkpoint, actual_digest = self._validated_sham_checkpoint(
            run_dir,
            role=role,
            checkpoint_value=evidence.get("checkpoint"),
            expected_digest=evidence.get("checkpoint_sha256"),
        )
        if digests.get(role) != actual_digest:
            raise MatchedShamEvidenceError(
                "matched sham reconciliation digest differs from completion"
            )
        metrics_path = Path(str(evidence.get("metrics"))).expanduser().resolve()
        try:
            metrics_path.relative_to(run_dir)
        except ValueError as error:
            raise MatchedShamEvidenceError(
                "matched sham reconciliation metrics are outside the run"
            ) from error
        if not metrics_path.is_file() or evidence.get("metrics_sha256") != _sha256(
            metrics_path
        ):
            raise MatchedShamEvidenceError(
                "matched sham reconciliation metrics are missing or drifted"
            )
        gate = Map50AccuracyGate(
            _metrics(self.plan.accepted_metrics),
            maximum_drop=self.plan.gates.map50_max_drop,
            maximum_map50_95_drop=self.plan.gates.map50_95_max_drop,
            matched_baseline=_metrics(self.plan.matched_metrics),
            matched_max_absolute_drift=(
                self.plan.gates.matched_sham_max_absolute_drift
            ),
        ).evaluate(_metrics(metrics_path))
        if (
            gate.matched_sham_passed is not True
            or evidence.get("deployment_passed") != gate.deployment_passed
        ):
            raise MatchedShamEvidenceError(
                "matched sham reconciliation no longer satisfies its gates"
            )
        return {
            "evidence_kind": "reconciled",
            "manifest": str(manifest_path),
            "reconciliation": str(evidence_path),
            "reconciliation_sha256": _sha256(evidence_path),
            "checkpoint_role": role,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_digest,
            "epochs_completed": epochs_completed,
            "selected_epoch": evidence.get("selected_epoch"),
        }

    def _external_control_evidence(self) -> dict[str, Any]:
        """Return hash-pinned external sham evidence for an unpaired continuation."""

        external = self.plan.external_control
        if external is None:
            raise RuntimeError("QAT plan has no external control reference")
        required = (
            "mode",
            "plan",
            "plan_sha256",
            "completion",
            "completion_sha256",
            "metrics",
            "metrics_sha256",
        )
        if any(key not in external for key in required):
            raise RuntimeError("external control reference is incomplete")
        completion = Path(str(external["completion"])).resolve()
        metrics = Path(str(external["metrics"])).resolve()
        if _sha256(completion) != str(external["completion_sha256"]):
            raise RuntimeError("external control completion SHA-256 drifted")
        if _sha256(metrics) != str(external["metrics_sha256"]):
            raise RuntimeError("external control metrics SHA-256 drifted")
        return {
            "evidence_kind": "external_control",
            "mode": str(external["mode"]),
            "plan": str(external["plan"]),
            "plan_sha256": str(external["plan_sha256"]),
            "completion": str(completion),
            "completion_sha256": str(external["completion_sha256"]),
            "metrics": str(metrics),
            "metrics_sha256": str(external["metrics_sha256"]),
            "unpaired_continuation": True,
        }

    def reconcile_matched_sham(self) -> dict[str, Any]:
        """Replay a completed historical sham and pin its best stable checkpoint."""

        run_dir, manifest_path, payload, epochs_completed = self._read_sham_completion()
        gate = Map50AccuracyGate(
            _metrics(self.plan.accepted_metrics),
            maximum_drop=self.plan.gates.map50_max_drop,
            maximum_map50_95_drop=self.plan.gates.map50_95_max_drop,
            matched_baseline=_metrics(self.plan.matched_metrics),
            matched_max_absolute_drift=(
                self.plan.gates.matched_sham_max_absolute_drift
            ),
        )
        selectors = Map50CheckpointSelectors()
        metric_paths: dict[int, Path] = {}
        for epoch in range(epochs_completed):
            metrics_path = (
                run_dir
                / "validation"
                / f"epoch-{epoch:04d}"
                / "bittrue"
                / "metrics.json"
            )
            if not metrics_path.is_file():
                raise MatchedShamEvidenceError(
                    f"matched sham epoch metrics are missing: {metrics_path}"
                )
            metric_paths[epoch] = metrics_path
            metrics = _metrics(metrics_path)
            selectors.observe(epoch=epoch, metrics=metrics, gate=gate.evaluate(metrics))
        selected = selectors.state_dict().get("best_matched_sham")
        if not isinstance(selected, dict):
            raise MatchedShamEvidenceError(
                "completed sham has no epoch within the matched drift budget"
            )
        selected_epoch = int(selected["epoch"])
        paths = payload.get("checkpoint_paths")
        digests = payload.get("checkpoint_sha256")
        if not isinstance(paths, dict) or not isinstance(digests, dict):
            raise MatchedShamEvidenceError(
                "matched sham completion evidence has no hash-pinned checkpoints"
            )
        selected_role: str | None = None
        selected_checkpoint: Path | None = None
        selected_digest: str | None = None
        for role in ("best_matched_sham", "best_pose", "best_detect", "last"):
            if role not in paths:
                continue
            checkpoint, digest = self._validated_sham_checkpoint(
                run_dir,
                role=role,
                checkpoint_value=paths.get(role),
                expected_digest=digests.get(role),
            )
            checkpoint_payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            progress = (
                checkpoint_payload.get("progress")
                if isinstance(checkpoint_payload, dict)
                else None
            )
            checkpoint_epoch = (
                int(progress.get("next_epoch")) - 1
                if isinstance(progress, dict)
                and isinstance(progress.get("next_epoch"), int)
                else None
            )
            del checkpoint_payload
            if checkpoint_epoch == selected_epoch:
                selected_role = role
                selected_checkpoint = checkpoint
                selected_digest = digest
                break
        if (
            selected_role is None
            or selected_checkpoint is None
            or selected_digest is None
        ):
            raise MatchedShamEvidenceError(
                "best drift-feasible sham epoch has no retained checkpoint; "
                "the sham must be rerun with the corrected selector"
            )
        metrics_path = metric_paths[selected_epoch]
        selected_metrics = _metrics(metrics_path)
        gate_report = gate.evaluate(selected_metrics)
        matched_deltas = gate_report.matched_sham_deltas
        report = {
            "schema_version": 1,
            "status": "completed",
            "plan": str(self.plan.config_path),
            "plan_sha256": self.plan.config_sha256,
            "candidate_id": self.plan.candidate_id,
            "run_name": self.run_name("sham"),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256(manifest_path),
            "selected_epoch": selected_epoch,
            "joint_score": float(selected["score"]),
            "checkpoint_role": selected_role,
            "checkpoint": str(selected_checkpoint),
            "checkpoint_sha256": selected_digest,
            "metrics": str(metrics_path),
            "metrics_sha256": _sha256(metrics_path),
            "deployment_passed": gate_report.deployment_passed,
            "matched_sham_passed": gate_report.matched_sham_passed,
            "maximum_absolute_drift": max(
                abs(value) for value in matched_deltas.values()
            ),
            "matched_sham_max_absolute_drift": (
                self.plan.gates.matched_sham_max_absolute_drift
            ),
            "formal_validation": False,
        }
        _atomic_json(run_dir / "matched-sham-evidence.json", report)
        return report

    def _imports(self) -> dict[str, Any]:
        source = str(_FULL35_SOURCE)
        if source not in sys.path:
            sys.path.insert(0, source)
        package = importlib.import_module("yolo_combine")
        package_file = Path(package.__file__).resolve()
        if _FULL35_SOURCE not in package_file.parents:
            raise RuntimeError(f"yolo_combine is shadowed by {package_file}")
        return {
            name: importlib.import_module(f"yolo_combine.{module}")
            for name, module in {
                "joint_config": "joint_config",
                "joint_data": "joint_data",
                "formal_training": "formal_training",
                "formal_impl": "_formal_training_impl",
                "stage_policy": "stage_policy",
                "resume": "resume",
            }.items()
        }

    def training_config(self, arm: QATArm) -> _QATConfigProxy:
        self.weight_blend_ratio(arm)
        modules = self._imports()
        base = modules["joint_config"].JointExperimentConfig.load(
            self.plan.joint_config
        )
        derived = replace(
            base,
            stages=(),
            enable_j3=True,
            seed=self.plan.training.seed,
            run_root=self.plan.run_root,
            detect_batch_size=self.plan.training.detect_logical_batch,
            detect_microbatch_size=self.plan.training.detect_microbatch,
            pose_batch_size=self.plan.training.pose_batch,
            optimizer=self.plan.optimizer.name,
            weight_decay=self.plan.optimizer.weight_decay,
            beta1=self.plan.optimizer.beta1,
            beta2=self.plan.optimizer.beta2,
            gradient_clip_norm=self.plan.training.gradient_clip_norm,
            validation_backends=("bittrue",),
            selection_backend="bittrue",
            maximum_map_drop=self.plan.gates.map50_max_drop,
            validation_plots=False,
            save_coco_json=False,
        )
        return _QATConfigProxy(
            derived,
            plan=self.plan,
            arm=arm,
            patience_continuation=self.patience_continuation,
        )

    def _build_graph(
        self,
        *,
        device: torch.device | None = None,
        calibrate: bool = False,
    ) -> _QATGraphHandles:
        if calibrate != (device is not None):
            raise ValueError(
                "QAT calibration and a target device must be requested together"
            )
        built = Full35ActivationAdapter(recipe=self.plan.activation_recipe).build(
            self.plan.activation,
            checkpoint=self.plan.parent_checkpoint,
            checkpoint_sha256=self.plan.parent_checkpoint_sha256,
        )
        prepared = prepare_folded_qat_training_graph(
            built.model,
            expected_catalog=self.plan.expected_master_catalog,
            assignments=self.plan.assignments,
        )
        warm_start_report = None
        if self.plan.warm_start is not None:
            warm_start_report = _apply_qat_full_resume_warm_start(
                prepared.model,
                prepared.policy,
                self.plan.warm_start,
            )
        activation = built.applied.rebind(prepared.model)
        calibration = None
        if calibrate and self.plan.warm_start is not None:
            if activation.mode != "fake_quant":
                raise RuntimeError(
                    "warm-start activation state is not frozen fake quant"
                )
            ranges = activation.observer_ranges()
            invalid = tuple(
                path
                for path, observed in ranges.items()
                if observed is None or observed[1] <= observed[0]
            )
            if invalid:
                raise RuntimeError(
                    "warm-start activation observers are invalid: " + ", ".join(invalid)
                )
            prepared.model.to(device).eval()
            valid_ranges = tuple(
                value for value in ranges.values() if value is not None
            )
            calibration = QATActivationCalibrationReport(
                samples={"detect": 0, "pose": 0},
                quantizers=activation.quantizer_count,
                valid_observers=len(valid_ranges),
                invalid_observers=(),
                observed_minimum=min(value[0] for value in valid_ranges),
                observed_maximum=max(value[1] for value in valid_ranges),
                seconds=0.0,
                peak_gpu_memory_mib=0.0,
            )
            warm_start_report["activation_calibration"] = (
                "reused_locked_v19_learned_state"
            )
        elif calibrate:
            manifest = DiagnosticManifest.from_json(
                self.plan.data.diagnostic_manifest,
                verify_files=True,
            )
            if manifest.sha256 != self.plan.data.diagnostic_manifest_identity:
                raise ValueError("diagnostic manifest identity differs from QAT plan")
            calibration = calibrate_activation_outputs(
                activation,
                samples={
                    "detect": _CalibrationImages(
                        manifest.paths("calibration", "coco_detect"),
                        image_size=640,
                    ),
                    "pose": _CalibrationImages(
                        manifest.paths("calibration", "bbat_pose"),
                        image_size=640,
                    ),
                },
                device=device,
            )
        return _QATGraphHandles(
            model=prepared.model,
            source=built.source,
            activation=activation,
            weights=prepared.policy,
            catalog=prepared.catalog,
            fold_report=prepared.fold_report,
            calibration=calibration,
            checkpoint_state_source=str(built.loaded_checkpoint.state_source),
            warm_start=warm_start_report,
        )

    def load_deployment_parent(
        self,
        checkpoint: str | Path,
        *,
        checkpoint_sha256: str,
        full_resume_sha256: str,
        epoch: int,
    ) -> LoadedQATDeploymentParent:
        """Reconstruct the reviewed folded graph, then strict-load one QAT export."""

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        actual_sha256 = _sha256(checkpoint_path)
        if actual_sha256 != checkpoint_sha256:
            raise ValueError(
                "QAT deployment checkpoint SHA-256 drifted: "
                f"{actual_sha256} != {checkpoint_sha256}"
            )
        if len(full_resume_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in full_resume_sha256
        ):
            raise ValueError("full-resume SHA-256 must be a lowercase digest")
        if epoch < 0:
            raise ValueError("locked QAT parent epoch cannot be negative")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError("QAT deployment checkpoint must be a mapping")
        if payload.get("schema_version") != 1:
            raise ValueError("QAT deployment checkpoint schema_version must be 1")
        if payload.get("checkpoint_kind") != "inference_only":
            raise ValueError("locked QAT parent must be an inference-only checkpoint")
        if payload.get("source") != "live":
            raise ValueError(
                "QAT deployment checkpoint source must be live materialized EMA"
            )
        metadata = payload.get("metadata")
        state = payload.get("state_dict")
        contract = payload.get("contract")
        if not isinstance(metadata, dict):
            raise TypeError("QAT deployment checkpoint metadata must be a mapping")
        if not isinstance(state, dict) or not state:
            raise TypeError("QAT deployment checkpoint state_dict must be non-empty")
        if not isinstance(contract, dict):
            raise TypeError("QAT deployment checkpoint contract must be a mapping")
        invalid_state = tuple(
            name
            for name, value in state.items()
            if not isinstance(name, str)
            or not isinstance(value, Tensor)
            or (value.dtype.is_floating_point and not bool(torch.isfinite(value).all()))
        )
        if invalid_state:
            raise ValueError(
                "QAT deployment state contains invalid tensors: "
                + ", ".join(str(value) for value in invalid_state[:10])
            )

        expected_metadata = {
            "epoch": epoch,
            "full_resume_sha256": full_resume_sha256,
            "qat_plan_sha256": self.plan.config_sha256,
            "qat_arm": "qat",
            "weight_blend_ratio": 1.0,
            "loader": "yolo_quantize.qat_runtime",
        }
        metadata_drift = {
            name: {"expected": value, "actual": metadata.get(name)}
            for name, value in expected_metadata.items()
            if metadata.get(name) != value
        }
        if metadata_drift:
            raise ValueError(
                "QAT deployment checkpoint lineage differs: "
                + json.dumps(metadata_drift, ensure_ascii=False, sort_keys=True)
            )
        graph_schema = metadata.get("deployment_graph_schema")
        if graph_schema not in {None, _QAT_DEPLOYMENT_GRAPH_SCHEMA}:
            raise ValueError("QAT deployment checkpoint graph schema is unsupported")

        handles = self._build_graph()
        contract_method = getattr(handles.model, "contract", None)
        if not callable(contract_method) or contract != contract_method():
            raise ValueError("QAT deployment checkpoint model contract mismatch")
        deployment = materialize_qat_deployment_graph(
            handles.weights,
            expected_catalog=self.plan.expected_deployment_catalog,
            blend_ratio=1.0,
        )
        result = deployment.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "strict QAT deployment load drifted: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        activation = handles.activation.rebind(deployment)
        if activation.mode != "fake_quant":
            raise RuntimeError(
                "locked QAT deployment activation quantizers are not in fake_quant mode"
            )
        if activation.quantizer_count != handles.activation.quantizer_count:
            raise RuntimeError("locked QAT deployment activation paths changed")
        catalog = Full35WeightRegionCatalog.inspect(deployment)
        if catalog.summary() != self.plan.expected_deployment_catalog:
            raise RuntimeError("locked QAT deployment weight catalog differs")
        if catalog.training_only_sites:
            raise RuntimeError("locked QAT deployment retains training-only weights")
        source = Full35DeploymentValidationSource(handles.source, activation)
        return LoadedQATDeploymentParent(
            model=deployment,
            source=source,
            activation=activation,
            catalog=catalog,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=actual_sha256,
            full_resume_sha256=full_resume_sha256,
            epoch=epoch,
            metadata=dict(metadata),
            legacy_schema_inferred=graph_schema is None,
        )

    def preflight(
        self,
        *,
        verify_graph: bool = False,
        verify_sample_files: bool = True,
    ) -> QATRuntimePreflightReport:
        blockers: list[str] = []
        warnings: list[str] = []
        try:
            config = self.training_config("qat")
            upstream = config.preflight()
            blockers.extend(upstream.blockers)
            warnings.extend(upstream.warnings)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            blockers.append(f"Full35 QAT config preflight failed: {error}")
        try:
            manifest = DiagnosticManifest.from_json(
                self.plan.data.diagnostic_manifest,
                verify_files=verify_sample_files,
            )
            if manifest.sha256 != self.plan.data.diagnostic_manifest_identity:
                raise ValueError("diagnostic manifest identity differs from QAT plan")
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            blockers.append(f"QAT calibration manifest failed: {error}")
        try:
            accepted = _metrics(self.plan.accepted_metrics)
            matched = _metrics(self.plan.matched_metrics)
            gate = Map50AccuracyGate(
                accepted,
                maximum_drop=self.plan.gates.map50_max_drop,
                maximum_map50_95_drop=self.plan.gates.map50_95_max_drop,
            )
            gate.evaluate(matched)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            blockers.append(f"QAT dual-metric baseline failed: {error}")
        graph: dict[str, Any] | None = None
        if verify_graph and not blockers:
            try:
                handles = self._build_graph()
                catalog = handles.catalog.summary()
                totals = catalog["totals"]
                modules = self._imports()
                stage = replace(
                    modules["formal_impl"].JOINT_STAGES["j3"],
                    epochs=self.plan.training.epochs,
                    patience=self.effective_patience,
                    warmup_epochs=self.plan.training.warmup_epochs,
                    learning_rates=MappingProxyType(
                        dict(self.plan.optimizer.learning_rates)
                    ),
                )
                optimizer, _ = modules["stage_policy"].build_joint_optimizer(
                    handles.model,
                    stage,
                    optimizer_name=self.plan.optimizer.name,
                    weight_decay=self.plan.optimizer.weight_decay,
                    beta1=self.plan.optimizer.beta1,
                    beta2=self.plan.optimizer.beta2,
                )
                optimizer_split = split_quantizer_parameter_groups(
                    optimizer,
                    qparam_lr_ratio=self.plan.optimizer.qparam_lr_ratio,
                )
                graph = {
                    "weight_quantizers": handles.weights.quantized_modules,
                    "activation_quantizers": handles.activation.quantizer_count,
                    "final_batch_norm_modules": (
                        handles.fold_report.final_batch_norm_modules
                    ),
                    "binary_qk_protected_modules": int(totals["protected_modules"]),
                    "quantizer_optimizer_parameters": (
                        optimizer_split.quantizer_parameters
                    ),
                    "quantizer_optimizer_groups": len(optimizer_split.quantizer_groups),
                    "fold": handles.fold_report.to_dict(),
                    "catalog": catalog,
                    "warm_start": handles.warm_start,
                }
                del optimizer, handles
                gc.collect()
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                blockers.append(f"QAT graph preflight failed: {error}")
        return QATRuntimePreflightReport(
            ready=not blockers,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
            resolved={
                "plan_id": self.plan.plan_id,
                "plan_sha256": self.plan.config_sha256,
                "formal_validation": False,
                "pose_data": str(self.plan.data.pose_search),
                "required_metric_count": len(QAT_GATE_METRICS),
                "gates": {
                    "map50_max_drop": self.plan.gates.map50_max_drop,
                    "map50_95_max_drop": self.plan.gates.map50_95_max_drop,
                },
                "graph_verified": graph is not None,
                "graph": graph,
            },
        )

    @contextmanager
    def _patched_training(
        self,
        modules: dict[str, Any],
        *,
        arm: QATArm,
        run_name: str,
        device: torch.device,
    ) -> Iterator[type]:
        plan = self.plan
        impl = modules["formal_impl"]
        joint_config_module = modules["joint_config"]
        stage_policy_module = modules["stage_policy"]
        formal_module = modules["formal_training"]
        joint_data_module = modules["joint_data"]
        resume_module = modules["resume"]
        original_stages = impl.JOINT_STAGES
        original_joint_stages = joint_config_module.JOINT_STAGES
        original_policy_stages = stage_policy_module.JOINT_STAGES
        original_factory = impl.FusionModelFactory
        original_validator = impl.JointValidator
        original_gate = impl.AccuracyGate
        original_selectors = impl.CheckpointSelectors
        original_guard = impl.HardwareContractGuard
        original_apply_stage = impl.apply_stage
        original_build_optimizer = impl.build_joint_optimizer
        original_joint_runner = impl.JointEpochRunner
        original_pose_runner = impl.PoseEpochRunner
        original_prepare_view = formal_module.prepare_bbt5_view
        original_pose_validator = joint_data_module.validate_canonical_pose_source
        loss_module = importlib.import_module("ultralytics.utils.loss")
        original_bbox_iou = loss_module.bbox_iou

        original_j3 = original_stages["j3"]
        matched_reference = _metrics(plan.matched_metrics)
        qat_j3 = replace(
            original_j3,
            epochs=plan.training.epochs,
            patience=self.effective_patience,
            warmup_epochs=plan.training.warmup_epochs,
            learning_rates=MappingProxyType(dict(plan.optimizer.learning_rates)),
        )
        stages = MappingProxyType({**dict(original_stages), "j3": qat_j3})
        run_dir = plan.run_root / run_name
        state: dict[str, Any] = {"epoch": None, "records": {}}
        runtime = self

        class PolicyFactory:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._inner = original_factory(*args, **kwargs)

            def build(self, *args: Any, **kwargs: Any) -> Any:
                if "handles" in state:
                    raise RuntimeError(
                        "QAT factory may build the training graph only once"
                    )
                seed_build = self._inner.build(*args, **kwargs)
                handles = runtime._build_graph(device=device, calibrate=True)
                if handles.calibration is None:
                    raise RuntimeError(
                        "QAT factory did not calibrate activation outputs"
                    )
                state["handles"] = handles
                state["quantization"] = QuantizationEpochController(
                    handles.weights,
                    plan.training.progressive_schedule,
                    sham=arm == "sham",
                )
                state["trainability"] = QATTrainabilityController(
                    scale_only_epochs=plan.training.scale_only_epochs
                )
                _atomic_json(
                    run_dir / "qat-graph.json",
                    {
                        "schema_version": 1,
                        "plan": str(plan.config_path),
                        "plan_sha256": plan.config_sha256,
                        "candidate_id": plan.candidate_id,
                        "arm": arm,
                        "parent_checkpoint": str(plan.parent_checkpoint),
                        "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
                        "checkpoint_state_source": handles.checkpoint_state_source,
                        "warm_start": handles.warm_start,
                        "activation_policy": plan.activation.policy_id,
                        "activation_quantizers": handles.activation.quantizer_count,
                        "weight_quantizers": handles.weights.quantized_modules,
                        "weight_sites": list(handles.weights.site_records),
                        "fold": handles.fold_report.to_dict(),
                        "calibration": handles.calibration.to_dict(),
                        "master_catalog": handles.catalog.summary(),
                        "formal_validation": False,
                    },
                )
                return replace(seed_build, model=handles.model)

        def qat_apply_stage(model: nn.Module, stage: Any) -> Any:
            report = original_apply_stage(model, stage)
            epoch = state["epoch"]
            if epoch is None:
                return report
            handles = state.get("handles")
            if not isinstance(handles, _QATGraphHandles):
                raise TypeError("QAT epoch began before graph construction")
            quantization = state["quantization"].begin_epoch(int(epoch))
            trainability = state["trainability"].apply(model, epoch=int(epoch))
            state["records"][str(epoch)] = {
                "quantization": quantization.to_dict(),
                "trainability": trainability.to_dict(),
            }
            _atomic_json(
                run_dir / "qat-epoch-controls.json",
                {
                    "schema_version": 1,
                    "arm": arm,
                    "epochs": state["records"],
                },
            )
            return report

        def qat_build_optimizer(model: nn.Module, stage: Any, **kwargs: Any) -> Any:
            optimizer, report = original_build_optimizer(model, stage, **kwargs)
            split = split_quantizer_parameter_groups(
                optimizer,
                qparam_lr_ratio=plan.optimizer.qparam_lr_ratio,
            )
            report = replace(
                report,
                group_names=tuple(
                    str(group.get("group_name")) for group in optimizer.param_groups
                ),
            )
            _atomic_json(
                run_dir / "qat-optimizer.json",
                {
                    "schema_version": 1,
                    "arm": arm,
                    "optimizer": plan.optimizer.name,
                    "weight_decay": plan.optimizer.weight_decay,
                    "beta1": plan.optimizer.beta1,
                    "beta2": plan.optimizer.beta2,
                    "split": split.to_dict(),
                    "groups": [
                        {
                            "group_name": str(group.get("group_name")),
                            "role": str(group.get("role")),
                            "lr": float(group["lr"]),
                            "weight_decay": float(group["weight_decay"]),
                            "parameters": len(group["params"]),
                        }
                        for group in optimizer.param_groups
                    ],
                },
            )
            return optimizer, report

        class QATJointEpochRunner(original_joint_runner):
            def run_epoch(self, *args: Any, **kwargs: Any) -> Any:
                epoch = kwargs.get("epoch", args[0] if args else None)
                if epoch is None:
                    raise TypeError("QAT JointEpochRunner requires epoch")
                state["epoch"] = int(epoch)
                try:
                    return super().run_epoch(*args, **kwargs)
                finally:
                    state["epoch"] = None

        class QATPoseEpochRunner(original_pose_runner):
            def run_epoch(self, *args: Any, **kwargs: Any) -> Any:
                epoch = kwargs.get("epoch", args[0] if args else None)
                if epoch is None:
                    raise TypeError("QAT PoseEpochRunner requires epoch")
                state["epoch"] = int(epoch)
                try:
                    return super().run_epoch(*args, **kwargs)
                finally:
                    state["epoch"] = None

        class QATAccuracyGate(Map50AccuracyGate):
            def __init__(
                self,
                baseline: dict[str, float],
                *,
                maximum_drop: float,
            ) -> None:
                super().__init__(
                    baseline,
                    maximum_drop=maximum_drop,
                    maximum_map50_95_drop=plan.gates.map50_95_max_drop,
                    matched_baseline=(matched_reference if arm == "sham" else None),
                    matched_max_absolute_drift=(
                        plan.gates.matched_sham_max_absolute_drift
                        if arm == "sham"
                        else None
                    ),
                )

        def qat_validator(source: Any, *args: Any, **kwargs: Any) -> Any:
            if args:
                raise TypeError("QAT JointValidator requires keyword settings")
            handles = state.get("handles")
            if not isinstance(handles, _QATGraphHandles):
                raise TypeError("QAT validator constructed before graph")
            return QATJointValidatorAdapter(
                source,
                activation_policy=handles.activation,
                weight_policy=handles.weights,
                expected_catalog=plan.expected_deployment_catalog,
                weight_blend_ratio=runtime.weight_blend_ratio(arm),
                upstream_validator_cls=original_validator,
                **kwargs,
            )

        def prepare_search_view(registry: str | Path, destination: str | Path) -> Any:
            if Path(registry).expanduser().resolve() != plan.data.registry:
                raise ValueError("QAT BBAT5 registry differs from the pinned plan")
            return prepare_bbat5_search_view(plan.data.pose_search, destination)

        def validate_search_pose_source(
            data_yaml: str | Path,
            *,
            registry: str | Path,
        ) -> Any:
            resolved_registry = Path(registry).expanduser().resolve()
            if resolved_registry != plan.data.registry:
                raise ValueError("QAT Pose runtime View registry differs from the plan")
            path = Path(data_yaml).expanduser().resolve()
            expected_root = (run_dir / "datasets" / "bbat5-v1-runtime").resolve()
            if path != expected_root / "data.yaml":
                raise ValueError(
                    "QAT Pose loader may only use its pinned search runtime View"
                )
            prepared = prepare_bbat5_search_view(plan.data.pose_search, expected_root)
            if prepared.yaml != path or (
                prepared.train_images,
                prepared.val_images,
            ) != (5364, 600):
                raise ValueError("QAT BBAT5 search runtime View identity drifted")
            return joint_data_module.CanonicalPoseSourceReport(
                dataset_id="bbat5-v1",
                source_kind="runtime_view",
                yaml=path,
                registry=resolved_registry,
                train_images=prepared.train_images,
                val_images=prepared.val_images,
                kpt_shape=(2, 3),
                flip_idx=(0, 1),
            )

        def stable_bbox_iou(
            box1: Tensor,
            box2: Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            with torch.autocast(device_type=box1.device.type, enabled=False):
                return original_bbox_iou(box1.float(), box2.float(), *args, **kwargs)

        original_session = formal_module.FormalJointTrainingSession

        class QATFormalSession(original_session):
            def _save_selected(
                self,
                labels: tuple[str, ...],
                **kwargs: Any,
            ) -> dict[str, Path]:
                outputs = super()._save_selected(labels, **kwargs)
                handles = state.get("handles")
                if not isinstance(handles, _QATGraphHandles):
                    raise TypeError("QAT checkpoint save has no graph handles")
                ema = kwargs["ema"]
                progress = kwargs["progress"]
                metrics = kwargs["metrics"]
                view = build_qat_deployment_view(
                    shared_ema=ema.ema,
                    source=handles.source,
                    activation_policy=handles.activation,
                    weight_policy=handles.weights,
                    expected_catalog=plan.expected_deployment_catalog,
                    weight_blend_ratio=runtime.weight_blend_ratio(arm),
                )
                for label, checkpoint in outputs.items():
                    _save_qat_deployment_inference_weights(
                        resume_module,
                        self.run_dir / "inference" / f"{label}.pt",
                        deployment_model=view.model,
                        contract_source=ema.ema,
                        metadata={
                            "stage": progress.stage,
                            "epoch": progress.next_epoch - 1,
                            "global_macro_step": progress.global_macro_step,
                            "metrics": dict(metrics),
                            "full_resume_sha256": _sha256(checkpoint),
                            "qat_plan_sha256": plan.config_sha256,
                            "qat_arm": arm,
                            "weight_blend_ratio": runtime.weight_blend_ratio(arm),
                            "loader": "yolo_quantize.qat_runtime",
                        },
                    )
                del view
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                return outputs

        impl.JOINT_STAGES = stages
        joint_config_module.JOINT_STAGES = stages
        stage_policy_module.JOINT_STAGES = stages
        impl.FusionModelFactory = PolicyFactory
        impl.JointValidator = qat_validator
        impl.AccuracyGate = QATAccuracyGate
        impl.CheckpointSelectors = Map50CheckpointSelectors
        impl.HardwareContractGuard = BNFoldedHardwareContractGuard
        impl.apply_stage = qat_apply_stage
        impl.build_joint_optimizer = qat_build_optimizer
        impl.JointEpochRunner = QATJointEpochRunner
        impl.PoseEpochRunner = QATPoseEpochRunner
        formal_module.prepare_bbt5_view = prepare_search_view
        joint_data_module.validate_canonical_pose_source = validate_search_pose_source
        loss_module.bbox_iou = stable_bbox_iou
        try:
            yield QATFormalSession
        finally:
            impl.JOINT_STAGES = original_stages
            joint_config_module.JOINT_STAGES = original_joint_stages
            stage_policy_module.JOINT_STAGES = original_policy_stages
            impl.FusionModelFactory = original_factory
            impl.JointValidator = original_validator
            impl.AccuracyGate = original_gate
            impl.CheckpointSelectors = original_selectors
            impl.HardwareContractGuard = original_guard
            impl.apply_stage = original_apply_stage
            impl.build_joint_optimizer = original_build_optimizer
            impl.JointEpochRunner = original_joint_runner
            impl.PoseEpochRunner = original_pose_runner
            formal_module.prepare_bbt5_view = original_prepare_view
            joint_data_module.validate_canonical_pose_source = original_pose_validator
            loss_module.bbox_iou = original_bbox_iou

    def run(
        self,
        arm: QATArm,
        *,
        device_index: int = 0,
        resume: str | Path | None = None,
    ) -> Any:
        self.weight_blend_ratio(arm)
        if self.patience_continuation is not None and arm != "qat":
            raise ValueError("patience continuation is valid only for the QAT arm")
        if self.patience_continuation is not None:
            source = self.patience_continuation.resume_checkpoint
            if resume is not None and Path(resume).expanduser().resolve() != source:
                raise ValueError(
                    "--resume differs from the patience continuation checkpoint"
                )
            resume = source
        if arm == "qat" and self.plan.external_control is not None:
            matched_sham_evidence = self._external_control_evidence()
        else:
            matched_sham_evidence = self._assert_sham_ready() if arm == "qat" else None
        if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device_index} is unavailable")
        preflight = self.preflight(verify_graph=False, verify_sample_files=True)
        if not preflight.ready:
            raise RuntimeError("QAT preflight failed: " + "; ".join(preflight.blockers))
        modules = self._imports()
        config = self.training_config(arm)
        run_name = self.run_name(arm)
        device = torch.device(f"cuda:{device_index}")
        continuation_evidence: dict[str, Any] | None = None
        with self._patched_training(
            modules,
            arm=arm,
            run_name=run_name,
            device=device,
        ) as session_cls:
            if self.patience_continuation is not None:
                destination = (
                    self.plan.run_root
                    / run_name
                    / "checkpoints"
                    / (
                        f"resume-patience{self.effective_patience}"
                        f"-epoch{self.patience_continuation.boundary_next_epoch:04d}.pt"
                    )
                )
                continuation_evidence = (
                    self.patience_continuation.prepare_resume_checkpoint(
                        destination,
                        expected_resolved_config=config.as_dict(),
                    )
                )
                _atomic_json(
                    self.plan.run_root / run_name / "patience-continuation.json",
                    continuation_evidence,
                )
                resume = destination
            report = session_cls(
                config,
                device=str(device_index),
                run_name=run_name,
                detect_microbatch_size=self.plan.training.detect_microbatch,
            ).run(resume=resume, enable_j3=True)
        checkpoint_paths = _checkpoint_paths(report.run_dir, report.checkpoint_paths)
        checkpoint_sha256 = {
            name: _sha256(path) for name, path in checkpoint_paths.items()
        }
        early_stop_event = _latest_log_event(
            report.run_dir / "logs" / "events.jsonl", "early_stop"
        )
        _atomic_json(
            report.run_dir / "qat-experiment.json",
            {
                "schema_version": 1,
                "plan": str(self.plan.config_path),
                "plan_sha256": self.plan.config_sha256,
                "candidate_id": self.plan.candidate_id,
                "authorization_id": self.plan.execution_authorization_id,
                "arm": arm,
                "run_name": run_name,
                "completed_stages": list(report.completed_stages),
                "epochs_completed": report.epochs_completed,
                "epochs_planned": self.plan.training.epochs,
                "effective_patience": self.effective_patience,
                "global_macro_steps": report.global_macro_steps,
                "checkpoint_paths": {
                    name: str(path) for name, path in checkpoint_paths.items()
                },
                "checkpoint_sha256": checkpoint_sha256,
                "matched_sham_evidence": matched_sham_evidence,
                "patience_continuation": continuation_evidence,
                "early_stop": early_stop_event,
                "formal_validation": False,
            },
        )
        return report

    def gpu_smoke(self, *, device_index: int = 0) -> dict[str, Any]:
        """Exercise calibrated full-strength QAT forward/backward before training."""

        if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device_index} is unavailable")
        preflight = self.preflight(verify_graph=False, verify_sample_files=True)
        if not preflight.ready:
            raise RuntimeError("QAT preflight failed: " + "; ".join(preflight.blockers))
        torch.manual_seed(self.plan.training.seed)
        torch.cuda.manual_seed_all(self.plan.training.seed)
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        handles: _QATGraphHandles | None = None
        optimizer: torch.optim.Optimizer | None = None
        try:
            handles = self._build_graph(device=device, calibrate=True)
            modules = self._imports()
            stage = replace(
                modules["formal_impl"].JOINT_STAGES["j3"],
                epochs=self.plan.training.epochs,
                patience=self.effective_patience,
                warmup_epochs=self.plan.training.warmup_epochs,
                learning_rates=MappingProxyType(
                    dict(self.plan.optimizer.learning_rates)
                ),
            )
            modules["stage_policy"].apply_stage(handles.model, stage)
            optimizer, _ = modules["stage_policy"].build_joint_optimizer(
                handles.model,
                stage,
                optimizer_name=self.plan.optimizer.name,
                weight_decay=self.plan.optimizer.weight_decay,
                beta1=self.plan.optimizer.beta1,
                beta2=self.plan.optimizer.beta2,
            )
            optimizer_split = split_quantizer_parameter_groups(
                optimizer,
                qparam_lr_ratio=self.plan.optimizer.qparam_lr_ratio,
            )
            quantization = QuantizationEpochController(
                handles.weights,
                self.plan.training.progressive_schedule,
            ).begin_epoch(self.plan.training.progressive_full_epoch)
            trainability = QATTrainabilityController(
                scale_only_epochs=self.plan.training.scale_only_epochs
            ).apply(handles.model, epoch=0)
            manifest = DiagnosticManifest.from_json(
                self.plan.data.diagnostic_manifest,
                verify_files=True,
            )
            task_contract = {
                "detect": (
                    "coco_detect",
                    self.plan.training.detect_microbatch,
                ),
                "pose": ("bbat_pose", self.plan.training.pose_batch),
            }
            handles.model.train()
            optimizer.zero_grad(set_to_none=True)
            task_reports: dict[str, Any] = {}
            for task, (manifest_task, batch_size) in task_contract.items():
                paths = manifest.paths("calibration", manifest_task)
                if len(paths) < batch_size:
                    raise RuntimeError(
                        f"QAT smoke {task} needs {batch_size} samples, got {len(paths)}"
                    )
                batch = torch.cat(
                    tuple(_letterbox(path, 640) for path in paths[:batch_size]),
                    dim=0,
                ).to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = handles.model(batch, task=task)
                    objective, output_tensors = _tensor_objective(outputs)
                if not bool(torch.isfinite(objective.detach())):
                    raise RuntimeError(f"QAT smoke {task} objective is not finite")
                objective.backward()
                task_reports[task] = {
                    "batch": batch_size,
                    "output_tensors": output_tensors,
                    "objective": float(objective.detach().float().cpu()),
                }
                del batch, outputs, objective

            quantizer_parameters = tuple(
                (name, parameter)
                for name, parameter in handles.model.named_parameters()
                if parameter.requires_grad and is_quantizer_parameter(name)
            )
            gradients = tuple(
                (name, parameter.grad)
                for name, parameter in quantizer_parameters
                if parameter.grad is not None
            )
            if not gradients:
                raise RuntimeError("QAT GPU smoke produced no quantizer gradients")
            nonfinite_gradients = tuple(
                name
                for name, gradient in gradients
                if not bool(torch.isfinite(gradient).all())
            )
            if nonfinite_gradients:
                raise RuntimeError(
                    "QAT GPU smoke produced non-finite quantizer gradients: "
                    + ", ".join(nonfinite_gradients[:20])
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                tuple(parameter for _, parameter in quantizer_parameters),
                self.plan.training.gradient_clip_norm,
            )
            optimizer.step()
            nonfinite_parameters = tuple(
                name
                for name, parameter in quantizer_parameters
                if not bool(torch.isfinite(parameter.detach()).all())
            )
            if nonfinite_parameters:
                raise RuntimeError(
                    "QAT GPU smoke optimizer step produced non-finite qparams: "
                    + ", ".join(nonfinite_parameters[:20])
                )
            optimizer.zero_grad(set_to_none=True)
            deployment = build_qat_deployment_view(
                shared_ema=handles.model,
                source=handles.source,
                activation_policy=handles.activation,
                weight_policy=handles.weights,
                expected_catalog=self.plan.expected_deployment_catalog,
                weight_blend_ratio=1.0,
            )
            del deployment
            torch.cuda.synchronize(device)
            return {
                "schema_version": 1,
                "kind": "full35_qat_gpu_smoke",
                "status": "passed",
                "gpu_used": True,
                "formal_training": False,
                "formal_validation": False,
                "plan": str(self.plan.config_path),
                "plan_sha256": self.plan.config_sha256,
                "candidate_id": self.plan.candidate_id,
                "device": {
                    "index": device_index,
                    "name": torch.cuda.get_device_name(device_index),
                },
                "calibration": (
                    handles.calibration.to_dict()
                    if handles.calibration is not None
                    else None
                ),
                "quantization": quantization.to_dict(),
                "trainability": trainability.to_dict(),
                "optimizer_split": optimizer_split.to_dict(),
                "tasks": task_reports,
                "quantizer_parameters": len(quantizer_parameters),
                "quantizer_gradients": len(gradients),
                "gradient_norm_before_clip": float(
                    gradient_norm.detach().float().cpu()
                ),
                "deployment_materialized": True,
                "peak_gpu_memory_mib": (
                    torch.cuda.max_memory_allocated(device) / 2**20
                ),
                "seconds": time.perf_counter() - started,
            }
        finally:
            if optimizer is not None:
                del optimizer
            if handles is not None:
                del handles
            gc.collect()
            torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired Full35 weight-QAT runtime")
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"),
    )
    parser.add_argument("--arm", choices=("sham", "qat"), default="qat")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--patience-continuation", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--reconcile-matched-sham", action="store_true")
    parser.add_argument("--gpu-smoke-only", action="store_true")
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=Path("artifacts/reports/v19-poly-shift-all-w8-qat-preflight-v1.json"),
    )
    parser.add_argument(
        "--gpu-smoke-output",
        type=Path,
        default=Path("artifacts/reports/v19-poly-shift-all-w8-qat-gpu-smoke-v1.json"),
    )
    parser.add_argument("--execute-reviewed-plan", action="store_true")
    args = parser.parse_args(argv)
    runtime = Full35QATRuntime.from_yaml(
        args.plan,
        patience_continuation=args.patience_continuation,
    )
    if args.reconcile_matched_sham:
        payload = runtime.reconcile_matched_sham()
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    if args.preflight_only:
        report = runtime.preflight(verify_graph=True, verify_sample_files=True)
        _atomic_json(args.preflight_output.resolve(), report.to_dict())
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0 if report.ready else 1
    if args.gpu_smoke_only:
        if not args.execute_reviewed_plan:
            parser.error("GPU smoke requires --execute-reviewed-plan")
        payload = runtime.gpu_smoke(device_index=args.device)
        _atomic_json(args.gpu_smoke_output.resolve(), payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "plan_sha256": payload["plan_sha256"],
                    "peak_gpu_memory_mib": payload["peak_gpu_memory_mib"],
                    "quantizer_gradients": payload["quantizer_gradients"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if not args.execute_reviewed_plan:
        parser.error("execution requires --execute-reviewed-plan")
    runtime.run(args.arm, device_index=args.device, resume=args.resume)
    return 0


__all__ = (
    "Full35QATRuntime",
    "MatchedShamEvidenceError",
    "QATRuntimePreflightReport",
    "main",
)


if __name__ == "__main__":
    raise SystemExit(main())
