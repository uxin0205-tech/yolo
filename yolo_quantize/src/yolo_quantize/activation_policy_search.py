"""Candidate-only Full35 activation-policy search under the active mAP50 gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections.abc import Sequence
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


@dataclass(frozen=True)
class ActivationPolicyCandidate:
    """One activation graph, recovery checkpoint, and analytic hardware order."""

    candidate_id: str
    policy: Full35ActivationPolicy
    checkpoint: Path
    checkpoint_sha256: str
    analytic_hardware_rank: int

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("activation candidate_id must not be empty")
        if self.analytic_hardware_rank < 0:
            raise ValueError("analytic hardware rank must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "policy_id": self.policy.policy_id,
            "default_activation": self.policy.activation,
            "region_assignments": [
                {"region": region, "activation": activation}
                for region, activation in self.policy.region_assignments
            ],
            "activation_bits": self.policy.bits,
            "signed_codes": self.policy.signed_codes,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "analytic_hardware_rank": self.analytic_hardware_rank,
        }


@dataclass(frozen=True)
class Full35ActivationPolicySearchPlan:
    """Reviewed Q3-derived activation search, using search-val rather than formal."""

    config_path: Path
    config_sha256: str
    plan_id: str
    execution_authorization_id: str
    metric_contract_id: str
    base_plan: Full35SearchValidationPlan
    reference_report: Path
    reference_report_sha256: str
    reference_regate: Path
    q3_evidence_path: Path
    q3_evidence_sha256: str
    candidates: tuple[ActivationPolicyCandidate, ...]
    gate_spec: Full35MetricGateSpec
    accuracy_tolerance: float
    formal_training: bool = False
    formal_validation: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> Full35ActivationPolicySearchPlan:
        config_path = Path(path).expanduser().resolve()
        payload = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "activation search plan",
        )
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported activation search plan schema")
        if payload.get("execution_authorized") is not True:
            raise ValueError("activation search execution is not authorized")
        if payload.get("formal_training") is not False:
            raise ValueError("activation search cannot authorize training")
        if payload.get("formal_validation") is not False:
            raise ValueError("activation search cannot use formal validation")
        authorization = _mapping(
            payload.get("execution_authorization"), "activation authorization"
        )
        authorization_id = str(authorization.get("authorization_id", "")).strip()
        if not authorization_id or authorization.get("validation_roles") != [
            "candidate"
        ]:
            raise ValueError("activation execution authorization is invalid")

        upstream = _mapping(payload.get("upstream"), "activation upstream")
        base_path, _ = _verified_file(
            upstream.get("base_search_plan"), "activation base search plan"
        )
        base_plan = Full35SearchValidationPlan.from_yaml(base_path)
        reference = _mapping(
            upstream.get("reference_search_report"), "activation reference report"
        )
        if set(reference) != {"path", "sha256", "reuse_roles"}:
            raise ValueError("activation reference report fields differ")
        if reference.get("reuse_roles") != ["accepted", "matched"]:
            raise ValueError("activation reference must reuse accepted and qSiLU")
        reference_report = _resolve(reference["path"])
        reference_sha256 = str(reference["sha256"])
        if _sha256(reference_report) != reference_sha256:
            raise ValueError("activation reference report SHA-256 drifted")
        reference_regate, _ = _verified_file(
            upstream.get("reference_map50_regate"),
            "activation reference map50 re-gate",
        )
        q3 = _mapping(upstream.get("q3_evidence"), "Q3 evidence")
        if set(q3) != {"path", "sha256", "use"}:
            raise ValueError("Q3 evidence fields differ")
        if q3.get("use") != "candidate_order_and_rationale_only":
            raise ValueError("Q3 evidence cannot be reused as current mAP50 metrics")
        q3_path = _resolve(q3["path"])
        q3_sha256 = str(q3["sha256"])
        if _sha256(q3_path) != q3_sha256:
            raise ValueError("Q3 evidence SHA-256 drifted")

        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("activation search candidates must be non-empty")
        candidates: list[ActivationPolicyCandidate] = []
        for index, raw in enumerate(raw_candidates):
            item = _mapping(raw, f"activation candidate {index}")
            expected_fields = {
                "candidate_id",
                "default_activation",
                "region_assignments",
                "bits",
                "signed_codes",
                "policy_id",
                "checkpoint",
                "analytic_hardware_rank",
            }
            if set(item) != expected_fields:
                raise ValueError(f"activation candidate {index} fields differ")
            assignments = item["region_assignments"]
            if not isinstance(assignments, dict):
                raise TypeError("activation region_assignments must be a mapping")
            policy = Full35ActivationPolicy(
                activation=str(item["default_activation"]),
                bits=int(item["bits"]),
                signed_codes=bool(item["signed_codes"]),
                region_assignments=tuple(
                    (str(region), str(activation))
                    for region, activation in assignments.items()
                ),
            )
            if policy.policy_id != item["policy_id"]:
                raise ValueError(f"activation candidate {index} policy_id is not canonical")
            checkpoint, checkpoint_sha256 = _verified_file(
                item["checkpoint"], f"activation candidate {index} checkpoint"
            )
            candidates.append(
                ActivationPolicyCandidate(
                    candidate_id=str(item["candidate_id"]),
                    policy=policy,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha256,
                    analytic_hardware_rank=int(item["analytic_hardware_rank"]),
                )
            )
        authorized_ids = tuple(
            str(value) for value in authorization.get("candidate_ids", ())
        )
        if authorized_ids != tuple(item.candidate_id for item in candidates):
            raise ValueError("activation candidate order differs from authorization")

        gates = _mapping(payload.get("gates"), "activation gates")
        if gates.get("all_eight_metrics_required") is not True:
            raise ValueError("activation gate must require all eight metrics")
        if gates.get("activation_replacement_included") is not True:
            raise ValueError("activation gate must include activation replacement")
        gate_spec = Full35MetricGateSpec(
            metric_family=str(gates["metric_family"]),  # type: ignore[arg-type]
            total_max_drop=float(gates["total_max_drop"]),
            w8_incremental_max_drop=float(gates["w8_incremental_max_drop"]),
            sham_max_absolute_drift=float(gates["sham_max_absolute_drift"]),
            recovery_floor=float(gates["recovery_floor"]),
        )
        if gate_spec.metric_family != "map50" or gate_spec.total_max_drop != 0.015:
            raise ValueError("activation search must use total mAP50 drop 0.015")
        plan_id = str(payload.get("plan_id", "")).strip()
        metric_contract_id = str(payload.get("metric_contract_id", "")).strip()
        if not plan_id or not metric_contract_id:
            raise ValueError("activation plan and metric contract IDs are required")
        selection = _mapping(payload.get("selection"), "activation selection")
        if selection.get("analytic_hardware_rank_is_not_measured_latency") is not True:
            raise ValueError("activation hardware rank evidence boundary is required")
        return cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            plan_id=plan_id,
            execution_authorization_id=authorization_id,
            metric_contract_id=metric_contract_id,
            base_plan=base_plan,
            reference_report=reference_report,
            reference_report_sha256=reference_sha256,
            reference_regate=reference_regate,
            q3_evidence_path=q3_path,
            q3_evidence_sha256=q3_sha256,
            candidates=tuple(candidates),
            gate_spec=gate_spec,
            accuracy_tolerance=float(selection["accuracy_tolerance"]),
        )

    @property
    def weight_study(self) -> Any:
        return self.base_plan.weight_study

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan": str(self.config_path),
            "plan_sha256": self.config_sha256,
            "execution_authorization_id": self.execution_authorization_id,
            "metric_contract_id": self.metric_contract_id,
            "base_plan": str(self.base_plan.config_path),
            "base_plan_sha256": self.base_plan.config_sha256,
            "reference_report": str(self.reference_report),
            "reference_report_sha256": self.reference_report_sha256,
            "reference_regate": str(self.reference_regate),
            "q3_evidence": {
                "path": str(self.q3_evidence_path),
                "sha256": self.q3_evidence_sha256,
                "use": "candidate_order_and_rationale_only",
            },
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
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


def _activation_selection(
    rows: list[dict[str, Any]], accuracy_tolerance: float
) -> dict[str, Any]:
    eligible = [row for row in rows if row["gate_passed"]]
    if not eligible:
        roles = {"accuracy": None, "balanced": None, "hardware": None}
    else:
        accuracy = min(
            eligible,
            key=lambda row: (-row["worst_total_delta"], row["candidate_id"]),
        )
        accuracy_band = [
            row
            for row in eligible
            if row["worst_total_delta"]
            >= accuracy["worst_total_delta"] - accuracy_tolerance
        ]
        balanced = min(
            accuracy_band,
            key=lambda row: (
                row["analytic_hardware_rank"],
                -row["worst_total_delta"],
                row["candidate_id"],
            ),
        )
        hardware = min(
            eligible,
            key=lambda row: (
                row["analytic_hardware_rank"],
                -row["worst_total_delta"],
                row["candidate_id"],
            ),
        )
        roles = {
            "accuracy": accuracy["candidate_id"],
            "balanced": balanced["candidate_id"],
            "hardware": hardware["candidate_id"],
        }
    return {
        "total_max_drop": 0.015,
        "accuracy_tolerance": accuracy_tolerance,
        "eligible_ids": [row["candidate_id"] for row in eligible],
        "rejected_ids": [row["candidate_id"] for row in rows if not row["gate_passed"]],
        "roles": roles,
        "analytic_hardware_rank_is_not_measured_latency": True,
    }


def run_activation_policy_search(
    *,
    plan: Full35ActivationPolicySearchPlan,
    output: Path,
    device_index: int,
    resume: bool,
) -> int:
    prepared, base_contract = _prepare_contract(plan, device_index=device_index)  # type: ignore[arg-type]
    contract = {
        **base_contract,
        "kind": "full35_activation_policy_map50_search",
        "reference_reuse": {
            "report": str(plan.reference_report),
            "sha256": plan.reference_report_sha256,
            "roles": ["accepted", "matched_qsilu_comparison"],
            "validation_rerun": False,
        },
        "q3_use": "candidate_order_and_rationale_only_not_metric_reuse",
        "weight_quantization": "disabled",
    }
    output_path = output.expanduser().resolve()
    validation_root = PROJECT_ROOT / "artifacts/runs" / plan.plan_id / "validation"
    results: dict[str, Any] = {}
    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"output already exists; use --resume or a new path: {output_path}"
            )
        existing = _mapping(
            json.loads(output_path.read_text(encoding="utf-8")),
            "existing activation report",
        )
        if existing.get("contract") != contract:
            raise RuntimeError("existing activation contract differs; refusing resume")
        if isinstance(existing.get("results"), dict):
            results.update(existing["results"])
    elif validation_root.exists():
        raise FileExistsError(
            "activation validation artifacts exist without report; refusing overwrite"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "full35_activation_policy_map50_search",
        "status": "running",
        "contract": contract,
        "results": results,
        "selection": None,
        "formal_training": False,
        "formal_validation": False,
    }
    _atomic_json(output_path, payload)

    regated = regate_search_report(
        source_report=plan.reference_report,
        expected_source_sha256=plan.reference_report_sha256,
        metric_contract_id=plan.metric_contract_id,
        spec=plan.gate_spec,
    )
    pinned_regate = _mapping(
        json.loads(plan.reference_regate.read_text(encoding="utf-8")),
        "activation pinned map50 re-gate",
    )
    if pinned_regate.get("gate") != regated.get("gate"):
        raise RuntimeError("fresh activation reference re-gate differs from pin")
    roles = _mapping(regated.get("roles"), "activation reference roles")
    accepted_role = _mapping(roles.get("accepted"), "accepted role")
    qsilu_role = _mapping(roles.get("matched"), "matched qSiLU role")
    accepted = Full35MetricSnapshot(
        run_id=f"{plan.plan_id}:accepted-reused",
        policy_id=str(accepted_role["policy_id"]),
        metric_contract_id=plan.metric_contract_id,
        metrics=accepted_role["metrics"],
    )
    qsilu_matched = Full35MetricSnapshot(
        run_id=f"{plan.plan_id}:qsilu-reused",
        policy_id=str(qsilu_role["policy_id"]),
        metric_contract_id=plan.metric_contract_id,
        metrics=qsilu_role["metrics"],
    )
    qsilu_gate = Full35MetricGate(plan.gate_spec).evaluate(
        Full35MetricCandidate(
            run_id=f"{plan.plan_id}:qsilu-comparison",
            format_id="activation-a8",
            stage="ptq",
            policy_id=qsilu_matched.policy_id,
            metric_contract_id=plan.metric_contract_id,
            metrics=qsilu_matched.metrics,
        ),
        accepted,
        qsilu_matched,
    )

    manifest = plan.weight_study.load_diagnostic_manifest(verify_files=True)
    probe_paths = {
        "detect": manifest.paths("probe", "coco_detect")[0],
        "pose": manifest.paths("probe", "bbat_pose")[0],
    }
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(device_index)
    for index, candidate in enumerate(plan.candidates, start=1):
        if results.get(candidate.candidate_id, {}).get("status") == "completed":
            continue
        payload["current_candidate"] = candidate.candidate_id
        _atomic_json(output_path, payload)
        print(
            f"[{index}/{len(plan.candidates)}] activation {candidate.candidate_id}",
            flush=True,
        )
        model = source = applied = build = calibration = None
        try:
            model, source, applied, build = _build_deployment_policy(
                activation=candidate.policy.activation,
                checkpoint=candidate.checkpoint,
                checkpoint_sha256=candidate.checkpoint_sha256,
                expected_catalog=plan.weight_study.expected_catalog,
                activation_region_assignments=candidate.policy.region_assignments,
            )
            if build["activation_policy_id"] != candidate.policy.policy_id:
                raise RuntimeError("built activation policy differs from reviewed ID")
            calibration = _calibrate_activation_a8(
                plan=plan,  # type: ignore[arg-type]
                model=model,
                applied=applied,
                device_index=device_index,
            )
            with torch.inference_mode():
                probe_outputs = {
                    task: model(
                        _letterbox(path, plan.image_size).cuda(device_index),
                        task=task,
                    )
                    for task, path in probe_paths.items()
                }
            diagnostics = {
                task: {
                    **_compare_outputs(value, value),
                    "deployment": _deployment_comparison(value, value, task=task),
                }
                for task, value in probe_outputs.items()
            }
            if not all(
                item["all_finite"] and item["same_structure"]
                for item in diagnostics.values()
            ):
                raise RuntimeError("activation fixed-probe diagnostic failed")
            record = _validate_role(
                role=candidate.candidate_id,
                model=model,
                source=source,
                plan=plan,  # type: ignore[arg-type]
                pose_yaml=prepared.yaml,
                output_root=validation_root,
                device_index=device_index,
            )
            matched = Full35MetricSnapshot(
                run_id=f"{plan.plan_id}:{candidate.candidate_id}:matched",
                policy_id=candidate.policy.policy_id,
                metric_contract_id=plan.metric_contract_id,
                metrics=record["metrics"],
            )
            metric_candidate = Full35MetricCandidate(
                run_id=f"{plan.plan_id}:{candidate.candidate_id}",
                format_id="activation-a8",
                stage="ptq",
                policy_id=candidate.policy.policy_id,
                metric_contract_id=plan.metric_contract_id,
                metrics=record["metrics"],
            )
            gate = Full35MetricGate(plan.gate_spec).evaluate(
                metric_candidate, accepted, matched
            )
            record.update(
                {
                    "status": "completed",
                    "candidate": candidate.to_dict(),
                    "activation_output_quantization": "calibrated_lsq_plus_a8",
                    "weight_quantization": "disabled",
                    "fixed_probe": diagnostics,
                    "build": build,
                    "calibration": calibration,
                    "gate": gate.to_dict(),
                }
            )
            results[candidate.candidate_id] = record
            payload["results"] = results
            _atomic_json(output_path, payload)
        except Exception as error:
            results[candidate.candidate_id] = {
                "status": "failed",
                "candidate": candidate.to_dict(),
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
            payload["results"] = results
            _atomic_json(output_path, payload)
            raise
        finally:
            del model, source, applied, build, calibration
            gc.collect()
            torch.cuda.empty_cache()

    rows = [
        {
            "candidate_id": "uniform-qsilu-comparison",
            "policy_id": qsilu_matched.policy_id,
            "worst_total_delta": qsilu_gate.worst_total_delta,
            "gate_passed": qsilu_gate.passed,
            "analytic_hardware_rank": 100,
            "source": "pinned_reference_matched_role",
        }
    ]
    for candidate in plan.candidates:
        result = _mapping(results[candidate.candidate_id], "activation result")
        if result.get("status") != "completed":
            continue
        gate = _mapping(result.get("gate"), "activation result gate")
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "policy_id": candidate.policy.policy_id,
                "worst_total_delta": float(gate["worst_total_delta"]),
                "gate_passed": bool(gate["passed"]),
                "analytic_hardware_rank": candidate.analytic_hardware_rank,
                "source": "current_gpu_search_val",
            }
        )
    payload.pop("current_candidate", None)
    payload["candidates"] = rows
    payload["selection"] = _activation_selection(rows, plan.accuracy_tolerance)
    payload["status"] = "completed"
    _atomic_json(output_path, payload)
    print(json.dumps(payload["selection"], ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full35 reviewed activation-policy mAP50 search"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/v12-activation-policy-map50-search-v1.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/reports/v12-activation-policy-map50-search-v1.json",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--execute-reviewed-plan", action="store_true")
    args = parser.parse_args(argv)
    plan = Full35ActivationPolicySearchPlan.from_yaml(args.plan)
    if args.list_only:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.execute_reviewed_plan:
        parser.error("execution requires --execute-reviewed-plan")
    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        parser.error(f"CUDA device {args.device} is unavailable")
    return run_activation_policy_search(
        plan=plan,
        output=args.output,
        device_index=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
