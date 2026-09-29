"""Resumable candidate-only Full35 search race with verified reference reuse."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .activation_smoke import _atomic_json
from .balance_selection import BalanceCandidate, QuantizationBalanceSelector
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
    _sha256,
    _validate_role,
)
from .weight_quantization import (
    Full35WeightRegionCatalog,
    UniformWeightSpec,
    WeightQuantizationAdapter,
)
from .weight_sensitivity import WeightSensitivityCell, WeightSensitivityStudy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    PROJECT_ROOT / "configs/experiments/v5-qsilu-backbone-early-bit-search-v1.yaml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/reports/v5-qsilu-backbone-early-bit-search-v1.json"
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _resolve(config_path: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _verified_file(config_path: Path, record: object, label: str) -> Path:
    payload = _mapping(record, label)
    if set(payload) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = _resolve(config_path, payload["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    expected = str(payload["sha256"])
    if actual != expected:
        raise ValueError(f"{label} SHA-256 drifted: {actual} != {expected}")
    return path


@dataclass(frozen=True)
class Full35SearchRacePlan:
    """One reviewed multi-candidate race over a shared immutable reference."""

    config_path: Path
    config_sha256: str
    plan_id: str
    execution_authorization_id: str
    metric_contract_id: str
    base_plan: Full35SearchValidationPlan
    weight_study: WeightSensitivityStudy
    cells: tuple[WeightSensitivityCell, ...]
    diagnostic_reports: tuple[Path, ...]
    reference_report: Path
    reference_report_sha256: str
    reference_regate: Path
    gate_spec: Full35MetricGateSpec
    accuracy_tolerance: float
    formal_training: bool = False
    formal_validation: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> Full35SearchRacePlan:
        config_path = Path(path).expanduser().resolve()
        payload = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "search race plan",
        )
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported search race plan schema")
        if payload.get("execution_authorized") is not True:
            raise ValueError("search race execution is not authorized")
        if payload.get("formal_training") is not False:
            raise ValueError("search race cannot authorize training")
        if payload.get("formal_validation") is not False:
            raise ValueError("search race cannot use formal validation")

        authorization = _mapping(
            payload.get("execution_authorization"), "execution authorization"
        )
        authorization_id = str(authorization.get("authorization_id", "")).strip()
        if not authorization_id:
            raise ValueError("search race authorization_id is empty")
        if authorization.get("validation_roles") != ["candidate"]:
            raise ValueError("search race may validate candidate roles only")

        upstream = _mapping(payload.get("upstream"), "upstream")
        base_path = _verified_file(
            config_path,
            upstream.get("base_search_plan"),
            "base search plan",
        )
        base_plan = Full35SearchValidationPlan.from_yaml(base_path)
        weight_path = _verified_file(
            config_path,
            upstream.get("weight_plan"),
            "weight plan",
        )
        weight_study = WeightSensitivityStudy.from_yaml(weight_path)
        reference_regate = _verified_file(
            config_path,
            upstream.get("reference_map50_regate"),
            "reference map50 re-gate",
        )

        reference_record = _mapping(
            upstream.get("reference_search_report"), "reference search report"
        )
        allowed_reference_keys = {
            "path",
            "sha256",
            "reuse_roles",
            "old_candidate_role",
        }
        if set(reference_record) != allowed_reference_keys:
            raise ValueError("reference search report fields differ from schema")
        if reference_record.get("reuse_roles") != ["accepted", "matched"]:
            raise ValueError("reference reuse must be accepted plus matched only")
        reference_report = _resolve(config_path, reference_record["path"])
        reference_sha256 = str(reference_record["sha256"])
        if _sha256(reference_report) != reference_sha256:
            raise ValueError("reference search report SHA-256 drifted")

        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("search race candidates must be a non-empty list")
        cells: list[WeightSensitivityCell] = []
        diagnostics: list[Path] = []
        for index, raw_candidate in enumerate(raw_candidates):
            candidate = _mapping(raw_candidate, f"candidate {index}")
            resolved = weight_study.cells(
                activations=(str(candidate["activation"]),),
                regions=(str(candidate["region"]),),
                bits=(int(candidate["weight_bits"]),),
                scale_method=str(candidate["scale_method"]),  # type: ignore[arg-type]
            )
            if len(resolved) != 1 or resolved[0].cell_id != candidate.get("cell_id"):
                raise ValueError(f"candidate {index} does not resolve to one cell")
            weight_study.require_execution_authorized(resolved)
            diagnostic_path = _verified_file(
                config_path,
                candidate.get("diagnostic_report"),
                f"candidate {index} diagnostic report",
            )
            diagnostic = _mapping(
                json.loads(diagnostic_path.read_text(encoding="utf-8")),
                f"candidate {index} diagnostic payload",
            )
            result = _mapping(
                _mapping(diagnostic.get("results"), "diagnostic results").get(
                    resolved[0].cell_id
                ),
                f"candidate {index} diagnostic result",
            )
            contract = _mapping(diagnostic.get("contract"), "diagnostic contract")
            if (
                diagnostic.get("status") != "diagnostic_completed"
                or result.get("status") != "diagnostic_pass"
                or result.get("selection_claim") is not False
                or contract.get("plan_sha256") != weight_study.config_sha256
            ):
                raise ValueError(f"candidate {index} diagnostic evidence is invalid")
            cells.append(resolved[0])
            diagnostics.append(diagnostic_path)

        authorized_cells = tuple(str(item) for item in authorization.get("cells", ()))
        if authorized_cells != tuple(cell.cell_id for cell in cells):
            raise ValueError("candidate order differs from execution authorization")
        parent_ids = {cell.parent.policy_id for cell in cells}
        if parent_ids != {base_plan.matched_policy_id}:
            raise ValueError(
                "search race must share the base matched activation parent"
            )

        gates = _mapping(payload.get("gates"), "gates")
        if gates.get("all_eight_metrics_required") is not True:
            raise ValueError("search race must require all eight metrics")
        if gates.get("activation_replacement_included") is not True:
            raise ValueError(
                "search race total gate must include activation replacement"
            )
        gate_spec = Full35MetricGateSpec(
            metric_family=str(gates["metric_family"]),  # type: ignore[arg-type]
            total_max_drop=float(gates["total_max_drop"]),
            w8_incremental_max_drop=float(gates["w8_incremental_max_drop"]),
            sham_max_absolute_drift=float(gates["sham_max_absolute_drift"]),
            recovery_floor=float(gates["recovery_floor"]),
        )
        if gate_spec.metric_family != "map50" or gate_spec.total_max_drop != 0.015:
            raise ValueError("active search race must use mAP50 total drop 0.015")
        selection = _mapping(payload.get("selection"), "selection")
        accuracy_tolerance = float(selection["accuracy_tolerance"])

        plan_id = str(payload.get("plan_id", "")).strip()
        metric_contract_id = str(payload.get("metric_contract_id", "")).strip()
        if not plan_id or not metric_contract_id:
            raise ValueError("search race plan and metric contract IDs are required")
        return cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            plan_id=plan_id,
            execution_authorization_id=authorization_id,
            metric_contract_id=metric_contract_id,
            base_plan=base_plan,
            weight_study=weight_study,
            cells=tuple(cells),
            diagnostic_reports=tuple(diagnostics),
            reference_report=reference_report,
            reference_report_sha256=reference_sha256,
            reference_regate=reference_regate,
            gate_spec=gate_spec,
            accuracy_tolerance=accuracy_tolerance,
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
        return self.base_plan.matched_policy_id

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
            "reference_report": str(self.reference_report),
            "reference_report_sha256": self.reference_report_sha256,
            "reference_regate": str(self.reference_regate),
            "cells": [cell.to_dict() for cell in self.cells],
            "diagnostic_reports": [str(path) for path in self.diagnostic_reports],
            "candidate_count": len(self.cells),
            "datasets": {
                "coco": str(self.coco_yaml),
                "bbat5_pose_search": str(self.pose_search_yaml),
                "bbat5_registry": str(self.bbat5_registry),
                "runtime_view": str(self.runtime_view),
                "assignment_changed": False,
            },
            "batch": {"detect": self.detect_batch_size, "pose": self.pose_batch_size},
            "workers": {"detect": self.detect_workers, "pose": self.pose_workers},
            "image_size": self.image_size,
            "plots": False,
            "save_coco_json": False,
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


def _candidate_cost(
    *,
    deployment_weight_elements: int,
    region_weight_elements: int,
    numeric: Mapping[str, Any],
) -> tuple[int, int]:
    reference = deployment_weight_elements * 4
    packed = (
        (deployment_weight_elements - region_weight_elements) * 4
        + int(numeric["weight_code_bytes"])
        + int(numeric["scale_bytes"])
    )
    return packed, reference


def run_search_race(
    *,
    plan: Full35SearchRacePlan,
    output: Path,
    device_index: int,
    resume: bool,
) -> int:
    """Validate only new candidates, then gate and Pareto-rank with pinned roles."""

    prepared, base_contract = _prepare_contract(plan, device_index=device_index)  # type: ignore[arg-type]
    contract = {
        **base_contract,
        "kind": "full35_candidate_only_map50_search_race",
        "reference_reuse": {
            "report": str(plan.reference_report),
            "sha256": plan.reference_report_sha256,
            "roles": ["accepted", "matched"],
            "validation_rerun": False,
        },
    }
    regated = regate_search_report(
        source_report=plan.reference_report,
        expected_source_sha256=plan.reference_report_sha256,
        metric_contract_id=plan.metric_contract_id,
        spec=plan.gate_spec,
    )
    pinned_regate = json.loads(plan.reference_regate.read_text(encoding="utf-8"))
    if pinned_regate.get("gate") != regated.get("gate"):
        raise RuntimeError("fresh reference re-gate differs from pinned v5 evidence")

    source_payload = json.loads(plan.reference_report.read_text(encoding="utf-8"))
    expected_calibration = _mapping(
        _mapping(source_payload["roles"], "reference roles")["matched"].get(
            "calibration"
        ),
        "reference matched calibration",
    )
    output_path = output.expanduser().resolve()
    validation_root = PROJECT_ROOT / "artifacts/runs" / plan.plan_id / "validation"
    results: dict[str, Any] = {}
    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"output already exists; use --resume or a new path: {output_path}"
            )
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("contract") != contract:
            raise RuntimeError("existing search race contract differs; refusing resume")
        if isinstance(existing.get("results"), dict):
            results.update(existing["results"])
    elif validation_root.exists():
        raise FileExistsError(
            "validation artifacts exist without the race report; refusing overwrite"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "full35_candidate_only_map50_search_race",
        "status": "running",
        "contract": contract,
        "reference": regated,
        "results": results,
        "selection": None,
        "formal_training": False,
        "formal_validation": False,
    }
    _atomic_json(output_path, payload)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(device_index)
    pending = tuple(
        cell
        for cell in plan.cells
        if results.get(cell.cell_id, {}).get("status") != "completed"
    )
    if pending:
        parent = pending[0].parent
        model, source, applied, build = _build_deployment_policy(
            activation=parent.activation,
            checkpoint=parent.checkpoint,
            checkpoint_sha256=parent.sha256,
            expected_catalog=plan.weight_study.expected_catalog,
        )
        calibration = _calibrate_activation_a8(
            plan=plan,  # type: ignore[arg-type]
            model=model,
            applied=applied,
            device_index=device_index,
        )
        if _calibration_identity(calibration) != _calibration_identity(
            expected_calibration
        ):
            raise RuntimeError(
                "recalibrated activation identity differs from reference"
            )
        catalog = Full35WeightRegionCatalog.inspect(model)
        if catalog.summary() != plan.weight_study.expected_catalog:
            raise RuntimeError("search race catalog differs from reviewed contract")
        accepted_metrics = regated["roles"]["accepted"]["metrics"]
        matched_metrics = regated["roles"]["matched"]["metrics"]
        accepted = Full35MetricSnapshot(
            run_id=f"{plan.plan_id}:accepted-reused",
            policy_id=regated["roles"]["accepted"]["policy_id"],
            metric_contract_id=plan.metric_contract_id,
            metrics=accepted_metrics,
        )
        matched = Full35MetricSnapshot(
            run_id=f"{plan.plan_id}:matched-reused",
            policy_id=plan.matched_policy_id,
            metric_contract_id=plan.metric_contract_id,
            metrics=matched_metrics,
        )
        adapter = WeightQuantizationAdapter()
        try:
            for index, cell in enumerate(pending, start=1):
                print(f"[{index}/{len(pending)}] validate {cell.cell_id}", flush=True)
                payload["current_cell"] = cell.cell_id
                _atomic_json(output_path, payload)
                with adapter.quantized(
                    model,
                    catalog=catalog,
                    region=cell.region,
                    spec=UniformWeightSpec(
                        bits=cell.bits,
                        scale_method=cell.scale_method,
                    ),
                ) as quantized:
                    record = _validate_role(
                        role=cell.cell_id,
                        model=model,
                        source=source,
                        plan=plan,  # type: ignore[arg-type]
                        pose_yaml=prepared.yaml,
                        output_root=validation_root,
                        device_index=device_index,
                    )
                    candidate = Full35MetricCandidate(
                        run_id=f"{plan.plan_id}:{cell.cell_id}",
                        format_id=f"uniform-{cell.format_id}-{cell.scale_method}",
                        stage="ptq",
                        policy_id=plan.matched_policy_id,
                        metric_contract_id=plan.metric_contract_id,
                        metrics=record["metrics"],
                    )
                    gate = Full35MetricGate(plan.gate_spec).evaluate(
                        candidate, accepted, matched
                    )
                    record.update(
                        {
                            "status": "completed",
                            "policy_id": plan.matched_policy_id,
                            "cell": cell.to_dict(),
                            "activation_output_quantization": "calibrated_lsq_plus_a8",
                            "weight_quantization": {
                                "format_id": quantized.format_id,
                                "region": quantized.region,
                                "quantized_modules": quantized.quantized_modules,
                                "weight_elements": quantized.weight_elements,
                                "numeric": quantized.numeric,
                                "sites": list(quantized.site_metrics),
                            },
                            "gate": gate.to_dict(),
                            "build": build,
                            "calibration": calibration,
                        }
                    )
                results[cell.cell_id] = record
                payload["results"] = results
                _atomic_json(output_path, payload)
        finally:
            del model, source, applied, build, calibration, catalog
            gc.collect()
            torch.cuda.empty_cache()

    all_candidates: list[BalanceCandidate] = []
    reference_candidate = _mapping(
        _mapping(source_payload["roles"], "reference roles")["candidate"],
        "reference W8 candidate",
    )
    reference_weight = _mapping(
        reference_candidate.get("weight_quantization"), "reference W8 weight"
    )
    reference_numeric = _mapping(
        reference_weight.get("numeric"), "reference W8 numeric"
    )
    deployment_elements = int(
        plan.weight_study.expected_catalog["totals"]["deployment_weight_elements"]  # type: ignore[index]
    )
    w8_packed, reference_fp32 = _candidate_cost(
        deployment_weight_elements=deployment_elements,
        region_weight_elements=int(reference_weight["weight_elements"]),
        numeric=reference_numeric,
    )
    reference_gate = regated["gate"]
    all_candidates.append(
        BalanceCandidate(
            candidate_id="reference-backbone_early-w8-mse_grid_v1",
            policy_id=plan.matched_policy_id,
            format_id="uniform-w8-mse_grid_v1",
            worst_total_delta=float(reference_gate["worst_total_delta"]),
            packed_weight_bytes=w8_packed,
            reference_fp32_bytes=reference_fp32,
            hard_gate_passed=bool(reference_gate["passed"]),
        )
    )
    for cell in plan.cells:
        record = _mapping(results.get(cell.cell_id), f"result {cell.cell_id}")
        if record.get("status") != "completed":
            raise RuntimeError(f"search race result incomplete: {cell.cell_id}")
        weight = _mapping(record.get("weight_quantization"), "candidate weight")
        numeric = _mapping(weight.get("numeric"), "candidate numeric")
        packed, reference_bytes = _candidate_cost(
            deployment_weight_elements=deployment_elements,
            region_weight_elements=int(weight["weight_elements"]),
            numeric=numeric,
        )
        gate = _mapping(record.get("gate"), "candidate gate")
        all_candidates.append(
            BalanceCandidate(
                candidate_id=cell.cell_id,
                policy_id=plan.matched_policy_id,
                format_id=f"uniform-{cell.format_id}-{cell.scale_method}",
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
    payload.pop("current_cell", None)
    payload["candidates"] = [candidate.to_dict() for candidate in all_candidates]
    payload["selection"] = selection.to_dict()
    payload["status"] = "completed"
    _atomic_json(output_path, payload)
    print(json.dumps(payload["selection"], ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full35 candidate-only mAP50 search race"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--execute-reviewed-plan", action="store_true")
    args = parser.parse_args(argv)
    plan = Full35SearchRacePlan.from_yaml(args.plan)
    if args.list_only:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.execute_reviewed_plan:
        parser.error("execution requires --execute-reviewed-plan")
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        parser.error(f"CUDA device {args.device} is unavailable")
    return run_search_race(
        plan=plan,
        output=args.output,
        device_index=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
