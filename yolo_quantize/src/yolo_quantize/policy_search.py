"""Monotonic combined-region PTQ policy chains and resumable validation."""

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
from .search_race import _calibration_identity, _mapping, _verified_file
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
    WeightRegionAssignment,
)
from .weight_sensitivity import WeightSensitivityStudy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    PROJECT_ROOT / "configs/experiments/v5-qsilu-cumulative-w8-policy-search-v1.yaml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/reports/v5-qsilu-cumulative-w8-policy-search-v1.json"
)


@dataclass(frozen=True)
class CombinedPolicyCandidate:
    """One combined weight policy with an explicit predecessor."""

    policy_id: str
    chain_id: str
    predecessor_id: str
    assignments: tuple[WeightRegionAssignment, ...]

    def __post_init__(self) -> None:
        if not self.policy_id or not self.chain_id or not self.predecessor_id:
            raise ValueError("combined policy identity must not be empty")
        if not self.assignments:
            raise ValueError("combined policy must contain assignments")
        regions = self.regions
        if len(set(regions)) != len(regions):
            raise ValueError("combined policy regions must be unique")
        bits = {assignment.spec.bits for assignment in self.assignments}
        methods = {
            getattr(assignment.spec, "scale_method", "paper_twn")
            for assignment in self.assignments
        }
        if bits != {8} or methods != {"mse_grid_v1"}:
            raise ValueError(
                "v5 cumulative bridge requires uniform W8 mse_grid_v1 assignments"
            )

    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(assignment.region for assignment in self.assignments)

    @property
    def format_id(self) -> str:
        return "mixed-w8-mse_grid_v1"


@dataclass(frozen=True)
class CombinedPolicyChain:
    """One strict cumulative chain that halts after its first failed node."""

    chain_id: str
    seed_candidate_id: str
    seed_regions: tuple[str, ...]
    candidates: tuple[CombinedPolicyCandidate, ...]

    def __post_init__(self) -> None:
        if not self.chain_id or not self.seed_candidate_id or not self.seed_regions:
            raise ValueError("combined policy chain seed must not be empty")
        if not self.candidates:
            raise ValueError("combined policy chain must contain candidates")
        previous_id = self.seed_candidate_id
        previous_regions = self.seed_regions
        for candidate in self.candidates:
            if candidate.chain_id != self.chain_id:
                raise ValueError("combined policy candidate belongs to another chain")
            if candidate.predecessor_id != previous_id:
                raise ValueError("combined policy predecessor is not sequential")
            if candidate.regions[: len(previous_regions)] != previous_regions or len(
                candidate.regions
            ) <= len(previous_regions):
                raise ValueError(
                    "combined policy must be a strict ordered extension of predecessor"
                )
            previous_id = candidate.policy_id
            previous_regions = candidate.regions

    def runnable_candidate_ids(self, passed_by_id: dict[str, bool]) -> tuple[str, ...]:
        """Return at most the next node whose predecessor is already green."""

        for candidate in self.candidates:
            if candidate.policy_id in passed_by_id:
                continue
            if passed_by_id.get(candidate.predecessor_id) is True:
                return (candidate.policy_id,)
            return ()
        return ()

    def blocked_candidate_ids(self, passed_by_id: dict[str, bool]) -> tuple[str, ...]:
        """Return unrun descendants after a known failed predecessor."""

        blocked: list[str] = []
        predecessor_failed = passed_by_id.get(self.seed_candidate_id) is False
        for candidate in self.candidates:
            if candidate.policy_id in passed_by_id:
                predecessor_failed = not passed_by_id[candidate.policy_id]
                continue
            if (
                predecessor_failed
                or passed_by_id.get(candidate.predecessor_id) is False
            ):
                predecessor_failed = True
                blocked.append(candidate.policy_id)
        return tuple(blocked)


@dataclass(frozen=True)
class Full35CombinedPolicySearchPlan:
    """Reviewed candidate-only search over monotonic multi-region W8 chains."""

    config_path: Path
    config_sha256: str
    plan_id: str
    execution_authorization_id: str
    metric_contract_id: str
    base_plan: Full35SearchValidationPlan
    weight_study: WeightSensitivityStudy
    chains: tuple[CombinedPolicyChain, ...]
    reference_report: Path
    reference_report_sha256: str
    reference_regate: Path
    isolated_report: Path
    isolated_report_sha256: str
    isolated_candidates: tuple[dict[str, Any], ...]
    gate_spec: Full35MetricGateSpec
    accuracy_tolerance: float
    formal_training: bool = False
    formal_validation: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> Full35CombinedPolicySearchPlan:
        config_path = Path(path).expanduser().resolve()
        payload = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "combined policy search plan",
        )
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported combined policy search schema")
        if payload.get("execution_authorized") is not True:
            raise ValueError("combined policy search execution is not authorized")
        if payload.get("formal_training") is not False:
            raise ValueError("combined policy search cannot authorize training")
        if payload.get("formal_validation") is not False:
            raise ValueError("combined policy search cannot use formal validation")

        authorization = _mapping(
            payload.get("execution_authorization"), "execution authorization"
        )
        authorization_id = str(authorization.get("authorization_id", "")).strip()
        if not authorization_id:
            raise ValueError("combined policy authorization_id is empty")
        if authorization.get("validation_roles") != ["candidate"]:
            raise ValueError("combined policy search may validate candidates only")
        if authorization.get("training") is not False:
            raise ValueError("combined policy search may not train")
        if (
            authorization.get("stop_rule")
            != "first_failed_predecessor_blocks_descendants"
        ):
            raise ValueError("combined policy stop rule changed")

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
        reference_report = _verified_file(
            config_path,
            upstream.get("reference_search_report"),
            "reference search report",
        )
        reference_regate = _verified_file(
            config_path,
            upstream.get("reference_map50_regate"),
            "reference map50 re-gate",
        )
        isolated_report = _verified_file(
            config_path,
            upstream.get("isolated_w8_search_report"),
            "isolated W8 search report",
        )
        isolated_payload = _mapping(
            json.loads(isolated_report.read_text(encoding="utf-8")),
            "isolated W8 search payload",
        )
        reviewed = _mapping(
            _mapping(isolated_payload.get("contract"), "isolated contract").get(
                "reviewed_plan"
            ),
            "isolated reviewed plan",
        )
        if (
            isolated_payload.get("status") != "completed"
            or reviewed.get("base_plan_sha256") != base_plan.config_sha256
            or reviewed.get("weight_plan_sha256") != weight_study.config_sha256
            or reviewed.get("reference_report_sha256") != _sha256(reference_report)
        ):
            raise ValueError("isolated W8 search evidence contract is invalid")

        pinned_regate = _mapping(
            json.loads(reference_regate.read_text(encoding="utf-8")),
            "reference mAP50 re-gate payload",
        )
        reference_gate = _mapping(pinned_regate.get("gate"), "reference W8 gate")
        if reference_gate.get("passed") is not True:
            raise ValueError("backbone_early did not pass its isolated W8 gate")

        isolated_results = _mapping(
            isolated_payload.get("results"), "isolated W8 results"
        )
        green_regions: dict[str, str] = {
            base_plan.cell.region: "reference-backbone_early-w8-mse_grid_v1"
        }
        for candidate_id, raw_record in isolated_results.items():
            record = _mapping(raw_record, f"isolated W8 result {candidate_id}")
            cell = _mapping(record.get("cell"), f"isolated W8 cell {candidate_id}")
            gate = _mapping(record.get("gate"), f"isolated W8 gate {candidate_id}")
            if (
                record.get("status") == "completed"
                and gate.get("passed") is True
                and int(cell.get("weight_bits", 0)) == 8
                and cell.get("scale_method") == "mse_grid_v1"
                and record.get("policy_id") == base_plan.matched_policy_id
            ):
                green_regions[str(cell["region"])] = str(candidate_id)

        weight_format = _mapping(payload.get("weight_format"), "weight format")
        if weight_format != {
            "bits": 8,
            "scale_method": "mse_grid_v1",
            "granularity": "per_output_channel",
            "rounding": "nearest_even",
        }:
            raise ValueError("combined policy weight format changed")

        raw_chains = payload.get("chains")
        if not isinstance(raw_chains, list) or not raw_chains:
            raise ValueError("combined policy chains must be a non-empty list")
        chains: list[CombinedPolicyChain] = []
        all_policy_ids: list[str] = []
        for chain_index, raw_chain in enumerate(raw_chains):
            chain_payload = _mapping(raw_chain, f"chain {chain_index}")
            chain_id = str(chain_payload.get("chain_id", "")).strip()
            seed = _mapping(chain_payload.get("seed"), f"chain {chain_id} seed")
            seed_id = str(seed.get("candidate_id", "")).strip()
            seed_regions = tuple(str(item) for item in seed.get("regions", ()))
            if len(seed_regions) != 1 or green_regions.get(seed_regions[0]) != seed_id:
                raise ValueError(
                    f"chain {chain_id} seed lacks a green isolated W8 gate"
                )
            raw_candidates = chain_payload.get("candidates")
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise ValueError(f"chain {chain_id} candidates must not be empty")
            candidates: list[CombinedPolicyCandidate] = []
            for candidate_index, raw_candidate in enumerate(raw_candidates):
                candidate_payload = _mapping(
                    raw_candidate,
                    f"chain {chain_id} candidate {candidate_index}",
                )
                policy_id = str(candidate_payload.get("policy_id", "")).strip()
                predecessor_id = str(
                    candidate_payload.get("predecessor_id", "")
                ).strip()
                regions = tuple(
                    str(region) for region in candidate_payload.get("regions", ())
                )
                unapproved = tuple(
                    region for region in regions if region not in green_regions
                )
                if unapproved:
                    raise ValueError(
                        "combined policy region lacks a green isolated W8 gate: "
                        + ",".join(unapproved)
                    )
                for region in regions:
                    cells = weight_study.cells(
                        activations=(base_plan.cell.parent.activation,),
                        regions=(region,),
                        bits=(8,),
                        scale_method="mse_grid_v1",
                    )
                    if len(cells) != 1 or (
                        cells[0].parent.policy_id != base_plan.matched_policy_id
                    ):
                        raise ValueError(
                            f"region does not resolve to the matched W8 cell: {region}"
                        )
                candidate = CombinedPolicyCandidate(
                    policy_id=policy_id,
                    chain_id=chain_id,
                    predecessor_id=predecessor_id,
                    assignments=tuple(
                        WeightRegionAssignment(
                            region=region,
                            spec=UniformWeightSpec(
                                bits=8,
                                scale_method="mse_grid_v1",
                            ),
                        )
                        for region in regions
                    ),
                )
                candidates.append(candidate)
                all_policy_ids.append(policy_id)
            chains.append(
                CombinedPolicyChain(
                    chain_id=chain_id,
                    seed_candidate_id=seed_id,
                    seed_regions=seed_regions,
                    candidates=tuple(candidates),
                )
            )

        if len({chain.chain_id for chain in chains}) != len(chains):
            raise ValueError("combined policy chain IDs must be unique")
        if len(set(all_policy_ids)) != len(all_policy_ids):
            raise ValueError("combined policy candidate IDs must be unique")
        authorized_ids = tuple(
            str(item) for item in authorization.get("candidate_ids", ())
        )
        if authorized_ids != tuple(all_policy_ids):
            raise ValueError("combined policy order differs from authorization")

        validation = _mapping(payload.get("validation"), "validation")
        expected_validation = {
            "source": "inherit_base_search_plan",
            "backend": "bittrue",
            "image_size": base_plan.image_size,
            "batch": {
                "detect": base_plan.detect_batch_size,
                "pose": base_plan.pose_batch_size,
            },
            "workers": {
                "detect": base_plan.detect_workers,
                "pose": base_plan.pose_workers,
            },
            "plots": False,
            "save_coco_json": False,
            "augmentation": False,
        }
        if validation != expected_validation:
            raise ValueError("combined policy validation differs from base search")

        gates = _mapping(payload.get("gates"), "gates")
        if gates.get("all_eight_metrics_required") is not True:
            raise ValueError("combined policy search requires all eight metrics")
        if gates.get("activation_replacement_included") is not True:
            raise ValueError("combined total gate must include activation replacement")
        gate_spec = Full35MetricGateSpec(
            metric_family=str(gates["metric_family"]),  # type: ignore[arg-type]
            total_max_drop=float(gates["total_max_drop"]),
            w8_incremental_max_drop=float(gates["w8_incremental_max_drop"]),
            sham_max_absolute_drift=float(gates["sham_max_absolute_drift"]),
            recovery_floor=float(gates["recovery_floor"]),
        )
        if gate_spec.metric_family != "map50" or gate_spec.total_max_drop != 0.015:
            raise ValueError("active combined policy gate must be mAP50 drop 0.015")

        selection = _mapping(payload.get("selection"), "selection")
        if selection.get("include_all_prior_isolated_w8_candidates") is not True:
            raise ValueError("combined selection must retain prior isolated candidates")
        raw_isolated_candidates = isolated_payload.get("candidates")
        if not isinstance(raw_isolated_candidates, list) or not raw_isolated_candidates:
            raise ValueError("isolated W8 report has no comparable candidates")
        isolated_candidates: list[dict[str, Any]] = []
        for raw_candidate in raw_isolated_candidates:
            candidate_record = _mapping(raw_candidate, "isolated balance candidate")
            BalanceCandidate(
                candidate_id=str(candidate_record["candidate_id"]),
                policy_id=str(candidate_record["policy_id"]),
                format_id=str(candidate_record["format_id"]),
                worst_total_delta=float(candidate_record["worst_total_delta"]),
                packed_weight_bytes=int(candidate_record["packed_weight_bytes"]),
                reference_fp32_bytes=int(candidate_record["reference_fp32_bytes"]),
                hard_gate_passed=bool(candidate_record["hard_gate_passed"]),
            )
            isolated_candidates.append(dict(candidate_record))

        plan_id = str(payload.get("plan_id", "")).strip()
        metric_contract_id = str(payload.get("metric_contract_id", "")).strip()
        if not plan_id or not metric_contract_id:
            raise ValueError(
                "combined policy plan and metric contract IDs are required"
            )
        if metric_contract_id != pinned_regate.get("metric_contract_id"):
            raise ValueError(
                "combined policy metric contract differs from pinned mAP50 re-gate"
            )
        return cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            plan_id=plan_id,
            execution_authorization_id=authorization_id,
            metric_contract_id=metric_contract_id,
            base_plan=base_plan,
            weight_study=weight_study,
            chains=tuple(chains),
            reference_report=reference_report,
            reference_report_sha256=_sha256(reference_report),
            reference_regate=reference_regate,
            isolated_report=isolated_report,
            isolated_report_sha256=_sha256(isolated_report),
            isolated_candidates=tuple(isolated_candidates),
            gate_spec=gate_spec,
            accuracy_tolerance=float(selection["accuracy_tolerance"]),
        )

    @property
    def candidates(self) -> tuple[CombinedPolicyCandidate, ...]:
        return tuple(
            candidate for chain in self.chains for candidate in chain.candidates
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
            "isolated_report": str(self.isolated_report),
            "isolated_report_sha256": self.isolated_report_sha256,
            "chains": [
                {
                    "chain_id": chain.chain_id,
                    "seed_candidate_id": chain.seed_candidate_id,
                    "seed_regions": list(chain.seed_regions),
                    "candidates": [
                        {
                            "policy_id": candidate.policy_id,
                            "predecessor_id": candidate.predecessor_id,
                            "regions": list(candidate.regions),
                            "format_id": candidate.format_id,
                        }
                        for candidate in chain.candidates
                    ],
                }
                for chain in self.chains
            ],
            "candidate_count": len(self.candidates),
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


def _combined_weight_record(applied: Any) -> dict[str, Any]:
    return {
        "format_id": "mixed-w8-mse_grid_v1",
        "regions": list(applied.regions),
        "quantized_modules": applied.quantized_modules,
        "weight_elements": applied.weight_elements,
        "numeric": {
            "weight_code_bytes": applied.weight_code_bytes,
            "scale_bytes": applied.scale_bytes,
            "encoded_bits": 8,
        },
        "per_region": [
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


def _combined_cost(
    *,
    deployment_weight_elements: int,
    weight_record: Mapping[str, Any],
) -> tuple[int, int]:
    numeric = _mapping(weight_record.get("numeric"), "combined weight numeric")
    packed = (
        (deployment_weight_elements - int(weight_record["weight_elements"])) * 4
        + int(numeric["weight_code_bytes"])
        + int(numeric["scale_bytes"])
    )
    return packed, deployment_weight_elements * 4


def run_combined_policy_search(
    *,
    plan: Full35CombinedPolicySearchPlan,
    output: Path,
    device_index: int,
    resume: bool,
) -> int:
    """Validate cumulative W8 nodes, blocking only descendants of failed nodes."""

    prepared, base_contract = _prepare_contract(plan, device_index=device_index)  # type: ignore[arg-type]
    contract = {
        **base_contract,
        "kind": "full35_candidate_only_cumulative_w8_policy_search",
        "reference_reuse": {
            "reference_report": str(plan.reference_report),
            "reference_report_sha256": plan.reference_report_sha256,
            "isolated_report": str(plan.isolated_report),
            "isolated_report_sha256": plan.isolated_report_sha256,
            "roles": ["accepted", "matched", "isolated_candidates"],
            "validation_rerun": False,
        },
        "stop_rule": "first_failed_predecessor_blocks_descendants",
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
    reference_roles = _mapping(source_payload.get("roles"), "reference roles")
    expected_calibration = _mapping(
        _mapping(reference_roles.get("matched"), "reference matched").get(
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
            raise RuntimeError(
                "existing combined search contract differs; refusing resume"
            )
        if isinstance(existing.get("results"), dict):
            results.update(existing["results"])
    elif validation_root.exists():
        raise FileExistsError(
            "validation artifacts exist without combined report; refusing overwrite"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "full35_candidate_only_cumulative_w8_policy_search",
        "status": "running",
        "contract": contract,
        "reference": regated,
        "results": results,
        "chain_summary": None,
        "selection": None,
        "formal_training": False,
        "formal_validation": False,
    }
    _atomic_json(output_path, payload)

    def completed_pass(candidate_id: str) -> bool | None:
        raw = results.get(candidate_id)
        if not isinstance(raw, dict) or raw.get("status") != "completed":
            return None
        gate = raw.get("gate")
        if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
            raise TypeError(f"completed candidate has invalid gate: {candidate_id}")
        return bool(gate["passed"])

    needs_model = False
    for chain in plan.chains:
        predecessor_passed = True
        for candidate in chain.candidates:
            prior = completed_pass(candidate.policy_id)
            if prior is not None:
                predecessor_passed = prior
            elif predecessor_passed:
                needs_model = True
                break
        if needs_model:
            break

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(device_index)
    if needs_model:
        parent = plan.base_plan.cell.parent
        model, source, applied_activation, build = _build_deployment_policy(
            activation=parent.activation,
            checkpoint=parent.checkpoint,
            checkpoint_sha256=parent.sha256,
            expected_catalog=plan.weight_study.expected_catalog,
        )
        calibration = _calibrate_activation_a8(
            plan=plan,  # type: ignore[arg-type]
            model=model,
            applied=applied_activation,
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
            raise RuntimeError("combined search catalog differs from reviewed contract")
        accepted = Full35MetricSnapshot(
            run_id=f"{plan.plan_id}:accepted-reused",
            policy_id=str(regated["roles"]["accepted"]["policy_id"]),
            metric_contract_id=plan.metric_contract_id,
            metrics=regated["roles"]["accepted"]["metrics"],
        )
        matched = Full35MetricSnapshot(
            run_id=f"{plan.plan_id}:matched-reused",
            policy_id=plan.matched_policy_id,
            metric_contract_id=plan.metric_contract_id,
            metrics=regated["roles"]["matched"]["metrics"],
        )
        adapter = WeightQuantizationAdapter()
        try:
            for chain in plan.chains:
                predecessor_passed = True
                failed_predecessor: str | None = None
                for index, candidate in enumerate(chain.candidates, start=1):
                    prior = completed_pass(candidate.policy_id)
                    if prior is not None:
                        predecessor_passed = prior
                        failed_predecessor = None if prior else candidate.policy_id
                        continue
                    if not predecessor_passed:
                        results[candidate.policy_id] = {
                            "status": "blocked_by_failed_predecessor",
                            "chain_id": chain.chain_id,
                            "policy_id": candidate.policy_id,
                            "predecessor_id": candidate.predecessor_id,
                            "failed_predecessor_id": failed_predecessor,
                            "validation_ran": False,
                            "selection_eligible": False,
                        }
                        payload["results"] = results
                        _atomic_json(output_path, payload)
                        continue
                    print(
                        f"[{chain.chain_id} {index}/{len(chain.candidates)}] "
                        f"validate {candidate.policy_id}",
                        flush=True,
                    )
                    payload["current_candidate"] = candidate.policy_id
                    _atomic_json(output_path, payload)
                    with adapter.quantized_policy(
                        model,
                        catalog=catalog,
                        assignments=candidate.assignments,
                    ) as quantized:
                        record = _validate_role(
                            role=candidate.policy_id,
                            model=model,
                            source=source,
                            plan=plan,  # type: ignore[arg-type]
                            pose_yaml=prepared.yaml,
                            output_root=validation_root,
                            device_index=device_index,
                        )
                        gate = Full35MetricGate(plan.gate_spec).evaluate(
                            Full35MetricCandidate(
                                run_id=f"{plan.plan_id}:{candidate.policy_id}",
                                format_id=candidate.format_id,
                                stage="ptq",
                                policy_id=plan.matched_policy_id,
                                metric_contract_id=plan.metric_contract_id,
                                metrics=record["metrics"],
                            ),
                            accepted,
                            matched,
                        )
                        record.update(
                            {
                                "status": "completed",
                                "chain_id": chain.chain_id,
                                "policy_id": candidate.policy_id,
                                "activation_policy_id": plan.matched_policy_id,
                                "predecessor_id": candidate.predecessor_id,
                                "activation_output_quantization": (
                                    "calibrated_lsq_plus_a8"
                                ),
                                "weight_quantization": _combined_weight_record(
                                    quantized
                                ),
                                "gate": gate.to_dict(),
                                "build": build,
                                "calibration": calibration,
                            }
                        )
                    results[candidate.policy_id] = record
                    predecessor_passed = gate.passed
                    failed_predecessor = None if gate.passed else candidate.policy_id
                    payload["results"] = results
                    _atomic_json(output_path, payload)
        finally:
            del model, source, applied_activation, build, calibration, catalog
            gc.collect()
            torch.cuda.empty_cache()
    else:
        for chain in plan.chains:
            predecessor_passed = True
            failed_predecessor: str | None = None
            for candidate in chain.candidates:
                prior = completed_pass(candidate.policy_id)
                if prior is not None:
                    predecessor_passed = prior
                    failed_predecessor = None if prior else candidate.policy_id
                elif not predecessor_passed:
                    results[candidate.policy_id] = {
                        "status": "blocked_by_failed_predecessor",
                        "chain_id": chain.chain_id,
                        "policy_id": candidate.policy_id,
                        "predecessor_id": candidate.predecessor_id,
                        "failed_predecessor_id": failed_predecessor,
                        "validation_ran": False,
                        "selection_eligible": False,
                    }

    deployment_elements = int(
        plan.weight_study.expected_catalog["totals"]["deployment_weight_elements"]  # type: ignore[index]
    )
    balance_candidates = tuple(
        BalanceCandidate(
            candidate_id=str(record["candidate_id"]),
            policy_id=str(record["policy_id"]),
            format_id=str(record["format_id"]),
            worst_total_delta=float(record["worst_total_delta"]),
            packed_weight_bytes=int(record["packed_weight_bytes"]),
            reference_fp32_bytes=int(record["reference_fp32_bytes"]),
            hard_gate_passed=bool(record["hard_gate_passed"]),
        )
        for record in plan.isolated_candidates
    )
    combined_balance: list[BalanceCandidate] = []
    for candidate in plan.candidates:
        record = results.get(candidate.policy_id)
        if not isinstance(record, dict) or record.get("status") != "completed":
            continue
        weight_record = _mapping(
            record.get("weight_quantization"), "combined weight record"
        )
        packed, reference_bytes = _combined_cost(
            deployment_weight_elements=deployment_elements,
            weight_record=weight_record,
        )
        gate = _mapping(record.get("gate"), "combined gate")
        combined_balance.append(
            BalanceCandidate(
                candidate_id=candidate.policy_id,
                policy_id=plan.matched_policy_id,
                format_id=candidate.format_id,
                worst_total_delta=float(gate["worst_total_delta"]),
                packed_weight_bytes=packed,
                reference_fp32_bytes=reference_bytes,
                hard_gate_passed=bool(gate["passed"]),
            )
        )
    selection_candidates = balance_candidates + tuple(combined_balance)
    selection = QuantizationBalanceSelector(
        total_max_drop=plan.gate_spec.total_max_drop,
        accuracy_tolerance=plan.accuracy_tolerance,
    ).select(selection_candidates)
    chain_summary = []
    for chain in plan.chains:
        completed_ids = tuple(
            candidate.policy_id
            for candidate in chain.candidates
            if results.get(candidate.policy_id, {}).get("status") == "completed"
        )
        blocked_ids = tuple(
            candidate.policy_id
            for candidate in chain.candidates
            if results.get(candidate.policy_id, {}).get("status")
            == "blocked_by_failed_predecessor"
        )
        failed_ids = tuple(
            candidate_id
            for candidate_id in completed_ids
            if results[candidate_id]["gate"]["passed"] is False
        )
        chain_summary.append(
            {
                "chain_id": chain.chain_id,
                "seed_candidate_id": chain.seed_candidate_id,
                "completed_ids": list(completed_ids),
                "first_failed_id": failed_ids[0] if failed_ids else None,
                "blocked_ids": list(blocked_ids),
            }
        )
    unresolved = tuple(
        candidate.policy_id
        for candidate in plan.candidates
        if results.get(candidate.policy_id, {}).get("status")
        not in {"completed", "blocked_by_failed_predecessor"}
    )
    if unresolved:
        raise RuntimeError(
            "combined search results remain unresolved: " + ",".join(unresolved)
        )
    payload.pop("current_candidate", None)
    payload["results"] = results
    payload["candidates"] = [item.to_dict() for item in selection_candidates]
    payload["chain_summary"] = chain_summary
    payload["selection"] = selection.to_dict()
    payload["status"] = "completed"
    _atomic_json(output_path, payload)
    print(json.dumps(payload["selection"], ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full35 cumulative multi-region W8 candidate search"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--execute-reviewed-plan", action="store_true")
    args = parser.parse_args(argv)
    plan = Full35CombinedPolicySearchPlan.from_yaml(args.plan)
    if args.list_only:
        print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.execute_reviewed_plan:
        parser.error("execution requires --execute-reviewed-plan")
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        parser.error(f"CUDA device {args.device} is unavailable")
    return run_combined_policy_search(
        plan=plan,
        output=args.output,
        device_index=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
