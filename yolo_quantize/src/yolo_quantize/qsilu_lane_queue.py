"""Fail-closed, low-frequency queue for the complete qSiLU lane."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .dual_metric_regate import regate_candidate_report
from .metric_gate import FULL35_MAP50_95_KEYS, FULL35_MAP50_KEYS
from .mixed_policy_search import Full35MixedPolicySearchPlan

PROJECT_ROOT = Path(__file__).resolve().parents[2]
Sleep = Callable[[float], None]
CommandRunner = Callable[[tuple[str, ...], Path], int]
GpuPidProbe = Callable[[int], tuple[int, ...]]
_TERMINAL_FAILURES = frozenset({"failed", "blocked", "cancelled", "aborted"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: object) -> Path:
    configured = Path(str(value)).expanduser()
    return (
        configured.resolve()
        if configured.is_absolute()
        else (PROJECT_ROOT / configured).resolve()
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class DependencyFailedError(RuntimeError):
    """Raised when an upstream queue reached a terminal failure state."""


@dataclass
class QueueRecorder:
    """Persist only state transitions so normal waiting remains low-volume."""

    root: Path
    queue_id: str
    last_signature: str | None = None

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


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"dependency state must be a mapping: {path}")
    return raw


def wait_for_dependency(
    path: Path,
    *,
    required_status: str,
    poll_seconds: int,
    recorder: QueueRecorder,
    expected_sha256: str | None = None,
    sleep: Sleep = time.sleep,
) -> dict[str, Any]:
    """Wait at a fixed low-frequency cadence and fail closed on terminal errors."""

    if poll_seconds < 30:
        raise ValueError("dependency poll interval must be at least 30 seconds")
    while True:
        payload = _read_state(path)
        if payload is not None and expected_sha256 is not None:
            actual_sha256 = _sha256(path)
            if actual_sha256 != expected_sha256:
                recorder.transition(
                    "dependency_hash_mismatch",
                    dependency=str(path),
                    expected_sha256=expected_sha256,
                    actual_sha256=actual_sha256,
                )
                raise DependencyFailedError(
                    f"dependency {path} hash mismatch: {actual_sha256}"
                )
        status = None if payload is None else str(payload.get("status", ""))
        if status == required_status:
            recorder.transition(
                "dependency_complete",
                dependency=str(path),
                status=status,
            )
            return payload
        if status in _TERMINAL_FAILURES:
            recorder.transition(
                "dependency_failed",
                dependency=str(path),
                status=status,
            )
            raise DependencyFailedError(
                f"dependency {path} reached terminal status {status}"
            )
        recorder.transition(
            "waiting_for_dependency",
            dependency=str(path),
            status=status or "missing",
            poll_seconds=poll_seconds,
        )
        sleep(float(poll_seconds))


@dataclass(frozen=True)
class QSiluLanePlan:
    """Hash-checked qSiLU lane registration and its first executable stage."""

    config_path: Path
    config_sha256: str
    lane_id: str
    dependency_state: Path
    dependency_sha256: str
    required_dependency_status: str
    poll_seconds: int
    retry_per_arm: int
    q1: Full35MixedPolicySearchPlan
    q1_candidate_id: str
    q1_output: Path
    q1_dual_output: Path
    q2_plan_output: Path
    q2_preflight: Path
    q3_parent_manifest: Path
    q4_profile: Path
    q4_profile_id: str
    q5_plan_output: Path
    q5_run_root: Path
    q6_output_root: Path
    q6_qat_run_root: Path
    q7_report: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> QSiluLanePlan:
        config_path = Path(path).expanduser().resolve()
        payload = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "qSiLU lane"
        )
        if (
            payload.get("schema_version") != 1
            or payload.get("execution_authorized") is not True
        ):
            raise ValueError("qSiLU lane is not an authorized schema-v1 plan")
        if (
            payload.get("formal_training") is not False
            or payload.get("formal_validation") is not False
        ):
            raise ValueError("qSiLU lane cannot authorize formal work")
        lane_id = str(payload.get("lane_id", "")).strip()
        if not lane_id:
            raise ValueError("qSiLU lane_id is empty")
        queue = _mapping(payload.get("queue"), "qSiLU queue")
        dependency_sha256 = str(queue.get("dependency_sha256", "")).lower()
        if len(dependency_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in dependency_sha256
        ):
            raise ValueError("qSiLU dependency_sha256 must be lowercase SHA-256")
        poll_seconds = int(queue.get("poll_seconds", 0))
        if poll_seconds != 600 or queue.get("monitoring") != "event_only_json":
            raise ValueError(
                "qSiLU queue must use the reviewed 600-second event-only policy"
            )
        if queue.get("fail_closed_on_hash_or_contract_drift") is not True:
            raise ValueError("qSiLU queue must fail closed")
        stages = payload.get("stages")
        if not isinstance(stages, list):
            raise TypeError("qSiLU stages must be a list")
        stage_records = {
            str(item.get("id")): item
            for item in stages
            if isinstance(item, dict) and item.get("id") is not None
        }
        if set(stage_records) != {f"q{index}" for index in range(8)}:
            raise ValueError("qSiLU lane must contain exactly q0 through q7")
        q1_record = _mapping(stage_records["q1"], "qSiLU q1 stage")
        q2_record = _mapping(stage_records["q2"], "qSiLU q2 stage")
        q3_record = _mapping(stage_records["q3"], "qSiLU q3 stage")
        q4_record = _mapping(stage_records["q4"], "qSiLU q4 stage")
        q5_record = _mapping(stage_records["q5"], "qSiLU q5 stage")
        q6_record = _mapping(stage_records["q6"], "qSiLU q6 stage")
        q7_record = _mapping(stage_records["q7"], "qSiLU q7 stage")
        q1_path = _resolve(q1_record.get("reviewed_plan"))
        q1 = Full35MixedPolicySearchPlan.from_yaml(q1_path)
        if len(q1.candidates) != 1:
            raise ValueError("qSiLU q1 must contain one candidate")
        candidate = q1.candidates[0]
        expected_regions = tuple(
            str(item)
            for item in _mapping(payload.get("coverage"), "coverage")[
                "deployment_regions"
            ]
        )
        actual_regions = tuple(item.region for item in candidate.region_defaults)
        if actual_regions != expected_regions or len(actual_regions) != 10:
            raise ValueError("qSiLU q1 does not cover the reviewed ten regions")
        activation_parent = _mapping(
            payload.get("immutable_activation_parent"), "activation parent"
        )
        checkpoint = _mapping(
            activation_parent.get("checkpoint"), "activation checkpoint"
        )
        if (
            q1.activation_policy.activation != "qsilu_pq"
            or q1.activation_policy.bits != 8
            or q1.activation_checkpoint_sha256 != checkpoint.get("sha256")
        ):
            raise ValueError("qSiLU q1 activation parent drifted")
        return cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            lane_id=lane_id,
            dependency_state=_resolve(queue.get("dependency_state")),
            dependency_sha256=dependency_sha256,
            required_dependency_status=str(queue.get("dependency_required_status", "")),
            poll_seconds=poll_seconds,
            retry_per_arm=int(queue.get("retry_per_arm", -1)),
            q1=q1,
            q1_candidate_id=candidate.candidate_id,
            q1_output=_resolve(q1_record.get("output")),
            q1_dual_output=_resolve(q1_record.get("dual_output")),
            q2_plan_output=_resolve(q2_record.get("materialized_plan")),
            q2_preflight=_resolve(q2_record.get("preflight")),
            q3_parent_manifest=_resolve(q3_record.get("manifest")),
            q4_profile=_resolve(q4_record.get("output")),
            q4_profile_id=str(q4_record.get("profile_id", "")).strip(),
            q5_plan_output=_resolve(q5_record.get("materialized_plan")),
            q5_run_root=_resolve(q5_record.get("run_root")),
            q6_output_root=_resolve(q6_record.get("output_root")),
            q6_qat_run_root=_resolve(q6_record.get("qat_run_root")),
            q7_report=_resolve(q7_record.get("report")),
        )


def _verified_metrics_from_reference(plan: QSiluLanePlan, role: str) -> dict[str, str]:
    reference = _mapping(
        json.loads(plan.q1.reference_report.read_text(encoding="utf-8")),
        "qSiLU reference report",
    )
    record = _mapping(
        _mapping(reference.get("roles"), "qSiLU reference roles").get(role),
        f"qSiLU {role} role",
    )
    metrics = _resolve(record.get("metrics_report"))
    expected = str(record.get("metrics_report_sha256", ""))
    if not metrics.is_file() or _sha256(metrics) != expected:
        raise ValueError(f"qSiLU {role} metrics hash drifted")
    return {"path": str(metrics), "sha256": expected}


def materialize_q2_plan(
    lane: QSiluLanePlan,
    *,
    q1_report: Path,
    dual_report: Path,
    output_path: Path,
    run_root: Path | None = None,
) -> Path:
    """Build the paired qSiLU all-W8 QAT plan only from completed q1 evidence."""

    q1_payload = _mapping(
        json.loads(q1_report.read_text(encoding="utf-8")), "q1 evidence"
    )
    q1_record = _mapping(
        _mapping(q1_payload.get("results"), "q1 evidence results").get(
            lane.q1_candidate_id
        ),
        "q1 evidence candidate",
    )
    q1_gate = _mapping(q1_record.get("gate"), "q1 evidence gate")
    if (
        q1_payload.get("status") != "completed"
        or q1_record.get("status") != "completed"
        or q1_gate.get("decision") not in {"green", "recover"}
    ):
        raise ValueError("qSiLU q1 evidence is not eligible for QAT")
    dual_payload = _mapping(
        json.loads(dual_report.read_text(encoding="utf-8")), "q1 dual evidence"
    )
    dual_record = _mapping(
        _mapping(dual_payload.get("candidates"), "q1 dual candidates").get(
            lane.q1_candidate_id
        ),
        "q1 dual candidate",
    )
    if dual_payload.get("status") != "completed" or dual_record.get("decision") not in {
        "green",
        "recover",
    }:
        raise ValueError("qSiLU q1 dual gate is not eligible for QAT")
    if _mapping(dual_payload.get("source"), "q1 dual source").get(
        "report_sha256"
    ) != _sha256(q1_report):
        raise ValueError("qSiLU q1 dual evidence does not pin the source report")

    template_path = (
        PROJECT_ROOT / "configs/experiments/v19-poly-shift-all-w8-qat-pilot-v1.yaml"
    )
    payload = _mapping(
        yaml.safe_load(template_path.read_text(encoding="utf-8")), "QAT template"
    )
    payload = dict(payload)
    payload["plan_id"] = "v30-qsilu-all-w8-qat-pilot-v1"
    payload["date"] = "2026-09-05"
    payload["execution_authorization"] = {
        "authorization_id": "user-2026-09-05-complete-qsilu-quantization-lane",
        "arms": ["sham", "qat"],
        "scope": "paired_qsilu_all_w8_search_qat_only_no_formal_validation",
    }
    sources = dict(_mapping(payload["sources"], "QAT template sources"))
    sources["parent_checkpoint"] = {
        "path": str(lane.q1.activation_checkpoint),
        "sha256": lane.q1.activation_checkpoint_sha256,
    }
    sources["accepted_metrics"] = _verified_metrics_from_reference(lane, "accepted")
    sources["matched_metrics"] = _verified_metrics_from_reference(lane, "matched")
    sources["weight_plan"] = {
        "path": str(lane.q1.weight_study.config_path),
        "sha256": lane.q1.weight_study.config_sha256,
    }
    sources["candidate_evidence"] = {
        "path": str(q1_report.resolve()),
        "sha256": _sha256(q1_report),
    }
    sources["candidate_dual_regate"] = {
        "path": str(dual_report.resolve()),
        "sha256": _sha256(dual_report),
    }
    payload["sources"] = sources
    payload["activation"] = {
        "name": "qsilu_pq",
        "bits": 8,
        "signed_codes": False,
        "region_assignments": [],
    }
    payload["weight_policy"] = {
        "candidate_id": lane.q1_candidate_id,
        "assignments": [
            {
                "region": default.region,
                "format": {
                    "family": "uniform",
                    "bits": 8,
                    "scale_method": "mse_grid_v1",
                },
            }
            for default in lane.q1.candidates[0].region_defaults
        ],
    }
    training = dict(_mapping(payload["training"], "QAT template training"))
    training["patience"] = 5
    payload["training"] = training
    if run_root is not None:
        payload["run_root"] = str(run_root.expanduser().resolve())
    destination = output_path.expanduser().resolve()
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"refusing to overwrite drifted qSiLU q2 plan: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(destination)
    from .qat_plan import Full35QATPlan

    parsed = Full35QATPlan.from_yaml(destination)
    if (
        parsed.candidate_id != lane.q1_candidate_id
        or parsed.activation.activation != "qsilu_pq"
        or len(parsed.assignments) != 10
        or parsed.training.patience != 5
    ):
        raise ValueError("materialized qSiLU q2 plan drifted")
    return destination


def validate_q1_report(plan: QSiluLanePlan, report_path: Path) -> dict[str, Any]:
    """Require a completed report pinned to the exact reviewed q1 plan."""

    report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "q1 report")
    if (
        report.get("status") != "completed"
        or report.get("kind") != "full35_mixed_weight_policy_map50_search"
        or report.get("formal_training") is not False
        or report.get("formal_validation") is not False
    ):
        raise ValueError("qSiLU q1 report is not a completed non-formal search")
    reviewed = _mapping(
        _mapping(report.get("contract"), "q1 contract").get("reviewed_plan"),
        "q1 reviewed plan",
    )
    if reviewed.get("plan_id") != plan.q1.plan_id:
        raise ValueError("qSiLU q1 report plan id drifted")
    if reviewed.get("plan_sha256") != plan.q1.config_sha256:
        raise ValueError("qSiLU q1 report plan hash drifted")
    result = _mapping(
        _mapping(report.get("results"), "q1 results").get(plan.q1_candidate_id),
        "q1 candidate result",
    )
    gate = _mapping(result.get("gate"), "q1 candidate gate")
    decision = str(gate.get("decision", ""))
    if result.get("status") != "completed" or decision not in {
        "green",
        "recover",
        "reject",
    }:
        raise ValueError("qSiLU q1 candidate result is incomplete")
    return {
        "candidate_id": plan.q1_candidate_id,
        "decision": decision,
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
    }


def _gpu_pids(device_index: int) -> tuple[int, ...]:
    from .progressive_queue import _foreign_gpu_pids

    return _foreign_gpu_pids(device_index)


def _run_command(command: tuple[str, ...], log_path: Path) -> int:
    environment = os.environ.copy()
    source = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).returncode


def select_best_joint_epoch(gate_csv: Path) -> tuple[int, float]:
    """Replay the checkpoint selector from the persisted best-joint score rows."""

    candidates: list[tuple[int, float]] = []
    with gate_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("metric") == "score/best_joint":
                candidates.append((int(str(row["step"])), float(str(row["value"]))))
    if not candidates:
        raise ValueError(f"best-joint score is missing: {gate_csv}")
    return max(candidates, key=lambda item: (item[1], -item[0]))


def _metric_values(path: Path) -> Mapping[str, float]:
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "metric report")
    values = _mapping(payload.get("metrics"), "metric values")
    return {str(key): float(value) for key, value in values.items()}


def evaluate_total_dual_gate(
    accepted_path: Path, candidate_path: Path
) -> dict[str, Any]:
    """Evaluate all eight mAP50 and eight mAP50-95 totals against accepted Full35."""

    accepted = _metric_values(accepted_path)
    candidate = _metric_values(candidate_path)
    keys = (*FULL35_MAP50_KEYS, *FULL35_MAP50_95_KEYS)
    missing = tuple(key for key in keys if key not in accepted or key not in candidate)
    if missing:
        raise ValueError("dual metric reports are incomplete: " + ",".join(missing))
    deltas = {key: candidate[key] - accepted[key] for key in keys}
    worst_map50_key = min(FULL35_MAP50_KEYS, key=deltas.__getitem__)
    worst_map50_95_key = min(FULL35_MAP50_95_KEYS, key=deltas.__getitem__)
    worst_map50 = deltas[worst_map50_key]
    worst_map50_95 = deltas[worst_map50_95_key]
    green = worst_map50 >= -0.015 and worst_map50_95 >= -0.04
    return {
        "decision": "green" if green else "reject",
        "map50_max_drop": 0.015,
        "map50_95_max_drop": 0.04,
        "worst_map50_metric": worst_map50_key,
        "worst_map50_delta": worst_map50,
        "worst_map50_95_metric": worst_map50_95_key,
        "worst_map50_95_delta": worst_map50_95,
        "total_deltas": deltas,
    }


def lock_qat_parent(
    q2_plan_path: Path,
    *,
    output_path: Path,
    parent_id: str,
) -> Path:
    """Lock the exact best-joint epoch only when all 16 total gates pass."""

    from .progressive_preparation import LockedQATParentSpec
    from .qat_plan import Full35QATPlan

    plan = Full35QATPlan.from_yaml(q2_plan_path)
    if any(assignment.spec.format_id != "w8" for assignment in plan.assignments):
        raise ValueError("locked qSiLU parent must be all-W8")
    run_dir = plan.run_root / f"{plan.plan_id}-qat-seed{plan.training.seed}"
    completion_path = run_dir / "qat-experiment.json"
    completion = _mapping(
        json.loads(completion_path.read_text(encoding="utf-8")), "QAT completion"
    )
    if (
        completion.get("schema_version") != 1
        or completion.get("plan_sha256") != plan.config_sha256
        or completion.get("candidate_id") != plan.candidate_id
        or completion.get("arm") != "qat"
        or completion.get("completed_stages") != ["j3"]
    ):
        raise ValueError("QAT completion cannot support a locked parent")
    selected_epoch, _ = select_best_joint_epoch(run_dir / "logs/gate.csv")
    metrics_path = (
        run_dir / f"validation/epoch-{selected_epoch:04d}/bittrue/metrics.json"
    )
    gate = evaluate_total_dual_gate(plan.accepted_metrics, metrics_path)
    if gate["decision"] != "green":
        raise ValueError("best-joint qSiLU checkpoint does not pass all 16 total gates")
    checkpoint_paths = _mapping(
        completion.get("checkpoint_paths"), "QAT checkpoint paths"
    )
    checkpoint_hashes = _mapping(
        completion.get("checkpoint_sha256"), "QAT checkpoint hashes"
    )
    full_resume = _resolve(checkpoint_paths.get("best_joint"))
    full_resume_sha256 = _sha256(full_resume)
    if checkpoint_hashes.get("best_joint") != full_resume_sha256:
        raise ValueError("QAT best-joint full-resume checkpoint hash drifted")
    inference = run_dir / "inference/best_joint.pt"
    if not inference.is_file():
        raise FileNotFoundError(inference)
    totals = _mapping(
        plan.expected_deployment_catalog.get("totals"), "deployment totals"
    )
    payload = {
        "schema_version": 1,
        "parent_id": parent_id,
        "status": "locked_search_parent",
        "formal_validation": False,
        "selected_epoch": selected_epoch,
        "activation": {
            "name": plan.activation.activation,
            "bits": plan.activation.bits,
            "quantizer": f"lsq_plus_a{plan.activation.bits}",
        },
        "weight_policy": {
            "format_id": "all-w8",
            "deployment_modules": int(totals["deployment_modules"]),
            "deployment_weight_elements": int(totals["deployment_weight_elements"]),
        },
        "plan": {"path": str(plan.config_path), "sha256": plan.config_sha256},
        "completion": {
            "path": str(completion_path),
            "sha256": _sha256(completion_path),
        },
        "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
        "checkpoints": {
            "full_resume": {"path": str(full_resume), "sha256": full_resume_sha256},
            "inference": {"path": str(inference), "sha256": _sha256(inference)},
        },
        "gate": {
            "decision": "green",
            "map50_max_drop": 0.015,
            "map50_95_max_drop": 0.04,
            "worst_map50_delta": gate["worst_map50_delta"],
            "worst_map50_95_delta": gate["worst_map50_95_delta"],
        },
    }
    destination = output_path.expanduser().resolve()
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"refusing to overwrite drifted locked parent: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(destination)
    locked = LockedQATParentSpec.from_yaml(destination)
    if locked.parent_id != parent_id or locked.selected_epoch != selected_epoch:
        raise ValueError("locked qSiLU parent failed round-trip verification")
    return destination


def materialize_progressive_plan(
    *,
    locked_parent: Path,
    cpu_profile: Path,
    output_path: Path,
    queue_id: str,
    run_root: Path,
    include_intermediate_uniform_bits: bool = False,
) -> Path:
    """Clone the reviewed V29 experiment schema for another locked activation parent."""

    from .progressive_preparation import LockedQATParentSpec
    from .progressive_queue import ProgressiveQueuePlan
    from .qat_plan import Full35QATPlan

    parent = LockedQATParentSpec.from_yaml(locked_parent)
    qat_plan = Full35QATPlan.from_yaml(parent.plan_path)
    template = (
        PROJECT_ROOT / "configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml"
    )
    payload = dict(
        _mapping(
            yaml.safe_load(template.read_text(encoding="utf-8")),
            "progressive template",
        )
    )
    if include_intermediate_uniform_bits:
        formats = _mapping(payload.get("formats"), "progressive formats")
        for bits in (7, 6, 5):
            formats[f"exact_w{bits}"] = {
                "family": "uniform",
                "bits": bits,
                "scale_method": "optimal_scaled_codebook",
            }
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list):
            raise TypeError("progressive stages must be a list")
        for raw_stage in raw_stages:
            stage = _mapping(raw_stage, "progressive stage")
            current = [str(value) for value in stage.get("format_ids", [])]
            for required in ("exact_w4", "fixed_sd4"):
                if required not in current:
                    raise ValueError(
                        f"progressive stage lacks required control: {required}"
                    )
            stage["format_ids"] = [
                "exact_w7",
                "exact_w6",
                "exact_w5",
                "exact_w4",
                "fixed_sd4",
                "paper_twn_v2",
                "twn_v3_filterwise",
                "exact_scaled_ternary",
            ]
    payload["queue_id"] = queue_id
    payload["date"] = "2026-09-05"
    payload["status"] = "ready"
    payload["execution_authorization"] = {
        "authorization_id": "user-2026-09-05-complete-qsilu-quantization-lane",
        "scope": "search_ptq_and_eligible_short_qat_only",
    }
    payload["sources"] = {
        "locked_parent": {
            "path": str(locked_parent.resolve()),
            "sha256": _sha256(locked_parent),
        },
        "cpu_profile": {
            "path": str(cpu_profile.resolve()),
            "sha256": _sha256(cpu_profile),
        },
        "accepted_metrics": {
            "path": str(qat_plan.accepted_metrics),
            "sha256": _sha256(qat_plan.accepted_metrics),
        },
        "locked_parent_metrics": {
            "path": str(parent.metrics_path),
            "sha256": parent.metrics_sha256,
        },
    }
    datasets = dict(_mapping(payload["datasets"], "progressive datasets"))
    datasets["runtime_view"] = str(
        PROJECT_ROOT / f"artifacts/datasets/{queue_id}-bbat5-v1"
    )
    payload["datasets"] = datasets
    gpu = dict(_mapping(payload["gpu_queue"], "progressive GPU queue"))
    gpu["poll_seconds"] = 600
    payload["gpu_queue"] = gpu
    payload["run_root"] = str(run_root.expanduser().resolve())
    destination = output_path.expanduser().resolve()
    encoded = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(
                f"refusing to overwrite drifted progressive plan: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(destination)
    parsed = ProgressiveQueuePlan.from_yaml(destination)
    if (
        parsed.queue_id != queue_id
        or parsed.poll_seconds != 600
        or len(parsed.stages) != 10
    ):
        raise ValueError("materialized progressive plan drifted")
    return destination


def _qat_completion(runtime: Any, arm: str) -> Path | None:
    run_dir = runtime.plan.run_root / runtime.run_name(arm)
    manifest = run_dir / "qat-experiment.json"
    if not manifest.is_file():
        return None
    payload = _mapping(
        json.loads(manifest.read_text(encoding="utf-8")), f"qSiLU {arm} completion"
    )
    expected = {
        "schema_version": 1,
        "plan_sha256": runtime.plan.config_sha256,
        "candidate_id": runtime.plan.candidate_id,
        "arm": arm,
        "run_name": runtime.run_name(arm),
        "completed_stages": ["j3"],
        "formal_validation": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"qSiLU {arm} completion manifest drifted")
    return manifest


def run_paired_qat(
    plan_path: Path,
    *,
    device_index: int,
    maximum_retries_per_arm: int,
    recorder: QueueRecorder,
    runtime_factory: Callable[[Path], Any] | None = None,
    gpu_pid_probe: GpuPidProbe = _gpu_pids,
    sleep: Sleep = time.sleep,
) -> dict[str, Any]:
    """Execute one hash-pinned matched sham followed by its QAT arm."""

    if maximum_retries_per_arm < 0:
        raise ValueError("QAT retry count must not be negative")
    if runtime_factory is None:
        from .qat_runtime import Full35QATRuntime

        runtime_factory = Full35QATRuntime.from_yaml
    runtime = runtime_factory(plan_path)
    completed: dict[str, str] = {}
    for arm in ("sham", "qat"):
        if _qat_completion(runtime, arm) is not None:
            completed[arm] = "completed"
            continue
        last_error: Exception | None = None
        for attempt in range(maximum_retries_per_arm + 1):
            while True:
                pids = gpu_pid_probe(device_index)
                if not pids:
                    break
                recorder.transition(
                    "waiting_for_gpu",
                    stage="q2",
                    arm=arm,
                    pids=list(pids),
                    poll_seconds=600,
                )
                sleep(600.0)
            last = (
                runtime.plan.run_root
                / runtime.run_name(arm)
                / "checkpoints"
                / "last.pt"
            )
            resume = last if last.is_file() else None
            recorder.transition(
                "q2_arm_started",
                arm=arm,
                attempt=attempt,
                resume=None if resume is None else str(resume),
            )
            try:
                runtime.run(arm, device_index=device_index, resume=resume)
                if _qat_completion(runtime, arm) is None:
                    raise RuntimeError(
                        f"qSiLU {arm} returned without completion manifest"
                    )
                recorder.transition("q2_arm_completed", arm=arm)
                completed[arm] = "completed"
                last_error = None
                break
            except Exception as error:  # noqa: BLE001 - persist reviewed arm error
                last_error = error
                retryable = getattr(error, "retryable", True) is not False
                recorder.transition(
                    "q2_arm_failed",
                    arm=arm,
                    attempt=attempt,
                    error_type=type(error).__name__,
                    message=str(error),
                    retryable=retryable,
                )
                if not retryable:
                    break
        if last_error is not None:
            raise last_error
    return {
        "plan": str(plan_path),
        "plan_sha256": runtime.plan.config_sha256,
        "arms": completed,
    }


def _preflight_plan_sha256(payload: Mapping[str, Any]) -> str:
    resolved = _mapping(payload.get("resolved"), "q2 preflight resolved")
    return str(resolved.get("plan_sha256", ""))


def ensure_q2_preflight(plan_path: Path, output_path: Path) -> dict[str, Any]:
    """Run the real graph/data/hash preflight once without using GPU."""

    from .qat_runtime import Full35QATRuntime

    runtime = Full35QATRuntime.from_yaml(plan_path)
    if output_path.is_file():
        payload = _mapping(
            json.loads(output_path.read_text(encoding="utf-8")), "q2 preflight"
        )
        if (
            payload.get("ready") is not True
            or _preflight_plan_sha256(payload) != runtime.plan.config_sha256
        ):
            raise ValueError("existing qSiLU q2 preflight drifted")
        return dict(payload)
    report = runtime.preflight(verify_graph=True, verify_sample_files=True).to_dict()
    if (
        report.get("ready") is not True
        or _preflight_plan_sha256(report) != runtime.plan.config_sha256
    ):
        raise RuntimeError("qSiLU q2 preflight did not pass")
    _atomic_json(output_path, report)
    return report


def ensure_cpu_profile(
    parent_manifest: Path,
    *,
    output_path: Path,
    profile_id: str,
) -> dict[str, Any]:
    """Create or verify the qSiLU-owned 148x8 CPU weight profile."""

    from .progressive_preparation import (
        LockedQATParentSpec,
        ProgressiveWeightPreparation,
    )

    parent = LockedQATParentSpec.from_yaml(parent_manifest)
    if output_path.is_file():
        payload = _mapping(
            json.loads(output_path.read_text(encoding="utf-8")), "qSiLU CPU profile"
        )
        profile_parent = _mapping(payload.get("parent"), "qSiLU CPU profile parent")
        coverage = _mapping(
            _mapping(payload.get("summary"), "qSiLU CPU summary").get("coverage"),
            "qSiLU CPU coverage",
        )
        if (
            payload.get("status") != "completed"
            or payload.get("profile_id") != profile_id
            or profile_parent.get("manifest_sha256") != parent.config_sha256
            or int(coverage.get("deployment_paths", -1)) != 148
            or int(coverage.get("measurements", -1)) != 1184
            or int(coverage.get("formats_per_path", -1)) != 8
        ):
            raise ValueError("existing qSiLU CPU profile drifted")
        return dict(payload)
    ProgressiveWeightPreparation().prepare_profile(
        parent,
        output_path=output_path,
        profile_id=profile_id,
        include_intermediate_uniform_bits=True,
    )
    return ensure_cpu_profile(
        parent_manifest, output_path=output_path, profile_id=profile_id
    )


def _complete_lane_summary(
    lane: QSiluLanePlan,
    *,
    parent_manifest: Path,
    cpu_profile: Path,
    progressive_state: Mapping[str, Any],
    short_qat_state: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = _mapping(
        progressive_state.get("current_metrics"), "qSiLU progressive metrics"
    )
    from .progressive_preparation import LockedQATParentSpec

    parent = LockedQATParentSpec.from_yaml(parent_manifest)
    from .qat_plan import Full35QATPlan

    q2_plan = Full35QATPlan.from_yaml(parent.plan_path)
    accepted = _metric_values(q2_plan.accepted_metrics)
    keys = (*FULL35_MAP50_KEYS, *FULL35_MAP50_95_KEYS)
    deltas = {key: float(metrics[key]) - accepted[key] for key in keys}
    return {
        "schema_version": 1,
        "kind": "full35_qsilu_complete_search_lane_summary",
        "lane_id": lane.lane_id,
        "lane_plan": str(lane.config_path),
        "lane_plan_sha256": lane.config_sha256,
        "status": "complete_search_pipeline_pending_finalist_review",
        "formal_training": False,
        "formal_validation": False,
        "monitoring": {"mode": "event_only_json", "poll_seconds": 600},
        "activation": {"name": "qsilu_pq", "bits": 8, "quantizer": "lsq_plus"},
        "locked_parent": {
            "path": str(parent_manifest),
            "sha256": _sha256(parent_manifest),
            "parent_id": parent.parent_id,
        },
        "cpu_profile": {
            "path": str(cpu_profile),
            "sha256": _sha256(cpu_profile),
            "coverage": "148_paths_x_8_formats",
        },
        "progressive": {
            "queue_id": progressive_state.get("queue_id"),
            "status": progressive_state.get("status"),
            "locked_assignments": progressive_state.get("locked_assignments"),
            "packed_weight_bytes": progressive_state.get("current_packed_bytes"),
            "total_deltas": deltas,
            "worst_map50_delta": min(deltas[key] for key in FULL35_MAP50_KEYS),
            "worst_map50_95_delta": min(deltas[key] for key in FULL35_MAP50_95_KEYS),
        },
        "short_qat": {
            "status": short_qat_state.get("status"),
            "jobs": len(short_qat_state.get("jobs", []))
            if isinstance(short_qat_state.get("jobs"), list)
            else 0,
            "state": str(lane.q6_output_root / "execution-state.json"),
        },
        "next": "compare_poly_shift_and_qsilu_pareto_then_review_hardswish",
        "long_or_formal_execution_authorized": False,
    }


class QSiluLaneQueue:
    """Run the complete qSiLU search lane after the active V29 queue finishes."""

    def __init__(
        self,
        plan: QSiluLanePlan,
        *,
        queue_root: Path,
        device_index: int = 0,
        dependency_state: Path | None = None,
        q1_output: Path | None = None,
        q1_dual_output: Path | None = None,
        command_runner: CommandRunner = _run_command,
        gpu_pid_probe: GpuPidProbe = _gpu_pids,
        sleep: Sleep = time.sleep,
    ) -> None:
        self.plan = plan
        self.root = queue_root.resolve()
        self.device_index = device_index
        self.dependency_state = (dependency_state or plan.dependency_state).resolve()
        self.q1_output = (q1_output or plan.q1_output).resolve()
        self.q1_dual_output = (q1_dual_output or plan.q1_dual_output).resolve()
        self.command_runner = command_runner
        self.gpu_pid_probe = gpu_pid_probe
        self.sleep = sleep
        self.recorder = QueueRecorder(self.root, plan.lane_id)

    def _wait_for_gpu(self) -> None:
        while True:
            pids = self.gpu_pid_probe(self.device_index)
            if not pids:
                self.recorder.transition("gpu_acquired", device=self.device_index)
                return
            self.recorder.transition(
                "waiting_for_gpu",
                device=self.device_index,
                pids=list(pids),
                poll_seconds=self.plan.poll_seconds,
            )
            self.sleep(float(self.plan.poll_seconds))

    def _run_q1(self) -> dict[str, Any]:
        if self.q1_output.is_file():
            try:
                return validate_q1_report(self.plan, self.q1_output)
            except (KeyError, TypeError, ValueError):
                resume = True
        else:
            resume = False
        last_return_code = -1
        for attempt in range(self.plan.retry_per_arm + 1):
            self._wait_for_gpu()
            command = [
                sys.executable,
                "-m",
                "yolo_quantize.mixed_policy_search",
                "--plan",
                str(self.plan.q1.config_path),
                "--output",
                str(self.q1_output),
                "--device",
                str(self.device_index),
                "--execute-reviewed-plan",
            ]
            if resume or attempt:
                command.append("--resume")
            self.recorder.transition(
                "q1_started",
                attempt=attempt,
                candidate_id=self.plan.q1_candidate_id,
                resume="--resume" in command,
            )
            last_return_code = self.command_runner(
                tuple(command), self.root / "q1-console.log"
            )
            if last_return_code == 0:
                return validate_q1_report(self.plan, self.q1_output)
            self.recorder.transition(
                "q1_failed_attempt", attempt=attempt, return_code=last_return_code
            )
            resume = True
        raise RuntimeError(
            f"qSiLU q1 failed after retries; return code {last_return_code}"
        )

    def _write_state(self, status: str, **values: object) -> dict[str, Any]:
        state = {
            "schema_version": 1,
            "lane_id": self.plan.lane_id,
            "lane_plan_sha256": self.plan.config_sha256,
            "status": status,
            "values": values,
            "formal_training": False,
            "formal_validation": False,
        }
        _atomic_json(self.root / "execution-state.json", state)
        self.recorder.transition(status, **values)
        return state

    def run(self) -> dict[str, Any]:
        state_path = self.root / "execution-state.json"
        existing = _read_state(state_path)
        terminal = {
            "complete_search_pipeline_pending_finalist_review",
            "halted_q1_dual_reject",
        }
        if existing is not None and existing.get("status") in terminal:
            return existing
        wait_for_dependency(
            self.dependency_state,
            required_status=self.plan.required_dependency_status,
            poll_seconds=self.plan.poll_seconds,
            recorder=self.recorder,
            expected_sha256=self.plan.dependency_sha256,
            sleep=self.sleep,
        )
        q1 = self._run_q1()
        dual = regate_candidate_report(
            source_report=self.q1_output,
            expected_source_sha256=str(q1["report_sha256"]),
            metric_contract_id=self.plan.q1.metric_contract_id,
        )
        if self.q1_dual_output.is_file():
            existing_dual = json.loads(self.q1_dual_output.read_text(encoding="utf-8"))
            if existing_dual != dual:
                raise FileExistsError("existing qSiLU q1 dual report drifted")
        else:
            _atomic_json(self.q1_dual_output, dual)
        dual_record = _mapping(
            _mapping(dual.get("candidates"), "q1 dual candidates").get(
                self.plan.q1_candidate_id
            ),
            "q1 dual candidate",
        )
        decision = str(dual_record.get("decision", ""))
        if decision not in {"green", "recover", "reject"}:
            raise ValueError("qSiLU q1 dual decision is invalid")
        if decision == "reject":
            return self._write_state(
                "halted_q1_dual_reject",
                q1_report=str(self.q1_output),
                q1_dual_report=str(self.q1_dual_output),
            )
        self._write_state("q1_complete_ready_for_q2", q1_dual_decision=decision)

        q2_plan = materialize_q2_plan(
            self.plan,
            q1_report=self.q1_output,
            dual_report=self.q1_dual_output,
            output_path=self.plan.q2_plan_output,
        )
        preflight = ensure_q2_preflight(q2_plan, self.plan.q2_preflight)
        self._write_state(
            "q2_preflight_complete",
            q2_plan=str(q2_plan),
            q2_plan_sha256=_sha256(q2_plan),
            quantizer_count=preflight.get("activation_quantizers"),
        )
        q2 = run_paired_qat(
            q2_plan,
            device_index=self.device_index,
            maximum_retries_per_arm=self.plan.retry_per_arm,
            recorder=self.recorder,
            gpu_pid_probe=self.gpu_pid_probe,
            sleep=self.sleep,
        )
        self._write_state("q2_paired_qat_complete", **q2)

        parent = lock_qat_parent(
            q2_plan,
            output_path=self.plan.q3_parent_manifest,
            parent_id="v30-qsilu-a8-all-w8-qat-best-joint",
        )
        self._write_state(
            "q3_parent_locked", parent=str(parent), parent_sha256=_sha256(parent)
        )

        profile = ensure_cpu_profile(
            parent,
            output_path=self.plan.q4_profile,
            profile_id=self.plan.q4_profile_id,
        )
        self._write_state(
            "q4_cpu_profile_complete",
            profile=str(self.plan.q4_profile),
            profile_sha256=_sha256(self.plan.q4_profile),
            measurements=_mapping(
                _mapping(profile["summary"], "profile summary")["coverage"],
                "profile coverage",
            )["measurements"],
        )

        q5_plan_path = materialize_progressive_plan(
            locked_parent=parent,
            cpu_profile=self.plan.q4_profile,
            output_path=self.plan.q5_plan_output,
            queue_id="v31-qsilu-progressive-ptq-to-short-qat-v1",
            run_root=self.plan.q5_run_root,
            include_intermediate_uniform_bits=True,
        )
        from .progressive_queue import ProgressiveExperimentQueue, ProgressiveQueuePlan

        q5_plan = ProgressiveQueuePlan.from_yaml(q5_plan_path)
        progressive_state = ProgressiveExperimentQueue(q5_plan).run()
        if progressive_state.get("status") not in {
            "ptq_complete_short_qat_ready",
            "complete_no_short_qat",
        }:
            raise RuntimeError("qSiLU progressive PTQ queue did not complete")
        self._write_state(
            "q5_progressive_ptq_complete",
            progressive_plan=str(q5_plan_path),
            progressive_state=str(self.plan.q5_run_root / "queue-state.json"),
            progressive_status=progressive_state["status"],
        )

        short_queue = self.plan.q5_run_root / "short-qat-queue.json"
        from .progressive_qat_queue import ProgressiveShortQATRunner

        short_qat_state = ProgressiveShortQATRunner(
            short_queue,
            output_root=self.plan.q6_output_root,
            base_qat_plan=q2_plan,
            plan_namespace="v31-qsilu",
            qat_run_root=self.plan.q6_qat_run_root,
            authorization_id="user-2026-09-05-complete-qsilu-quantization-lane",
            plan_date="2026-09-05",
            maximum_retries_per_arm=self.plan.retry_per_arm,
            gpu_pid_probe=self.gpu_pid_probe,
            sleep=self.sleep,
        ).run()
        if short_qat_state.get("status") not in {"complete", "complete_no_jobs"}:
            raise RuntimeError("qSiLU progressive short-QAT queue did not complete")
        self._write_state(
            "q6_short_qat_complete",
            short_qat_status=short_qat_state["status"],
            jobs=len(short_qat_state.get("jobs", [])),
        )

        summary = _complete_lane_summary(
            self.plan,
            parent_manifest=parent,
            cpu_profile=self.plan.q4_profile,
            progressive_state=progressive_state,
            short_qat_state=short_qat_state,
        )
        if self.plan.q7_report.is_file():
            if json.loads(self.plan.q7_report.read_text(encoding="utf-8")) != summary:
                raise FileExistsError("existing qSiLU complete-lane summary drifted")
        else:
            _atomic_json(self.plan.q7_report, summary)
        return self._write_state(
            "complete_search_pipeline_pending_finalist_review",
            summary=str(self.plan.q7_report),
            summary_sha256=_sha256(self.plan.q7_report),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deferred complete qSiLU quantization lane"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/v30-qsilu-complete-quantization-lane-v1.yaml",
    )
    parser.add_argument(
        "--queue-root",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/queues/v30-qsilu-complete-quantization-lane-v1",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--execute-reviewed-queue", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_reviewed_queue:
        parser.error("qSiLU queue execution requires --execute-reviewed-queue")
    plan = QSiluLanePlan.from_yaml(args.plan)
    queue = QSiluLaneQueue(plan, queue_root=args.queue_root, device_index=args.device)
    try:
        result = queue.run()
    except (
        DependencyFailedError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        queue.recorder.transition(
            "queue_failed", error_type=type(error).__name__, message=str(error)
        )
        print(
            json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


__all__ = (
    "DependencyFailedError",
    "QSiluLanePlan",
    "QSiluLaneQueue",
    "QueueRecorder",
    "evaluate_total_dual_gate",
    "lock_qat_parent",
    "materialize_progressive_plan",
    "materialize_q2_plan",
    "run_paired_qat",
    "select_best_joint_epoch",
    "validate_q1_report",
    "wait_for_dependency",
)


if __name__ == "__main__":
    raise SystemExit(main())
