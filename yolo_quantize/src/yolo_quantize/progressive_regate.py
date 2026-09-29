"""Immutable recovery re-gate for completed progressive PTQ evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .progressive_queue import (
    ProgressiveDualMetricGate,
    ProgressiveQueuePlan,
    ProgressiveStage,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PLAN = (
    _PROJECT_ROOT / "configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml"
)
_DEFAULT_QUEUE_ROOT = (
    _PROJECT_ROOT / "artifacts/queues/v29-v19-progressive-ptq-to-short-qat-v1"
)


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


def _immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _stage_role(stage: ProgressiveStage) -> str:
    if not stage.promotion_allowed or "sentinel" in stage.role:
        return "sentinel"
    return "main"


def _stage_for_candidate(
    plan: ProgressiveQueuePlan, candidate_id: str
) -> ProgressiveStage:
    matches = tuple(
        stage for stage in plan.stages if candidate_id.startswith(f"{stage.stage_id}--")
    )
    if len(matches) != 1:
        raise ValueError(f"recovery candidate has ambiguous stage: {candidate_id}")
    return matches[0]


def _recovery_rank(record: Mapping[str, object]) -> tuple[float, int, str]:
    gate = _mapping(record.get("gate"), "recovery gate")
    return (
        -float(gate["worst_total_map50_delta"]),
        -int(record["savings_from_locked_parent_bytes"]),
        str(record["candidate_id"]),
    )


def select_recovery_candidates(
    plan: ProgressiveQueuePlan, pool: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Apply the reviewed 6-main/2-sentinel caps without mixing their quotas."""

    buckets: dict[str, list[dict[str, object]]] = {"main": [], "sentinel": []}
    seen: set[str] = set()
    for raw in pool:
        record = dict(raw)
        candidate_id = str(record.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            raise ValueError("recovery candidate ids must be non-empty and unique")
        seen.add(candidate_id)
        stage = _stage_for_candidate(plan, candidate_id)
        gate = _mapping(record.get("gate"), "recovery candidate gate")
        if gate.get("decision") != "recover":
            raise ValueError("only corrected recover candidates may enter short QAT")
        if (
            int(record.get("savings_from_locked_parent_bytes", 0))
            < stage.promotion_min_savings_bytes
        ):
            raise ValueError("recovery candidate does not meet stage capacity floor")
        role = _stage_role(stage)
        record["queue_role"] = role
        buckets[role].append(record)

    selected = [
        *sorted(buckets["main"], key=_recovery_rank)[: plan.short_qat.main_candidates],
        *sorted(buckets["sentinel"], key=_recovery_rank)[
            : plan.short_qat.sentinel_candidates
        ],
    ]
    if len(selected) > plan.short_qat.maximum_candidates:
        raise RuntimeError("short-QAT role caps exceed the reviewed maximum")
    return selected


def build_short_qat_queue_payload(
    plan: ProgressiveQueuePlan, jobs: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    if len(jobs) > plan.short_qat.maximum_candidates:
        raise ValueError("short-QAT jobs exceed the reviewed maximum")
    encoded_jobs = [dict(job) for job in jobs]
    return {
        "schema_version": 1,
        "queue_id": f"{plan.queue_id}--short-qat",
        "source_queue_plan": str(plan.config_path),
        "source_queue_plan_sha256": plan.config_sha256,
        "source_parent_manifest": str(plan.parent.config_path),
        "source_parent_manifest_sha256": plan.parent.config_sha256,
        "status": (
            "ready_for_qat_plan_materialization"
            if encoded_jobs
            else "no_eligible_recovery_candidates"
        ),
        "execution_authorized": plan.short_qat.execution_authorized,
        "matched_sham_required": True,
        "recipe": {
            "optimizer": plan.short_qat.optimizer,
            "epochs": plan.short_qat.epochs,
            "patience": plan.short_qat.patience,
            "schedule": {
                "fp32_epochs": plan.short_qat.fp32_epochs,
                "progressive_ramp_epochs": (plan.short_qat.progressive_ramp_epochs),
                "full_quant_epochs": plan.short_qat.full_quant_epochs,
            },
            "batch": {
                "detect_logical": plan.short_qat.detect_logical_batch,
                "detect_microbatch": plan.short_qat.detect_microbatch,
                "pose": plan.short_qat.pose_batch,
            },
            "added_noise": plan.short_qat.added_noise,
            "augmentation": "accepted_full35_exact",
        },
        "jobs": encoded_jobs,
        "formal_validation": False,
        "long_qat_authorized": False,
    }


class ProgressiveRecoveryRegater:
    """Recompute only recovery labels; preserve completed GPU metrics and selection."""

    def __init__(self, plan: ProgressiveQueuePlan, source_state: str | Path) -> None:
        self.plan = plan
        self.source_state = Path(source_state).expanduser().resolve()

    def _load(self) -> dict[str, Any]:
        if not self.source_state.is_file():
            raise FileNotFoundError(self.source_state)
        state = _mapping(
            json.loads(self.source_state.read_text(encoding="utf-8")),
            "completed progressive queue state",
        )
        if (
            state.get("schema_version") != 1
            or state.get("queue_id") != self.plan.queue_id
            or state.get("queue_plan_sha256") != self.plan.config_sha256
            or Path(str(state.get("queue_plan"))).resolve() != self.plan.config_path
            or state.get("status")
            not in {"ptq_complete_short_qat_ready", "complete_no_short_qat"}
            or state.get("formal_validation") is not False
        ):
            raise ValueError("completed progressive queue state drifted")
        stages = _mapping(state.get("stages"), "completed queue stages")
        if set(stages) != {stage.stage_id for stage in self.plan.stages}:
            raise ValueError("completed progressive queue stage set drifted")
        return state

    @staticmethod
    def _gate_payload(result: object) -> dict[str, object]:
        return {
            "decision": result.decision,
            "worst_total_map50_delta": result.worst_total_map50_delta,
            "worst_total_map50_95_delta": result.worst_total_map50_95_delta,
            "worst_incremental_map50_delta": result.worst_incremental_map50_delta,
            "worst_incremental_map50_95_delta": (
                result.worst_incremental_map50_95_delta
            ),
            "total_deltas": result.total_deltas,
            "incremental_deltas": result.incremental_deltas,
        }

    def regate(
        self, *, output_queue: str | Path, output_report: str | Path
    ) -> dict[str, object]:
        state = self._load()
        source_sha256 = _sha256(self.source_state)
        gate = ProgressiveDualMetricGate(
            map50_max_drop=self.plan.map50_max_drop,
            map50_95_max_drop=self.plan.map50_95_max_drop,
            recover_map50_floor=self.plan.recover_map50_floor,
            recover_map50_95_floor=self.plan.recover_map50_95_floor,
        )
        stages = _mapping(state["stages"], "completed queue stages")
        corrected_pool: list[dict[str, object]] = []
        audit: list[dict[str, object]] = []
        before: Counter[str] = Counter()
        after: Counter[str] = Counter()
        for stage in self.plan.stages:
            stage_state = _mapping(stages[stage.stage_id], stage.stage_id)
            if (
                stage_state.get("status") != "completed"
                or stage_state.get("stage_id") != stage.stage_id
            ):
                raise ValueError(f"completed stage drifted: {stage.stage_id}")
            parent_metrics = _mapping(
                stage_state.get("parent_metrics"), f"{stage.stage_id} parent metrics"
            )
            candidates = _mapping(
                stage_state.get("candidates"), f"{stage.stage_id} candidates"
            )
            allowed_formats = {item.format_id for item in stage.formats}
            for candidate_id, raw in candidates.items():
                record = _mapping(raw, f"completed candidate {candidate_id}")
                if (
                    record.get("status") != "completed"
                    or record.get("candidate_id") != candidate_id
                    or record.get("format_id") not in allowed_formats
                    or not candidate_id.startswith(f"{stage.stage_id}--")
                ):
                    raise ValueError(f"completed candidate drifted: {candidate_id}")
                old_gate = _mapping(record.get("gate"), f"old gate {candidate_id}")
                old_decision = str(old_gate.get("decision"))
                result = gate.evaluate(
                    _mapping(record.get("metrics"), f"metrics {candidate_id}"),
                    self.plan.accepted_metrics,
                    parent_metrics,
                )
                if old_decision == "green" and result.decision != "green":
                    raise ValueError("re-gate would change a promoted green decision")
                if old_decision not in {"green", "reject", "recover"}:
                    raise ValueError("source queue has an unknown gate decision")
                before[old_decision] += 1
                after[result.decision] += 1
                corrected = dict(record)
                corrected["gate"] = self._gate_payload(result)
                corrected["regated_from_gate_decision"] = old_decision
                eligible = (
                    result.decision == "recover"
                    and int(record.get("savings_from_locked_parent_bytes", 0))
                    >= stage.promotion_min_savings_bytes
                )
                role = _stage_role(stage)
                if eligible:
                    corrected["queue_role"] = role
                    corrected_pool.append(corrected)
                audit.append(
                    {
                        "candidate_id": candidate_id,
                        "stage_id": stage.stage_id,
                        "format_id": record["format_id"],
                        "old_decision": old_decision,
                        "corrected_decision": result.decision,
                        "eligible_for_short_qat": eligible,
                        "queue_role": role,
                        "savings_from_locked_parent_bytes": int(
                            record.get("savings_from_locked_parent_bytes", 0)
                        ),
                        "worst_total_map50_delta": (result.worst_total_map50_delta),
                        "worst_total_map50_95_delta": (
                            result.worst_total_map50_95_delta
                        ),
                    }
                )

        selected = select_recovery_candidates(self.plan, corrected_pool)
        selected_ids = {str(item["candidate_id"]) for item in selected}
        for item in audit:
            item["selected_for_short_qat"] = item["candidate_id"] in selected_ids
        queue_path = Path(output_queue).expanduser().resolve()
        report_path = Path(output_report).expanduser().resolve()
        report = {
            "schema_version": 1,
            "report_id": f"{self.plan.queue_id}--recovery-regate-v2",
            "status": "completed",
            "cause": "negative_delta_floor_was_double_negated_in_plan_parser",
            "source_state": str(self.source_state),
            "source_state_sha256": source_sha256,
            "source_queue_plan": str(self.plan.config_path),
            "source_queue_plan_sha256": self.plan.config_sha256,
            "output_queue": str(queue_path),
            "gate_contract": {
                "map50_green_floor": -self.plan.map50_max_drop,
                "map50_95_green_floor": -self.plan.map50_95_max_drop,
                "map50_recover_floor": -self.plan.recover_map50_floor,
                "map50_95_recover_floor": -self.plan.recover_map50_95_floor,
            },
            "decision_counts_before": dict(sorted(before.items())),
            "decision_counts_corrected": dict(sorted(after.items())),
            "corrected_recover_candidates": len(corrected_pool),
            "selected_main": sum(item.get("queue_role") == "main" for item in selected),
            "selected_sentinel": sum(
                item.get("queue_role") == "sentinel" for item in selected
            ),
            "selected_candidate_ids": [item["candidate_id"] for item in selected],
            "candidates": audit,
            "source_artifacts_modified": False,
            "gpu_validation_repeated": False,
            "formal_validation": False,
        }
        _immutable_json(report_path, report)
        payload = build_short_qat_queue_payload(self.plan, selected)
        payload["recovery_regate"] = {
            "report": str(report_path),
            "report_sha256": _sha256(report_path),
            "source_state": str(self.source_state),
            "source_state_sha256": source_sha256,
            "corrected_recover_candidates": len(corrected_pool),
            "selected_main": report["selected_main"],
            "selected_sentinel": report["selected_sentinel"],
        }
        _immutable_json(queue_path, payload)
        return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-gate completed V29 PTQ metrics without repeating GPU validation"
    )
    parser.add_argument("--plan", type=Path, default=_DEFAULT_PLAN)
    parser.add_argument(
        "--source-state", type=Path, default=_DEFAULT_QUEUE_ROOT / "queue-state.json"
    )
    parser.add_argument(
        "--output-queue",
        type=Path,
        default=_DEFAULT_QUEUE_ROOT / "short-qat-queue-regated-v2.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=(
            _PROJECT_ROOT
            / "artifacts/reports/v29-v19-progressive-recovery-regate-v2.json"
        ),
    )
    parser.add_argument("--regate-completed-queue", action="store_true")
    args = parser.parse_args(argv)
    if not args.regate_completed_queue:
        parser.error("re-gate requires --regate-completed-queue acknowledgement")
    plan = ProgressiveQueuePlan.from_yaml(args.plan)
    payload = ProgressiveRecoveryRegater(plan, args.source_state).regate(
        output_queue=args.output_queue,
        output_report=args.output_report,
    )
    regate = _mapping(payload["recovery_regate"], "recovery re-gate result")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "jobs": len(payload["jobs"]),
                "corrected_recover_candidates": regate["corrected_recover_candidates"],
                "selected_main": regate["selected_main"],
                "selected_sentinel": regate["selected_sentinel"],
                "output_queue": str(args.output_queue.resolve()),
                "report": regate["report"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


__all__ = (
    "ProgressiveRecoveryRegater",
    "build_short_qat_queue_payload",
    "main",
    "select_recovery_candidates",
)


if __name__ == "__main__":
    raise SystemExit(main())
