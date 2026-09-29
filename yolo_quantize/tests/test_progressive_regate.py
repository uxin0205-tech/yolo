from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from yolo_quantize.metric_gate import FULL35_MAP50_95_KEYS, FULL35_MAP50_KEYS
from yolo_quantize.progressive_queue import ProgressiveQueuePlan
from yolo_quantize.progressive_regate import ProgressiveRecoveryRegater

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PLAN = PROJECT_ROOT / "configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml"


def _candidate(plan, stage, named_format, *, map50_drop, map50_95_drop, savings):
    metrics = {
        **{key: plan.accepted_metrics[key] - map50_drop for key in FULL35_MAP50_KEYS},
        **{
            key: plan.accepted_metrics[key] - map50_95_drop
            for key in FULL35_MAP50_95_KEYS
        },
    }
    return {
        "candidate_id": f"{stage.stage_id}--{named_format.format_id}",
        "format_id": named_format.format_id,
        "weight_format_id": named_format.spec.format_id,
        "cpu_format_id": named_format.cpu_format_id,
        "assignments": [
            {
                "stage_id": stage.stage_id,
                "region": stage.region,
                "path": stage.path,
                "format_id": named_format.format_id,
                "weight_format_id": named_format.spec.format_id,
                "cpu_format_id": named_format.cpu_format_id,
            }
        ],
        "metrics": metrics,
        "packed_bytes": 1000 - savings,
        "savings_from_locked_parent_bytes": savings,
        "gate": {"decision": "reject"},
        "seconds": 1.0,
        "status": "completed",
    }


def test_completed_queue_can_be_immutably_regated_into_role_capped_qat_jobs(
    tmp_path: Path,
) -> None:
    parsed = ProgressiveQueuePlan.from_yaml(QUEUE_PLAN)
    main_stage = parsed.stages[0]
    sentinel_stage = parsed.stages[-1]
    plan = replace(
        parsed,
        stages=(main_stage, sentinel_stage),
        short_qat=replace(
            parsed.short_qat,
            maximum_candidates=2,
            main_candidates=1,
            sentinel_candidates=1,
        ),
        run_root=tmp_path / "source-queue",
    )
    main_candidates = {
        record["candidate_id"]: record
        for record in (
            _candidate(
                plan,
                main_stage,
                named_format,
                map50_drop=0.02 + index * 0.001,
                map50_95_drop=0.03,
                savings=main_stage.promotion_min_savings_bytes + index + 1,
            )
            for index, named_format in enumerate(main_stage.formats[:2])
        )
    }
    sentinel_candidates = {
        record["candidate_id"]: record
        for record in (
            _candidate(
                plan,
                sentinel_stage,
                named_format,
                map50_drop=0.025 + index * 0.001,
                map50_95_drop=0.05,
                savings=index + 1,
            )
            for index, named_format in enumerate(sentinel_stage.formats[:2])
        )
    }
    state = {
        "schema_version": 1,
        "queue_id": plan.queue_id,
        "queue_plan": str(plan.config_path),
        "queue_plan_sha256": plan.config_sha256,
        "status": "complete_no_short_qat",
        "stages": {
            main_stage.stage_id: {
                "status": "completed",
                "stage_index": 0,
                "stage_id": main_stage.stage_id,
                "parent_metrics": plan.locked_parent_metrics,
                "candidates": main_candidates,
            },
            sentinel_stage.stage_id: {
                "status": "completed",
                "stage_index": 1,
                "stage_id": sentinel_stage.stage_id,
                "parent_metrics": plan.locked_parent_metrics,
                "candidates": sentinel_candidates,
            },
        },
        "formal_validation": False,
    }
    source = tmp_path / "queue-state.json"
    source.write_text(json.dumps(state), encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    result = ProgressiveRecoveryRegater(plan, source).regate(
        output_queue=tmp_path / "short-qat-queue-regated-v2.json",
        output_report=tmp_path / "recovery-regate-v2.json",
    )
    repeated = ProgressiveRecoveryRegater(plan, source).regate(
        output_queue=tmp_path / "short-qat-queue-regated-v2.json",
        output_report=tmp_path / "recovery-regate-v2.json",
    )

    assert result == repeated
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    assert result["status"] == "ready_for_qat_plan_materialization"
    assert [job["queue_role"] for job in result["jobs"]] == ["main", "sentinel"]
    assert all(job["gate"]["decision"] == "recover" for job in result["jobs"])
    assert result["recovery_regate"]["corrected_recover_candidates"] == 4
    assert result["recovery_regate"]["selected_main"] == 1
    assert result["recovery_regate"]["selected_sentinel"] == 1
