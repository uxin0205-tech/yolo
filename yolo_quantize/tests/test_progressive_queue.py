from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from yolo_quantize.metric_gate import FULL35_MAP50_95_KEYS, FULL35_MAP50_KEYS
from yolo_quantize.progressive_queue import (
    ProgressiveCandidateEvaluation,
    ProgressiveDualMetricGate,
    ProgressiveExperimentQueue,
    ProgressivePolicyOutcome,
    ProgressiveQueuePlan,
    ProgressiveStageSelector,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _metrics(map50: float, map50_95: float) -> dict[str, float]:
    return {
        **{key: map50 for key in FULL35_MAP50_KEYS},
        **{key: map50_95 for key in FULL35_MAP50_95_KEYS},
    }


def test_v29_queue_pins_cpu_parent_stage_order_and_short_qat_recipe() -> None:
    plan = ProgressiveQueuePlan.from_yaml(
        PROJECT_ROOT / "configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml"
    )

    assert plan.queue_id == "v29-v19-progressive-ptq-to-short-qat-v1"
    assert plan.parent.parent_id == "v19-poly-shift-a8-all-w8-qat-epoch5"
    assert len(plan.stages) == 10
    assert sum(len(stage.formats) for stage in plan.stages) == 40
    assert tuple(stage.region for stage in plan.stages[:3]) == (
        "backbone_early",
        "backbone_deep",
        "backbone_attention_safe",
    )
    assert tuple(stage.region for stage in plan.stages[-4:]) == (
        "detect_one2one_tower",
        "detect_one2one_predictor",
        "pose_one2one_tower",
        "pose_one2one_predictor",
    )
    assert plan.validation.detect_batch == 32
    assert plan.validation.pose_batch == 16
    assert plan.poll_seconds == 300
    assert plan.short_qat.epochs == 15
    assert plan.short_qat.patience == 5
    assert plan.short_qat.detect_logical_batch == 128
    assert plan.short_qat.detect_microbatch == 16
    assert plan.short_qat.pose_batch == 16
    assert plan.formal_validation is False


def test_v29_yaml_negative_delta_floors_parse_to_positive_gate_magnitudes() -> None:
    plan = ProgressiveQueuePlan.from_yaml(
        PROJECT_ROOT / "configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml"
    )
    assert plan.recover_map50_floor == 0.04
    assert plan.recover_map50_95_floor == 0.08
    gate = ProgressiveDualMetricGate(
        map50_max_drop=plan.map50_max_drop,
        map50_95_max_drop=plan.map50_95_max_drop,
        recover_map50_floor=plan.recover_map50_floor,
        recover_map50_95_floor=plan.recover_map50_95_floor,
    )

    result = gate.evaluate(
        _metrics(0.984215571241, 0.983941586407),
        _metrics(1.0, 1.0),
        _metrics(0.986158239915, 0.986902059338),
    )

    assert result.decision == "recover"


def test_dual_metric_gate_separates_green_recover_and_reject() -> None:
    accepted = _metrics(1.0, 1.0)
    current = _metrics(0.986, 0.987)
    gate = ProgressiveDualMetricGate(
        map50_max_drop=0.015,
        map50_95_max_drop=0.04,
        recover_map50_floor=0.04,
        recover_map50_95_floor=0.08,
    )

    green = gate.evaluate(_metrics(0.9855, 0.98), accepted, current)
    recover = gate.evaluate(_metrics(0.98, 0.95), accepted, current)
    reject = gate.evaluate(_metrics(0.95, 0.90), accepted, current)

    assert green.decision == "green"
    assert green.worst_total_map50_delta == -0.0145
    assert green.worst_incremental_map50_delta == -0.0005
    assert recover.decision == "recover"
    assert reject.decision == "reject"


def test_stage_selector_uses_size_inside_accuracy_tolerance_and_can_hold_parent() -> (
    None
):
    baseline = ProgressivePolicyOutcome(
        candidate_id="unchanged",
        format_id="w8",
        metrics=_metrics(0.986, 0.987),
        packed_bytes=1000,
        gate_decision="green",
    )
    accuracy = ProgressivePolicyOutcome(
        candidate_id="accuracy",
        format_id="w4",
        metrics=_metrics(0.987, 0.986),
        packed_bytes=900,
        gate_decision="green",
    )
    smaller = ProgressivePolicyOutcome(
        candidate_id="smaller",
        format_id="twn",
        metrics=_metrics(0.9855, 0.985),
        packed_bytes=700,
        gate_decision="green",
    )
    recover = ProgressivePolicyOutcome(
        candidate_id="recover",
        format_id="paper-twn",
        metrics=_metrics(0.98, 0.95),
        packed_bytes=500,
        gate_decision="recover",
    )
    selector = ProgressiveStageSelector(
        map50_tolerance=0.002,
        map50_95_tolerance=0.005,
    )

    selected = selector.select(
        baseline=baseline,
        candidates=(accuracy, smaller, recover),
        promotion_min_savings_bytes=100,
        promotion_allowed=True,
    )
    held = selector.select(
        baseline=baseline,
        candidates=(accuracy, smaller),
        promotion_min_savings_bytes=100,
        promotion_allowed=False,
    )

    assert selected.candidate_id == "smaller"
    assert held.candidate_id == "unchanged"


class _ScriptedEvaluator:
    requires_gpu = False

    def __init__(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics
        self.calls: list[str] = []

    def evaluate(self, *, candidate_id, assignments, output_dir):
        self.calls.append(candidate_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps({"schema_version": 1, "metrics": self.metrics}) + "\n",
            encoding="utf-8",
        )
        return ProgressiveCandidateEvaluation(
            candidate_id=candidate_id,
            metrics=dict(self.metrics),
            output_dir=output_dir,
            metrics_path=metrics_path,
            metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            weight_quantization=(),
            seconds=0.01,
        )


class _FaultingEvaluator(_ScriptedEvaluator):
    def __init__(
        self,
        metrics: dict[str, float],
        *,
        fail_candidate: str | None = None,
        failures: int = 1,
    ) -> None:
        super().__init__(metrics)
        self.fail_candidate = fail_candidate
        self.failures = failures
        self.output_dirs: list[Path] = []

    def evaluate(self, *, candidate_id, assignments, output_dir):
        self.calls.append(candidate_id)
        self.output_dirs.append(output_dir)
        if candidate_id == self.fail_candidate and self.failures:
            self.failures -= 1
            raise RuntimeError("injected evaluator failure")
        self.calls.pop()
        return super().evaluate(
            candidate_id=candidate_id,
            assignments=assignments,
            output_dir=output_dir,
        )


def test_queue_runs_one_stage_through_its_evaluator_seam_and_resumes(
    tmp_path: Path,
) -> None:
    parsed = ProgressiveQueuePlan.from_yaml(
        PROJECT_ROOT / "configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml"
    )
    plan = replace(
        parsed,
        run_root=tmp_path / "queue",
        stages=(parsed.stages[0],),
        maximum_retries_per_candidate=0,
    )
    candidate_metrics = {
        key: min(1.0, value + 0.0001)
        for key, value in plan.locked_parent_metrics.items()
    }
    evaluator = _ScriptedEvaluator(candidate_metrics)
    queue = ProgressiveExperimentQueue(plan, evaluator=evaluator)

    completed = queue.run()
    resumed = queue.run()

    assert completed["status"] == "complete_no_short_qat"
    assert len(completed["locked_assignments"]) == 1
    assert len(evaluator.calls) == 5
    assert resumed == completed
    assert (plan.run_root / "queue-state.json").is_file()
    assert (plan.run_root / "short-qat-queue.json").is_file()


def test_queue_resumes_running_stage_without_repeating_completed_candidate(
    tmp_path: Path,
) -> None:
    parsed = ProgressiveQueuePlan.from_yaml(
        PROJECT_ROOT / "configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml"
    )
    plan = replace(
        parsed,
        run_root=tmp_path / "queue",
        stages=(parsed.stages[0],),
        maximum_retries_per_candidate=0,
    )
    metrics = {
        key: min(1.0, value + 0.0001)
        for key, value in plan.locked_parent_metrics.items()
    }
    failed_id = f"{plan.stages[0].stage_id}--fixed_sd4"
    first = _FaultingEvaluator(metrics, fail_candidate=failed_id)

    try:
        ProgressiveExperimentQueue(plan, evaluator=first).run()
    except RuntimeError as error:
        assert str(error) == "injected evaluator failure"
    else:
        raise AssertionError("injected evaluator failure did not escape")

    resumed = _ScriptedEvaluator(metrics)
    completed = ProgressiveExperimentQueue(plan, evaluator=resumed).run()

    completed_id = f"{plan.stages[0].stage_id}--exact_w4"
    assert completed["status"] == "complete_no_short_qat"
    assert completed_id not in resumed.calls
    assert len(resumed.calls) == 4


def test_queue_retries_candidates_in_immutable_attempt_directories(
    tmp_path: Path,
) -> None:
    parsed = ProgressiveQueuePlan.from_yaml(
        PROJECT_ROOT / "configs/experiments/v29-v19-progressive-ptq-queue-v1.yaml"
    )
    stage = replace(parsed.stages[0], formats=(parsed.stages[0].formats[0],))
    plan = replace(
        parsed,
        run_root=tmp_path / "queue",
        stages=(stage,),
        maximum_retries_per_candidate=1,
    )
    metrics = {
        key: min(1.0, value + 0.0001)
        for key, value in plan.locked_parent_metrics.items()
    }
    candidate_id = f"{stage.stage_id}--exact_w4"
    evaluator = _FaultingEvaluator(
        metrics,
        fail_candidate=candidate_id,
        failures=1,
    )

    completed = ProgressiveExperimentQueue(plan, evaluator=evaluator).run()

    assert completed["status"] == "complete_no_short_qat"
    assert [path.name for path in evaluator.output_dirs] == ["attempt-0", "attempt-1"]
