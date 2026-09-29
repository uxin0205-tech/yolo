from __future__ import annotations

from yolo_quantize import (
    ActivationIntakeReport,
    ActivationJobSummary,
    CoupledExperimentPlanner,
)


def _job(
    activation: str,
    *,
    upstream_gate_status: str,
    quantization_eligibility: str,
) -> ActivationJobSummary:
    return ActivationJobSummary(
        queue_name="activation-queue",
        job_id=f"short-recovery-{activation}",
        run_name=f"{activation}-run",
        activation=activation,
        kind="train",
        phase="short_recovery",
        status="completed",
        upstream_gate_status=upstream_gate_status,
        quantization_eligibility=quantization_eligibility,
        classification=quantization_eligibility,
        gate_passed=upstream_gate_status == "passed",
        worst_delta=-0.02,
        failed_metrics=(),
        deltas={
            "coco/box/map50_95": -0.02,
            "bbat/box/map50_95": -0.01,
            "bbat/pose/map50_95": -0.003,
        },
        metrics={
            "coco/box/map50_95": 0.48,
            "bbat/box/map50_95": 0.62,
            "bbat/pose/map50_95": 0.897,
        },
        primary_metrics={
            "coco/box/map50_95": 0.48,
            "bbat/box/map50_95": 0.62,
            "bbat/pose/map50_95": 0.897,
        },
    )


def test_activation_screen_keeps_upstream_failures_that_pass_quantization_gate() -> (
    None
):
    report = ActivationIntakeReport(
        ready=False,
        blockers=("finalist queue is pending",),
        queues=(),
        jobs=(
            _job(
                "qsilu_pq",
                upstream_gate_status="passed",
                quantization_eligibility="screen_candidate",
            ),
            _job(
                "poly_quality",
                upstream_gate_status="failed",
                quantization_eligibility="screen_candidate",
            ),
            _job(
                "hardswish",
                upstream_gate_status="failed",
                quantization_eligibility="screen_candidate",
            ),
            _job(
                "relu",
                upstream_gate_status="failed",
                quantization_eligibility="outside_screen_gate",
            ),
        ),
        files=(),
    )

    plan = CoupledExperimentPlanner().activation_screen(report)

    assert plan.candidate_activations == ("silu", "qsilu_pq", "hardswish")
    assert plan.preferred_seed == "qsilu_pq"
    assert plan.is_final_selection is False
    assert {
        cell.activation_bits
        for cell in plan.cells
        if cell.activation == "hardswish" and cell.activation_bits is not None
    } == {3, 4, 5, 6, 7, 8}
    assert all(cell.activation != "poly_quality" for cell in plan.cells)
    assert all(cell.weight_format == "fp32" for cell in plan.cells)


def test_weight_matrix_crosses_each_complete_activation_policy_without_splicing() -> (
    None
):
    report = ActivationIntakeReport(
        ready=False,
        blockers=(),
        queues=(),
        jobs=(
            _job(
                "qsilu_pq",
                upstream_gate_status="passed",
                quantization_eligibility="screen_candidate",
            ),
            _job(
                "hardswish",
                upstream_gate_status="failed",
                quantization_eligibility="screen_candidate",
            ),
        ),
        files=(),
    )
    planner = CoupledExperimentPlanner()
    activation_plan = planner.activation_screen(report)
    finalists = (
        "qsilu_pq--lsq-plus-a6",
        "hardswish--lsq-plus-a8",
    )

    weight_plan = planner.weight_matrix(activation_plan, finalists)

    assert weight_plan.activation_policy_ids == finalists
    assert weight_plan.weight_families == (
        "int8",
        "int4",
        "fixed_sd4",
        "ls_sd4",
        "paper_twn",
        "channel_twn_ttq",
    )
    assert len(weight_plan.cells) == len(finalists) * len(weight_plan.weight_families)
    assert {cell.activation_policy_id for cell in weight_plan.cells} == set(finalists)
    assert all(
        cell.joint_policy_id == f"{cell.activation_policy_id}--{cell.weight_family}"
        for cell in weight_plan.cells
    )


def test_pending_finalist_blocks_finalization_but_not_authorized_screening() -> None:
    pending = "finalist-queue has 1 pending job(s)"
    interrupted = (
        "finalist-queue is interrupted: pending jobs remain but its runner lock "
        "is not held"
    )
    report = ActivationIntakeReport(
        ready=False,
        blockers=(pending, interrupted),
        queues=(),
        jobs=(
            _job(
                "qsilu_pq",
                upstream_gate_status="passed",
                quantization_eligibility="screen_candidate",
            ),
        ),
        files=(),
        finalization_only_blockers=(pending, interrupted),
        warnings=(interrupted,),
    )

    assert report.screening_ready
    assert not report.finalization_ready
    assert report.screening_blockers == ()
