from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from yolo_attention.config import BasisKind, BiasKind, ScaleMode, VariantConfig
from yolo_attention.queue_model import JobStatus, QueueResult
from yolo_attention.queue_policy import SelectionDecision
from yolo_attention.queue_workflow import (
    create_initial_state,
    materialize_after_selection,
    next_runnable_job,
    refresh_readiness,
)

ROOT = Path(__file__).resolve().parents[1]


def _succeed(state, job_id: str, *, checkpoint: str | None = None, metric: float | None = None):
    jobs = tuple(
        replace(
            job,
            status=JobStatus.SUCCEEDED,
            result=QueueResult(map50_95=metric, checkpoint_path=checkpoint),
            checkpoint_path=checkpoint,
        )
        if job.id == job_id
        else job
        for job in state.jobs
    )
    return replace(state, jobs=jobs)


def test_initial_queue_is_serial_but_screening_model_parent_stays_p0() -> None:
    state = create_initial_state(ROOT)

    assert [job.id for job in state.jobs] == [
        "b26-fp",
        "p0",
        "i-scr",
        "h-scr",
        "t5-scr",
        "architecture-select",
    ]
    assert state.job("b26-fp").status is JobStatus.READY
    assert state.job("h-scr").parent_job_ids == ("p0", "i-scr")
    assert state.job("h-scr").model_parent_job_id == "p0"
    assert state.job("t5-scr").parent_job_ids == ("p0", "i-scr", "h-scr")
    assert all(not job.id.startswith("q") for job in state.jobs)


def test_refresh_requires_existing_model_parent_checkpoint(tmp_path: Path) -> None:
    state = create_initial_state(tmp_path)
    missing = tmp_path / "weights" / "yolo26m.pt"
    state = _succeed(state, "b26-fp", checkpoint=str(missing), metric=0.4)

    still_blocked = refresh_readiness(state)
    missing.parent.mkdir(parents=True)
    missing.touch()
    ready = refresh_readiness(state)

    assert still_blocked.job("p0").status is JobStatus.BLOCKED
    assert ready.job("p0").status is JobStatus.READY


def test_next_runnable_prefers_explicit_queued_retry() -> None:
    state = create_initial_state(ROOT)
    jobs = tuple(
        replace(job, status=JobStatus.QUEUED) if job.id == "p0" else replace(job, status=JobStatus.BLOCKED)
        for job in state.jobs
    )
    state = replace(state, jobs=jobs)

    assert next_runnable_job(state).id == "p0"


def test_architecture_winner_expansion_preserves_winner_configuration(tmp_path: Path) -> None:
    winner_path = tmp_path / "h-winner.yaml"
    VariantConfig(
        name="H-SCR",
        basis=BasisKind.HADAMARD,
        bias=BiasKind.DECOMPOSED_2D,
        scale_mode=ScaleMode.FIXED_HEAD,
    ).to_yaml(winner_path)
    state = create_initial_state(tmp_path)
    jobs = tuple(
        replace(job, variant_path=str(winner_path)) if job.id == "h-scr" else job for job in state.jobs
    )
    state = replace(state, jobs=jobs)
    decision = SelectionDecision(
        winners=("h-scr",),
        skipped=("i-scr", "t5-scr"),
        reason="test winner",
    )

    expanded = materialize_after_selection(
        state,
        "architecture-select",
        decision,
        generated_root=tmp_path / "generated",
    )

    direct = VariantConfig.from_yaml(expanded.job("w-dir").variant_path)
    progressive = VariantConfig.from_yaml(expanded.job("w-prog").variant_path)
    assert direct.basis is BasisKind.HADAMARD
    assert direct.bias is BiasKind.DECOMPOSED_2D
    assert direct.scale_mode is ScaleMode.FIXED_HEAD
    assert direct.progressive is False
    assert progressive.progressive is True
    assert expanded.job("w-dir").model_parent_job_id == "h-scr"
    assert [job.id for job in expanded.jobs].count("w-dir") == 1

    repeated = materialize_after_selection(
        expanded,
        "architecture-select",
        decision,
        generated_root=tmp_path / "generated",
    )
    assert repeated == expanded


def test_selection_expansions_materialize_complete_pre_quantization_funnel(tmp_path: Path) -> None:
    winner_path = tmp_path / "winner.yaml"
    VariantConfig(name="I-SCR", basis=BasisKind.IDENTITY).to_yaml(winner_path)
    state = create_initial_state(tmp_path)
    state = replace(
        state,
        jobs=tuple(
            replace(job, variant_path=str(winner_path)) if job.id == "i-scr" else job for job in state.jobs
        ),
    )
    generated = tmp_path / "generated"

    state = materialize_after_selection(
        state,
        "architecture-select",
        SelectionDecision(("i-scr",), ("h-scr", "t5-scr"), "architecture"),
        generated_root=generated,
    )
    state = materialize_after_selection(
        state,
        "recovery-select",
        SelectionDecision(("w-dir",), ("w-prog",), "recovery"),
        generated_root=generated,
    )
    assert {"v1-dyn", "v1-shead", "v1-p2", "scale-select"} <= {job.id for job in state.jobs}
    assert VariantConfig.from_yaml(state.job("v1-p2").variant_path).scale_mode is ScaleMode.POWER_OF_TWO

    state = materialize_after_selection(
        state,
        "scale-select",
        SelectionDecision(("v1-p2",), ("v1-dyn", "v1-shead"), "scale"),
        generated_root=generated,
    )
    assert {"v1-b0", "v1-bd", "v1-br", "a0-select"} <= {job.id for job in state.jobs}
    assert VariantConfig.from_yaml(state.job("v1-br").variant_path).bias is BiasKind.DECOMPOSED_2D

    state = materialize_after_selection(
        state,
        "a0-select",
        SelectionDecision(("v1-br",), ("v1-b0", "v1-bd"), "a0"),
        generated_root=generated,
    )
    assert {"n0-exact", "n0-lut", "n0-mk1", "n0-mk3", "n0-mk5", "n0-select"} <= {job.id for job in state.jobs}

    state = materialize_after_selection(
        state,
        "n0-select",
        SelectionDecision(("n0-lut", "n0-mk3"), ("n0-relu",), "n0"),
        generated_root=generated,
    )
    assert {"n1-lut", "n1-mk3", "normalization-select"} <= {job.id for job in state.jobs}

    state = materialize_after_selection(
        state,
        "normalization-select",
        SelectionDecision(("n1-lut",), ("n0-exact", "n1-mk3"), "normalization"),
        generated_root=generated,
    )
    assert {"d0-idx", "d0-select"} <= {job.id for job in state.jobs}
    assert state.job("d0-idx").model_parent_job_id == "v1-br"

    state = materialize_after_selection(
        state,
        "d0-select",
        SelectionDecision(("d0-idx",), (), "d0"),
        generated_root=generated,
    )
    assert {
        "d1-shared",
        "d1-pattn",
        "d1-phead",
        "d1-shared-10",
        "d1-pattn-10",
        "d1-phead-10",
        "d1-select",
    } <= {job.id for job in state.jobs}
    assert state.job("d1-shared-10").model_parent_job_id == "d1-shared"
    assert state.job("d1-pattn-10").model_parent_job_id == "d1-pattn"
    assert state.job("d1-phead-10").model_parent_job_id == "d1-phead"
    assert state.job("d1-select").parent_job_ids == (
        "d1-shared-10",
        "d1-pattn-10",
        "d1-phead-10",
    )

    state = materialize_after_selection(
        state,
        "d1-select",
        SelectionDecision(("d1-shared-10",), ("d1-pattn-10", "d1-phead-10"), "d1"),
        generated_root=generated,
    )
    assert {"d2-fp", "d2-1p", "d2-2p", "d2-select"} <= {job.id for job in state.jobs}
    assert state.job("d2-fp").model_parent_job_id == "d1-shared-10"

    state = materialize_after_selection(
        state,
        "d2-select",
        SelectionDecision(("d2-2p",), ("d2-fp", "d2-1p"), "d2"),
        generated_root=generated,
    )
    assert {"r0-div", "r1-rlut", "r2-pshift", "denominator-select"} <= {job.id for job in state.jobs}

    state = materialize_after_selection(
        state,
        "denominator-select",
        SelectionDecision(("r1-rlut",), ("r0-div", "r2-pshift"), "denominator"),
        generated_root=generated,
    )
    assert {"bdcn-select", "final-select"} <= {job.id for job in state.jobs}
    assert all(not job.id.startswith("q") for job in state.jobs)

    state = materialize_after_selection(
        state,
        "final-select",
        SelectionDecision(("r1-rlut",), ("n1-lut",), "final"),
        generated_root=generated,
    )
    assert state.job("a-final").model_parent_job_id == "r1-rlut"
    assert VariantConfig.from_yaml(state.job("a-final").variant_path).name == "A-FINAL"


def test_denominator_gate_can_materialize_one_newton_job(tmp_path: Path) -> None:
    parent_path = tmp_path / "r1.yaml"
    VariantConfig(
        name="R1-RLUT",
        normalization="bdcn",
        bdcn_codebook="learned",
        bdcn_sharing="global",
        bdcn_projection="two_pot",
        bdcn_denominator="reciprocal_lut",
    ).to_yaml(parent_path)
    state = create_initial_state(tmp_path)
    placeholder = replace(
        state.job("p0"),
        id="r1-rlut",
        run_name="R1-RLUT",
        order=6,
        variant_path=str(parent_path),
        parent_job_ids=(),
        model_parent_job_id=None,
    )
    denominator_select = replace(
        state.job("architecture-select"),
        id="denominator-select",
        run_name="DENOMINATOR-SELECT",
        order=7,
        parent_job_ids=("r1-rlut",),
    )
    normalization_select = replace(
        state.job("architecture-select"),
        id="normalization-select",
        run_name="NORMALIZATION-SELECT",
        order=8,
        parent_job_ids=(),
    )
    state = replace(
        state,
        jobs=state.jobs[:-1] + (placeholder, denominator_select, normalization_select),
    )

    expanded = materialize_after_selection(
        state,
        "denominator-select",
        SelectionDecision((), (), "missed gate", ("r1-newton",)),
        generated_root=tmp_path / "generated",
    )

    newton = VariantConfig.from_yaml(expanded.job("r1-newton").variant_path)
    assert newton.bdcn_reciprocal_newton_steps == 1
    assert expanded.job("bdcn-select").parent_job_ids == ("r1-newton",)


def test_d1_uncertainty_runs_winner_seed_one_before_d2(tmp_path: Path) -> None:
    winner_path = tmp_path / "d1-shared-10.yaml"
    VariantConfig(
        name="D1-SHARED-10",
        basis=BasisKind.HADAMARD,
        normalization="bdcn",
        bdcn_codebook="learned",
        bdcn_sharing="global",
        bdcn_projection="float",
        bdcn_denominator="exact",
    ).to_yaml(winner_path)
    state = create_initial_state(tmp_path)
    base = replace(
        state.job("p0"),
        id="d1-shared-10",
        run_name="D1-SHARED-10",
        order=6,
        variant_path=str(winner_path),
        parent_job_ids=(),
        model_parent_job_id=None,
    )
    selection = replace(
        state.job("architecture-select"),
        id="d1-select",
        run_name="D1-SELECT",
        order=7,
        parent_job_ids=("d1-shared-10",),
    )
    d0 = replace(
        state.job("p0"),
        id="d0-idx",
        run_name="D0-IDX",
        order=8,
        parent_job_ids=(),
        model_parent_job_id=None,
    )
    state = replace(state, jobs=state.jobs[:-1] + (base, selection, d0))

    seeded = materialize_after_selection(
        state,
        "d1-select",
        SelectionDecision(
            ("d1-shared-10",),
            (),
            "D1 requires seed confirmation",
            ("d1-seed1",),
        ),
        generated_root=tmp_path / "generated",
    )

    assert seeded.job("d1-seed1").model_parent_job_id == "d0-idx"
    assert seeded.job("d1-confirm-select").parent_job_ids == ("d1-seed1",)
    assert "d2-fp" not in {job.id for job in seeded.jobs}

    confirmed = materialize_after_selection(
        seeded,
        "d1-confirm-select",
        SelectionDecision(("d1-shared-10",), ("d1-seed1",), "seed recorded"),
        generated_root=tmp_path / "generated",
    )

    assert confirmed.job("d2-fp").model_parent_job_id == "d1-shared-10"
    assert confirmed.job("d2-fp").parent_job_ids == ("d1-confirm-select",)
