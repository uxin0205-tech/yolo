"""Experiment queue 的 declarative graph construction 與 dynamic expansion。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from .config import (
    BDCNCodebookKind,
    BDCNDenominator,
    BDCNProjection,
    BDCNSharing,
    BiasKind,
    NormalizationKind,
    ScaleMode,
    VariantConfig,
)
from .queue_model import JobKind, JobStatus, QueueJob, QueueState
from .queue_policy import SelectionDecision


def validate_queue_environment(state: QueueState) -> dict[str, object]:
    errors: list[str] = []
    warnings = ["optional quantization is locked"]
    try:
        state.validate()
    except Exception as exc:  # noqa: BLE001 - validation reports malformed persisted state.
        errors.append(str(exc))
    root = Path(state.project_root)
    checked_data: set[Path] = set()
    for job in state.jobs:
        for field, parser in (
            (job.variant_path, VariantConfig.from_yaml),
            (job.training_path, None),
            (job.evaluation_path, None),
        ):
            if field is None:
                continue
            path = Path(field)
            if not path.is_file():
                errors.append(f"missing configuration: {path}")
                continue
            try:
                if parser is not None:
                    parser(path)
                else:
                    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise TypeError("configuration must contain a mapping")
                    if "data" in payload:
                        data_path = Path(payload["data"])
                        if not data_path.is_absolute():
                            data_path = root / data_path
                        checked_data.add(data_path.resolve())
            except Exception as exc:  # noqa: BLE001 - config validators expose several error types.
                errors.append(f"invalid configuration {path}: {exc}")
    for data_path in sorted(checked_data):
        if not data_path.is_file():
            errors.append(f"missing dataset YAML: {data_path}")
            continue
        try:
            data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            dataset_root = Path(data["path"])
            if not dataset_root.is_dir():
                errors.append(f"missing dataset root: {dataset_root}")
        except Exception as exc:  # noqa: BLE001 - report every malformed dataset configuration.
            errors.append(f"invalid dataset YAML {data_path}: {exc}")
    baseline = state.job("b26-fp")
    if baseline.parent_checkpoint is None or not Path(baseline.parent_checkpoint).is_file():
        errors.append(f"missing baseline weights: {baseline.parent_checkpoint}")
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def _project_path(root: Path, relative: str) -> str:
    return str((root / relative).resolve())


def create_initial_state(project_root: Path) -> QueueState:
    root = project_root.resolve()
    evaluation = _project_path(root, "configs/evaluation/coco2017.yaml")
    jobs = (
        QueueJob(
            id="b26-fp",
            run_name="B26-FP",
            stage="baseline",
            kind=JobKind.EVALUATE,
            order=0,
            status=JobStatus.READY,
            evaluation_path=evaluation,
            parent_checkpoint=_project_path(root, "weights/yolo26m.pt"),
            requires_gpu=True,
        ),
        QueueJob(
            id="p0",
            run_name="P0",
            stage="validation",
            kind=JobKind.VALIDATE,
            order=1,
            variant_path=_project_path(root, "configs/variants/p0.yaml"),
            parent_job_ids=("b26-fp",),
            model_parent_job_id="b26-fp",
        ),
        QueueJob(
            id="i-scr",
            run_name="I-SCR",
            stage="architecture-screening",
            kind=JobKind.TRAIN,
            order=2,
            variant_path=_project_path(root, "configs/variants/i-screen.yaml"),
            training_path=_project_path(root, "configs/training/screening.yaml"),
            parent_job_ids=("p0",),
            model_parent_job_id="p0",
            requires_gpu=True,
        ),
        QueueJob(
            id="h-scr",
            run_name="H-SCR",
            stage="architecture-screening",
            kind=JobKind.TRAIN,
            order=3,
            variant_path=_project_path(root, "configs/variants/h-screen.yaml"),
            training_path=_project_path(root, "configs/training/screening.yaml"),
            parent_job_ids=("p0", "i-scr"),
            model_parent_job_id="p0",
            requires_gpu=True,
        ),
        QueueJob(
            id="t5-scr",
            run_name="T5-SCR",
            stage="architecture-screening",
            kind=JobKind.TRAIN,
            order=4,
            variant_path=_project_path(root, "configs/variants/t5-screen.yaml"),
            training_path=_project_path(root, "configs/training/screening.yaml"),
            parent_job_ids=("p0", "i-scr", "h-scr"),
            model_parent_job_id="p0",
            requires_gpu=True,
        ),
        QueueJob(
            id="architecture-select",
            run_name="ARCHITECTURE-SELECT",
            stage="architecture-selection",
            kind=JobKind.SELECT,
            order=5,
            parent_job_ids=("i-scr", "h-scr", "t5-scr"),
        ),
    )
    state = QueueState.initial(jobs, project_root=str(root))
    state.validate()
    return state


def refresh_readiness(state: QueueState) -> QueueState:
    jobs: list[QueueJob] = []
    for job in state.jobs:
        if job.status not in {JobStatus.BLOCKED, JobStatus.READY}:
            jobs.append(job)
            continue
        parents_succeeded = all(
            state.job(parent).status is JobStatus.SUCCEEDED for parent in job.parent_job_ids
        )
        checkpoint: str | None = job.parent_checkpoint
        model_parent_ready = True
        if job.model_parent_job_id is not None:
            parent = state.job(job.model_parent_job_id)
            checkpoint = (
                parent.result.checkpoint_path if parent.result is not None else parent.checkpoint_path
            )
            model_parent_ready = checkpoint is not None and Path(checkpoint).is_file()
        ready = parents_succeeded and model_parent_ready
        jobs.append(
            replace(
                job,
                status=JobStatus.READY if ready else JobStatus.BLOCKED,
                parent_checkpoint=checkpoint,
            )
        )
    refreshed = replace(state, jobs=tuple(jobs), revision=state.revision + 1)
    refreshed.validate()
    return refreshed


def next_runnable_job(state: QueueState) -> QueueJob | None:
    if any(job.status is JobStatus.RUNNING for job in state.jobs):
        return None
    queued = sorted((job for job in state.jobs if job.status is JobStatus.QUEUED), key=lambda job: job.order)
    if queued:
        return queued[0]
    ready = sorted((job for job in state.jobs if job.status is JobStatus.READY), key=lambda job: job.order)
    return ready[0] if ready else None


def derive_variant(
    parent: VariantConfig,
    *,
    name: str,
    destination: Path,
    changes: dict[str, object],
) -> str:
    derived = replace(parent, name=name, **changes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if VariantConfig.from_yaml(destination) != derived:
            raise FileExistsError(f"generated variant differs from existing file: {destination}")
    else:
        derived.to_yaml(destination)
    return str(destination.resolve())


def _record_selection(
    state: QueueState,
    selection_job_id: str,
    decision: SelectionDecision,
) -> QueueState:
    payload = {
        "winners": list(decision.winners),
        "skipped": list(decision.skipped),
        "reason": decision.reason,
        "expand": list(decision.expand),
    }
    existing = state.selections.get(selection_job_id)
    if existing is not None:
        if existing != payload:
            raise ValueError(f"selection {selection_job_id!r} was already recorded differently")
        return state
    selections = dict(state.selections)
    selections[selection_job_id] = payload
    return replace(state, selections=selections, revision=state.revision + 1)


def _append_jobs(state: QueueState, jobs: tuple[QueueJob, ...]) -> QueueState:
    existing = {job.id for job in state.jobs}
    if any(job.id in existing for job in jobs):
        raise ValueError("dynamic expansion attempted to append an existing job")
    expanded = replace(state, jobs=state.jobs + jobs, revision=state.revision + 1)
    expanded.validate()
    return expanded


def _variant(
    parent: VariantConfig,
    *,
    job_id: str,
    name: str,
    generated_root: Path,
    **changes: object,
) -> str:
    return derive_variant(
        parent,
        name=name,
        destination=generated_root / job_id / "variant.yaml",
        changes=changes,
    )


def _evaluation_job(
    state: QueueState,
    *,
    job_id: str,
    run_name: str,
    stage: str,
    order: int,
    variant_path: str,
    parent_job_ids: tuple[str, ...],
    model_parent_job_id: str,
) -> QueueJob:
    return QueueJob(
        id=job_id,
        run_name=run_name,
        stage=stage,
        kind=JobKind.EVALUATE,
        order=order,
        variant_path=variant_path,
        evaluation_path=_project_path(Path(state.project_root), "configs/evaluation/coco2017.yaml"),
        parent_job_ids=parent_job_ids,
        model_parent_job_id=model_parent_job_id,
        requires_gpu=True,
    )


def _training_job(
    state: QueueState,
    *,
    job_id: str,
    run_name: str,
    stage: str,
    order: int,
    variant_path: str,
    training_file: str,
    parent_job_ids: tuple[str, ...],
    model_parent_job_id: str,
) -> QueueJob:
    return QueueJob(
        id=job_id,
        run_name=run_name,
        stage=stage,
        kind=JobKind.TRAIN,
        order=order,
        variant_path=variant_path,
        training_path=_project_path(Path(state.project_root), training_file),
        parent_job_ids=parent_job_ids,
        model_parent_job_id=model_parent_job_id,
        requires_gpu=True,
    )


def materialize_after_selection(
    state: QueueState,
    selection_job_id: str,
    decision: SelectionDecision,
    *,
    generated_root: Path,
) -> QueueState:
    first_by_selection = {
        "architecture-select": "w-dir",
        "recovery-select": "v1-dyn",
        "scale-select": "v1-b0",
        "a0-select": "n0-exact",
        "n0-select": "normalization-select",
        "normalization-select": "d0-idx",
        "d0-select": "d1-shared",
        "d1-select": "d2-fp",
        "d1-confirm-select": "d2-fp",
        "d2-select": "r0-div",
        "denominator-select": "bdcn-select",
        "final-select": "a-final",
    }
    if selection_job_id not in first_by_selection:
        raise ValueError(f"unsupported selection expansion: {selection_job_id}")
    state = _record_selection(state, selection_job_id, decision)
    if selection_job_id == "d1-select" and decision.expand:
        if decision.expand != ("d1-seed1",):
            raise ValueError(f"unsupported D1 expansion: {decision.expand}")
        if any(job.id == "d1-seed1" for job in state.jobs):
            return state
        return _expand_d1_seed_confirmation(state, decision, generated_root)
    if any(job.id == first_by_selection[selection_job_id] for job in state.jobs):
        return state
    if selection_job_id == "denominator-select" and decision.expand == ("r1-newton",):
        return _expand_newton(state, generated_root)
    if len(decision.winners) != 1 and (selection_job_id != "n0-select" or len(decision.winners) > 2):
        raise ValueError(f"selection {selection_job_id!r} has an invalid winner count")
    if selection_job_id == "architecture-select":
        return _expand_architecture(state, decision, generated_root)
    if selection_job_id == "recovery-select":
        return _expand_recovery(state, decision, generated_root)
    if selection_job_id == "scale-select":
        return _expand_bias(state, decision, generated_root)
    if selection_job_id == "a0-select":
        return _expand_n0(state, decision, generated_root)
    if selection_job_id == "n0-select":
        return _expand_n1(state, decision, generated_root)
    if selection_job_id == "normalization-select":
        return _expand_d0(state, generated_root)
    if selection_job_id == "d0-select":
        return _expand_d1(state, decision, generated_root)
    if selection_job_id == "d1-select":
        return _expand_d2(state, decision, generated_root, selection_job_id="d1-select")
    if selection_job_id == "d1-confirm-select":
        return _expand_d2(state, decision, generated_root, selection_job_id="d1-confirm-select")
    if selection_job_id == "d2-select":
        return _expand_denominators(state, decision, generated_root)
    if selection_job_id == "denominator-select":
        return _expand_final_selection_nodes(state, bdcn_parent="denominator-select")
    return _expand_a_final(state, decision, generated_root)


def append_bdcn_v2_fix(state: QueueState) -> QueueState:
    """不改寫歷史，附加直接由 A0 衍生的 BDCN defect fix。"""

    if any(job.status not in {JobStatus.SUCCEEDED, JobStatus.SKIPPED} for job in state.jobs):
        raise ValueError("BDCN v2 can only be appended after the existing queue is complete")
    if any(job.id.startswith("bdcn-v2-") for job in state.jobs):
        raise ValueError("BDCN v2 rerun branch already exists")
    a0_id = _a0_winner(state)
    root = Path(state.project_root)
    order = max(job.order for job in state.jobs) + 1
    previous = state.jobs[-1].id
    jobs = (
        _training_job(
            state,
            job_id="bdcn-v2-learn",
            run_name="BDCN-V2-LEARN",
            stage="bdcn-v2-recovery",
            order=order,
            variant_path=_project_path(root, "BCND/configs/bdcn-v2.yaml"),
            training_file="configs/training/bdcn-codebook.yaml",
            parent_job_ids=(previous,),
            model_parent_job_id=a0_id,
        ),
        _evaluation_job(
            state,
            job_id="bdcn-v2-r1",
            run_name="BDCN-V2-R1",
            stage="bdcn-v2-denominator",
            order=order + 1,
            variant_path=_project_path(root, "BCND/configs/bdcn-v2-r1.yaml"),
            parent_job_ids=("bdcn-v2-learn",),
            model_parent_job_id="bdcn-v2-learn",
        ),
    )
    return _append_jobs(state, jobs)


def append_bdcn_v3_stable(state: QueueState) -> QueueState:
    """不改寫歷史，附加 fixed-exp control 與 bounded learned-codebook recovery。"""

    if any(job.status not in {JobStatus.SUCCEEDED, JobStatus.SKIPPED} for job in state.jobs):
        raise ValueError("BDCN v3 can only be appended after the existing queue is complete")
    if any(job.id.startswith("bdcn-v3-") for job in state.jobs):
        raise ValueError("BDCN v3 stabilized branch already exists")
    a0_id = _a0_winner(state)
    root = Path(state.project_root)
    order = max(job.order for job in state.jobs) + 1
    previous = state.jobs[-1].id
    jobs = (
        _evaluation_job(
            state,
            job_id="bdcn-v3-fixed",
            run_name="BDCN-V3-FIXED",
            stage="bdcn-v3-control",
            order=order,
            variant_path=_project_path(root, "BCND/configs/bdcn-v3-fixed.yaml"),
            parent_job_ids=(previous,),
            model_parent_job_id=a0_id,
        ),
        _training_job(
            state,
            job_id="bdcn-v3-learn",
            run_name="BDCN-V3-LEARN",
            stage="bdcn-v3-recovery",
            order=order + 1,
            variant_path=_project_path(root, "BCND/configs/bdcn-v3-learn.yaml"),
            training_file="configs/training/bdcn-codebook-stable.yaml",
            parent_job_ids=("bdcn-v3-fixed",),
            model_parent_job_id=a0_id,
        ),
        _evaluation_job(
            state,
            job_id="bdcn-v3-r1",
            run_name="BDCN-V3-R1",
            stage="bdcn-v3-denominator",
            order=order + 2,
            variant_path=_project_path(root, "BCND/configs/bdcn-v3-r1.yaml"),
            parent_job_ids=("bdcn-v3-learn",),
            model_parent_job_id="bdcn-v3-learn",
        ),
    )
    return _append_jobs(state, jobs)


def append_pwl_validation(state: QueueState) -> QueueState:
    """附加不訓練的 Exact／Float／Bit-True PWL validation branch。"""

    if any(job.id.startswith("pwl-") for job in state.jobs):
        raise ValueError("PWL validation branch already exists")
    if any(job.status not in {JobStatus.SUCCEEDED, JobStatus.SKIPPED} for job in state.jobs):
        raise ValueError("PWL validation can only be appended after the existing queue is complete")
    parent = state.job("v1-br")
    checkpoint = None
    if parent.result is not None:
        checkpoint = parent.result.checkpoint_path
    checkpoint = checkpoint or parent.checkpoint_path
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError("PWL validation requires the retained V1-BR best checkpoint")
    root = Path(state.project_root)
    order = max(job.order for job in state.jobs) + 1
    analysis = QueueJob(
        id="pwl-score-analysis",
        run_name="PWL-SCORE-ANALYSIS",
        stage="pwl-final-validation",
        kind=JobKind.VALIDATE,
        order=order,
        variant_path=_project_path(root, "PWL/configs/exact.yaml"),
        evaluation_path=_project_path(root, "configs/evaluation/coco2017.yaml"),
        parent_job_ids=(state.jobs[-1].id,),
        model_parent_job_id="v1-br",
        parent_checkpoint=str(Path(checkpoint).resolve()),
        requires_gpu=True,
    )
    comparison = QueueJob(
        id="pwl-compare",
        run_name="PWL-COMPARE",
        stage="pwl-final-validation",
        kind=JobKind.VALIDATE,
        order=order + 1,
        variant_path=_project_path(root, "PWL/configs/exact.yaml"),
        evaluation_path=_project_path(root, "configs/evaluation/coco2017.yaml"),
        parent_job_ids=(analysis.id,),
        model_parent_job_id="v1-br",
        parent_checkpoint=str(Path(checkpoint).resolve()),
        requires_gpu=True,
    )
    return _append_jobs(state, (analysis, comparison))


def _expand_architecture(
    state: QueueState,
    decision: SelectionDecision,
    generated_root: Path,
) -> QueueState:
    winner = state.job(decision.winners[0])
    if winner.variant_path is None:
        raise ValueError("architecture winner has no variant configuration")
    parent = VariantConfig.from_yaml(winner.variant_path)
    direct_path = _variant(
        parent,
        job_id="w-dir",
        name="W-DIR",
        generated_root=generated_root,
        progressive=False,
    )
    progressive_path = _variant(
        parent,
        job_id="w-prog",
        name="W-PROG",
        generated_root=generated_root,
        progressive=True,
    )
    root = Path(state.project_root)
    first_order = max(job.order for job in state.jobs) + 1
    added = (
        QueueJob(
            id="w-dir",
            run_name="W-DIR",
            stage="architecture-recovery",
            kind=JobKind.TRAIN,
            order=first_order,
            variant_path=direct_path,
            training_path=_project_path(root, "configs/training/recovery.yaml"),
            parent_job_ids=("architecture-select",),
            model_parent_job_id=winner.id,
            requires_gpu=True,
        ),
        QueueJob(
            id="w-prog",
            run_name="W-PROG",
            stage="architecture-recovery",
            kind=JobKind.TRAIN,
            order=first_order + 1,
            variant_path=progressive_path,
            training_path=_project_path(root, "configs/training/recovery.yaml"),
            parent_job_ids=("architecture-select", "w-dir"),
            model_parent_job_id=winner.id,
            requires_gpu=True,
        ),
        QueueJob(
            id="recovery-select",
            run_name="RECOVERY-SELECT",
            stage="recovery-selection",
            kind=JobKind.SELECT,
            order=first_order + 2,
            parent_job_ids=("w-dir", "w-prog"),
        ),
    )
    return _append_jobs(state, added)


def _expand_recovery(
    state: QueueState,
    decision: SelectionDecision,
    generated_root: Path,
) -> QueueState:
    winner = state.job(decision.winners[0])
    parent = VariantConfig.from_yaml(winner.variant_path)
    specs = (
        ("v1-dyn", "V1-DYN", ScaleMode.DYNAMIC),
        ("v1-shead", "V1-SHEAD", ScaleMode.FIXED_HEAD),
        ("v1-p2", "V1-P2", ScaleMode.POWER_OF_TWO),
    )
    order = max(job.order for job in state.jobs) + 1
    jobs: list[QueueJob] = []
    previous: str | None = None
    for index, (job_id, name, scale_mode) in enumerate(specs):
        variant_path = _variant(
            parent,
            job_id=job_id,
            name=name,
            generated_root=generated_root,
            scale_mode=scale_mode,
            progressive=False,
        )
        parents = ("recovery-select",) + ((previous,) if previous else ())
        jobs.append(
            _evaluation_job(
                state,
                job_id=job_id,
                run_name=name,
                stage="scale-screening",
                order=order + index,
                variant_path=variant_path,
                parent_job_ids=parents,
                model_parent_job_id=winner.id,
            )
        )
        previous = job_id
    jobs.append(
        QueueJob(
            id="scale-select",
            run_name="SCALE-SELECT",
            stage="scale-selection",
            kind=JobKind.SELECT,
            order=order + len(specs),
            parent_job_ids=tuple(job_id for job_id, _, _ in specs),
        )
    )
    return _append_jobs(state, tuple(jobs))


def _expand_bias(
    state: QueueState,
    decision: SelectionDecision,
    generated_root: Path,
) -> QueueState:
    winner = state.job(decision.winners[0])
    parent = VariantConfig.from_yaml(winner.variant_path)
    specs = (
        ("v1-b0", "V1-B0", BiasKind.NONE),
        ("v1-bd", "V1-BD", BiasKind.DENSE_2D),
        ("v1-br", "V1-BR", BiasKind.DECOMPOSED_2D),
    )
    order = max(job.order for job in state.jobs) + 1
    jobs: list[QueueJob] = []
    previous: str | None = None
    for index, (job_id, name, bias) in enumerate(specs):
        path = _variant(parent, job_id=job_id, name=name, generated_root=generated_root, bias=bias)
        parents = ("scale-select",) + ((previous,) if previous else ())
        jobs.append(
            _training_job(
                state,
                job_id=job_id,
                run_name=name,
                stage="bias",
                order=order + index,
                variant_path=path,
                training_file="configs/training/bias.yaml",
                parent_job_ids=parents,
                model_parent_job_id=winner.id,
            )
        )
        previous = job_id
    jobs.append(
        QueueJob(
            id="a0-select",
            run_name="A0-SELECT",
            stage="a0-selection",
            kind=JobKind.SELECT,
            order=order + len(specs),
            parent_job_ids=tuple(job_id for job_id, _, _ in specs),
        )
    )
    return _append_jobs(state, tuple(jobs))


def _expand_n0(
    state: QueueState,
    decision: SelectionDecision,
    generated_root: Path,
) -> QueueState:
    winner = state.job(decision.winners[0])
    parent = VariantConfig.from_yaml(winner.variant_path)
    specs = (
        ("n0-exact", "N0-EXACT", NormalizationKind.EXACT, {}),
        ("n0-lut", "N0-LUT", NormalizationKind.LUT, {}),
        ("n0-pwl", "N0-PWL", NormalizationKind.PIECEWISE_LINEAR, {}),
        ("n0-shift", "N0-SHIFT", NormalizationKind.POWER_OF_TWO, {}),
        ("n0-hsig", "N0-HSIG", NormalizationKind.HARD_SIGMOID, {}),
        ("n0-relu", "N0-RELU", NormalizationKind.RELU, {}),
        ("n0-mk1", "N0-MK1", NormalizationKind.MULTIMAX, {"multimax_top_k": 1}),
        ("n0-mk3", "N0-MK3", NormalizationKind.MULTIMAX, {"multimax_top_k": 3}),
        ("n0-mk5", "N0-MK5", NormalizationKind.MULTIMAX, {"multimax_top_k": 5}),
    )
    order = max(job.order for job in state.jobs) + 1
    jobs: list[QueueJob] = []
    previous: str | None = None
    for index, (job_id, name, normalization, extra) in enumerate(specs):
        path = _variant(
            parent,
            job_id=job_id,
            name=name,
            generated_root=generated_root,
            normalization=normalization,
            normalization_progressive=False,
            **extra,
        )
        parents = ("a0-select",) + ((previous,) if previous else ())
        jobs.append(
            _evaluation_job(
                state,
                job_id=job_id,
                run_name=name,
                stage="normalization-screening",
                order=order + index,
                variant_path=path,
                parent_job_ids=parents,
                model_parent_job_id=winner.id,
            )
        )
        previous = job_id
    jobs.append(
        QueueJob(
            id="n0-select",
            run_name="N0-SELECT",
            stage="normalization-selection",
            kind=JobKind.SELECT,
            order=order + len(specs),
            parent_job_ids=tuple(job_id for job_id, *_ in specs),
        )
    )
    return _append_jobs(state, tuple(jobs))


def _expand_n1(
    state: QueueState,
    decision: SelectionDecision,
    generated_root: Path,
) -> QueueState:
    order = max(job.order for job in state.jobs) + 1
    jobs: list[QueueJob] = []
    previous: str | None = None
    n1_ids: list[str] = []
    for winner_id in decision.winners:
        if winner_id == "n0-exact":
            continue
        winner = state.job(winner_id)
        parent = VariantConfig.from_yaml(winner.variant_path)
        suffix = winner_id.removeprefix("n0-")
        job_id = f"n1-{suffix}"
        path = _variant(
            parent,
            job_id=job_id,
            name=job_id.upper(),
            generated_root=generated_root,
            normalization_progressive=True,
            normalization_transition_epochs=5,
        )
        parents = ("n0-select",) + ((previous,) if previous else ())
        jobs.append(
            _training_job(
                state,
                job_id=job_id,
                run_name=job_id.upper(),
                stage="normalization-recovery",
                order=order + len(jobs),
                variant_path=path,
                training_file="configs/training/normalization-recovery.yaml",
                parent_job_ids=parents,
                model_parent_job_id=winner_id,
            )
        )
        n1_ids.append(job_id)
        previous = job_id
    jobs.append(
        QueueJob(
            id="normalization-select",
            run_name="NORMALIZATION-SELECT",
            stage="normalization-final-selection",
            kind=JobKind.SELECT,
            order=order + len(jobs),
            parent_job_ids=("n0-select", *n1_ids),
        )
    )
    return _append_jobs(state, tuple(jobs))


def _a0_winner(state: QueueState) -> str:
    record = state.selections.get("a0-select")
    if record is None or len(record["winners"]) != 1:
        raise ValueError("A0 selection is unavailable")
    return record["winners"][0]


def _expand_d0(state: QueueState, generated_root: Path) -> QueueState:
    a0_id = _a0_winner(state)
    parent = VariantConfig.from_yaml(state.job(a0_id).variant_path)
    path = _variant(
        parent,
        job_id="d0-idx",
        name="D0-IDX",
        generated_root=generated_root,
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.FIXED_EXP,
        bdcn_sharing=BDCNSharing.GLOBAL,
        bdcn_projection=BDCNProjection.FLOAT,
        bdcn_denominator=BDCNDenominator.EXACT,
    )
    order = max(job.order for job in state.jobs) + 1
    jobs = (
        _evaluation_job(
            state,
            job_id="d0-idx",
            run_name="D0-IDX",
            stage="bdcn-reference",
            order=order,
            variant_path=path,
            parent_job_ids=("normalization-select",),
            model_parent_job_id=a0_id,
        ),
        QueueJob(
            id="d0-select",
            run_name="D0-SELECT",
            stage="bdcn-reference-selection",
            kind=JobKind.SELECT,
            order=order + 1,
            parent_job_ids=("d0-idx",),
        ),
    )
    return _append_jobs(state, jobs)


def _expand_d1(
    state: QueueState,
    decision: SelectionDecision,
    generated_root: Path,
) -> QueueState:
    winner = state.job(decision.winners[0])
    parent = VariantConfig.from_yaml(winner.variant_path)
    specs = (
        ("d1-shared", "D1-SHARED", BDCNSharing.GLOBAL),
        ("d1-pattn", "D1-PATTN", BDCNSharing.PER_ATTENTION),
        ("d1-phead", "D1-PHEAD", BDCNSharing.PER_HEAD),
    )
    order = max(job.order for job in state.jobs) + 1
    jobs: list[QueueJob] = []
    previous: str | None = None
    for index, (job_id, name, sharing) in enumerate(specs):
        path = _variant(
            parent,
            job_id=job_id,
            name=name,
            generated_root=generated_root,
            bdcn_codebook=BDCNCodebookKind.LEARNED,
            bdcn_sharing=sharing,
        )
        parents = ("d0-select",) + ((previous,) if previous else ())
        jobs.append(
            _training_job(
                state,
                job_id=job_id,
                run_name=name,
                stage="bdcn-learning",
                order=order + index,
                variant_path=path,
                training_file="configs/training/bdcn-codebook.yaml",
                parent_job_ids=parents,
                model_parent_job_id=winner.id,
            )
        )
        previous = job_id
    jobs.append(
        QueueJob(
            id="d1-select",
            run_name="D1-SELECT",
            stage="bdcn-sharing-selection",
            kind=JobKind.SELECT,
            order=order + len(specs),
            parent_job_ids=tuple(job_id for job_id, _, _ in specs),
        )
    )
    return _append_jobs(state, tuple(jobs))


def _expand_d2(
    state: QueueState,
    decision: SelectionDecision,
    generated_root: Path,
    *,
    selection_job_id: str,
) -> QueueState:
    winner = state.job(decision.winners[0])
    parent = VariantConfig.from_yaml(winner.variant_path)
    specs = (
        ("d2-fp", "D2-FP", BDCNProjection.FLOAT),
        ("d2-1p", "D2-1P", BDCNProjection.ONE_POT),
        ("d2-2p", "D2-2P", BDCNProjection.TWO_POT),
    )
    order = max(job.order for job in state.jobs) + 1
    jobs: list[QueueJob] = []
    previous: str | None = None
    for index, (job_id, name, projection) in enumerate(specs):
        path = _variant(
            parent, job_id=job_id, name=name, generated_root=generated_root, bdcn_projection=projection
        )
        parents = (selection_job_id,) + ((previous,) if previous else ())
        jobs.append(
            _evaluation_job(
                state,
                job_id=job_id,
                run_name=name,
                stage="bdcn-projection",
                order=order + index,
                variant_path=path,
                parent_job_ids=parents,
                model_parent_job_id=winner.id,
            )
        )
        previous = job_id
    jobs.append(
        QueueJob(
            id="d2-select",
            run_name="D2-SELECT",
            stage="bdcn-projection-selection",
            kind=JobKind.SELECT,
            order=order + len(specs),
            parent_job_ids=tuple(job_id for job_id, _, _ in specs),
        )
    )
    return _append_jobs(state, tuple(jobs))


def _expand_d1_seed_confirmation(
    state: QueueState,
    decision: SelectionDecision,
    generated_root: Path,
) -> QueueState:
    winner = state.job(decision.winners[0])
    parent = VariantConfig.from_yaml(winner.variant_path)
    path = _variant(
        parent,
        job_id="d1-seed1",
        name=f"{parent.name}-S1",
        generated_root=generated_root,
    )
    order = max(job.order for job in state.jobs) + 1
    jobs = (
        _training_job(
            state,
            job_id="d1-seed1",
            run_name=f"{winner.run_name}-S1",
            stage="bdcn-seed-confirmation",
            order=order,
            variant_path=path,
            training_file="configs/training/bdcn-codebook-seed1.yaml",
            parent_job_ids=("d1-select",),
            model_parent_job_id="d0-idx",
        ),
        QueueJob(
            id="d1-confirm-select",
            run_name="D1-CONFIRM-SELECT",
            stage="bdcn-seed-confirmation-selection",
            kind=JobKind.SELECT,
            order=order + 1,
            parent_job_ids=("d1-seed1",),
        ),
    )
    return _append_jobs(state, jobs)


def _expand_denominators(
    state: QueueState,
    decision: SelectionDecision,
    generated_root: Path,
) -> QueueState:
    winner = state.job(decision.winners[0])
    parent = VariantConfig.from_yaml(winner.variant_path)
    specs = (
        ("r0-div", "R0-DIV", BDCNDenominator.EXACT),
        ("r1-rlut", "R1-RLUT", BDCNDenominator.RECIPROCAL_LUT),
        ("r2-pshift", "R2-PSHIFT", BDCNDenominator.POT_SHIFT),
    )
    order = max(job.order for job in state.jobs) + 1
    jobs: list[QueueJob] = []
    previous: str | None = None
    for index, (job_id, name, denominator) in enumerate(specs):
        path = _variant(
            parent, job_id=job_id, name=name, generated_root=generated_root, bdcn_denominator=denominator
        )
        parents = ("d2-select",) + ((previous,) if previous else ())
        jobs.append(
            _evaluation_job(
                state,
                job_id=job_id,
                run_name=name,
                stage="bdcn-denominator",
                order=order + index,
                variant_path=path,
                parent_job_ids=parents,
                model_parent_job_id=winner.id,
            )
        )
        previous = job_id
    jobs.append(
        QueueJob(
            id="denominator-select",
            run_name="DENOMINATOR-SELECT",
            stage="bdcn-denominator-selection",
            kind=JobKind.SELECT,
            order=order + len(specs),
            parent_job_ids=tuple(job_id for job_id, _, _ in specs),
        )
    )
    return _append_jobs(state, tuple(jobs))


def _expand_newton(state: QueueState, generated_root: Path) -> QueueState:
    parent_job = state.job("r1-rlut")
    parent = VariantConfig.from_yaml(parent_job.variant_path)
    path = _variant(
        parent,
        job_id="r1-newton",
        name="R1-NEWTON",
        generated_root=generated_root,
        bdcn_reciprocal_newton_steps=1,
    )
    order = max(job.order for job in state.jobs) + 1
    newton = _evaluation_job(
        state,
        job_id="r1-newton",
        run_name="R1-NEWTON",
        stage="bdcn-denominator-recovery",
        order=order,
        variant_path=path,
        parent_job_ids=("denominator-select",),
        model_parent_job_id="r1-rlut",
    )
    state = _append_jobs(state, (newton,))
    return _expand_final_selection_nodes(state, bdcn_parent="r1-newton")


def _expand_final_selection_nodes(state: QueueState, *, bdcn_parent: str) -> QueueState:
    order = max(job.order for job in state.jobs) + 1
    jobs = (
        QueueJob(
            id="bdcn-select",
            run_name="BDCN-SELECT",
            stage="bdcn-final-selection",
            kind=JobKind.SELECT,
            order=order,
            parent_job_ids=(bdcn_parent,),
        ),
        QueueJob(
            id="final-select",
            run_name="FINAL-SELECT",
            stage="final-selection",
            kind=JobKind.SELECT,
            order=order + 1,
            parent_job_ids=("normalization-select", "bdcn-select"),
        ),
    )
    return _append_jobs(state, jobs)


def _expand_a_final(
    state: QueueState,
    decision: SelectionDecision,
    generated_root: Path,
) -> QueueState:
    winner = state.job(decision.winners[0])
    if winner.variant_path is None:
        raise ValueError("A-FINAL winner has no variant configuration")
    parent = VariantConfig.from_yaml(winner.variant_path)
    path = _variant(parent, job_id="a-final", name="A-FINAL", generated_root=generated_root)
    order = max(job.order for job in state.jobs) + 1
    job = _evaluation_job(
        state,
        job_id="a-final",
        run_name="A-FINAL",
        stage="final",
        order=order,
        variant_path=path,
        parent_job_ids=("final-select",),
        model_parent_job_id=winner.id,
    )
    return _append_jobs(state, (job,))
