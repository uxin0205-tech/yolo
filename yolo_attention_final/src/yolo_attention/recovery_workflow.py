"""Static staged-recovery queue for controlled downstream PWL adaptation."""

from __future__ import annotations

from pathlib import Path

from .queue_model import JobKind, JobStatus, QueueJob, QueueState

RECOVERY_STAGES = ("block", "neck", "backbone", "full")


def create_pwl_recovery_state(project_root: str | Path) -> QueueState:
    """Build an immutable single-seed recovery graph with Bit-True gates."""

    root = Path(project_root).resolve()
    variant_float = str(root / "configs/variants/float-pwl-final.yaml")
    variant_bittrue = str(root / "configs/variants/bittrue-pwl-final.yaml")
    evaluation = str(root / "configs/evaluation/coco2017.yaml")
    parent = root / "artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt"
    if not parent.is_file():
        raise FileNotFoundError(f"recovery parent checkpoint does not exist: {parent}")
    jobs: list[QueueJob] = []

    def add(
        job_id: str,
        kind: JobKind,
        *,
        parents: tuple[str, ...] = (),
        model_parent: str | None = None,
        variant: str | None = None,
        training: str | None = None,
        ready: bool = False,
    ) -> None:
        jobs.append(
            QueueJob(
                id=job_id,
                run_name=job_id.upper(),
                stage="pwl-recovery",
                kind=kind,
                order=len(jobs),
                status=JobStatus.READY if ready else JobStatus.BLOCKED,
                variant_path=variant,
                training_path=training,
                evaluation_path=evaluation if kind is JobKind.EVALUATE else None,
                parent_job_ids=parents,
                model_parent_job_id=model_parent,
                parent_checkpoint=str(parent) if ready else None,
                requires_gpu=kind in {JobKind.TRAIN, JobKind.EVALUATE},
            )
        )

    add("recovery-parent-bittrue", JobKind.EVALUATE, variant=variant_bittrue, ready=True)
    parent_gate = "recovery-parent-bittrue"
    candidates = [parent_gate]
    for stage in RECOVERY_STAGES:
        train = f"recovery-{stage}"
        add(
            train,
            JobKind.TRAIN,
            parents=(parent_gate,),
            model_parent=parent_gate,
            variant=variant_float,
            training=str(root / "configs/training" / f"recovery-{stage}.yaml"),
        )
        evaluated = f"{train}-bittrue"
        add(evaluated, JobKind.EVALUATE, parents=(train,), model_parent=train, variant=variant_bittrue)
        gate = f"{train}-gate"
        add(gate, JobKind.SELECT, parents=(parent_gate, evaluated))
        parent_gate = gate
        candidates.append(evaluated)
    add("recovery-select", JobKind.SELECT, parents=tuple(candidates))
    add("export-recovery", JobKind.VALIDATE, parents=("recovery-select",))
    state = QueueState.initial(tuple(jobs), project_root=str(root))
    state.validate()
    return state
