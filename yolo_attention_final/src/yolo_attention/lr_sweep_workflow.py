"""LR sweep followed by gated downstream recovery from one immutable parent."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .queue_model import JobKind, JobStatus, QueueJob, QueueState
from .run_config import TrainingRecipe

LR_MULTIPLIERS = {"x1": 1.0, "x2": 2.0, "x4": 4.0}
DOWNSTREAM_STAGES = ("neck", "backbone", "full")


def materialize_lr_recovery_recipes(project_root: str | Path, label: str) -> tuple[Path, ...]:
    """Scale every discriminative LR together after the Bit-True block pilot selection."""

    if label not in LR_MULTIPLIERS:
        raise ValueError(f"unknown LR multiplier label: {label}")
    root = Path(project_root).resolve()
    generated = root / "artifacts/lr-sweep-queue/generated"
    multiplier = LR_MULTIPLIERS[label]
    paths: list[Path] = []
    parent = "lr-block-select"
    for stage in DOWNSTREAM_STAGES:
        template = TrainingRecipe.from_yaml(root / "configs/training" / f"recovery-{stage}.yaml")
        if template.layer_lrs is None:
            raise ValueError(f"recovery-{stage} template has no layer_lrs")
        recipe = replace(
            template,
            phase=f"lr-recovery-{stage}",
            parent=parent,
            lr0=template.lr0 * multiplier,
            layer_lrs={name: value * multiplier for name, value in template.layer_lrs.items()},
        )
        destination = generated / f"recovery-{stage}.yaml"
        if destination.exists():
            if TrainingRecipe.from_yaml(destination) != recipe:
                raise FileExistsError(f"immutable recipe differs from selected LR: {destination}")
        else:
            recipe.to_yaml(destination)
        paths.append(destination)
        parent = f"lr-recovery-{stage}-gate"
    return tuple(paths)


def create_lr_sweep_state(project_root: str | Path) -> QueueState:
    """Build three same-parent block pilots before any Neck or Backbone adaptation."""

    root = Path(project_root).resolve()
    float_variant = str(root / "configs/variants/float-pwl-final.yaml")
    bittrue_variant = str(root / "configs/variants/bittrue-pwl-final.yaml")
    evaluation = str(root / "configs/evaluation/coco2017.yaml")
    parent = root / "artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt"
    if not parent.is_file():
        raise FileNotFoundError(f"LR sweep parent checkpoint does not exist: {parent}")
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
                stage="pwl-lr-sweep",
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

    add("lr-parent-bittrue", JobKind.EVALUATE, variant=bittrue_variant, ready=True)
    block_results: list[str] = []
    for label in LR_MULTIPLIERS:
        train = f"lr-block-{label}"
        add(
            train,
            JobKind.TRAIN,
            parents=("lr-parent-bittrue",),
            model_parent="lr-parent-bittrue",
            variant=float_variant,
            training=str(root / "configs/training" / f"{train}.yaml"),
        )
        evaluated = f"{train}-bittrue"
        add(evaluated, JobKind.EVALUATE, parents=(train,), model_parent=train, variant=bittrue_variant)
        block_results.append(evaluated)
    add("lr-block-select", JobKind.SELECT, parents=tuple(block_results))

    parent_gate = "lr-block-select"
    downstream_results: list[str] = []
    for stage in DOWNSTREAM_STAGES:
        train = f"lr-recovery-{stage}"
        add(
            train,
            JobKind.TRAIN,
            parents=(parent_gate,),
            model_parent=parent_gate,
            variant=float_variant,
            training=str(root / "artifacts/lr-sweep-queue/generated" / f"recovery-{stage}.yaml"),
        )
        evaluated = f"{train}-bittrue"
        add(evaluated, JobKind.EVALUATE, parents=(train,), model_parent=train, variant=bittrue_variant)
        gate = f"{train}-gate"
        add(gate, JobKind.SELECT, parents=(parent_gate, evaluated))
        parent_gate = gate
        downstream_results.append(evaluated)

    candidates = ("lr-parent-bittrue", *block_results, *downstream_results)
    add("lr-recovery-select", JobKind.SELECT, parents=candidates)
    add("export-lr-recovery", JobKind.VALIDATE, parents=("lr-recovery-select",))
    state = QueueState.initial(tuple(jobs), project_root=str(root))
    state.validate()
    return state
