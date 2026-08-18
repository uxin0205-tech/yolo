"""Static, immutable PWL-final queue graph."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .queue_model import JobKind, JobStatus, QueueJob, QueueState


def create_pwl_final_state(project_root: str | Path) -> QueueState:
    root = Path(project_root).resolve()
    variant_float = str(root / "configs/variants/float-pwl-final.yaml")
    variant_bittrue = str(root / "configs/variants/bittrue-pwl-final.yaml")
    evaluation = str(root / "configs/evaluation/coco2017.yaml")
    parent = str(root / "weights/v1-br-best.pt")
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
                stage="pwl-final",
                kind=kind,
                order=len(jobs),
                status=JobStatus.READY if ready else JobStatus.BLOCKED,
                variant_path=variant,
                training_path=training,
                evaluation_path=evaluation if kind is JobKind.EVALUATE else None,
                parent_job_ids=parents,
                model_parent_job_id=model_parent,
                parent_checkpoint=parent if ready else None,
                requires_gpu=kind in {JobKind.TRAIN, JobKind.EVALUATE},
            )
        )

    add("epoch0-float", JobKind.EVALUATE, variant=variant_float, ready=True)
    add(
        "epoch0-bittrue",
        JobKind.EVALUATE,
        parents=("epoch0-float",),
        model_parent="epoch0-float",
        variant=variant_bittrue,
    )
    for label, recipe in (("1e-5", "pilot-1e-5.yaml"), ("5e-6", "pilot-5e-6.yaml")):
        train = f"pilot-{label}"
        add(
            train,
            JobKind.TRAIN,
            parents=("epoch0-bittrue",),
            model_parent="epoch0-bittrue",
            variant=variant_float,
            training=str(root / "configs/training" / recipe),
        )
        add(
            f"{train}-bittrue",
            JobKind.EVALUATE,
            parents=(train,),
            model_parent=train,
            variant=variant_bittrue,
        )
    add("pilot-select", JobKind.SELECT, parents=("pilot-1e-5-bittrue", "pilot-5e-6-bittrue"))
    for seed in range(3):
        parent_gate = "epoch0-bittrue"
        candidates = ["epoch0-bittrue"]
        for phase in "abc":
            train = f"s{seed}-phase-{phase}"
            recipe = root / "artifacts/queue/generated" / f"phase-{phase}-s{seed}.yaml"
            deps = ("pilot-select", parent_gate) if phase == "a" else (parent_gate,)
            add(
                train,
                JobKind.TRAIN,
                parents=deps,
                model_parent=parent_gate,
                variant=variant_float,
                training=str(recipe),
            )
            evaluated = f"{train}-bittrue"
            add(evaluated, JobKind.EVALUATE, parents=(train,), model_parent=train, variant=variant_bittrue)
            gate = f"s{seed}-phase-{phase}-gate"
            add(gate, JobKind.SELECT, parents=(parent_gate, evaluated))
            parent_gate = gate
            candidates.append(evaluated)
        add(f"s{seed}-select", JobKind.SELECT, parents=tuple(candidates))
    add("final-select", JobKind.SELECT, parents=("s0-select", "s1-select", "s2-select"))
    add("export-final", JobKind.VALIDATE, parents=("final-select",))
    state = QueueState.initial(tuple(jobs), project_root=str(root))
    state.validate()
    return state


def materialize_phase_recipes(project_root: str | Path, lr0: float) -> tuple[Path, ...]:
    """Create seed-specific immutable recipes after the pilot decision."""

    from .run_config import TrainingRecipe

    root = Path(project_root).resolve()
    generated = root / "artifacts/queue/generated"
    paths: list[Path] = []
    for seed in range(3):
        for phase in "abc":
            template = TrainingRecipe.from_yaml(root / "configs/training" / f"phase-{phase}.yaml")
            phase_lr = lr0 / 2 if phase == "c" else lr0
            destination = generated / f"phase-{phase}-s{seed}.yaml"
            if destination.exists():
                raise FileExistsError(f"immutable recipe already exists: {destination}")
            replace(
                template.with_seed_and_lr(seed=seed, lr0=phase_lr),
                parent=("epoch0-bittrue" if phase == "a" else f"s{seed}-phase-{chr(ord(phase) - 1)}-gate"),
            ).to_yaml(destination)
            paths.append(destination)
    return tuple(paths)
