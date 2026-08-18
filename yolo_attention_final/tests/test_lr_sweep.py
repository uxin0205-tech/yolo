from __future__ import annotations

from pathlib import Path

from yolo_attention.cli import build_parser
from yolo_attention.lr_sweep_workflow import (
    LR_MULTIPLIERS,
    create_lr_sweep_state,
    materialize_lr_recovery_recipes,
)
from yolo_attention.run_config import TrainingRecipe

ROOT = Path(__file__).resolve().parents[1]


def test_lr_sweep_queue_compares_same_parent_before_downstream() -> None:
    state = create_lr_sweep_state(ROOT)
    assert len(state.jobs) == 19
    assert state.job("lr-parent-bittrue").status.value == "ready"
    for label in LR_MULTIPLIERS:
        job = state.job(f"lr-block-{label}")
        assert job.model_parent_job_id == "lr-parent-bittrue"
        assert job.parent_job_ids == ("lr-parent-bittrue",)
    assert state.job("lr-recovery-neck").model_parent_job_id == "lr-block-select"
    assert state.job("export-lr-recovery").parent_job_ids == ("lr-recovery-select",)


def test_lr_sweep_materializes_selected_discriminative_rates(tmp_path: Path) -> None:
    configs = tmp_path / "configs/training"
    configs.mkdir(parents=True)
    for stage in ("neck", "backbone", "full"):
        source = ROOT / "configs/training" / f"recovery-{stage}.yaml"
        (configs / source.name).write_bytes(source.read_bytes())
    paths = materialize_lr_recovery_recipes(tmp_path, "x4")
    assert len(paths) == 3
    neck = TrainingRecipe.from_yaml(paths[0])
    assert neck.lr0 == 2e-5
    assert neck.layer_lrs == {"attention": 2e-5, "adjacent_block": 4e-6, "neck_detect": 2e-6}
    backbone = TrainingRecipe.from_yaml(paths[1])
    assert backbone.layer_lrs["backbone"] == 4e-7


def test_lr_sweep_cli_has_independent_default_root() -> None:
    args = build_parser().parse_args(["queue", "init-pwl-lr-sweep"])
    assert args.queue_root == Path("artifacts/lr-sweep-queue")
    assert args.project_root == Path.cwd()
