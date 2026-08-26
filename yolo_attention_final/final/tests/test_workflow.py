from __future__ import annotations

from pathlib import Path

import pytest

from yolo_attention.final_selection import choose_final, choose_pilot, phase_gate
from yolo_attention.final_workflow import create_pwl_final_state, materialize_phase_recipes
from yolo_attention.run_config import TrainingRecipe


def test_pilot_phase_and_final_rules() -> None:
    assert choose_pilot({1e-5: 0.5070, 5e-6: 0.5065}) == 5e-6
    assert choose_pilot({1e-5: 0.5080, 5e-6: 0.5065}) == 1e-5
    assert phase_gate(parent_id="p", parent_map=0.507, child_id="c", child_map=0.5059) == "p"
    decision = choose_final({"0": ("a", 0.508), "1": ("b", 0.508), "2": ("c", 0.508)})
    assert decision.stable_improvement
    assert decision.formal_winner in {"a", "b", "c"}
    fallback = choose_final({"0": ("a", 0.507), "1": ("b", 0.506), "2": ("c", 0.5065)})
    assert fallback.formal_winner == "epoch0-bittrue"


def test_queue_shape_and_recipe_materialization(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    state = create_pwl_final_state(root)
    assert len(state.jobs) == 39
    assert state.job("epoch0-bittrue").variant_path.endswith("bittrue-pwl-final.yaml")
    generated_root = tmp_path / "project"
    (generated_root / "configs/training").mkdir(parents=True)
    for phase in "abc":
        source = root / "configs/training" / f"phase-{phase}.yaml"
        (generated_root / "configs/training" / f"phase-{phase}.yaml").write_bytes(source.read_bytes())
    paths = materialize_phase_recipes(generated_root, 1e-5)
    assert len(paths) == 9
    assert TrainingRecipe.from_yaml(generated_root / "artifacts/queue/generated/phase-c-s2.yaml").lr0 == 5e-6
    with pytest.raises(FileExistsError):
        materialize_phase_recipes(generated_root, 1e-5)


def test_training_recipe_disables_warmup() -> None:
    root = Path(__file__).resolve().parents[1]
    recipe = TrainingRecipe.from_yaml(root / "configs/training/pilot-1e-5.yaml")
    args = recipe.to_ultralytics_args()
    assert args["warmup_epochs"] == args["warmup_bias_lr"] == 0
    assert args["lrf"] == 1.0 and args["cos_lr"] is False
