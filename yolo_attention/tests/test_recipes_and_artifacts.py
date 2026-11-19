from __future__ import annotations

import json
from pathlib import Path

from yolo_attention.artifacts import ArtifactStore
from yolo_attention.config import VariantConfig
from yolo_attention.run_config import TrainingRecipe

ROOT = Path(__file__).resolve().parents[1]


def test_all_committed_variant_configs_parse() -> None:
    paths = sorted((ROOT / "configs" / "variants").glob("*.yaml"))
    assert paths
    assert {VariantConfig.from_yaml(path).name for path in paths} == {
        "D0-IDX",
        "D1-SHARED",
        "D1-PATTN",
        "D1-PHEAD",
        "D2-1P",
        "D2-2P",
        "D2-FP",
        "R0-DIV",
        "R1-RLUT",
        "R2-PSHIFT",
        "P0",
        "I-SCR",
        "H-SCR",
        "T5-SCR",
        "Q1-L3A",
    }


def test_all_training_recipes_parse_and_map_to_ultralytics() -> None:
    paths = sorted((ROOT / "configs" / "training").glob("*.yaml"))
    recipes = [TrainingRecipe.from_yaml(path) for path in paths]

    assert {recipe.stage for recipe in recipes} == {
        "screening",
        "recovery",
        "normalization",
        "bias",
        "bdcn_codebook",
        "q2",
    }
    assert all(recipe.to_ultralytics_args()["data"].endswith("coco2017.yaml") for recipe in recipes)


def test_artifact_store_records_immutable_run_inputs(tmp_path: Path) -> None:
    variant = VariantConfig.from_yaml(ROOT / "configs" / "variants" / "h-screen.yaml")
    recipe = TrainingRecipe.from_yaml(ROOT / "configs" / "training" / "screening.yaml")

    run = ArtifactStore(tmp_path).create_run("h-screen-seed0", variant, recipe)

    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["run_id"] == "h-screen-seed0"
    assert manifest["variant"]["basis"] == "hadamard"
    assert manifest["training"]["epochs"] == 10
    assert (run / "metrics").is_dir()
    assert (run / "checkpoints").is_dir()
