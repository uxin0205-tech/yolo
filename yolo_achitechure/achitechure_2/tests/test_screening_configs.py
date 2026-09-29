from __future__ import annotations

from pathlib import Path

import yaml

from achitechure_2.config import load_training_template


def test_float20_template_uses_manifests_not_ultralytics_fraction() -> None:
    config = load_training_template("configs/training/float-screen-20.yaml")
    assert config["candidate_policy"]["base_candidates"] == ["C0", "C1", "C2", "C3"]
    assert config["candidate_policy"]["screening_only"] is True
    assert config["execution"]["formal_training"] is False
    assert config["routes"]["pose"]["enabled_by_default"] is False
    assert config["adjustable"]["fraction"] == {"source": "local", "value": 1.0}
    assert config["adjustable"]["patience"] == {"source": "local", "value": 0}
    assert config["validation"]["automatic_acceptance"] is False
    assert config["validation"]["formal_split_used"] is False


def test_qat_lite_template_has_fixed_small_budget_and_no_auto_acceptance() -> None:
    config = load_training_template("configs/training/quant-qat-lite.yaml")
    assert config["recipe"]["optimizer_steps"] == {"source": "local", "value": 200}
    assert config["recipe"]["validation_interval"] == {"source": "local", "value": 50}
    assert config["adjustable"]["epochs"] == {"source": "local", "value": 3}
    assert config["validation"]["observer_update_steps"] == 50
    assert config["validation"]["simulation_only"] is True
    assert config["validation"]["automatic_acceptance"] is False
    assert config["transition"]["input"] == "candidate_q1_calibrated_checkpoint"


def test_catalog_and_quantization_expose_float20_and_q2_lite() -> None:
    catalog = yaml.safe_load(Path("configs/catalog.yaml").read_text(encoding="utf-8"))
    assert catalog["training"]["float-screen-20"] == (
        "configs/training/float-screen-20.yaml"
    )
    assert catalog["training"]["quant-qat-lite"] == (
        "configs/training/quant-qat-lite.yaml"
    )
    assert catalog["datasets"]["coco2017-screen-20"] == (
        "configs/data/coco2017-screen-20.yaml"
    )
    quant = yaml.safe_load(
        Path(catalog["quantization"]).read_text(encoding="utf-8")
    )
    assert tuple(quant["stages"]) == ("Q0", "Q1", "Q2L", "Q2")
    assert quant["stages"]["Q2L"]["max_optimizer_steps"] == 200
    assert quant["stages"]["Q2L"]["observer_update_steps"] == 50
    assert quant["stages"]["Q2L"]["automatic_acceptance"] is False
