from pathlib import Path

import yaml

from masf_yolo.retest.contracts import P2_VARIANTS, P3_VARIANTS, RetestConfig, VARIANT_SPECS, load_retest_config


def _config():
    return {
        "schema_version": 1,
        "pipeline_name": "b1r-p2-p3-retest",
        "artifacts_root": "artifacts/b1r-p2-p3-retest",
        "variants": {"p2": list(P2_VARIANTS), "p3": list(P3_VARIANTS)},
        "environment": {},
        "dataset": {"source": "bbt5-detect-baseline/dataset", "locked_artifacts": "artifacts/static-phase1/dataset", "split_ratios": [0.8, 0.1, 0.1], "class_names": ["ball", "bat"], "seed": 42},
        "model": {},
        "training": {"optimizer": "SGD", "momentum": 0.937, "cos_lr": True, "deterministic": True, "amp": True, "nbs": 64, "batch": 16, "seed": 42, "b1_a_epochs": 10, "b1_b_epochs": 90, "direct_epochs": 100, "smoke_epochs": 3, "formal_epochs": 100, "b1_a_lr0": 0.01, "formal_lr0": 0.001, "freeze": list(range(11))},
        "profiling": {},
        "pipeline": {},
    }


def test_retest_contract_locks_dataset_schedule_and_order():
    config = RetestConfig.from_mapping(_config())
    assert config.values["dataset"]["source"] == "bbt5-detect-baseline/dataset"
    assert config.values["training"]["freeze"] == list(range(11))
    assert config.values["training"]["direct_epochs"] == 100
    assert P2_VARIANTS == ("PaperFormula-Full", "Lite-35", "Lite-35-F7", "Partial50-35", "Partial25-35")


def test_variant_specs_record_formula_and_adaptation():
    full = VARIANT_SPECS["PaperFormula-Full"]
    assert full.kernels == (3, 5, 7, 9)
    assert full.formula_version == "paper-equations-1-6"
    assert VARIANT_SPECS["Partial50-35"].processed_ratio == 0.5


def test_unknown_contract_key_is_rejected():
    raw = _config()
    raw["training"]["distillation"] = False
    try:
        RetestConfig.from_mapping(raw)
    except ValueError as error:
        assert "unknown training keys" in str(error)
    else:
        raise AssertionError("unknown keys must be rejected")


def test_checked_in_yaml_is_loadable_and_has_locked_source():
    path = Path(__file__).parents[2] / "configs" / "retest" / "b1r_p2_p3_retest.yaml"
    config = RetestConfig.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert config.values["model"]["source_weights"].endswith("yolo11m_bat_detect_init.pt")
    assert config.values["training"]["formal_epochs"] == 100
    assert load_retest_config(path).config_hash == config.config_hash
