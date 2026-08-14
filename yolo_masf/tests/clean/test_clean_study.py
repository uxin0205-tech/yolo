from pathlib import Path

import pytest
import yaml

from masf_yolo.clean.contracts import CLEAN_EXPERIMENTS, load_clean_config
from masf_yolo.clean.data_view import write_train_val_view
from masf_yolo.clean.plan import build_clean_plan
from masf_yolo.clean.profiles import clean_profile


CONFIG = Path(__file__).parents[2] / "configs" / "clean" / "clean_ablation.yaml"


def test_clean_config_locks_initializer_visibility_matrix_and_seeds():
    config = load_clean_config(CONFIG)
    assert config.values["seeds"] == [42, 43, 44]
    assert config.values["environment"]["ultralytics"] == "8.4.90"
    assert list(CLEAN_EXPERIMENTS) == config.values["experiments"]
    assert config.values["dataset"]["visibility"]["test"] == "historical_already_observed"
    config.assert_split_use(split="train", purpose="fit")
    config.assert_split_use(split="val", purpose="selection")
    with pytest.raises(ValueError, match="forbidden split use"):
        config.assert_split_use(split="test", purpose="selection")
    with pytest.raises(RuntimeError, match="new final holdout"):
        config.assert_split_use(split="final_holdout", purpose="unseen_claim")


def test_strict_fair_profiles_have_identical_training_settings():
    names = [name for name, spec in CLEAN_EXPERIMENTS.items() if spec.comparison_tier == "strict_fair"]
    profiles = [clean_profile(name, seed=42, model=name, data="data", project="project") for name in names]
    ignored = {"model", "name"}
    reference = {key: value for key, value in profiles[0].items() if key not in ignored}
    assert all({key: value for key, value in profile.items() if key not in ignored} == reference for profile in profiles[1:])
    assert reference["epochs"] == 100
    assert reference["freeze"] is None


def test_clean_plan_is_prepared_only_and_preserves_control_dependency():
    plan = build_clean_plan(load_clean_config(CONFIG))
    assert len(plan) == 18
    assert {job["status"] for job in plan} == {"prepared_not_queued"}
    full = [job for job in plan if job["experiment"] == "P2-Control-Clean-Full"]
    assert [job["depends_on"] for job in full] == [
        "P2-Control-Clean-Head:seed42",
        "P2-Control-Clean-Head:seed43",
        "P2-Control-Clean-Head:seed44",
    ]


def test_train_val_view_physically_omits_historical_test(tmp_path):
    source = tmp_path / "locked.yaml"
    source.write_text(yaml.safe_dump({
        "path": str(tmp_path), "train": "train.txt", "val": "val.txt",
        "test": "test.txt", "nc": 2, "names": ["ball", "bat"],
    }), encoding="utf-8")
    output = tmp_path / "clean" / "train_val.yaml"
    view = write_train_val_view(source, output)
    assert "test" not in view
    assert "test" not in yaml.safe_load(output.read_text(encoding="utf-8"))
