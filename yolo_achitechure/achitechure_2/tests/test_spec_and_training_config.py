from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from achitechure_2.config import (
    ConfigContractError,
    check_configs,
    load_candidate_specs,
    load_training_template,
    validate_runtime_overrides,
)


def test_config_check_describes_v2_matrix_and_blocked_handoff_values() -> None:
    report = check_configs()

    assert report.valid
    assert report.spec_version == "2.0.1"
    assert report.ultralytics_version == "8.4.90"
    assert report.candidate_ids == ("C0", "C1", "C2", "C3")
    assert report.checked_training_files == (
        "configs/training/cpu-smoke.yaml",
        "configs/training/float-extension.yaml",
        "configs/training/float-main.yaml",
        "configs/training/quant-qat.yaml",
    )
    assert "Pose 正式執行：等待使用者 opt-in" in report.accepted_but_inactive
    assert "batch_size：請使用 Ultralytics 正式鍵名 batch" in report.deprecated
    assert "上游 winner training recipe 尚未 handoff" in report.blocked
    assert "Float extension：等待 late gate 與未 strip continuation state" in report.blocked
    assert "Pose RLE active evidence：等待 handoff model/loss dry-run" in report.blocked
    assert "fusion template：融合選型屬於 yolo_combine" in report.deprecated
    assert report.runtime_overridable == (
        "cache",
        "device",
        "name",
        "project",
        "workers",
    )


def test_candidate_yaml_has_one_factor_and_resolves_paths_from_handoff() -> None:
    candidates = load_candidate_specs()

    assert tuple(candidates) == ("C0", "C1", "C2", "C3")
    assert candidates["C0"].factor_name == "none"
    assert candidates["C1"].changed_fields == ("e",)
    assert candidates["C2"].changed_fields == ("inner_n",)
    assert candidates["C3"].changed_fields == ("kernel_mode",)
    assert all(candidate.target_source == "handoff.candidate_regions" for candidate in candidates.values())
    assert all(not candidate.hardcoded_layer_indices for candidate in candidates.values())


def test_candidate_matrix_and_spec_hash_fail_closed(tmp_path: Path) -> None:
    destination = tmp_path / "candidates"
    shutil.copytree("configs/candidates", destination)
    c1_path = destination / "c1-e0375.yaml"
    c1 = yaml.safe_load(c1_path.read_text(encoding="utf-8"))
    c1["factors"]["inner_n"] = 1
    c1_path.write_text(yaml.safe_dump(c1, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigContractError, match="只能改變一個 factor"):
        load_candidate_specs(destination, spec_path=Path("EXPERIMENT_SPEC.md"))

    altered_spec = tmp_path / "EXPERIMENT_SPEC.md"
    altered_spec.write_text(Path("EXPERIMENT_SPEC.md").read_text(encoding="utf-8") + "\nchanged\n")
    with pytest.raises(ConfigContractError, match="spec_sha256"):
        load_candidate_specs(Path("configs/candidates"), spec_path=altered_spec)


def test_training_yaml_exposes_common_controls_without_inventing_handoff_recipe() -> None:
    main = load_training_template("configs/training/float-main.yaml")

    assert main["model_scale"] == "m"
    assert main["execution"]["requires_gpu_authorization"] is True
    assert main["routes"]["pose"]["enabled_by_default"] is False
    assert main["formal_ranking_requires"] == ["detect", "pose"]
    assert main["adjustable"]["batch"] == {"source": "handoff", "value": None}
    assert main["adjustable"]["fraction"] == {"source": "local", "value": 1.0}
    assert main["adjustable"]["scale"] == {"source": "handoff", "value": None}
    assert main["adjustable"]["cache"] == {"source": "local", "value": False}
    assert main["adjustable"]["imgsz"] == {"source": "handoff", "value": None}
    assert main["recipe"]["optimizer"] == {"source": "handoff", "value": None}
    assert main["recipe"]["task_ratio"] == {"source": "handoff", "value": None}
    assert main["validation"]["selection_backend"] == "float_model"
    assert main["validation"]["checkpoint_selection"] == {
        "detect": "handoff_defined",
        "pose_research": "pose_map50_95",
        "pose_official": "combined_fitness_recorded_separately",
    }


def test_catalog_exposes_formal_train_only_bbat5_search_yamls() -> None:
    catalog = yaml.safe_load(Path("configs/catalog.yaml").read_text(encoding="utf-8"))
    assert catalog["datasets"]["bbat5-pose-search"] == (
        "configs/data/bbat5-pose-search.yaml"
    )
    assert catalog["datasets"]["bbat5-detect-search"] == (
        "configs/data/bbat5-detect-search.yaml"
    )
    pose = yaml.safe_load(
        Path(catalog["datasets"]["bbat5-pose-search"]).read_text(encoding="utf-8")
    )
    detect = yaml.safe_load(
        Path(catalog["datasets"]["bbat5-detect-search"]).read_text(encoding="utf-8")
    )
    assert pose["project_metadata"]["role"] == "search_pose"
    assert detect["project_metadata"]["role"] == "search_diagnostic_detect"
    assert pose["project_metadata"]["formal_val_excluded"] is True
    assert detect["project_metadata"]["formal_val_excluded"] is True
    assert pose["group_key"] == detect["group_key"] == "prefix_before_.rf."
    assert Path(pose["train"]).name == Path(detect["train"]).name == "search-train.txt"
    assert Path(pose["val"]).name == Path(detect["val"]).name == "search-val.txt"


def test_float_extension_requires_unstripped_continuation_state(tmp_path: Path) -> None:
    extension = load_training_template("configs/training/float-extension.yaml")

    assert extension["transition"]["input"] == "own_unstripped_continuation_checkpoint"
    assert extension["transition"]["requires_unstripped_state"] is True
    assert extension["transition"]["stripped_last_or_best_policy"] == "reject"

    altered = tmp_path / "float-extension.yaml"
    extension["transition"]["input"] = "own_float_main_last_checkpoint"
    altered.write_text(yaml.safe_dump(extension, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigContractError, match="unstripped continuation"):
        load_training_template(altered)


def test_learning_cli_override_fails_closed() -> None:
    with pytest.raises(ConfigContractError, match="learning-field"):
        validate_runtime_overrides({"lr0": 0.1})
    assert validate_runtime_overrides({"device": "cpu", "workers": 2, "cache": False}) == {
        "device": "cpu",
        "workers": 2,
        "cache": False,
    }
