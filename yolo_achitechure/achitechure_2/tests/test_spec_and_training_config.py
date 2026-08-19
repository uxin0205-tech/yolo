from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from achitechure_2.cli import main
from achitechure_2.config import (
    ConfigContractError,
    check_configs,
    compose_training_config,
    load_candidate_specs,
    load_formal_training_config,
    validate_runtime_overrides,
)
from achitechure_2.graph import graph_snapshot
from achitechure_2.training import (
    require_c0_or_c_best,
    require_pose_opt_in,
    validate_stage_transition,
)


def test_config_check_covers_matrix_local_version_and_inactive_keys() -> None:
    report = check_configs()
    assert report.valid
    assert report.ultralytics_version == "8.4.90"
    assert report.candidate_ids == ("C0", "C1", "C2", "C3", "C3-P5", "R1")
    assert report.accepted_but_inactive == ("pose:rle（只有 Pose.flow_model 存在時才生效）",)
    assert report.catalog_file == "configs/catalog.yaml"
    assert report.spec_version == "1.2.0"
    assert "pose（必須由使用者明確啟用）" in report.optional_routes
    assert report.runtime_overridable == ("cache", "device", "model", "name", "project", "workers")
    assert len(report.checked_training_files) == 9


def test_candidate_yaml_matrix_fails_closed_on_factor_drift(tmp_path: Path) -> None:
    source = Path("configs/candidates")
    destination = tmp_path / "candidates"
    shutil.copytree(source, destination)
    path = destination / "c1-e0375.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["factors"]["inner_n"] = 1
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigContractError, match="matrix entry drift"):
        load_candidate_specs(destination, spec_path=Path("EXPERIMENT_SPEC.md"))


def test_spec_hash_drift_and_learning_cli_override_fail_closed(tmp_path: Path) -> None:
    source = Path("configs/candidates")
    destination = tmp_path / "candidates"
    shutil.copytree(source, destination)
    altered_spec = tmp_path / "EXPERIMENT_SPEC.md"
    altered_spec.write_text(Path("EXPERIMENT_SPEC.md").read_text() + "\nchanged\n", encoding="utf-8")
    with pytest.raises(ConfigContractError, match="spec_sha256"):
        load_candidate_specs(destination, spec_path=altered_spec)
    with pytest.raises(ConfigContractError, match="learning-field"):
        validate_runtime_overrides({"lr0": 0.1})
    assert validate_runtime_overrides({"device": "1", "workers": 4}) == {
        "device": "1",
        "workers": 4,
    }


def test_detect_ablation_composition_differs_only_by_name() -> None:
    configs = [
        compose_training_config(task="detect", candidate_id=candidate, stage="D1")
        for candidate in ("C0", "C1", "C2", "C3")
    ]
    canonical = {key: value for key, value in configs[0].args.items() if key != "name"}
    assert all(
        {key: value for key, value in config.args.items() if key != "name"} == canonical
        for config in configs[1:]
    )
    assert canonical["batch"] == 16
    assert canonical["nbs"] == 64
    assert canonical["optimizer"] == "MuSGD"


def test_pose_recipe_and_stage_transitions_are_explicit(tmp_path: Path) -> None:
    p1 = compose_training_config(task="pose", candidate_id="C0", stage="P1")
    p2 = compose_training_config(task="pose", candidate_id="C0", stage="P2")
    p3 = compose_training_config(task="pose", candidate_id="C0", stage="P3")
    p4 = compose_training_config(task="pose", candidate_id="C0", stage="P4")
    assert (p1.args["box"], p1.args["cls"], p1.args["dfl"]) == (7.5, 0.5, 1.5)
    assert (p1.args["pose"], p1.args["kobj"], p1.args["rle"]) == (12.0, 1.0, 1.0)
    assert p1.args["fliplr"] == 0.0
    assert p2.args["freeze"] == 11
    assert p3.args["freeze"] is None
    best = tmp_path / "best.pt"
    last = tmp_path / "last.pt"
    best.touch()
    last.touch()
    validate_stage_transition(p2, best)
    validate_stage_transition(p3, best)
    validate_stage_transition(p4, last)
    with pytest.raises(ValueError, match="last.pt resume"):
        validate_stage_transition(p4, best)


def test_graph_snapshot_is_review_only_and_matches_graph(toy_parent, pose_parent) -> None:
    detect = graph_snapshot(toy_parent, "C0")
    pose = graph_snapshot(pose_parent, "C0")
    assert not detect["standalone_loadable"]
    assert detect["builder"] == "achitechure_2"
    assert detect["head_contract"]["inputs"] == [16, 19, 22]
    assert detect["task"] == "detect"
    assert pose["task"] == "pose"
    assert pose["head_contract"]["type"] == "Pose26"


def test_training_unknown_top_level_and_nonzero_formal_seed_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "project"
    shutil.copytree("configs", root / "configs")
    root.mkdir(exist_ok=True)
    shutil.copy2("EXPERIMENT_SPEC.md", root / "EXPERIMENT_SPEC.md")
    formal = root / "configs/training/detect/d1-main.yaml"
    payload = yaml.safe_load(formal.read_text(encoding="utf-8"))
    payload["silent_typo"] = True
    formal.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigContractError, match="正式訓練欄位不完整"):
        check_configs(root)
    with pytest.raises(ValueError, match="seed 固定為 0"):
        main(["build", "--candidate", "C0", "--seed", "1"])


def test_pose_and_q2_downstream_gate_uses_recorded_c_best(tmp_path: Path) -> None:
    require_c0_or_c_best(tmp_path, "C0")
    with pytest.raises(RuntimeError, match="尚未選出 C_best"):
        require_c0_or_c_best(tmp_path, "C2")
    selection = tmp_path / "artifacts/selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(
        '{"c_best": {"metrics": {"candidate_id": "C2"}}}\n',
        encoding="utf-8",
    )
    require_c0_or_c_best(tmp_path, "C2")
    with pytest.raises(RuntimeError, match="已記錄的 C_best"):
        require_c0_or_c_best(tmp_path, "C1")



def test_pose_is_user_opt_in_at_cli_and_python_entry() -> None:
    require_pose_opt_in("detect", False)
    require_pose_opt_in("pose", True)
    with pytest.raises(ValueError, match="Pose 預設停用"):
        require_pose_opt_in("pose", False)
    with pytest.raises(ValueError, match="--enable-pose"):
        main(
            [
                "build-pose",
                "--candidate",
                "C0",
                "--checkpoint",
                "unused.pt",
                "--execute",
            ]
        )
    assert main(
        [
            "train",
            "--candidate",
            "C0",
            "--checkpoint",
            "unused.pt",
            "--task",
            "pose",
            "--stage",
            "P0",
            "--run-id",
            "dry-only",
        ]
    ) == 0


def test_fusion_template_is_disabled_and_mode_is_unset() -> None:
    payload = yaml.safe_load(Path("configs/fusion/source-pair.template.yaml").read_text(encoding="utf-8"))
    assert payload["enabled"] is False
    assert payload["fusion_mode"] is None
    assert payload["switch_policy"] is None
    assert set(payload["source_a"]) == set(payload["source_b"])



def test_formal_training_yaml_is_single_source_and_cli_accepts_config() -> None:
    formal = load_formal_training_config(
        "configs/training/detect/d1-main.yaml",
        candidate_id="C0",
    )
    assert formal.config_id == "detect-d1-main"
    assert formal.title_zh == "Detect D1：正式候選比較"
    assert len(formal.sources) == 1
    assert formal.sources[0].name == "d1-main.yaml"
    assert formal.args["epochs"] == 100
    assert formal.args["batch"] == 16
    assert formal.args["fraction"] == 1.0
    assert formal.args["scale"] == pytest.approx(0.95)
    assert formal.args["cache"] is False
    assert formal.args["multi_scale"] == 0.0
    assert main(
        [
            "train",
            "--config",
            "configs/training/detect/d1-main.yaml",
            "--candidate",
            "C0",
            "--checkpoint",
            "unused.pt",
            "--run-id",
            "dry-formal-config",
        ]
    ) == 0


def test_q2_lr_is_explicit_in_formal_yaml() -> None:
    q2 = load_formal_training_config(
        "configs/training/detect/q2-qat.yaml",
        candidate_id="C0",
    )
    assert q2.args["lr0"] == pytest.approx(0.000038)
    payload = yaml.safe_load(q2.sources[0].read_text(encoding="utf-8"))
    assert payload["derived_training"]["lr0"] == {
        "formula": "parent_checkpoint_lr0 * 0.1",
        "expected_if_parent_uses_spec": pytest.approx(0.000038),
    }
