from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from yolo_combine.variants import VariantConfigError, VariantWorkspace, load_variant

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("full35", "partial75")


def _files(name: str) -> set[Path]:
    root = PROJECT_ROOT / "variants" / name
    files: set[Path] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_file() and "artifacts" not in relative.parts:
            files.add(relative)
    return files


def _json(name: str) -> dict[str, object]:
    path = PROJECT_ROOT / "variants" / name / "baselines" / "independent.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(name: str, relative: str) -> dict[str, object]:
    path = PROJECT_ROOT / "variants" / name / relative
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_variant_workspaces_load_and_reference_existing_inputs() -> None:
    full = load_variant("full35", project_root=PROJECT_ROOT)
    partial = load_variant("partial75", project_root=PROJECT_ROOT)

    assert full.architecture == "full35"
    assert full.role == "primary"
    assert partial.architecture == "partial75"
    assert partial.role == "fallback"
    assert full.bbat5_version == partial.bbat5_version == "bbat5-v1"
    assert full.bbat5_registry == partial.bbat5_registry
    assert full.run_root != partial.run_root
    assert full.run_root == full.root / "artifacts"
    assert partial.run_root == partial.root / "artifacts"
    assert full.pose_run_root == full.root / "artifacts" / "pose"
    assert partial.pose_run_root == partial.root / "artifacts" / "pose"
    assert full.fusion_run_root == full.root / "artifacts" / "fusion"
    assert partial.fusion_run_root == partial.root / "artifacts" / "fusion"
    assert full.pose_view_root == full.run_root / "cache-views" / "bbat5-v1"
    assert (
        partial.pose_view_root == partial.run_root / "cache-views" / "bbat5-v1"
    )
    assert full.cpu_report_path != partial.cpu_report_path
    assert full.joint_smoke_report_path != partial.joint_smoke_report_path
    assert full.audit().ok
    assert partial.audit().ok


def test_variant_folders_have_the_same_relative_interface() -> None:
    assert _files("full35") == _files("partial75")


def test_independent_baselines_have_matching_schema_and_shared_history() -> None:
    full = _json("full35")

    partial = _json("partial75")

    assert set(full) == set(partial)
    assert full["schema_version"] == partial["schema_version"] == 1
    assert full["original_bbt5_pose"] == partial["original_bbt5_pose"]
    assert set(full["coco_detect"]) == set(partial["coco_detect"])
    assert set(full["architecture_matched_pose26"]) == set(
        partial["architecture_matched_pose26"]
    )


def test_each_folder_has_an_architecture_locked_entrypoint() -> None:
    for name in VARIANTS:
        workspace = load_variant(name, project_root=PROJECT_ROOT)
        runner = workspace.root / "run.py"
        content = runner.read_text(encoding="utf-8")

        assert runner.is_file()
        assert "--architecture" not in content
        assert "Path(__file__).resolve().parent" in content
        assert workspace.run_root.is_relative_to(workspace.root)
        assert workspace.pose_view_root.is_relative_to(workspace.root)
        assert workspace.fusion_run_root.is_relative_to(workspace.root)


def test_training_configs_distinguish_official_reference_from_actual_p1() -> None:
    for name in VARIANTS:
        config = _yaml(name, "configs/training.yaml")
        assert config["architecture"] == name
        official = config["official_yolo26m_coco_reference"]
        observed = config["observed_full35_p1"]
        schedule = config["pose_schedule"]
        assert official["published_batch"] == 128
        assert official["epochs"] == 80
        assert official["scope"] == "reference_only"
        assert observed["physical_batch"] == 16
        assert observed["nbs"] == 64
        assert observed["steady_state_accumulate"] == 4
        assert observed["epochs"] == 10
        assert schedule["batch_argument"] == 128
        assert schedule["p1"]["epochs"] == 17
        assert schedule["p2"]["epochs"] == 22
        assert schedule["p3"]["epochs"] == 100
        assert config["batch_decision"]["status"] == "locked_by_user"
        assert config["gpu_execution"]["performed_by_workspace_setup"] is False


def test_fusion_configs_keep_the_same_ratios_and_accuracy_gate() -> None:
    full = _yaml("full35", "configs/fusion.yaml")
    partial = _yaml("partial75", "configs/fusion.yaml")

    assert full["ratios"] == partial["ratios"]
    assert full["accuracy_gate"] == partial["accuracy_gate"]
    assert full["accuracy_gate"]["maximum_map50_95_drop"] == 0.08
    assert full["run_root"] == "artifacts/fusion"
    assert partial["run_root"] == "artifacts/fusion"


def test_variant_loader_fails_when_folder_name_and_architecture_disagree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wrong-name"
    root.mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    source = PROJECT_ROOT / "variants" / "full35" / "variant.yaml"
    (root / "variant.yaml").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )

    try:
        VariantWorkspace.load(root)
    except ValueError as error:
        assert "does not match architecture" in str(error)
    else:
        raise AssertionError("mismatched variant folder was accepted")


def test_variant_run_root_cannot_escape_its_architecture_folder(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    root = tmp_path / "variants" / "full35"
    root.mkdir(parents=True)
    payload = yaml.safe_load(
        (PROJECT_ROOT / "variants/full35/variant.yaml").read_text(encoding="utf-8")
    )
    payload["runs"]["root"] = "../partial75/artifacts"
    (root / "variant.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(VariantConfigError, match="escapes"):
        VariantWorkspace.load(root)
