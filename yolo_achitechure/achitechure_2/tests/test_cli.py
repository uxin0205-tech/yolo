from __future__ import annotations

import json
from pathlib import Path

import pytest

from achitechure_2.cli import _float20_preflight, build_parser, main


def test_cli_exposes_phase_a_commands_and_no_guessing_train_command() -> None:
    parser = build_parser()
    assert parser.parse_args(["status"]).command == "status"
    assert parser.parse_args(["config-check"]).command == "config-check"
    assert parser.parse_args(["prepare-pose-data"]).command == "prepare-pose-data"
    assert parser.parse_args(["validate-pose-data"]).command == "validate-pose-data"
    assert parser.parse_args(["export-github-dataset"]).command == "export-github-dataset"
    assert parser.parse_args(["validate-github-dataset"]).command == "validate-github-dataset"
    assert parser.parse_args(["show-candidates"]).command == "show-candidates"
    assert parser.parse_args(["float20-plan"]).command == "float20-plan"
    assert parser.parse_args(["queue-status"]).command == "queue-status"
    assert parser.parse_args(["float20-profile", "--queue-state", "state.json"]).command == "float20-profile"
    assert parser.parse_args(["export-float20-results"]).command == "export-float20-results"
    assert parser.parse_args(["native-loss-smoke", "--output", "report.json"]).command == "native-loss-smoke"
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


def test_status_is_chinese_cpu_phase_a_and_pose_is_user_choice(tmp_path: Path, capsys) -> None:
    (tmp_path / "configs/runs").mkdir(parents=True)
    (tmp_path / "EXPERIMENT_SPEC.md").write_text("temporary spec\n", encoding="utf-8")
    (tmp_path / "configs/runs/full35-float-screen-20.yaml").write_text(
        "authorization:\n  pose: pending_user_decision\n",
        encoding="utf-8",
    )
    assert main(["--project-root", str(tmp_path), "status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["current_phase"] == "A"
    assert payload["gpu_actions"] == "waiting_for_pose_decision"
    assert payload["pose_formal_execution"] == "pending_user_decision"
    assert payload["fusion_winner"] == "waiting_for_yolo_combine_handoff"
    assert payload["spec_version"] == "2.3.0"
    assert len(payload["spec_sha256"]) == 64


def test_config_check_is_available_from_cli(capsys) -> None:
    assert main(["config-check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"]
    assert payload["candidate_ids"] == ["C0", "C1", "C2", "C3"]


def test_float20_run_cannot_bypass_gpu_queue(capsys) -> None:
    state = Path("artifacts/queue/full35-j3-float20-seed0.json")
    assert (
        main(
            [
                "float20-run",
                "--queue-state",
                str(state),
                "--execute",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().err)
    assert payload["error_type"] == "PermissionError"
    assert "gpu_queue" in payload["error"]


def test_float20_profile_cannot_bypass_gpu_queue(capsys) -> None:
    state = Path("artifacts/queue/full35-j3-float20-seed0.json")
    assert (
        main(
            [
                "float20-profile",
                "--queue-state",
                str(state),
                "--execute",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().err)
    assert payload["error_type"] == "PermissionError"
    assert "gpu_queue" in payload["error"]


def test_float20_result_export_requires_explicit_execute(capsys) -> None:
    assert main(["export-float20-results"]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["error_type"] == "PermissionError"
    assert "--execute" in payload["error"]


def _write_test_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_float20_preflight_requires_all_materialized_cpu_evidence(tmp_path: Path) -> None:
    revision = "full35-final-j3-seed0-d67fb45c"
    cpu_root = tmp_path / "artifacts/cpu-validation"
    _write_test_json(
        tmp_path / "artifacts/intake/accepted.json",
        {"accepted": True, "revision_id": revision},
    )
    for candidate in ("C0", "C1", "C2", "C3"):
        _write_test_json(
            cpu_root / f"full35-{candidate.lower()}-dry-run.json",
            {
                "build": {"resolved_id": candidate},
                "handoff": {"accepted": True, "revision_id": revision},
                "cpu_validation": {
                    "passed": True,
                    "geometry_imgsz": 640,
                    "loss_is_finite": True,
                    "gradients_are_finite": True,
                    "state_dict_reload": True,
                    "contract_unchanged": True,
                    "frozen_gradient_count": 0,
                },
            },
        )
    _write_test_json(
        cpu_root / "full35-native-loss-smoke.json",
        {
            "schema_version": 2,
            "handoff": {"revision_id": revision},
            "losses": {"detect": {"finite": True}, "pose": {"finite": True}},
            "pose_rle": {"active": True},
            "cache_policy": {"source_adjacent_caches_absent": True},
        },
    )

    isolated_source_caches = (
        tmp_path / "source/coco.cache",
        tmp_path / "source/bbat5.cache",
    )
    passed = _float20_preflight(tmp_path, source_cache_paths=isolated_source_caches)
    assert passed["ready"]
    assert passed["passed_candidates"] == ["C0", "C1", "C2", "C3"]
    assert passed["native_loss_passed"] is True

    assert passed["source_adjacent_caches_absent"] is True
    c2_path = cpu_root / "full35-c2-dry-run.json"
    c2 = json.loads(c2_path.read_text(encoding="utf-8"))
    c2["cpu_validation"]["state_dict_reload"] = False
    _write_test_json(c2_path, c2)

    rejected = _float20_preflight(tmp_path, source_cache_paths=isolated_source_caches)
    assert rejected["ready"] is False
    assert rejected["invalid"] == ["cpu_c2:contract"]
