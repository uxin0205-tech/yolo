from __future__ import annotations

import json

import pytest

from achitechure_2.cli import build_parser, main


def test_cli_exposes_phase_a_commands_and_no_guessing_train_command() -> None:
    parser = build_parser()
    assert parser.parse_args(["status"]).command == "status"
    assert parser.parse_args(["config-check"]).command == "config-check"
    assert parser.parse_args(["prepare-pose-data"]).command == "prepare-pose-data"
    assert parser.parse_args(["validate-pose-data"]).command == "validate-pose-data"
    assert parser.parse_args(["show-candidates"]).command == "show-candidates"
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


def test_status_is_chinese_cpu_phase_a_and_pose_is_user_choice(capsys) -> None:
    assert main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["current_phase"] == "A"
    assert payload["gpu_actions"] == "blocked_until_user_authorization"
    assert payload["pose_formal_execution"] == "requires_user_opt_in"
    assert payload["fusion_winner"] == "waiting_for_yolo_combine_handoff"
    assert payload["spec_version"] == "2.0.1"
    assert len(payload["spec_sha256"]) == 64
    assert payload["bbat5_v1_lineage"]["spec_version"] == "2.0.0"


def test_config_check_is_available_from_cli(capsys) -> None:
    assert main(["config-check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"]
    assert payload["candidate_ids"] == ["C0", "C1", "C2", "C3"]
