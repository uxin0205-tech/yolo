from __future__ import annotations

from pathlib import Path

import pytest

from masf_yolo.cli import (
    build_parser,
    build_systemd_command,
    ensure_tracked_clean,
    load_config,
)
from masf_yolo.runtime import launch_python_path


def test_repository_config_loads_with_locked_hash() -> None:
    config = load_config(Path("configs/static-phase1.yaml"))

    assert config.values["variants"] == ["B1", "M7", "M0", "M1", "M2", "M3", "P3M", "SP2"]
    assert config.values["pipeline"]["systemd_unit_prefix"] == "masf-yolo-phase1"
    assert config.values["pipeline"]["wait_for_units"] == ["yolo-p2-study.service"]
    assert config.values["model"]["sp2_hidden_channels"] == 32
    assert config.values["model"]["sp2_auxiliary_loss_weight"] == 1.0
    assert config.values["profiling"] == {
        "precision": "fp16",
        "batch": 1,
        "warmup": 100,
        "iterations": 1000,
    }
    assert len(config.config_hash) == 64


def test_cli_exposes_only_documented_operator_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["audit", "--config", "x.yaml"]).command == "audit"
    assert parser.parse_args(["verify", "--config", "x.yaml"]).command == "verify"
    assert parser.parse_args(["pipeline", "start", "--config", "x.yaml"]).pipeline_command == "start"
    assert parser.parse_args(["pipeline", "execute", "--config", "x.yaml"]).pipeline_command == "execute"
    assert parser.parse_args(["pipeline", "status", "--config", "x.yaml"]).pipeline_command == "status"
    assert parser.parse_args(["report", "--config", "x.yaml"]).command == "report"
    cleanup = parser.parse_args(["cleanup", "--config", "x.yaml"])
    assert cleanup.command == "cleanup"
    assert cleanup.apply is False
    assert parser.parse_args(["cleanup", "--config", "x.yaml", "--apply"]).apply is True


def test_systemd_command_uses_absolute_paths_and_one_service() -> None:
    command = build_systemd_command(
        config_path=Path("/work/configs/static-phase1.yaml"),
        unit="masf-yolo-phase1-abcd",
        python=Path("/work/.venv/bin/python"),
    )

    assert command[:4] == ["systemd-run", "--user", "--unit", "masf-yolo-phase1-abcd"]
    assert command.count("--unit") == 1
    assert "--property=OOMPolicy=continue" in command
    assert "--property=MemoryAccounting=yes" in command
    assert "/work/.venv/bin/python" in command
    assert command[-5:] == [
        "-m",
        "masf_yolo.cli",
        "pipeline",
        "execute",
        "--config=/work/configs/static-phase1.yaml",
    ]


def test_tracked_clean_gate_ignores_untracked_but_rejects_tracked_changes() -> None:
    ensure_tracked_clean("")
    ensure_tracked_clean("?? local-data/")

    with pytest.raises(RuntimeError, match="tracked worktree"):
        ensure_tracked_clean(" M yolo_masf/masf_yolo/cli.py")


def test_systemd_launch_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    virtualenv_python = tmp_path / ".venv" / "bin" / "python"
    virtualenv_python.parent.mkdir(parents=True)
    virtualenv_python.symlink_to("/usr/bin/python3.14")

    selected = launch_python_path(str(virtualenv_python))

    assert selected == virtualenv_python
    assert selected != virtualenv_python.resolve()
