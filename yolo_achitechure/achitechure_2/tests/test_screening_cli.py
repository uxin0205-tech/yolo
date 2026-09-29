from __future__ import annotations

import json

from achitechure_2.cli import build_parser, main


def test_cli_exposes_screening_manifest_and_qat_lite_checks() -> None:
    parser = build_parser()
    assert (
        parser.parse_args(["prepare-screening-data"]).command
        == "prepare-screening-data"
    )
    assert (
        parser.parse_args(["validate-screening-data"]).command
        == "validate-screening-data"
    )
    assert (
        parser.parse_args(["qat-lite-check", "--candidate", "C0"]).command
        == "qat-lite-check"
    )


def test_qat_lite_cli_is_a_non_executing_fail_closed_gate(capsys) -> None:
    assert main(["qat-lite-check", "--candidate", "C0"]) == 2
    blocked = json.loads(capsys.readouterr().err)
    assert blocked["error_type"] == "PermissionError"
    assert "GPU" in blocked["error"]

    assert (
        main(
            [
                "qat-lite-check",
                "--candidate",
                "C0",
                "--gpu-authorized",
            ]
        )
        == 0
    )
    allowed = json.loads(capsys.readouterr().out)
    assert allowed["stage"] == "Q2L"
    assert allowed["simulation_only"] is True
    assert "不會開始訓練" in allowed["執行"]
