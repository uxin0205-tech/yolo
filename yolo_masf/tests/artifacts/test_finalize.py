from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from masf_yolo.artifacts.finalize import build_finalize_command, finalize_native_checkpoint
from masf_yolo.models.builder import build_model


def _native_checkpoint(path: Path) -> None:
    model = build_model("B1").half()
    torch.save(
        {
            "model": model,
            "ema": None,
            "epoch": -1,
            "train_args": {"task": "detect", "model": "b1.pt", "imgsz": 640, "epochs": 1},
        },
        path,
    )


def test_finalize_native_checkpoint_writes_strict_canonical_artifact(tmp_path: Path) -> None:
    source = tmp_path / "best.pt"
    canonical = tmp_path / "canonical.pt"
    _native_checkpoint(source)

    report = finalize_native_checkpoint(
        source,
        canonical,
        variant_id="B1",
        data_hash="d" * 64,
        config_hash="c" * 64,
        environment_hash="e" * 64,
    )

    assert report["variant"] == "B1"
    assert report["source"] == str(source.resolve())
    assert report["source_hash"]
    assert report["checkpoint_hash"]
    assert canonical.is_file()
    assert canonical.with_suffix(".manifest.json").is_file()


def test_finalize_cli_emits_machine_readable_report(tmp_path: Path) -> None:
    source = tmp_path / "best.pt"
    canonical = tmp_path / "canonical.pt"
    report_path = tmp_path / "finalize.json"
    _native_checkpoint(source)
    command = [
        sys.executable,
        "-m",
        "masf_yolo.artifacts.finalize",
        "--source",
        str(source),
        "--checkpoint",
        str(canonical),
        "--variant",
        "B1",
        "--data-hash",
        "d" * 64,
        "--config-hash",
        "c" * 64,
        "--environment-hash",
        "e" * 64,
        "--output",
        str(report_path),
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert json.loads(result.stdout.splitlines()[-1]) == report
    assert report["checkpoint_hash"]


def test_finalize_command_has_all_hash_gates() -> None:
    command = build_finalize_command(
        python=Path("/work/python"),
        source=Path("/work/best.pt"),
        checkpoint=Path("/work/canonical.pt"),
        variant_id="M7",
        data_hash="d" * 64,
        config_hash="c" * 64,
        environment_hash="e" * 64,
        output=Path("/work/finalize.json"),
    )

    assert command[:3] == ["/work/python", "-m", "masf_yolo.artifacts.finalize"]
    assert command[command.index("--variant") + 1] == "M7"
    assert command[command.index("--data-hash") + 1] == "d" * 64
    assert command[command.index("--config-hash") + 1] == "c" * 64
    assert command[command.index("--environment-hash") + 1] == "e" * 64
