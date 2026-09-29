from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from masf_yolo.artifacts.checkpoints import (
    load_canonical_checkpoint,
    save_canonical_checkpoint,
)
from masf_yolo.models.builder import build_model
from masf_yolo.variants import get_variant


def _save(tmp_path: Path, variant: str = "M1") -> Path:
    model = build_model(variant).half()
    checkpoint = tmp_path / f"{variant.lower()}-canonical.pt"
    save_canonical_checkpoint(
        model,
        checkpoint,
        get_variant(variant),
        data_hash="d" * 64,
        config_hash="c" * 64,
        environment_hash="e" * 64,
    )
    return checkpoint


def test_canonical_checkpoint_contains_only_cpu_float32_state_dict(tmp_path: Path) -> None:
    checkpoint = _save(tmp_path)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert set(payload) == {"schema_version", "metadata", "state_dict"}
    assert payload["metadata"]["variant_id"] == "M1"
    assert all(not tensor.is_cuda for tensor in payload["state_dict"].values())
    assert all(
        tensor.dtype == torch.float32
        for tensor in payload["state_dict"].values()
        if tensor.is_floating_point()
    )


def test_strict_load_rejects_wrong_variant_and_manifest_hash(tmp_path: Path) -> None:
    checkpoint = _save(tmp_path)

    with pytest.raises(ValueError, match="variant"):
        load_canonical_checkpoint(build_model("M0"), checkpoint, get_variant("M0"))

    manifest_path = checkpoint.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="checkpoint hash"):
        load_canonical_checkpoint(build_model("M1"), checkpoint, get_variant("M1"))


def test_fresh_subprocess_rebuilds_and_strict_loads_checkpoint(tmp_path: Path) -> None:
    checkpoint = _save(tmp_path)
    command = [
        sys.executable,
        "-m",
        "masf_yolo.artifacts.strict_reload",
        "--checkpoint",
        str(checkpoint),
        "--variant",
        "M1",
        "--data-hash",
        "d" * 64,
        "--config-hash",
        "c" * 64,
        "--environment-hash",
        "e" * 64,
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["strict_load"] is True
    assert report["strides"] == [4.0, 8.0, 16.0, 32.0]
    assert report["detect_scales"] == 4


def test_fresh_subprocess_can_run_real_validation_after_strict_load(tmp_path: Path) -> None:
    checkpoint = _save(tmp_path)
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        images = dataset / split / "images"
        labels = dataset / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (64, 64), "black").save(images / "frame.jpg")
        (labels / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset}\ntrain: train/images\nval: val/images\nnc: 2\nnames: [ball, bat]\n"
    )
    command = [
        sys.executable,
        "-m",
        "masf_yolo.artifacts.strict_reload",
        "--checkpoint",
        str(checkpoint),
        "--variant",
        "M1",
        "--data-hash",
        "d" * 64,
        "--config-hash",
        "c" * 64,
        "--environment-hash",
        "e" * 64,
        "--data",
        str(data_yaml),
        "--device",
        "cpu",
        "--imgsz",
        "64",
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["validation_ran"] is True
    assert "metrics/mAP50-95(B)" in report["validation"]
