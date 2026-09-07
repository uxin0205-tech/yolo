from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from yolo_quantize import (
    Full35SearchValidationPlan,
    prepare_bbat5_search_view,
    search_validation,
)
from yolo_quantize.search_validation import main


def _sample(root: Path, name: str) -> Path:
    image = root / "images/train" / f"{name}.jpg"
    label = root / "labels/train" / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(name.encode())
    label.write_text("1 0.5 0.5 0.2 0.2 0.4 0.4 2 0.6 0.6 2\n")
    return image


def test_search_view_preserves_assignment_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "canonical/pose"
    train = [_sample(source, "a.rf.1"), _sample(source, "b.rf.1")]
    val = [_sample(source, "c.rf.1")]
    splits = source / "splits"
    splits.mkdir()
    (splits / "search-train.txt").write_text(
        "\n".join(str(path) for path in train) + "\n"
    )
    (splits / "search-val.txt").write_text("\n".join(str(path) for path in val) + "\n")
    source_yaml = tmp_path / "pose-search.yaml"
    source_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(source),
                "train": str(splits / "search-train.txt"),
                "val": str(splits / "search-val.txt"),
                "names": {0: "ball", 1: "bat"},
                "kpt_shape": [2, 3],
                "flip_idx": [0, 1],
            }
        )
    )

    destination = tmp_path / "runtime"
    first = prepare_bbat5_search_view(
        source_yaml,
        destination,
        expected_train=2,
        expected_val=1,
    )
    second = prepare_bbat5_search_view(
        source_yaml,
        destination,
        expected_train=2,
        expected_val=1,
    )

    assert first == second
    assert first.train_images == 2
    assert first.val_images == 1
    assert (destination / "val/images/c.rf.1.jpg").resolve() == val[0]
    assert (destination / "val/labels/c.rf.1.txt").is_symlink()
    runtime_yaml = yaml.safe_load(first.yaml.read_text())
    assert runtime_yaml["path"] == str(destination.resolve())
    assert runtime_yaml["val"] == "val/images"
    manifest = json.loads(first.manifest.read_text())
    assert manifest["assignment_changed"] is False
    assert manifest["split_counts"] == {"train": 2, "val": 1}
    assert manifest["storage"] == "symlink-only-runtime-view"


def test_search_view_keeps_canonical_symlink_as_its_direct_lineage(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    archive_image = archive / "source.jpg"
    archive_label = archive / "source.txt"
    archive_image.write_bytes(b"image")
    archive_label.write_text("0 0.5 0.5 0.2 0.2 0.4 0.4 2 0.6 0.6 2\n")

    source = tmp_path / "canonical/pose"
    image = source / "images/train/a.rf.1.jpg"
    label = source / "labels/train/a.rf.1.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.symlink_to(archive_image)
    label.symlink_to(archive_label)
    splits = source / "splits"
    splits.mkdir()
    (splits / "search-train.txt").write_text(f"{image}\n")
    (splits / "search-val.txt").write_text("")
    source_yaml = tmp_path / "pose-search.yaml"
    source_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(source),
                "train": str(splits / "search-train.txt"),
                "val": str(splits / "search-val.txt"),
                "names": {0: "ball", 1: "bat"},
                "kpt_shape": [2, 3],
                "flip_idx": [0, 1],
            }
        )
    )

    prepared = prepare_bbat5_search_view(
        source_yaml,
        tmp_path / "runtime",
        expected_train=1,
        expected_val=0,
    )

    runtime_image = prepared.root / "train/images/a.rf.1.jpg"
    runtime_label = prepared.root / "train/labels/a.rf.1.txt"
    assert runtime_image.readlink() == image
    assert runtime_label.readlink() == label


def test_reviewed_search_plan_pins_one_matched_w8_comparison() -> None:
    project = Path(__file__).resolve().parents[1]
    plan = Full35SearchValidationPlan.from_yaml(
        project / "configs/experiments/v4-qsilu-backbone-early-w8-search-v1.yaml"
    )

    assert plan.execution_authorization_id == (
        "user-2026-09-03-qsilu-backbone-early-w8-search"
    )
    assert plan.metric_contract_id == ("full35-coco-val-bbat5-search-val-bittrue-v1")
    assert plan.gate_spec.metric_family == "map50_95"
    assert plan.cell.cell_id == (
        "qsilu_pq--lsq-plus-a8--backbone_early--w8-mse_grid_v1"
    )
    assert plan.detect_batch_size == 32
    assert plan.pose_batch_size == 16
    assert plan.formal_training is False
    assert plan.formal_validation is False


def test_search_cli_lists_contract_without_loading_cuda(
    capsys,
) -> None:
    project = Path(__file__).resolve().parents[1]

    result = main(
        [
            "--plan",
            str(
                project
                / "configs/experiments/v4-qsilu-backbone-early-w8-search-v1.yaml"
            ),
            "--list-only",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cell"]["weight_bits"] == 8
    assert payload["formal_training"] is False
    assert payload["formal_validation"] is False


def test_search_cli_requires_explicit_execution_acknowledgement(capsys) -> None:
    project = Path(__file__).resolve().parents[1]

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--plan",
                str(
                    project / "configs/experiments/"
                    "v4-qsilu-backbone-early-w8-search-v1.yaml"
                ),
            ]
        )

    assert "--execute-reviewed-plan" in capsys.readouterr().err


def test_search_cli_rejects_an_unavailable_cuda_device(capsys) -> None:
    project = Path(__file__).resolve().parents[1]

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--plan",
                str(
                    project / "configs/experiments/"
                    "v4-qsilu-backbone-early-w8-search-v1.yaml"
                ),
                "--device",
                "99",
                "--execute-reviewed-plan",
            ]
        )

    assert "CUDA device 99 is unavailable" in capsys.readouterr().err


def test_search_cli_executes_only_the_reviewed_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / "search.json"
    observed = {}

    monkeypatch.setattr(search_validation.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(search_validation.torch.cuda, "device_count", lambda: 1)

    def fake_runner(*, plan, output, device_index, resume):
        observed.update(
            {
                "plan_id": plan.plan_id,
                "output": output,
                "device_index": device_index,
                "resume": resume,
            }
        )
        return 0

    monkeypatch.setattr(search_validation, "run_search_validation", fake_runner)

    result = main(
        [
            "--plan",
            str(
                project / "configs/experiments/"
                "v4-qsilu-backbone-early-w8-search-v1.yaml"
            ),
            "--output",
            str(output),
            "--device",
            "0",
            "--resume",
            "--execute-reviewed-plan",
        ]
    )

    assert result == 0
    assert observed == {
        "plan_id": "v4-qsilu-backbone-early-w8-search-v1",
        "output": output,
        "device_index": 0,
        "resume": True,
    }
