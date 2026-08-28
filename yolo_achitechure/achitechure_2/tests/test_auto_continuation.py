from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from achitechure_2.cli import build_parser
from achitechure_2.downstream_export import _select
from achitechure_2.full_training import (
    FullRunConfig,
    eligible_full_candidates,
    run_full_matrix,
)
from achitechure_2.quant_training import (
    QuantRouteModel,
    _quant_lineage,
    _take_cycling,
    run_quant_matrix,
)


def _metrics(candidate: str, *, drop: float, params: int) -> dict[str, object]:
    return {
        "metrics": {
            "candidate_id": candidate,
            "coco_box_map50_95": 0.8 - drop,
            "bbat5_pose_box_map50_95": 0.7 - drop,
            "bbat5_keypoint_map50_95": 0.6 - drop,
            "macro_f1": 0.9 - drop,
            "params": params,
            "gflops": float(params),
            "latency_ms": float(params),
        }
    }


def test_full_run_config_is_formal_and_closed_after_rejection() -> None:
    config = FullRunConfig.load(
        Path("configs/runs/full35-c2-c3-auto-continuation.yaml")
    )

    assert config.candidates == ("C2", "C3")
    assert config.epochs == 100
    assert config.patience == 20
    assert config.pose_enabled is True
    assert config.detect_data.name == "coco2017.yaml"
    assert config.pose_data.name == "bbat5-pose.yaml"
    assert config.payload["status"] == "closed_rejected_by_user"
    assert config.payload["authorization"] == {
        "gpu": True,
        "pose": True,
        "full_training": True,
        "ptq": True,
        "qat_lite": True,
        "authorized_at": "2026-08-27",
        "note_zh": "使用者於2026-08-28決定不採用本階段；full、PTQ與QAT-lite永久關閉，不得重新排入queue。",
    }


def test_closed_continuation_rejects_full_and_quant_execution() -> None:
    path = Path("configs/runs/full35-c2-c3-auto-continuation.yaml")

    with pytest.raises(PermissionError, match="已封存且不採用"):
        run_full_matrix(path, execute=True)
    with pytest.raises(PermissionError, match="已封存且不採用"):
        run_quant_matrix(path, execute=True)


def test_quant_lineage_preserves_full_ancestry_and_sets_direct_parent(
    tmp_path: Path,
) -> None:
    config = FullRunConfig.load(
        Path("configs/runs/full35-c2-c3-auto-continuation.yaml")
    )
    run_dir = tmp_path / "c2-full"
    checkpoint = run_dir / "inference/best-joint-formal.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"full checkpoint")
    inherited = {
        key: "inherited"
        for key in (
            "spec_version",
            "spec_sha256",
            "architecture_yaml",
            "architecture_yaml_sha256",
            "training_yaml",
            "training_yaml_sha256",
            "detect_dataset_yaml",
            "detect_dataset_yaml_sha256",
            "pose_dataset_yaml",
            "pose_dataset_yaml_sha256",
            "handoff_manifest",
            "handoff_manifest_sha256",
            "parent_checkpoint",
            "parent_checkpoint_sha256",
            "candidate",
        )
    }
    inherited["candidate"] = "C2"
    inherited["parent_checkpoint"] = "/immutable/j3-parent.pt"
    inherited["parent_checkpoint_sha256"] = "a" * 64
    (run_dir / "complete.json").write_text(
        json.dumps({"lineage": inherited}),
        encoding="utf-8",
    )

    lineage = _quant_lineage(
        config,
        candidate="C2",
        full_checkpoint=checkpoint,
    )

    assert lineage["parent_checkpoint"] == str(checkpoint)
    assert len(lineage["parent_checkpoint_sha256"]) == 64
    assert lineage["architecture_parent_checkpoint"] == "/immutable/j3-parent.pt"
    assert lineage["architecture_parent_checkpoint_sha256"] == "a" * 64
    assert lineage["spec_version"] == "inherited"
    assert lineage["quantization_policy_yaml"].endswith("w8a8-simulation.yaml")


def test_eligibility_accepts_exact_limit_and_rejects_larger_drop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = FullRunConfig.load(
        Path("configs/runs/full35-c2-c3-auto-continuation.yaml")
    )
    config = replace(config, run_root=tmp_path / "full")
    payload = {
        "candidates": [
            _metrics("C0", drop=0.0, params=100),
            _metrics("C1", drop=0.1, params=90),
            _metrics("C2", drop=0.008, params=80),
            _metrics("C3", drop=0.009, params=70),
        ]
    }
    monkeypatch.setattr(
        "achitechure_2.full_training._verify_export_manifest",
        lambda _: payload,
    )
    metrics_path = config.float_results / "metrics.json"
    monkeypatch.setattr(
        "achitechure_2.full_training.file_sha256",
        lambda path: "a" * 64 if path == metrics_path else "b" * 64,
    )

    report = eligible_full_candidates(config)

    assert report["eligible_candidates"] == ["C2"]
    decisions = {item["candidate"]: item for item in report["decisions"]}
    assert decisions["C2"]["accuracy_pass"] is True
    assert decisions["C2"]["cost_pass"] is True
    assert decisions["C3"]["accuracy_pass"] is False
    saved = json.loads(
        (config.run_root / "eligibility.json").read_text(encoding="utf-8")
    )
    assert saved["eligible_candidates"] == ["C2"]


class _RouteShared(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.detect_head = SimpleNamespace(
            stride=torch.tensor([8.0, 16.0, 32.0]),
            end2end=False,
        )
        self.pose_head = SimpleNamespace(
            stride=torch.tensor([8.0, 16.0, 32.0]),
            end2end=False,
            kpt_shape=(2, 3),
        )
        self.detect_names = {0: "person"}
        self.pose_names = {0: "ball", 1: "bat"}

    def forward(self, images: torch.Tensor, task: str) -> dict[str, torch.Tensor]:
        return {task: images * self.weight + (1 if task == "pose" else 0)}


def test_quant_route_model_returns_one_task_and_exposes_contract() -> None:
    shared = _RouteShared()
    images = torch.ones(1, 3, 4, 4)

    detect = QuantRouteModel(shared, "detect")
    pose = QuantRouteModel(shared, "pose")

    assert torch.equal(detect(images), torch.full_like(images, 2.0))
    assert torch.equal(pose(images), torch.full_like(images, 3.0))
    assert detect.names == {0: "person"}
    assert pose.names == {0: "ball", 1: "bat"}
    assert pose.kpt_shape == (2, 3)
    with pytest.raises(ValueError, match="detect 或 pose"):
        QuantRouteModel(shared, "both")


def test_cycling_batches_records_wraps() -> None:
    loader = [{"value": 1}, {"value": 2}]
    values, iterator, wraps = _take_cycling(iter(loader), loader, 3)

    assert [item["value"] for item in values] == [1, 2, 1]
    assert wraps == 1
    more, _, wraps = _take_cycling(iterator, loader, 2)
    assert [item["value"] for item in more] == [2, 1]
    assert wraps == 1


def test_cli_exposes_queue_only_full_and_quant_commands() -> None:
    parser = build_parser()

    full = parser.parse_args(
        ["full-run", "--queue-state", "/tmp/full.json", "--execute"]
    )
    quant = parser.parse_args(
        ["quant-run", "--queue-state", "/tmp/quant.json", "--execute"]
    )

    assert full.command == "full-run"
    assert quant.command == "quant-run"
    assert full.execute is quant.execute is True

def _selection_record(candidate: str, joint: float, latency: float) -> dict:
    return {
        "candidate": candidate,
        "formal_joint_score": joint,
        "cost": {"latency_ms": latency, "gflops": latency},
        "formal_metrics": {"macro_f1": 0.8},
        "quantization": {"recommended_stage": "q2l"},
    }


def test_final_selection_prefers_cost_only_inside_accuracy_near_tie() -> None:
    close = _select(
        [
            _selection_record("C2", 0.8, 5.0),
            _selection_record("C3", 0.795, 4.0),
        ]
    )
    separated = _select(
        [
            _selection_record("C2", 0.8, 5.0),
            _selection_record("C3", 0.79, 4.0),
        ]
    )

    assert close["near_tie_candidates"] == ["C2", "C3"]
    assert close["c_best"] == "C3"
    assert separated["near_tie_candidates"] == ["C2"]
    assert separated["c_best"] == "C2"


def test_cli_exposes_downstream_export_command() -> None:
    args = build_parser().parse_args(
        ["export-downstream-results", "--execute"]
    )

    assert args.command == "export-downstream-results"
    assert args.execute is True
