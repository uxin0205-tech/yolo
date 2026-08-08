"""Isolated native Ultralytics training-stage worker."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from masf_yolo.artifacts.io import atomic_write_json
from masf_yolo.contracts import sha256_file
from masf_yolo.models.builder import build_model
from masf_yolo.models.transfer import transfer_b1_canonical

from .runner import run_training
from .resume import PermanentTrainingError, TransientTrainingError


@dataclass(frozen=True, slots=True)
class TrainingWorkerRequest:
    config_path: Path
    stage: str
    variant_id: str
    profile: dict[str, Any]
    resume_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "stage": self.stage,
            "variant_id": self.variant_id,
            "profile": self.profile,
            "resume_path": str(self.resume_path) if self.resume_path is not None else None,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrainingWorkerRequest":
        expected = {"config_path", "stage", "variant_id", "profile", "resume_path"}
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(f"unknown training worker keys: {sorted(unknown)}")
        missing = expected - set(raw)
        if missing:
            raise ValueError(f"missing training worker keys: {sorted(missing)}")
        profile = raw["profile"]
        if not isinstance(profile, dict):
            raise ValueError("training worker profile must be a mapping")
        resume = raw["resume_path"]
        return cls(
            config_path=Path(str(raw["config_path"])),
            stage=str(raw["stage"]),
            variant_id=str(raw["variant_id"]),
            profile=dict(profile),
            resume_path=Path(str(resume)) if resume is not None else None,
        )


def build_worker_command(*, python: Path, request_path: Path, output_path: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "masf_yolo.training.worker",
        "--request",
        str(request_path),
        "--output",
        str(output_path),
    ]


def launch_worker_process(
    request: TrainingWorkerRequest,
    *,
    request_path: Path,
    output_path: Path,
    python: Path,
) -> dict[str, Any]:
    request_path = request_path.resolve()
    output_path = output_path.resolve()
    atomic_write_json(request_path, request.to_dict())
    output_path.unlink(missing_ok=True)
    command = build_worker_command(
        python=python,
        request_path=request_path,
        output_path=output_path,
    )
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "no worker diagnostics"
        if result.returncode in {-9, 137} or "TransientTrainingError" in message:
            raise TransientTrainingError(
                f"training worker exited {result.returncode}: {message}"
            )
        raise PermanentTrainingError(
            f"training worker exited {result.returncode}: {message}"
        )
    if not output_path.is_file():
        raise PermanentTrainingError("training worker returned success without result manifest")
    try:
        report = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as error:
        raise PermanentTrainingError("training worker result manifest is unreadable") from error
    if not isinstance(report, dict) or not all(report.get(key) for key in ("best", "last")):
        raise PermanentTrainingError("training worker result manifest is incomplete")
    return report


def _read_training_record(artifact_root: Path, stage: str) -> dict[str, Any]:
    path = artifact_root / "training" / stage / "run.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _initial_model(request: TrainingWorkerRequest) -> tuple[torch.nn.Module | None, dict[str, Any] | None]:
    if request.resume_path is not None:
        return None, None
    from masf_yolo.cli import load_config

    config_path = request.config_path.resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    artifact_root = root / config.values["artifacts_root"]
    if request.stage == "b1_a":
        model = build_model(
            "B1",
            source_weights=(root / config.values["model"]["source_weights"]).resolve(),
        )
        return model, getattr(model, "masf_transfer_report", None)
    if request.stage == "b1_b":
        from ultralytics import YOLO

        model = YOLO(_read_training_record(artifact_root, "b1_a")["best"], task="detect").model
        from masf_yolo.variants import get_variant

        model.masf_variant = "B1"
        model.masf_variant_hash = get_variant("B1").config_hash
        return model, None
    b1_checkpoint = artifact_root / "training" / "b1_b" / "canonical.pt"
    b1 = build_model("B1", checkpoint=b1_checkpoint)
    model = build_model(request.variant_id)
    transfer = transfer_b1_canonical(model, b1.state_dict())
    return model, transfer.to_dict()


def execute_request(request: TrainingWorkerRequest) -> dict[str, Any]:
    model, transfer = _initial_model(request)
    result = run_training(model, request.profile, resume_path=request.resume_path)
    return {
        "stage": request.stage,
        "variant": request.variant_id,
        "best": str(result.best),
        "best_hash": sha256_file(result.best),
        "last": str(result.last),
        "last_hash": sha256_file(result.last),
        "save_dir": str(result.save_dir),
        "transfer": transfer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = TrainingWorkerRequest.from_dict(
        json.loads(args.request.read_text(encoding="utf-8"))
    )
    report = execute_request(request)
    atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
