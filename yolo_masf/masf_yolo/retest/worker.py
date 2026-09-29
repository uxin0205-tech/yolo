"""Isolated worker entry point for one resumable retest stage."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..artifacts.io import atomic_write_json
from ..artifacts.checkpoints import save_canonical_checkpoint
from ..contracts import sha256_file
from ..training.runner import run_training
from .builder import build_retest_model
from .profiles import b0_fair_profile, b1r_a_profile, b1r_b_profile, control_full_profile, control_head_profile, direct_profile, retest_formal_profile, retest_smoke_profile


@dataclass(frozen=True, slots=True)
class RetestWorkerRequest:
    config_path: Path
    stage: str
    family: str
    variant: str | None
    source_weights: Path | None
    data_yaml: Path
    project: Path
    resume_path: Path | None = None
    parent_checkpoint: Path | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RetestWorkerRequest":
        expected = {"config_path", "stage", "family", "variant", "source_weights", "data_yaml", "project", "resume_path", "parent_checkpoint"}
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(f"unknown retest worker keys: {sorted(unknown)}")
        missing = (expected - {"parent_checkpoint"}) - set(raw)
        if missing:
            raise ValueError(f"missing retest worker keys: {sorted(missing)}")
        return cls(
            config_path=Path(str(raw["config_path"])), stage=str(raw["stage"]), family=str(raw["family"]),
            variant=str(raw["variant"]) if raw["variant"] is not None else None,
            source_weights=Path(str(raw["source_weights"])) if raw["source_weights"] is not None else None,
            data_yaml=Path(str(raw["data_yaml"])), project=Path(str(raw["project"])),
            resume_path=Path(str(raw["resume_path"])) if raw["resume_path"] is not None else None,
            parent_checkpoint=Path(str(raw["parent_checkpoint"])) if raw.get("parent_checkpoint") is not None else None,
        )


def profile_for(request: RetestWorkerRequest, model_path: str) -> dict[str, Any]:
    if request.stage == "b0_fair":
        return b0_fair_profile(model_path, str(request.project))
    if request.stage == "b1r_a":
        return b1r_a_profile(model_path, str(request.project))
    if request.stage == "b1r_b":
        return b1r_b_profile(model_path, str(request.project))
    if request.stage == "control_head":
        return control_head_profile(model_path, str(request.project))
    if request.stage == "control_full":
        return control_full_profile(model_path, str(request.project))
    if request.stage == "direct":
        return direct_profile(model_path, str(request.project))
    if request.variant is None:
        raise ValueError("MFAM smoke/formal stages require a variant")
    if request.stage == "smoke":
        return retest_smoke_profile(request.variant, model_path, str(request.project))
    if request.stage == "formal":
        return retest_formal_profile(request.variant, model_path, str(request.project))
    raise ValueError(f"unsupported retest stage: {request.stage}")


def build_request_model(request: RetestWorkerRequest):
    if request.stage == "b1r_b" and request.resume_path is not None:
        from ultralytics import YOLO

        model = YOLO(str(request.resume_path), task="detect").model
        model.masf_variant = "B1R"
        model.masf_variant_hash = "b1r-per-scale-detect-v1"
        return model
    return build_retest_model(request.family, request.variant, source_weights=request.source_weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = RetestWorkerRequest.from_dict(json.loads(args.request.read_text(encoding="utf-8")))
    model = build_request_model(request)
    if request.parent_checkpoint is not None:
        from ultralytics import YOLO

        parent_state = YOLO(str(request.parent_checkpoint), task="detect").model.state_dict()
        # MFAM is intentionally new for every ablation; all other compatible
        # B1R/B0 parent tensors are inherited explicitly.
        if request.stage == "control_full":
            inherited = parent_state
        else:
            slot_prefix = "model.20." if request.family == "P2" else "model.16.1."
            inherited = {key: value for key, value in parent_state.items() if not key.startswith(slot_prefix)}
        model.load_state_dict(inherited, strict=False)
    profile = profile_for(request, "retest-in-memory")
    profile["data"] = str(request.data_yaml)
    # B1R-B is a fresh 90-epoch full-unfreeze stage initialized from A-best;
    # it must not inherit the trainer's previous epoch counter.
    result = run_training(model, profile, resume_path=None if request.stage in {"b1r_b", "control_full"} else request.resume_path)
    report = {"stage": request.stage, "family": request.family, "variant": request.variant, "best": str(result.best), "last": str(result.last), "best_hash": sha256_file(result.best), "last_hash": sha256_file(result.last), "save_dir": str(result.save_dir)}
    atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
