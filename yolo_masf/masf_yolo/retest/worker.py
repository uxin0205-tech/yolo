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
from .profiles import b1r_a_profile, b1r_b_profile, direct_profile, retest_formal_profile, retest_smoke_profile


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

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RetestWorkerRequest":
        expected = {"config_path", "stage", "family", "variant", "source_weights", "data_yaml", "project", "resume_path"}
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(f"unknown retest worker keys: {sorted(unknown)}")
        missing = expected - set(raw)
        if missing:
            raise ValueError(f"missing retest worker keys: {sorted(missing)}")
        return cls(
            config_path=Path(str(raw["config_path"])), stage=str(raw["stage"]), family=str(raw["family"]),
            variant=str(raw["variant"]) if raw["variant"] is not None else None,
            source_weights=Path(str(raw["source_weights"])) if raw["source_weights"] is not None else None,
            data_yaml=Path(str(raw["data_yaml"])), project=Path(str(raw["project"])),
            resume_path=Path(str(raw["resume_path"])) if raw["resume_path"] is not None else None,
        )


def profile_for(request: RetestWorkerRequest, model_path: str) -> dict[str, Any]:
    if request.stage == "b1r_a":
        return b1r_a_profile(model_path, str(request.project))
    if request.stage == "b1r_b":
        return b1r_b_profile(model_path, str(request.project))
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
    return build_retest_model(request.family, request.variant, source_weights=request.source_weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = RetestWorkerRequest.from_dict(json.loads(args.request.read_text(encoding="utf-8")))
    model = build_request_model(request)
    profile = profile_for(request, "retest-in-memory")
    profile["data"] = str(request.data_yaml)
    result = run_training(model, profile, resume_path=request.resume_path)
    report = {"stage": request.stage, "family": request.family, "variant": request.variant, "best": str(result.best), "last": str(result.last), "best_hash": sha256_file(result.best), "last_hash": sha256_file(result.last), "save_dir": str(result.save_dir)}
    atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
