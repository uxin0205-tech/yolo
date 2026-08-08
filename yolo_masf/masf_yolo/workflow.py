"""The sole dependency DAG and hash-gated stage orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from masf_yolo.artifacts.io import atomic_write_json
from masf_yolo.contracts import PipelineState


@dataclass(frozen=True, slots=True)
class StageDefinition:
    name: str
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StageResult:
    output_hashes: dict[str, str]


_STAGE_NAMES = (
    "audit",
    "verify",
    "preflight",
    "batch_probe",
    "b1_a",
    "b1_b",
    "smoke_m0",
    "smoke_m1",
    "smoke_m2",
    "smoke_m3",
    "formal_m0",
    "formal_m1",
    "formal_m2",
    "formal_m3",
    "val_all",
    "selection",
    "test_all",
    "profile_all",
    "final_audit",
    "report",
)

PHASE1_STAGES = tuple(
    StageDefinition(name, () if index == 0 else (_STAGE_NAMES[index - 1],))
    for index, name in enumerate(_STAGE_NAMES)
)
_STAGES = {stage.name: stage for stage in PHASE1_STAGES}


class PipelineWorkflow:
    def __init__(
        self,
        artifact_root: Path,
        *,
        pipeline_id: str,
        common_input_hashes: dict[str, str],
    ) -> None:
        self.artifact_root = artifact_root
        self.pipeline_id = pipeline_id
        self.common_input_hashes = dict(common_input_hashes)
        self.stage_root = artifact_root / "stages"

    def _stage_path(self, name: str) -> Path:
        return self.stage_root / f"{name}.json"

    def _load(self, name: str) -> PipelineState | None:
        path = self._stage_path(name)
        if not path.is_file():
            return None
        return PipelineState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _inputs(self, stage: StageDefinition) -> dict[str, str]:
        inputs = dict(self.common_input_hashes)
        for dependency in stage.dependencies:
            state = self._load(dependency)
            if state is None or state.status != "completed":
                raise RuntimeError(f"stage {stage.name} dependency is incomplete: {dependency}")
            for key, value in state.output_hashes.items():
                inputs[f"{dependency}:{key}"] = value
        return inputs

    def run_stage(self, name: str, action: Callable[[], StageResult]) -> StageResult:
        if name not in _STAGES:
            raise ValueError(f"unknown Phase 1 stage: {name}")
        stage = _STAGES[name]
        inputs = self._inputs(stage)
        if name == "test_all" and not (self.artifact_root / "selection.json").is_file():
            raise RuntimeError("test stage requires frozen selection.json")
        existing = self._load(name)
        if existing is not None and existing.status == "completed" and existing.input_hashes == inputs:
            return StageResult(existing.output_hashes)
        attempt = 1 if existing is None else existing.attempt + 1
        running = PipelineState(
            pipeline_id=self.pipeline_id,
            stage=name,
            status="running",
            attempt=attempt,
            epoch=None,
            input_hashes=inputs,
            output_hashes={},
        )
        self.stage_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._stage_path(name), running.to_dict())
        atomic_write_json(self.artifact_root / "state.json", running.to_dict())
        try:
            result = action()
        except BaseException as error:
            failed = PipelineState(
                pipeline_id=self.pipeline_id,
                stage=name,
                status="failed",
                attempt=attempt,
                epoch=None,
                input_hashes=inputs,
                output_hashes={},
                error=f"{type(error).__name__}: {error}",
            )
            atomic_write_json(self._stage_path(name), failed.to_dict())
            atomic_write_json(self.artifact_root / "state.json", failed.to_dict())
            raise
        completed = PipelineState(
            pipeline_id=self.pipeline_id,
            stage=name,
            status="completed",
            attempt=attempt,
            epoch=None,
            input_hashes=inputs,
            output_hashes=result.output_hashes,
        )
        atomic_write_json(self._stage_path(name), completed.to_dict())
        atomic_write_json(self.artifact_root / "state.json", completed.to_dict())
        return result
