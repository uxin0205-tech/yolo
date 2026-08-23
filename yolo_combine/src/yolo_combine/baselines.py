"""Independent Full35 baselines required before shared-trunk comparison."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.modules.head import Pose26
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils.torch_utils import unwrap_model

from .freezing import InheritedFreezeGuard, enforce_inherited_eval
from .source import SourceBundle, TrunkTransferReport, file_sha256


class MaterializedPoseTrainer(PoseTrainer):
    """Use an already grafted model instead of reconstructing an official YAML graph."""

    def __init__(self, *args: Any, source_model: PoseModel, **kwargs: Any) -> None:
        if not isinstance(source_model.model[-1], Pose26):
            raise TypeError("P0 source model must end in Pose26")
        self._materialized_source: PoseModel | None = source_model
        self._inherited_guard: InheritedFreezeGuard | None = None
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_end", self._verify_inherited_scope)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        del cfg, weights, verbose
        if self._materialized_source is None:
            raise RuntimeError("materialized P0 source model was already consumed")
        model = copy.deepcopy(self._materialized_source).float()
        self._materialized_source = None
        return model

    def build_optimizer(self, model, *args: Any, **kwargs: Any):
        self._inherited_guard = InheritedFreezeGuard.capture(unwrap_model(model))
        return super().build_optimizer(model, *args, **kwargs)

    def _model_train(self) -> None:
        super()._model_train()
        enforce_inherited_eval(unwrap_model(self.model))

    @staticmethod
    def _verify_inherited_scope(trainer: MaterializedPoseTrainer) -> None:
        if trainer._inherited_guard is None:
            raise AssertionError("inherited freeze guard was not initialized")
        trainer._inherited_guard.assert_unchanged(unwrap_model(trainer.model))

    @property
    def inherited_paths(self) -> tuple[str, ...]:
        return self._inherited_guard.paths if self._inherited_guard is not None else ()


@dataclass(frozen=True)
class PoseBaselineReport:
    run_dir: Path
    last_checkpoint: Path
    best_checkpoint: Path | None
    completed_epochs: int
    parameters: int
    transfer: TrunkTransferReport
    frozen_paths: tuple[str, ...]
    initial_checkpoint: Path | None
    initial_checkpoint_sha256: str | None


def train_pose_baseline(
    source: SourceBundle,
    *,
    data_yaml: str | Path,
    project: str | Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    seed: int,
    fraction: float = 1.0,
    val: bool = True,
    plots: bool = False,
    exist_ok: bool = False,
    overrides: dict[str, Any] | None = None,
    initial_checkpoint: str | Path | None = None,
) -> PoseBaselineReport:
    """Train P0 with official trainer mechanics and the accepted Full35 graph."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch < 1:
        raise ValueError("batch must be positive")
    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be a positive multiple of 32")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    initial_path = (
        Path(initial_checkpoint).expanduser().resolve()
        if initial_checkpoint is not None
        else None
    )
    template, transfer = source.build_pose_model()
    pose = source.load_pose_checkpoint(initial_path) if initial_path is not None else template
    if initial_path is not None:
        del template
    if not transfer.complete:
        raise RuntimeError(f"incomplete P0 trunk initialization: {transfer}")
    requested: dict[str, Any] = {
        "model": "yolo26m-pose.yaml",
        "data": str(Path(data_yaml).expanduser().resolve()),
        "project": str(Path(project).expanduser().resolve()),
        "name": name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "seed": seed,
        "fraction": fraction,
        "val": val,
        "plots": plots,
        "exist_ok": exist_ok,
        "pretrained": False,
        "save": True,
    }
    extra = dict(overrides or {})
    conflicts = sorted(requested.keys() & extra.keys())
    if conflicts:
        raise ValueError(f"P0 overrides cannot replace required fields: {conflicts}")
    requested.update(extra)
    trainer = MaterializedPoseTrainer(overrides=requested, source_model=pose)
    trainer.train()
    run_dir = Path(trainer.save_dir).resolve()
    last = Path(trainer.last).resolve()
    best = Path(trainer.best).resolve()
    if not last.is_file():
        raise FileNotFoundError(f"P0 trainer produced no last checkpoint: {last}")
    trained = trainer.model
    return PoseBaselineReport(
        run_dir=run_dir,
        last_checkpoint=last,
        best_checkpoint=best if best.is_file() else None,
        completed_epochs=int(trainer.epoch) + 1,
        parameters=sum(parameter.numel() for parameter in trained.parameters()),
        transfer=transfer,
        frozen_paths=trainer.inherited_paths,
        initial_checkpoint=initial_path,
        initial_checkpoint_sha256=file_sha256(initial_path) if initial_path is not None else None,
    )
