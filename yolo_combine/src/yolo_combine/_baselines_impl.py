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

    def __init__(
        self,
        *args: Any,
        source_model: PoseModel,
        physical_train_batch_size: int,
        validation_batch_size: int,
        **kwargs: Any,
    ) -> None:
        if not isinstance(source_model.model[-1], Pose26):
            raise TypeError("P0 source model must end in Pose26")
        if validation_batch_size < 1:
            raise ValueError("validation_batch_size must be positive")
        if physical_train_batch_size < 1:
            raise ValueError("physical_train_batch_size must be positive")
        self._materialized_source: PoseModel | None = source_model
        self.physical_train_batch_size = int(physical_train_batch_size)
        self.validation_batch_size = int(validation_batch_size)
        self.resume_weights_loaded = False
        self._inherited_guard: InheritedFreezeGuard | None = None
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_end", self._verify_inherited_scope)
        self.add_callback(
            "on_train_batch_start",
            self._enforce_physical_batch_contract,
        )

    @staticmethod
    def _enforce_physical_batch_contract(
        trainer: "MaterializedPoseTrainer",
    ) -> None:
        """Fail on OOM instead of accepting Ultralytics' first-epoch halving.

        Ultralytics 8.4.90 retries first-epoch CUDA memory failures by mutating
        ``batch_size`` and rebuilding the optimizer/dataloaders. That silently
        changes the formal experiment. Setting the retry counter to its limit
        makes the original exception propagate through the pinned upstream
        handler, while this explicit comparison catches any prior mutation.
        """

        if trainer.batch_size != trainer.physical_train_batch_size:
            raise RuntimeError(
                "physical train batch changed during formal Pose training: "
                f"expected={trainer.physical_train_batch_size}, "
                f"actual={trainer.batch_size}"
            )
        trainer._oom_retries = max(int(trainer._oom_retries), 3)

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        """Keep the physical train batch while independently capping validation.

        Ultralytics 8.4.90 normally builds Pose validation at twice the train
        batch. Full35 attention makes that unsafe for a physical train batch
        of 128, so the explicit validation contract takes precedence here.
        """

        effective_batch = self.validation_batch_size if mode == "val" else batch_size
        return super().get_dataloader(
            dataset_path,
            batch_size=effective_batch,
            rank=rank,
            mode=mode,
        )

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        del cfg, verbose
        if self._materialized_source is None:
            raise RuntimeError("materialized P0 source model was already consumed")
        model = copy.deepcopy(self._materialized_source).float()
        self._materialized_source = None
        if weights is not None:
            if not isinstance(weights, PoseModel):
                raise TypeError(
                    "resume checkpoint must contain a PoseModel EMA, got "
                    f"{type(weights).__name__}"
                )
            expected = model.state_dict()
            received = weights.state_dict()
            missing = sorted(expected.keys() - received.keys())
            unexpected = sorted(received.keys() - expected.keys())
            shape_mismatches = {
                name: (tuple(expected[name].shape), tuple(received[name].shape))
                for name in expected.keys() & received.keys()
                if expected[name].shape != received[name].shape
            }
            if missing or unexpected or shape_mismatches:
                raise RuntimeError(
                    "resume Pose graph/state mismatch: "
                    f"missing={missing[:20]}, unexpected={unexpected[:20]}, "
                    f"shape_mismatches={dict(list(shape_mismatches.items())[:20])}"
                )
            model.load_state_dict(received, strict=True)
            self.resume_weights_loaded = True
        return model

    def build_optimizer(self, model, *args: Any, **kwargs: Any):
        self._inherited_guard = InheritedFreezeGuard.capture(unwrap_model(model))
        return super().build_optimizer(model, *args, **kwargs)

    def _model_train(self) -> None:
        super()._model_train()
        enforce_inherited_eval(unwrap_model(self.model))

    @staticmethod
    def _verify_inherited_scope(trainer: "MaterializedPoseTrainer") -> None:
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
    resume_checkpoint: Path | None
    resume_checkpoint_sha256: str | None
    validation_batch_size: int
    resume_weights_loaded: bool


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
    resume_checkpoint: str | Path | None = None,
    validation_batch_size: int = 16,
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
    if validation_batch_size < 1:
        raise ValueError("validation_batch_size must be positive")
    if initial_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("initial_checkpoint and resume_checkpoint are mutually exclusive")
    initial_path = (
        Path(initial_checkpoint).expanduser().resolve()
        if initial_checkpoint is not None
        else None
    )
    resume_path = (
        Path(resume_checkpoint).expanduser().resolve()
        if resume_checkpoint is not None
        else None
    )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    # last.pt is overwritten in place as training continues, so capture the
    # exact input checkpoint identity before constructing the trainer.
    resume_sha256 = file_sha256(resume_path) if resume_path is not None else None
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
    if resume_path is not None:
        requested["resume"] = str(resume_path)
    extra = dict(overrides or {})
    conflicts = sorted(requested.keys() & extra.keys())
    if conflicts:
        raise ValueError(f"P0 overrides cannot replace required fields: {conflicts}")
    requested.update(extra)
    trainer = MaterializedPoseTrainer(
        overrides=requested,
        source_model=pose,
        physical_train_batch_size=batch,
        validation_batch_size=validation_batch_size,
    )
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
        resume_checkpoint=resume_path,
        resume_checkpoint_sha256=resume_sha256,
        validation_batch_size=trainer.validation_batch_size,
        resume_weights_loaded=trainer.resume_weights_loaded,
    )
