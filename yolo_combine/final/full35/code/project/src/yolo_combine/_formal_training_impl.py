"""Executable single-GPU Full35/Partial75 formal joint-training session."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from ultralytics.utils.torch_utils import ModelEMA

from .contracts import Task
from .data import prepare_bbt5_view
from .early_stop import StageEarlyStopping
from .experiment_log import ExperimentLogger
from .factory import FusionModelFactory
from .hardware_contract import HardwareContractGuard
from .joint_config import JointExperimentConfig
from .joint_data import TaskLoaderSettings, build_task_loader
from .joint_loss import MacroStepEngine, NativeTaskLossRouter
from .joint_trainer import (
    JointEpochRunner,
    PoseEpochRunner,
    StageWarmupCosineScheduler,
)
from .metrics import AccuracyGate, CheckpointSelectors
from .resume import (
    TrainingProgress,
    load_training_snapshot,
    save_inference_weights,
    save_training_snapshot,
)
from .source import SourceBundle, file_sha256
from .stage_policy import (
    JOINT_STAGES,
    JointStage,
    apply_stage,
    build_joint_optimizer,
    update_optimizer_stage,
)
from .validation import JointValidator, ValidationSettings
from .xnor import XNORExecutionConfig


@dataclass(frozen=True)
class FormalRunReport:
    run_dir: Path
    completed_stages: tuple[str, ...]
    epochs_completed: int
    global_macro_steps: int
    best_state: dict[str, Any]
    checkpoint_paths: dict[str, Path]
    plots: tuple[Path, ...]


class _AutocastTaskLossRouter:
    """Add AMP around native criteria without changing criterion ownership."""

    def __init__(
        self,
        router: NativeTaskLossRouter,
        *,
        device: torch.device,
        enabled: bool,
    ) -> None:
        self.router = router
        self.device = device
        self.enabled = bool(enabled and device.type == "cuda")

    def loss_for(self, task: Task | str, batch: Mapping[str, Any]):
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
            enabled=self.enabled,
        ):
            return self.router.loss_for(task, batch)

    def advance_epoch(self, tasks=None) -> None:
        self.router.advance_epoch(tasks)

    def state_dict(self):
        return self.router.state_dict()

    def load_state_dict(self, state):
        self.router.load_state_dict(state)


def seed_everything(seed: int) -> None:
    """Lock all process-local RNG sources used by the single-GPU recipe."""

    if seed < 0:
        raise ValueError("seed cannot be negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reseed_loader_for_epoch(loader: Any, *, seed: int, epoch: int, offset: int) -> int:
    """Reset persistent workers from an epoch-derived seed for resume stability."""

    if min(seed, epoch, offset) < 0:
        raise ValueError("loader seed inputs cannot be negative")
    resolved = 6148914691236517205 + seed * 1000003 + epoch * 1009 + offset
    generator = getattr(loader, "generator", None)
    if not isinstance(generator, torch.Generator):
        raise TypeError("formal loader exposes no torch.Generator")
    generator.manual_seed(resolved)
    sampler = getattr(loader, "sampler", None)
    sampler_generator = getattr(sampler, "generator", None)
    if isinstance(sampler_generator, torch.Generator) and sampler_generator is not generator:
        sampler_generator.manual_seed(resolved)
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)
    reset = getattr(loader, "reset", None)
    if not callable(reset):
        raise TypeError("formal loader exposes no reset()")
    reset()
    return resolved


def _gradient_groups(model: nn.Module) -> dict[str, tuple[nn.Parameter, ...]]:
    groups: dict[str, list[nn.Parameter]] = {
        "shared": [],
        "detect_head": [],
        "pose_head": [],
    }
    for name, parameter in model.named_parameters():
        if ".detect_head." in name:
            groups["detect_head"].append(parameter)
        elif ".pose_head." in name:
            groups["pose_head"].append(parameter)
        else:
            groups["shared"].append(parameter)
    if any(not values for values in groups.values()):
        raise ValueError(
            f"gradient ownership group is empty: "
            f"{ {name: len(values) for name, values in groups.items()} }"
        )
    return {name: tuple(values) for name, values in groups.items()}


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not value.isdigit():
        raise ValueError("v1 device must be a single CUDA index or cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    index = int(value)
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {index} does not exist")
    return torch.device(f"cuda:{index}")


class FormalJointTrainingSession:
    """Build, train, validate, gate, log, and checkpoint one formal run."""

    def __init__(
        self,
        config: JointExperimentConfig,
        *,
        device: str = "0",
        run_name: str | None = None,
        detect_microbatch_size: int | None = None,
    ) -> None:
        self.config = config
        self.device = _device(device)
        self.run_name = run_name or f"{config.architecture}-joint-seed{config.seed}"
        if "/" in self.run_name or "\\" in self.run_name:
            raise ValueError("run_name must be one path component")
        self.run_dir = (config.run_root / self.run_name).resolve()
        requested_microbatch = (
            config.detect_microbatch_size
            if detect_microbatch_size is None
            else int(detect_microbatch_size)
        )
        if (
            requested_microbatch < 1
            or config.detect_batch_size % requested_microbatch
        ):
            raise ValueError(
                "detect microbatch must divide logical Detect batch "
                f"{config.detect_batch_size}, got {requested_microbatch}"
            )
        self.detect_microbatch_size = requested_microbatch
        self.detect_microbatches_per_logical_batch = (
            config.detect_batch_size // requested_microbatch
        )
        self.detect_microbatches_per_macro = (
            config.detect_batches_per_macro
            * self.detect_microbatches_per_logical_batch
        )
        self._detect_microbatch_overridden = (
            requested_microbatch != config.detect_microbatch_size
        )

    @property
    def runtime_batch_plan(self) -> dict[str, int]:
        """Return the actual Detect accumulation plan used by this process."""

        return {
            "detect_train_logical": self.config.detect_batch_size,
            "detect_train_physical_microbatch": self.detect_microbatch_size,
            "detect_logical_batches_per_macro": self.config.detect_batches_per_macro,
            "detect_physical_microbatches_per_macro": (
                self.detect_microbatches_per_macro
            ),
            "detect_images_per_full_macro": (
                self.config.detect_batch_size
                * self.config.detect_batches_per_macro
            ),
        }

    def _resolved_config(self) -> dict[str, Any]:
        """Include a stage-transition runtime batch override in new snapshots."""

        resolved = self.config.as_dict()
        if self._detect_microbatch_overridden:
            resolved["runtime_overrides"] = dict(self.runtime_batch_plan)
        return resolved

    def _loaders(self, model):
        pose_view = prepare_bbt5_view(
            self.config.registry,
            self.run_dir / "datasets" / "bbat5-v1-runtime",
        )
        detect = build_task_loader(
            model,
            data_yaml=self.config.detect_data,
            settings=TaskLoaderSettings.for_detect(
                batch_size=self.detect_microbatch_size,
                workers=self.config.detect_workers,
                imgsz=self.config.imgsz,
                seed=self.config.seed,
            ),
            device=self.device,
            registry=self.config.registry,
        )
        pose = build_task_loader(
            model,
            data_yaml=pose_view.yaml,
            settings=TaskLoaderSettings.for_pose(
                batch_size=self.config.pose_batch_size,
                workers=self.config.pose_workers,
                imgsz=self.config.imgsz,
                seed=self.config.seed,
            ),
            device=self.device,
            registry=self.config.registry,
        )
        return detect, pose, pose_view

    def _save_selected(
        self,
        labels: tuple[str, ...],
        *,
        model,
        ema,
        optimizer,
        scheduler,
        scaler,
        criteria,
        progress: TrainingProgress,
        provenance: Mapping[str, Any],
        loader_state: Mapping[str, Any],
        selectors: CheckpointSelectors,
        metrics: Mapping[str, float],
    ) -> dict[str, Path]:
        outputs: dict[str, Path] = {}
        for label in labels:
            checkpoint = self.run_dir / "checkpoints" / f"{label}.pt"
            saved = save_training_snapshot(
                checkpoint,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                criteria=criteria,
                progress=progress,
                resolved_config=self._resolved_config(),
                provenance=provenance,
                loader_state=loader_state,
                best_state=selectors.state_dict(),
            )
            outputs[label] = saved.path
            save_inference_weights(
                self.run_dir / "inference" / f"{label}.pt",
                model=model,
                ema=ema,
                use_ema=True,
                metadata={
                    "stage": progress.stage,
                    "epoch": progress.next_epoch - 1,
                    "global_macro_step": progress.global_macro_step,
                    "metrics": dict(metrics),
                    "full_resume_sha256": saved.sha256,
                },
            )
        return outputs

    def run(
        self,
        *,
        resume: str | Path | None = None,
        enable_j3: bool | None = None,
    ) -> FormalRunReport:
        preflight = self.config.preflight()
        if not preflight.ready or preflight.baseline is None:
            raise RuntimeError(
                "formal joint training preflight failed:\n- "
                + "\n- ".join(preflight.blockers)
            )
        if self.run_dir.exists() and resume is None:
            raise FileExistsError(
                f"run directory already exists; choose another --name: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(self.config.seed)
        source = SourceBundle(
            self.config.source_bundle,
            architecture=self.config.architecture,
        )
        factory = FusionModelFactory(
            source,
            detect_data_yaml=self.config.detect_data,
            pose_data_yaml=self.config.pose_data,
            xnor=XNORExecutionConfig(
                backend="tiled_exact",
                token_tile=self.config.xnor_token_tile,
            ),
        )
        built = factory.build(
            pose_head_checkpoint=self.config.pose_checkpoint,
            checkpoint_kind="float",
        )
        model = built.model.to(self.device)
        stages = list(self.config.stages)
        should_j3 = self.config.enable_j3 if enable_j3 is None else enable_j3
        if should_j3:
            stages.append("j3")
        total_epochs = sum(JOINT_STAGES[name].epochs for name in stages)
        detect_epochs = sum(
            JOINT_STAGES[name].epochs
            for name in stages
            if JOINT_STAGES[name].task_mode == "joint"
        )
        detect_loader, pose_loader, pose_view = self._loaders(model)
        joint_macros_per_epoch = math.ceil(
            len(detect_loader.loader) / self.detect_microbatches_per_macro
        )
        pose_steps_per_epoch = len(pose_loader.loader)
        if joint_macros_per_epoch < 1:
            raise RuntimeError("Detect loader has no macro-steps")
        if pose_steps_per_epoch < 1:
            raise RuntimeError("Pose loader has no macro-steps")

        resume_payload: dict[str, Any] | None = None
        resume_stage: str | None = None
        if resume is not None:
            loaded = torch.load(
                Path(resume).expanduser().resolve(),
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(loaded, dict) or not isinstance(loaded.get("progress"), dict):
                raise ValueError("resume checkpoint has no training progress")
            resume_payload = loaded
            resume_stage = str(loaded["progress"].get("stage"))
            if resume_stage not in stages:
                raise ValueError(f"resume stage {resume_stage!r} is not enabled")
            if self._detect_microbatch_overridden:
                prior_loader_state = loaded.get("loader_state")
                if not isinstance(prior_loader_state, Mapping):
                    raise ValueError(
                        "runtime Detect microbatch override requires loader state"
                    )
                if resume_stage == "j2":
                    if not bool(prior_loader_state.get("stage_complete")):
                        raise ValueError(
                            "J3 Detect microbatch override requires completed J2"
                        )
                elif resume_stage == "j3":
                    if prior_loader_state.get("runtime_batch_plan") != (
                        self.runtime_batch_plan
                    ):
                        raise ValueError(
                            "J3 resume runtime batch plan differs from checkpoint"
                        )
                else:
                    raise ValueError(
                        "Detect microbatch override is restricted to J3 transition/resume"
                    )
            first_stage_index = stages.index(resume_stage)
        else:
            first_stage_index = 0

        stage = JOINT_STAGES[stages[first_stage_index]]
        optimizer, optimizer_report = build_joint_optimizer(
            model,
            stage,
            optimizer_name=self.config.optimizer,
            weight_decay=self.config.weight_decay,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
        )
        router = NativeTaskLossRouter(
            model,
            epochs=total_epochs,
            imgsz=self.config.imgsz,
            detect_overrides={"epochs": detect_epochs},
            pose_overrides={"epochs": total_epochs},
        )
        amp_router = _AutocastTaskLossRouter(
            router,
            device=self.device,
            enabled=self.config.amp,
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.config.amp and self.device.type == "cuda",
        )
        ema = ModelEMA(model, decay=0.9999, tau=2000)
        gate = AccuracyGate(
            preflight.baseline,
            maximum_drop=self.config.maximum_map_drop,
        )
        selectors = CheckpointSelectors()
        validator = JointValidator(
            source,
            detect_data_yaml=self.config.detect_data,
            pose_data_yaml=pose_view.yaml,
            output_root=self.run_dir / "validation",
            settings=ValidationSettings(
                imgsz=self.config.imgsz,
                detect_batch_size=self.config.detect_val_batch_size,
                pose_batch_size=self.config.pose_val_batch_size,
                detect_workers=self.config.detect_workers,
                pose_workers=self.config.pose_workers,
                device=str(self.device),
                plots=self.config.validation_plots,
                save_coco_json=self.config.save_coco_json,
            ),
        )
        provenance = {
            "factory": built.report.as_dict(),
            "config_sha256": file_sha256(self.config.path),
            "pose_view": asdict(pose_view),
            "optimizer": asdict(optimizer_report),
            "runtime_batch_plan": dict(self.runtime_batch_plan),
        }
        (self.run_dir / "resolved-config.json").write_text(
            json.dumps(self._resolved_config(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.run_dir / "factory-report.json").write_text(
            json.dumps(built.report.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        global_macro = 0
        joint_epochs = 0
        local_start = 0
        active_scheduler = StageWarmupCosineScheduler(
            optimizer,
            stage=stage.name,
            epochs=stage.epochs,
            steps_per_epoch=(
                pose_steps_per_epoch
                if stage.task_mode == "pose"
                else joint_macros_per_epoch
            ),
            warmup_epochs=stage.warmup_epochs,
            warmup_start_factor=self.config.warmup_start_factor,
            final_lr_factor=self.config.cosine_final_lr_factor,
            plateau_policy=(
                self.config.j2_plateau_policy if stage.name == "j2" else None
            ),
        )
        restored_loader_state: dict[str, Any] = {}
        if resume_payload is not None:
            restored = load_training_snapshot(
                resume,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=active_scheduler,
                scaler=scaler,
                criteria=amp_router,
                restore_rng=True,
            )
            restored_loader_state = restored.loader_state
            expected_resolved_config = self._resolved_config()
            transition_from_completed_j2 = bool(
                self._detect_microbatch_overridden
                and resume_stage == "j2"
                and restored_loader_state.get("stage_complete")
                and restored.resolved_config == self.config.as_dict()
            )
            if (
                restored.resolved_config != expected_resolved_config
                and not transition_from_completed_j2
            ):
                raise ValueError("resume config differs from current resolved config")
            local_start = restored.progress.next_epoch
            global_macro = restored.progress.global_macro_step
            joint_epochs = restored.progress.joint_epochs_completed
            selectors.load_state_dict(restored.best_state)
            if bool(restored_loader_state.get("stage_complete")):
                local_start = stage.epochs

        guard = HardwareContractGuard.capture(model)
        checkpoint_paths: dict[str, Path] = {}
        completed: list[str] = []
        with ExperimentLogger(
            self.run_dir / "logs",
            tensorboard=self.config.tensorboard,  # type: ignore[arg-type]
        ) as logger:
            for stage_index in range(first_stage_index, len(stages)):
                stage = JOINT_STAGES[stages[stage_index]]
                if stage_index != first_stage_index:
                    update_optimizer_stage(optimizer, model, stage)
                    active_scheduler = StageWarmupCosineScheduler(
                        optimizer,
                        stage=stage.name,
                        epochs=stage.epochs,
                        steps_per_epoch=(
                            pose_steps_per_epoch
                            if stage.task_mode == "pose"
                            else joint_macros_per_epoch
                        ),
                        warmup_epochs=stage.warmup_epochs,
                        warmup_start_factor=self.config.warmup_start_factor,
                        final_lr_factor=self.config.cosine_final_lr_factor,
                        plateau_policy=(
                            self.config.j2_plateau_policy if stage.name == "j2" else None
                        ),
                    )
                    local_start = 0
                engine = MacroStepEngine(
                    model=model,
                    losses=amp_router,
                    optimizer=optimizer,
                    reference_batch_size=self.config.reference_batch_size,
                    task_weights={
                        Task.DETECT: self.config.detect_weight,
                        Task.POSE: self.config.pose_weight,
                    },
                    gradient_groups=_gradient_groups(model),
                    scaler=scaler,
                    ema=ema,
                    max_grad_norm=self.config.gradient_clip_norm,
                    max_amp_retries=self.config.amp_max_overflow_retries,
                    preprocess=lambda task, batch: (
                        detect_loader.preprocess(batch)
                        if task is Task.DETECT
                        else pose_loader.preprocess(batch)
                    ),
                )
                if stage.task_mode == "pose":
                    runner = PoseEpochRunner(
                        engine=engine,
                        pose_loader=pose_loader.loader,
                        scheduler=active_scheduler,
                        logger=logger,
                        apply_training_mode=lambda stage=stage: apply_stage(model, stage),
                        assert_hardware_contract=lambda: guard.assert_unchanged(model),
                    )
                else:
                    runner = JointEpochRunner(
                        engine=engine,
                        detect_loader=detect_loader.loader,
                        pose_loader=pose_loader.loader,
                        scheduler=active_scheduler,
                        logger=logger,
                        apply_training_mode=lambda stage=stage: apply_stage(model, stage),
                        assert_hardware_contract=lambda: guard.assert_unchanged(model),
                        detect_batches_per_macro=self.detect_microbatches_per_macro,
                        gradient_statistics_interval=self.config.gradient_statistics_interval,
                    )
                early_stopper = (
                    StageEarlyStopping(
                        stage=stage.name,
                        patience=stage.patience,
                    )
                    if stage.name in {"j1", "j3"} and stage.patience
                    else None
                )
                if (
                    stage_index == first_stage_index
                    and early_stopper is not None
                    and isinstance(restored_loader_state.get("early_stop"), Mapping)
                ):
                    early_stopper.load_state_dict(
                        restored_loader_state["early_stop"]
                    )
                for local_epoch in range(local_start, stage.epochs):
                    detect_seed = None
                    if stage.task_mode == "joint":
                        detect_seed = reseed_loader_for_epoch(
                            detect_loader.loader,
                            seed=self.config.seed,
                            epoch=joint_epochs,
                            offset=0,
                        )
                    pose_seed = reseed_loader_for_epoch(
                        pose_loader.loader,
                        seed=self.config.seed,
                        epoch=joint_epochs,
                        offset=1,
                    )
                    training_report = runner.run_epoch(
                        epoch=joint_epochs,
                        global_macro_step=global_macro,
                        stage=stage.name,
                    )
                    global_macro = training_report.next_global_macro_step
                    joint_epochs += 1
                    validations = validator.validate_backends(
                        ema.ema,
                        epoch=joint_epochs - 1,
                        kinds=self.config.validation_backends,  # type: ignore[arg-type]
                    )
                    for backend, result in validations.items():
                        logger.log(
                            "validation",
                            step=joint_epochs - 1,
                            values=result.metrics,
                            context={"stage": stage.name, "backend": backend},
                        )
                    selected_metrics = validations[self.config.selection_backend].metrics
                    gate_report = gate.evaluate(selected_metrics)
                    selection = selectors.observe(
                        epoch=joint_epochs - 1,
                        metrics=selected_metrics,
                        gate=gate_report,
                    )
                    logger.log(
                        "gate",
                        step=joint_epochs - 1,
                        values={
                            "passed": int(gate_report.passed),
                            **{f"delta/{name}": value for name, value in gate_report.deltas.items()},
                            **{f"score/{name}": value for name, value in selection.scores.items()},
                        },
                        context={
                            "stage": stage.name,
                            "failed_metrics": list(gate_report.failed_metrics),
                            "selected": list(selection.selected),
                        },
                    )
                    plateau_decision = None
                    if stage.name == "j2":
                        plateau_decision = active_scheduler.observe_metric(
                            selection.scores["best_joint"]
                        )
                        logger.log(
                            "plateau",
                            step=joint_epochs - 1,
                            values={
                                "score": plateau_decision.score,
                                "best_score": plateau_decision.best_score,
                                "improved": int(plateau_decision.improved),
                                "stale_epochs": plateau_decision.stale_epochs,
                                "recovery_applied": int(
                                    plateau_decision.recovery_applied
                                ),
                                "reductions": plateau_decision.reductions,
                                "lr_multiplier": plateau_decision.lr_multiplier,
                                "should_stop": int(plateau_decision.should_stop),
                            },
                            context={
                                "stage": stage.name,
                                "monitor": self.config.j2_plateau_policy.monitor,
                                "action": (
                                    "lr_x0.5"
                                    if plateau_decision.recovery_applied
                                    else "none"
                                ),
                            },
                        )
                    early_decision = (
                        early_stopper.observe(selection.scores["best_joint"])
                        if early_stopper is not None
                        else None
                    )
                    if early_decision is not None:
                        logger.log(
                            "early_stop",
                            step=joint_epochs - 1,
                            values={
                                "score": early_decision.score,
                                "best_score": early_decision.best_score,
                                "improved": int(early_decision.improved),
                                "stale_epochs": early_decision.stale_epochs,
                                "patience": stage.patience,
                                "should_stop": int(early_decision.should_stop),
                            },
                            context={
                                "stage": stage.name,
                                "monitor": "joint_score",
                            },
                        )
                    should_stop = bool(
                        (
                            plateau_decision is not None
                            and plateau_decision.should_stop
                        )
                        or (
                            early_decision is not None
                            and early_decision.should_stop
                        )
                    )
                    stage_complete = bool(
                        local_epoch + 1 >= stage.epochs or should_stop
                    )
                    progress = TrainingProgress(
                        stage=stage.name,
                        next_epoch=local_epoch + 1,
                        global_macro_step=global_macro,
                        joint_epochs_completed=joint_epochs,
                    )
                    loader_state = {
                        "snapshot_boundary": "formal_epoch_end",
                        "task_mode": stage.task_mode,
                        "stage_complete": stage_complete,
                        "detect_epoch_seed": detect_seed,
                        "pose_epoch_seed": pose_seed,
                        "early_stop": (
                            early_stopper.state_dict()
                            if early_stopper is not None
                            else None
                        ),
                        "runtime_batch_plan": dict(self.runtime_batch_plan),
                        **asdict(training_report),
                    }
                    checkpoint_paths.update(
                        self._save_selected(
                            selection.selected,
                            model=model,
                            ema=ema,
                            optimizer=optimizer,
                            scheduler=active_scheduler,
                            scaler=scaler,
                            criteria=amp_router,
                            progress=progress,
                            provenance=provenance,
                            loader_state=loader_state,
                            selectors=selectors,
                            metrics=selected_metrics,
                        )
                    )
                    if should_stop:
                        del validations
                        break
                    del validations
                completed.append(stage.name)
                local_start = 0
                restored_loader_state = {}
            plots: list[Path] = []
            for kind in (
                "macro",
                "epoch",
                "validation",
                "gate",
                "plateau",
                "early_stop",
            ):
                try:
                    plots.append(logger.plot(kind))
                except ValueError:
                    continue
        return FormalRunReport(
            run_dir=self.run_dir,
            completed_stages=tuple(completed),
            epochs_completed=joint_epochs,
            global_macro_steps=global_macro,
            best_state=selectors.state_dict(),
            checkpoint_paths=checkpoint_paths,
            plots=tuple(plots),
        )
