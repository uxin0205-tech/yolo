"""Ultralytics adapters for fair Detect/Pose training with Bit-True ranking."""

from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.utils.torch_utils import unwrap_model

from .config import (
    CANDIDATES,
    EffectiveTrainingConfig,
    compose_training_config,
    file_sha256,
    load_formal_training_config,
    manifest_hashes,
    validate_runtime_overrides,
    write_effective_training,
)
from .freezing import (
    FrozenStateGuard,
    StageFrozenStateGuard,
    apply_stage_freeze,
    enforce_frozen_eval,
    enforce_stage_eval,
)
from .graph import inspect_graph
from .intake import require_accepted_intake

DETECT_SELECTION_METRIC = "metrics/mAP50-95(B)"
POSE_SELECTION_METRIC = "metrics/mAP50-95(P)"
MAP_SELECTION_METRIC = DETECT_SELECTION_METRIC
STAGE_ALIASES = {
    "smoke": "D0",
    "formal": "D1",
    "extension": "D2",
    "qat": "Q2",
}
STAGE_RULES = {
    "D0": {"epochs": 3, "patience": 3, "transition": "fresh"},
    "D1": {"epochs": 100, "patience": 20, "transition": "fresh"},
    "D2": {"epochs": 140, "patience": 15, "transition": "resume"},
    "P0": {"epochs": 3, "patience": 3, "transition": "fresh"},
    "P1": {"epochs": 10, "patience": 5, "transition": "fresh"},
    "P2": {"epochs": 20, "patience": 8, "transition": "fresh"},
    "P3": {"epochs": 100, "patience": 20, "transition": "fresh"},
    "P4": {"epochs": 130, "patience": 10, "transition": "resume"},
    "Q2": {"epochs": 15, "patience": 5, "transition": "fresh"},
}
# Backward-compatible read-only lookup for older command previews.
STAGE_RULES.update({alias: STAGE_RULES[target] for alias, target in STAGE_ALIASES.items()})


def normalize_stage(stage: str, task: str) -> str:
    value = STAGE_ALIASES.get(stage.lower(), stage.upper())
    allowed = {"D0", "D1", "D2", "Q2"} if task == "detect" else {"P0", "P1", "P2", "P3", "P4"}
    if value not in allowed:
        raise ValueError(f"階段 {stage!r} 不適用於任務 {task}；請選擇 {sorted(allowed)}")
    return value


def _source_model(weights: Any) -> nn.Module:
    source = weights.get("model") if isinstance(weights, dict) else weights
    if not isinstance(source, nn.Module):
        raise TypeError("training weights must contain the materialized candidate model")
    return source


def make_bittrue_validation_copy(model: nn.Module, bittrue_config: Path) -> nn.Module:
    """Deep-copy Float EMA and change only inherited attention normalization."""

    from yolo_attention.config import VariantConfig
    from yolo_attention.integration import convert_yolo26_model

    validation_model = copy.deepcopy(unwrap_model(model))
    convert_yolo26_model(validation_model, VariantConfig.from_yaml(bittrue_config))
    report = inspect_graph(validation_model)
    if report.attention_normalizations != ("bit_true_pwl", "bit_true_pwl"):
        raise AssertionError("Bit-True 驗證副本未完整轉換")
    return validation_model


def assert_pose_rle_contract(model: nn.Module, rle_weight: float) -> None:
    """Fail when an accepted rle key would be inactive on the selected Pose head."""

    report = inspect_graph(model)
    if report.task != "pose":
        raise TypeError(f"Pose training requires a Pose head, got {report.head_type}")
    head = model.model[-1]
    if rle_weight > 0 and getattr(head, "flow_model", None) is None:
        raise ValueError("rle=1 已接受但未生效：Pose head 沒有 flow_model")


def optimizer_group_report(optimizer: torch.optim.Optimizer | None) -> list[dict[str, Any]]:
    if optimizer is None:
        return []
    report: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        params = [value for value in group.get("params", ()) if isinstance(value, torch.Tensor)]
        report.append(
            {
                "index": index,
                "lr": float(group.get("lr", 0.0)),
                "initial_lr": float(group.get("initial_lr", group.get("lr", 0.0))),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameters": sum(value.numel() for value in params),
                "trainable_parameters": sum(value.numel() for value in params if value.requires_grad),
                "param_group": group.get("param_group"),
                "muon": bool(group.get("use_muon", group.get("muon", False))),
            }
        )
    return report


class _ResearchTrainerMixin:
    bittrue_config: Path
    stage: str
    research_metric: str
    _frozen_guard: FrozenStateGuard | None
    official_fitness: float | None

    def _research_init(self, *, bittrue_config: Path, stage: str, research_metric: str) -> None:
        self.bittrue_config = bittrue_config
        self.stage = stage
        self.research_metric = research_metric
        self.qat = stage == "Q2"
        self._frozen_guard = None
        self._stage_guard: StageFrozenStateGuard | None = None
        self.official_fitness = None
        self.official_fitness_at_research_best: float | None = None
        self.official_fitness_history: list[dict[str, float | int | None]] = []

    def check_resume(self, overrides: dict[str, Any]) -> None:
        super().check_resume(overrides)
        if self.args.resume and "epochs" in overrides:
            requested = int(overrides["epochs"])
            checkpoint_epochs = int(self.args.epochs)
            if requested <= checkpoint_epochs:
                raise ValueError("續訓的總 epochs 必須大於 checkpoint 已完成的 epochs")
            self.args.epochs = requested

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> nn.Module:
        model = copy.deepcopy(_source_model(weights)).float()
        inspect_graph(model)
        return model

    def build_optimizer(self, model: nn.Module, *args: Any, **kwargs: Any):
        apply_stage_freeze(model, self.stage)
        optimizer = super().build_optimizer(model, *args, **kwargs)
        self._frozen_guard = FrozenStateGuard.capture_preserving_stage(model)
        self._stage_guard = StageFrozenStateGuard.capture(model, self.stage)
        return optimizer

    @staticmethod
    def _train_start(trainer: Any) -> None:
        if int(trainer.args.batch) != 16 or int(trainer.batch_size) != 16:
            raise ValueError("正式契約要求 physical batch 為 16")
        # Ultralytics 8.4.90 otherwise halves batch on first-epoch OOM.
        trainer._oom_retries = 3

    def _epoch_start(self, trainer: Any) -> None:
        model = unwrap_model(trainer.model)
        enforce_frozen_eval(model)
        enforce_stage_eval(model, self.stage)
        if self.qat:
            from .quantization import configure_qat_epoch

            configure_qat_epoch(model, int(trainer.epoch) + 1)

    def _epoch_end(self, trainer: Any) -> None:
        if self._frozen_guard is None:
            raise AssertionError("凍結狀態檢查器尚未初始化")
        model = unwrap_model(trainer.model)
        self._frozen_guard.assert_unchanged(model)
        if self._stage_guard is None:
            raise AssertionError("階段凍結狀態檢查器尚未初始化")
        self._stage_guard.assert_unchanged(model)

    @staticmethod
    def _batch_end(trainer: Any) -> None:
        loss = getattr(trainer, "loss", None)
        if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite training loss")

    def validate(self):
        float_ema = self.ema.ema
        self.ema.ema = make_bittrue_validation_copy(float_ema or self.model, self.bittrue_config)
        try:
            metrics = self.validator(self)
        finally:
            self.ema.ema = float_ema
        if metrics is None:
            return None, None
        official = metrics.get("fitness")
        self.official_fitness = float(official) if official is not None else None
        self.official_fitness_history.append(
            {
                "epoch": int(getattr(self, "epoch", -1)) + 1,
                "fitness": self.official_fitness,
            }
        )
        metrics.pop("fitness", None)
        if self.research_metric not in metrics:
            raise KeyError(f"validator omitted research metric {self.research_metric}")
        fitness = float(metrics[self.research_metric])
        if not torch.isfinite(torch.tensor(fitness)):
            raise FloatingPointError("non-finite Bit-True validation metric")
        if not self.best_fitness or fitness > self.best_fitness:
            self.best_fitness = fitness
            self.official_fitness_at_research_best = self.official_fitness
        return metrics, fitness

    def _register_research_callbacks(self) -> None:
        self.add_callback("on_train_start", self._train_start)
        self.add_callback("on_train_epoch_start", self._epoch_start)
        self.add_callback("on_train_epoch_end", self._epoch_end)
        self.add_callback("on_train_batch_end", self._batch_end)


class LiteC3k2Trainer(_ResearchTrainerMixin, DetectionTrainer):
    """Detect trainer preserving runtime grafts and inherited frozen state."""

    def __init__(self, *args: Any, bittrue_config: Path, stage: str, **kwargs: Any) -> None:
        self._research_init(
            bittrue_config=bittrue_config,
            stage=stage,
            research_metric=DETECT_SELECTION_METRIC,
        )
        super().__init__(*args, **kwargs)
        self._register_research_callbacks()


class LiteC3k2PoseTrainer(_ResearchTrainerMixin, PoseTrainer):
    """Pose trainer ranking by Pose mAP while retaining official combined fitness."""

    def __init__(self, *args: Any, bittrue_config: Path, stage: str, **kwargs: Any) -> None:
        self._research_init(
            bittrue_config=bittrue_config,
            stage=stage,
            research_metric=POSE_SELECTION_METRIC,
        )
        super().__init__(*args, **kwargs)
        self._register_research_callbacks()


@dataclass(frozen=True)
class TrainerFactory:
    bittrue_config: Path
    task: str
    stage: str

    def __call__(self, *args: Any, **kwargs: Any):
        trainer = LiteC3k2PoseTrainer if self.task == "pose" else LiteC3k2Trainer
        return trainer(*args, bittrue_config=self.bittrue_config, stage=self.stage, **kwargs)


def _git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _best_epoch(path: Path, metric: str) -> int | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or metric not in rows[0]:
        return None
    return max(range(len(rows)), key=lambda index: float(rows[index][metric])) + 1


def require_c0_or_c_best(project_root: Path, candidate_id: str) -> None:
    """Gate Pose/Q2 to C0 and the exact recorded C_best."""

    if candidate_id == "C0":
        return
    path = project_root / "artifacts/selection.json"
    if not path.is_file():
        raise RuntimeError("尚未選出 C_best；下游工作目前只允許 C0")
    selected = json.loads(path.read_text(encoding="utf-8")).get("c_best")
    if not isinstance(selected, dict) or selected.get("metrics", {}).get("candidate_id") != candidate_id:
        raise RuntimeError(f"下游工作只允許 C0 與已記錄的 C_best，不允許 {candidate_id}")


def require_pose_opt_in(task: str, enabled: bool) -> None:
    """防止任何入口在未經使用者明確同意時執行 Pose。"""

    if task.lower() == "pose" and not enabled:
        raise ValueError("Pose 預設停用；必須明確設定 pose_opt_in=True（CLI 使用 --enable-pose）")


def validate_stage_transition(config: EffectiveTrainingConfig, checkpoint: Path) -> None:
    name = checkpoint.name
    if config.transition_mode == "resume" and name != "last.pt":
        raise ValueError(f"{config.stage} 必須從 last.pt resume，目前是 {name}")
    if config.transition_mode == "fresh" and config.stage in {"P2", "P3"} and name != "best.pt":
        raise ValueError(f"{config.stage} 必須用前一階段 best.pt 建立新 optimizer，目前是 {name}")
    if config.stage == "P0" and name in {"best.pt", "last.pt"}:
        # P0 may use a checkpoint named best.pt from the graft, but it is never a stage input.
        return


def launch_training(
    *,
    project_root: Path,
    checkpoint: Path,
    candidate_id: str,
    stage: str,
    run_id: str,
    task: str = "detect",
    smoke_epochs: int = 3,
    runtime_overrides: dict[str, Any] | None = None,
    pose_opt_in: bool = False,
    training_config_path: Path | None = None,
) -> Path:
    """Launch one gated stage; formal work never uses the development fixture."""

    task = task.lower()
    require_pose_opt_in(task, pose_opt_in)
    intake = require_accepted_intake(project_root)
    normalized_stage = normalize_stage(stage, task)
    if task == "pose" or normalized_stage == "Q2":
        require_c0_or_c_best(project_root, candidate_id)
    if training_config_path is None:
        config = compose_training_config(
            project_root=project_root,
            task=task,
            candidate_id=candidate_id,
            stage=normalized_stage,
        )
    else:
        config = load_formal_training_config(
            training_config_path,
            candidate_id=candidate_id,
            project_root=project_root,
        )
        if config.task != task or config.stage != normalized_stage:
            raise ValueError(
                "正式 YAML 的 task/stage 與 launch_training 請求不一致"
            )
    validate_stage_transition(config, checkpoint)
    args = dict(config.args)
    if normalized_stage in {"D0", "P0"}:
        if smoke_epochs not in (3, 4, 5) or (normalized_stage == "P0" and smoke_epochs != 3):
            raise ValueError("D0 smoke 必須為 3–5 epochs，P0 smoke 必須正好 3 epochs")
        args["epochs"] = smoke_epochs
    args.update(validate_runtime_overrides(runtime_overrides))

    run = project_root / "artifacts/runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    args.update(project=str(run), name="ultralytics", exist_ok=False)
    dataset_path = Path(str(args["data"]))
    if not dataset_path.is_absolute():
        dataset_path = (project_root / dataset_path).resolve()
    args["data"] = str(dataset_path)
    if config.transition_mode == "resume":
        args["resume"] = str(checkpoint.resolve())

    architecture = CANDIDATES[candidate_id].config_path
    hashes = manifest_hashes(
        spec_path=project_root / "EXPERIMENT_SPEC.md",
        architecture_path=architecture,
        training=config,
        dataset_path=dataset_path,
        parent_checkpoint=checkpoint,
    )
    effective_path = run / "effective-training.yaml"
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "task": task,
        "stage": normalized_stage,
        "transition": {
            "mode": config.transition_mode,
            "input_contract": config.transition_input,
        },
        "parent": {"path": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint)},
        "lineage": hashes,
        "architecture_yaml": str(architecture),
        "formal_training_yaml": str(config.sources[0]),
        "effective_training_yaml": str(effective_path),
        "dataset_yaml": str(dataset_path),
        "handoff_manifest_sha256": intake["manifest_sha256"],
        "requested_args": args,
        "git_revision": _git_revision(project_root),
        "selection_backend": "bit_true_pwl",
        "research_metric": POSE_SELECTION_METRIC if task == "pose" else DETECT_SELECTION_METRIC,
        "official_combined_fitness_recorded": task == "pose",
        "head_seed": 0 if task == "pose" else None,
        "frozen_modules": ["model.16.p3_masf", "model.10.m.0.attn", "model.22.m.0.1.attn"],
    }

    model = YOLO(str(checkpoint.resolve()))
    graph = inspect_graph(model.model)
    if graph.task != task:
        raise TypeError(f"checkpoint task {graph.task} does not match requested {task}")
    if task == "pose":
        assert_pose_rle_contract(model.model, float(args["rle"]))
    if normalized_stage == "Q2":
        from .quantization import Conv2dSimulationAdapter

        if not any(isinstance(module, Conv2dSimulationAdapter) for module in model.model.modules()):
            raise ValueError("Q2 需要 quant-prepare 產生的模擬 checkpoint")
        inherited_lr = float(model.overrides.get("lr0", 0.00038))
        expected_qat_lr = inherited_lr * 0.1
        if not math.isclose(float(args["lr0"]), expected_qat_lr, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"Q2 正式 YAML 的 lr0={args['lr0']}，但 parent lr0 的 0.1 倍是 {expected_qat_lr}"
            )
        manifest["qat_lr_ratio"] = 0.1
        manifest["inherited_architecture_lr0"] = inherited_lr
    write_effective_training(config, effective_path, args=args)
    hashes["formal_training_yaml_sha256"] = config.sha256
    hashes["training_yaml_sha256"] = file_sha256(effective_path)
    hashes["effective_training_yaml_sha256"] = file_sha256(effective_path)
    manifest["requested_args"] = args
    (run / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    started = time.perf_counter()
    model.train(
        trainer=TrainerFactory(
            project_root.parent / "achitechure_1/configs/attention/bittrue-pwl-final.yaml",
            task,
            normalized_stage,
        ),
        **args,
    )
    trainer = model.trainer
    if int(trainer.batch_size) != 16:
        raise AssertionError(f"physical batch 漂移為 {trainer.batch_size}")
    completed_epochs = int(getattr(trainer, "epoch", -1)) + 1
    resolved_args_path = run / "resolved-args.json"
    resolved_args_path.write_text(
        json.dumps(vars(trainer.args), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    metrics_csv = run / "ultralytics/results.csv"
    metric = POSE_SELECTION_METRIC if task == "pose" else DETECT_SELECTION_METRIC
    completion = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_epochs": completed_epochs,
        "requested_epochs": int(args["epochs"]),
        "best_epoch": _best_epoch(metrics_csv, metric),
        "best_research_fitness": float(trainer.best_fitness),
        "official_combined_fitness_last": getattr(trainer, "official_fitness", None),
        "official_combined_fitness_at_research_best": getattr(
            trainer, "official_fitness_at_research_best", None
        ),
        "official_combined_fitness_history": getattr(trainer, "official_fitness_history", []),
        "stop_reason": "early-stopping" if completed_epochs < int(args["epochs"]) else "max-epochs",
        "best_checkpoint": str((run / "ultralytics/weights/best.pt").resolve()),
        "last_checkpoint": str((run / "ultralytics/weights/last.pt").resolve()),
        "metrics_csv": str(metrics_csv.resolve()),
        "resolved_args": str(resolved_args_path.resolve()),
        "optimizer_groups": optimizer_group_report(getattr(trainer, "optimizer", None)),
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "simulation_only": normalized_stage == "Q2",
        "lineage": hashes,
    }
    destination = run / "training-complete.json"
    destination.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
