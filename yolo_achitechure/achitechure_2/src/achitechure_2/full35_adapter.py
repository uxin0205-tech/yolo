"""不可變 Full35 final 交付包的 architecture_2 Adapter。"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import ultralytics
from torch import nn

from .candidate import ResolvedCandidate, build_candidate
from .intake import CandidateRegion, file_sha256

DEFAULT_FULL35_ROOT = Path("/home/uxin/yolo/yolo_combine/final/full35")
EXPECTED_TORCH = "2.11.0+cu128"
EXPECTED_ULTRALYTICS = "8.4.90"
CANDIDATE_MODULE_PATHS = (
    "graph.model.6",
    "graph.model.8",
    "graph.model.13",
    "graph.model.19",
)
HEAD_MODULE_PATHS = (
    "graph.model.23.detect_head",
    "graph.model.23.pose_head",
)
FROZEN_MODULE_PATHS = (
    "graph.model.10.m.0.attn",
    "graph.model.16.p3_masf",
    "graph.model.22.m.0.1.attn",
)


@dataclass(frozen=True)
class Full35Parent:
    """已嚴格載入、尚未 graft 的 C0-Handoff。"""

    model: nn.Module
    source: Any
    factory_report: dict[str, Any]
    checkpoint_report: dict[str, Any]


@dataclass(frozen=True)
class Full35TrainingAPI:
    """Full35 final 交付包中，本專案允許組合的穩定訓練 primitives。"""

    Task: Any
    TaskLoaderSettings: Any
    PreparedTaskLoader: Any
    NativeTaskLossRouter: Any
    MacroStepEngine: Any
    JointEpochRunner: Any
    StageWarmupCosineScheduler: Any
    JointStage: Any
    apply_stage: Any
    build_joint_optimizer: Any
    ExperimentLogger: Any
    TrainingProgress: Any
    load_training_snapshot: Any
    save_training_snapshot: Any
    save_inference_weights: Any
    seed_everything: Any
    reseed_loader_for_epoch: Any


class _TaskGraphWrapper(nn.Module):
    """讓單任務 Ultralytics graph 走與 shared model 相同的 graft seam。"""

    def __init__(self, graph: nn.Module, contract: dict[str, Any]) -> None:
        super().__init__()
        self.graph = graph
        self._contract = copy.deepcopy(contract)

    def contract(self) -> dict[str, Any]:
        return copy.deepcopy(self._contract)


def _activate_full35_code(root: Path) -> None:
    code = (root / "code/project/src").resolve()
    if not code.is_dir():
        raise FileNotFoundError(code)
    value = str(code)
    if value not in sys.path:
        sys.path.insert(0, value)
    module = importlib.import_module("yolo_combine")
    location = Path(module.__file__).resolve()
    if code not in location.parents:
        raise RuntimeError(f"yolo_combine 被其他路徑遮蔽：{location}")


@dataclass(frozen=True)
class Full35Release:
    """隱藏 Full35 factory、checkpoint 與 validation template 的深模組。"""

    root: Path = DEFAULT_FULL35_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def checkpoint(self) -> Path:
        return self.root / "weights/combined/inference/best_joint.pt"

    @property
    def release_status(self) -> Path:
        return self.root / "RELEASE_STATUS.json"

    @property
    def release_manifest(self) -> Path:
        return self.root / "MANIFEST.json"

    @property
    def joint_config(self) -> Path:
        return self.root / "configs/joint.yaml"

    @property
    def architecture_config(self) -> Path:
        return self.root / "configs/experiment-joint.yaml"

    @property
    def builder_artifact(self) -> Path:
        return self.root / "code/project/src/yolo_combine/factory.py"

    @property
    def candidate_region(self) -> CandidateRegion:
        return CandidateRegion(
            region_id="shared-c3k2",
            role="shared",
            tasks=("detect", "pose"),
            module_paths=CANDIDATE_MODULE_PATHS,
            head_paths=HEAD_MODULE_PATHS,
        )

    @property
    def protected_module_paths(self) -> tuple[str, ...]:
        return (*FROZEN_MODULE_PATHS, *HEAD_MODULE_PATHS)

    @property
    def frozen_module_paths(self) -> tuple[str, ...]:
        return FROZEN_MODULE_PATHS

    def verify_layout(self) -> dict[str, Any]:
        required = {
            "checkpoint": self.checkpoint,
            "release_status": self.release_status,
            "release_manifest": self.release_manifest,
            "joint_config": self.joint_config,
            "architecture_config": self.architecture_config,
            "builder": self.builder_artifact,
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Full35 final 交付不完整：{missing}")
        status = json.loads(self.release_status.read_text(encoding="utf-8"))
        accepted = status.get("accepted_candidate", {})
        if (
            status.get("release_state") != "accepted_j3_with_j2_rollback"
            or accepted.get("stage") != "j3"
            or accepted.get("inference_weight") != "weights/combined/inference/best_joint.pt"
        ):
            raise ValueError("Full35 release 不再是本輪鎖定的 accepted J3")
        return {
            "root": str(self.root),
            "accepted_stage": "j3",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": file_sha256(self.checkpoint),
            "release_status_sha256": file_sha256(self.release_status),
            "release_manifest_sha256": file_sha256(self.release_manifest),
        }

    @staticmethod
    def verify_environment() -> None:
        if torch.__version__ != EXPECTED_TORCH:
            raise RuntimeError(f"Full35 需要 torch {EXPECTED_TORCH}，目前是 {torch.__version__}")
        if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
            raise RuntimeError(
                f"Full35 需要 ultralytics {EXPECTED_ULTRALYTICS}，目前是 {ultralytics.__version__}"
            )

    def _imports(self) -> dict[str, Any]:
        self.verify_layout()
        self.verify_environment()
        _activate_full35_code(self.root)
        return {
            "JointExperimentConfig": importlib.import_module(
                "yolo_combine.joint_config"
            ).JointExperimentConfig,
            "SourceBundle": importlib.import_module("yolo_combine.source").SourceBundle,
            "FusionModelFactory": importlib.import_module("yolo_combine.factory").FusionModelFactory,
            "load_combined_weights": importlib.import_module("yolo_combine.inference").load_combined_weights,
        }

    def training_api(self) -> Full35TrainingAPI:
        """集中解析 immutable release 的訓練 API，避免呼叫端散落 sys.path 操作。"""

        self._imports()
        contracts = importlib.import_module("yolo_combine.contracts")
        data = importlib.import_module("yolo_combine.joint_data")
        loss = importlib.import_module("yolo_combine.joint_loss")
        trainer = importlib.import_module("yolo_combine.joint_trainer")
        stage = importlib.import_module("yolo_combine.stage_policy")
        logging = importlib.import_module("yolo_combine.experiment_log")
        resume = importlib.import_module("yolo_combine.resume")
        formal = importlib.import_module("yolo_combine.formal_training")
        return Full35TrainingAPI(
            Task=contracts.Task,
            TaskLoaderSettings=data.TaskLoaderSettings,
            PreparedTaskLoader=data.PreparedTaskLoader,
            NativeTaskLossRouter=loss.NativeTaskLossRouter,
            MacroStepEngine=loss.MacroStepEngine,
            JointEpochRunner=trainer.JointEpochRunner,
            StageWarmupCosineScheduler=trainer.StageWarmupCosineScheduler,
            JointStage=stage.JointStage,
            apply_stage=stage.apply_stage,
            build_joint_optimizer=stage.build_joint_optimizer,
            ExperimentLogger=logging.ExperimentLogger,
            TrainingProgress=resume.TrainingProgress,
            load_training_snapshot=resume.load_training_snapshot,
            save_training_snapshot=resume.save_training_snapshot,
            save_inference_weights=resume.save_inference_weights,
            seed_everything=formal.seed_everything,
            reseed_loader_for_epoch=formal.reseed_loader_for_epoch,
        )

    def load_parent(self, checkpoint: str | Path | None = None) -> Full35Parent:
        """從 final source bundle 重建 graph，再嚴格載入 1,238 個 tensors。"""

        modules = self._imports()
        config = modules["JointExperimentConfig"].load(self.joint_config)
        source = modules["SourceBundle"](
            config.source_bundle,
            architecture=config.architecture,
        )
        factory = modules["FusionModelFactory"](
            source,
            detect_data_yaml=config.detect_data,
            pose_data_yaml=config.pose_data,
        )
        built = factory.build(
            checkpoint_kind="float",
            allow_untrained_pose_head=True,
        )
        path = Path(checkpoint).expanduser().resolve() if checkpoint is not None else self.checkpoint
        loaded = modules["load_combined_weights"](
            built.model,
            path,
            prefer_ema=True,
        )
        built.model.requires_grad_(True)
        return Full35Parent(
            model=built.model,
            source=source,
            factory_report=built.report.as_dict(),
            checkpoint_report={
                **asdict(loaded),
                "path": str(loaded.path),
                "checkpoint_sha256": file_sha256(path),
            },
        )

    def resolved_candidate(self, candidate_id: str) -> ResolvedCandidate:
        normalized = candidate_id.upper()
        if normalized == "C0":
            return ResolvedCandidate("C0", "C0", "shared_dual_head", None)
        if normalized not in {"C1", "C2", "C3"}:
            raise ValueError(f"未知 Full35 候選：{candidate_id}")
        return ResolvedCandidate(
            normalized,
            normalized,
            "shared_dual_head",
            self.candidate_region,
        )

    def build(self, candidate_id: str, *, seed: int = 0) -> tuple[nn.Module, Any]:
        parent = self.load_parent()
        resolved = self.resolved_candidate(candidate_id)
        return build_candidate(parent.model, resolved, seed=seed)

    def graft_task_template(
        self,
        graph: nn.Module,
        resolved: ResolvedCandidate,
        *,
        contract: dict[str, Any],
        seed: int = 0,
    ) -> tuple[nn.Module, Any]:
        """把同一候選 graft 到官方單任務 template，供嚴格 materialize。"""

        wrapped = _TaskGraphWrapper(graph, contract)
        candidate, report = build_candidate(wrapped, resolved, seed=seed)
        return candidate.graph, report

    def materialize_validation_models(
        self,
        shared: nn.Module,
        source: Any,
        resolved: ResolvedCandidate,
        *,
        kind: Literal["float", "bittrue"] = "float",
        seed: int = 0,
    ) -> Any:
        """以同一 C0–C3 結構建立 Detect/Pose 官方驗證 graph。"""

        _activate_full35_code(self.root)
        contract = shared.contract()
        templates = source.build_task_models(kind)
        detect, _ = self.graft_task_template(
            templates.detect,
            resolved,
            contract=contract,
            seed=seed,
        )
        pose, _ = self.graft_task_template(
            templates.pose,
            resolved,
            contract=contract,
            seed=seed,
        )
        materialize = importlib.import_module("yolo_combine.graph_materialize")
        task = importlib.import_module("yolo_combine.contracts").Task
        detect, detect_report = materialize.materialize_graph_task_model(
            shared,
            detect,
            task.DETECT,
        )
        pose, pose_report = materialize.materialize_graph_task_model(
            shared,
            pose,
            task.POSE,
        )
        if not detect_report.complete or not pose_report.complete:
            raise RuntimeError(
                f"候選 validation graph materialize 不完整：detect={detect_report}, pose={pose_report}"
            )
        return materialize.GraphValidationModels(
            detect=detect,
            pose=pose,
            detect_report=detect_report,
            pose_report=pose_report,
            checkpoint_kind=kind,
        )


def load_full35_parent(checkpoint: str | Path | None = None) -> nn.Module:
    """供 handoff 驗收 CLI 使用的穩定 loader seam。"""

    return Full35Release().load_parent(checkpoint).model


def _shape_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {str(key): _shape_tree(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shape_tree(child) for child in value]
    return type(value).__name__


def build_full35_fresh_process_report(
    root: str | Path = DEFAULT_FULL35_ROOT,
    *,
    imgsz: int = 64,
) -> dict[str, Any]:
    """在獨立 CPU 程序嚴格載入 J3，並驗證 shared detect/pose forward。"""

    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz 必須是至少 32 的 32 倍數")
    release = Full35Release(Path(root))
    layout = release.verify_layout()
    parent = release.load_parent()
    model = parent.model.cpu().eval()
    images = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        outputs = model(images, task="both")
    if not isinstance(outputs, dict) or set(outputs) != {"detect", "pose"}:
        raise ValueError(f"Full35 task=both 輸出漂移：{type(outputs).__name__} {list(outputs)}")
    contract = model.contract()
    checkpoint = parent.checkpoint_report
    loaded_tensors = int(checkpoint.get("tensors", 0))
    if loaded_tensors != 1238:
        raise ValueError(f"Full35 strict-load tensor 數量漂移：{loaded_tensors}")
    return {
        "schema_version": 2,
        "result": "passed",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "gpu_used": False,
        "release_state": "accepted_j3_with_j2_rollback",
        "accepted_stage": layout["accepted_stage"],
        "checkpoint": str(release.checkpoint),
        "checkpoint_sha256": layout["checkpoint_sha256"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "state_source": checkpoint["state_source"],
        "strict_load": True,
        "loaded_tensors": loaded_tensors,
        "factory_complete": bool(parent.factory_report.get("complete")),
        "task": "both",
        "output_tasks": ["detect", "pose"],
        "input": {
            "batch": 1,
            "channels": 3,
            "height": imgsz,
            "width": imgsz,
            "values": "zeros",
        },
        "output_shapes": _shape_tree(outputs),
        "model_kind": contract["model_kind"],
        "shared_layers": contract["shared_layers"],
        "shared_parameters": parent.factory_report["assembly"]["shared_parameters"],
        "head_inputs": contract["head_inputs"],
        "feature_channels": contract["feature_channels"],
        "strides": contract["strides"],
        "detect_nc": contract["detect_nc"],
        "pose_nc": contract["pose_nc"],
        "kpt_shape": contract["kpt_shape"],
    }
