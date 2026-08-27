"""YOLO26 Detect/Pose fusion modules."""

from .checkpoint import load_checkpoint, save_checkpoint
from .contracts import Task, normalize_tasks
from .data import (
    BBT5AuditReport,
    CanonicalBBAT5,
    DatasetContractError,
    PreparedDataset,
    PreparedDetectSubset,
    audit_bbt5,
    prepare_bbt5_view,
    prepare_coco_detect_subset,
)
from .freezing import InheritedFreezeGuard, freeze_inherited
from .joint import JointRunReport, run_joint_steps
from .loaders import JointLoaders, TaskLoader, build_joint_train_loaders
from .materialize import (
    MaterializedTaskModels,
    TaskMaterializationReport,
    build_validation_models,
)
from .models import RoutedDualModel, SharedDualHeadModel
from .pose_stages import POSE_STAGES, PoseStageSpec, pose_stage
from .source import (
    HeadTransferReport,
    SourceBundle,
    TrunkTransferReport,
    transfer_pose_head,
)
from .training import TaskLossRouter
from .variants import VariantAudit, VariantWorkspace, load_variant

__all__ = [
    "POSE_STAGES",
    "BBT5AuditReport",
    "CanonicalBBAT5",
    "DatasetContractError",
    "HeadTransferReport",
    "InheritedFreezeGuard",
    "JointLoaders",
    "JointRunReport",
    "MaterializedTaskModels",
    "PoseStageSpec",
    "PreparedDataset",
    "PreparedDetectSubset",
    "RoutedDualModel",
    "SharedDualHeadModel",
    "SourceBundle",
    "Task",
    "TaskLoader",
    "TaskLossRouter",
    "TaskMaterializationReport",
    "TrunkTransferReport",
    "VariantAudit",
    "VariantWorkspace",
    "audit_bbt5",
    "build_joint_train_loaders",
    "build_validation_models",
    "freeze_inherited",
    "load_checkpoint",
    "load_variant",
    "normalize_tasks",
    "pose_stage",
    "prepare_bbt5_view",
    "prepare_coco_detect_subset",
    "run_joint_steps",
    "save_checkpoint",
    "transfer_pose_head",
]
