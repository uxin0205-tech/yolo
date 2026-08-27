"""YOLO26 Detect/Pose fusion public API."""

from ._legacy_init import *  # noqa: F403
from ._legacy_init import __all__ as _legacy_all
from .factory import FusionModelFactory
from .formal_training import FormalJointTrainingSession
from .fusion_model import GraphSharedDualHeadModel, assemble_graph_shared_model
from .inference import SharedDualPredictor, load_combined_weights
from .joint_config import JointExperimentConfig
from .joint_data import JointEpochScheduler, TaskLoaderSettings, build_task_loader
from .joint_loss import MacroStepEngine, NativeTaskLossRouter
from .metrics import AccuracyGate, CheckpointSelectors
from .standalone_baseline import StandaloneBaselineValidator
from .validation import JointValidator
from .xnor import XNORExecutionConfig, install_xnor_backend

__all__ = tuple(_legacy_all) + (
    "AccuracyGate",
    "CheckpointSelectors",
    "FormalJointTrainingSession",
    "FusionModelFactory",
    "GraphSharedDualHeadModel",
    "JointEpochScheduler",
    "JointExperimentConfig",
    "JointValidator",
    "MacroStepEngine",
    "NativeTaskLossRouter",
    "SharedDualPredictor",
    "StandaloneBaselineValidator",
    "TaskLoaderSettings",
    "XNORExecutionConfig",
    "assemble_graph_shared_model",
    "build_task_loader",
    "install_xnor_backend",
    "load_combined_weights",
)

