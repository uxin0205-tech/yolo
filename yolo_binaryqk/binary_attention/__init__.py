"""Reproducible YOLO11 BinaryAttention experiment package."""
import os
from pathlib import Path

_cache = Path(__file__).resolve().parents[1] / ".cache"
(_cache / "matplotlib").mkdir(parents=True, exist_ok=True)
(_cache / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_cache / "ultralytics"))

from .attention.base import (
    FPAttention,
    MagnitudeSideChannelAttention,
    ParallelDualBinaryAttention,
    ResidualDualFullBasisAttention,
    ResidualDualMatchedBasisAttention,
    ScaledBinaryAttention,
    SignOnlyBinaryAttention,
)
from .variants.definitions import VARIANTS, VariantDefinition

__all__ = [
    "FPAttention", "SignOnlyBinaryAttention", "ScaledBinaryAttention",
    "ParallelDualBinaryAttention", "ResidualDualFullBasisAttention",
    "ResidualDualMatchedBasisAttention", "MagnitudeSideChannelAttention",
    "VARIANTS", "VariantDefinition",
]
