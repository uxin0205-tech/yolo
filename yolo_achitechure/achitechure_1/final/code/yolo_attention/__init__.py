"""YOLO26 硬體友善二值 Attention 研究框架。"""

from .config import (
    BasisKind,
    BDCNCodebookKind,
    BDCNDenominator,
    BDCNProjection,
    BDCNSharing,
    BiasKind,
    NormalizationKind,
    RowCorrection,
    ScaleMode,
    VariantConfig,
)

__all__ = [
    "BDCNCodebookKind",
    "BDCNDenominator",
    "BDCNProjection",
    "BDCNSharing",
    "BasisKind",
    "BiasKind",
    "NormalizationKind",
    "RowCorrection",
    "ScaleMode",
    "VariantConfig",
]

__version__ = "0.1.0"
