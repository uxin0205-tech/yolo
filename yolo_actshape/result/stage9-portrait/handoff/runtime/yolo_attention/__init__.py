"""YOLO26 hardware-friendly binary attention research framework."""

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
