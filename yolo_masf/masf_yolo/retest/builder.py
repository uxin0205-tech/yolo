"""Builders for the isolated paper-formula P2/P3 retest families."""

from pathlib import Path

from torch import nn

from ..models.builder import build_p2_retest_model, build_p3_model


def build_retest_model(
    family: str,
    variant: str | None = None,
    *,
    source_weights: Path | None = None,
):
    if family == "P2":
        if variant is None:
            raise ValueError("P2 retest models require a variant")
        return build_p2_retest_model(variant, source_weights=source_weights)
    if family == "P3":
        return build_p3_model(variant, source_weights=source_weights)
    if family == "B1R":
        from ..models.builder import build_b1r_model

        return build_b1r_model(source_weights=source_weights)
    raise ValueError("family must be B1R, P2, or P3")
