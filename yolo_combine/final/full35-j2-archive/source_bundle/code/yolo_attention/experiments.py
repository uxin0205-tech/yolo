"""已核准 experiment funnel 的 immutable registry。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import (
    BasisKind,
    BiasKind,
    NormalizationKind,
    RowCorrection,
    ScaleMode,
    VariantConfig,
)


class Stage(str, Enum):
    BASELINE = "baseline"
    VALIDATION = "validation"
    SCREENING = "screening"
    RECOVERY = "recovery"
    SCALE = "scale"
    BIAS = "bias"
    PTQ = "ptq"
    LUT = "lut"
    QAT = "qat"


@dataclass(frozen=True)
class ExperimentRun:
    variant: VariantConfig
    stage: Stage
    epochs: int
    parent: str
    conditional: bool = False
    trainable_scope: str = "none"


class ExperimentRegistry:
    def __init__(self, runs: list[ExperimentRun]) -> None:
        self._runs = {run.variant.name: run for run in runs}
        if len(self._runs) != len(runs):
            raise ValueError("experiment names must be unique")

    def get(self, name: str) -> ExperimentRun:
        try:
            return self._runs[name]
        except KeyError as exc:
            raise KeyError(f"unknown experiment {name!r}") from exc

    def for_stage(self, stage: Stage) -> list[ExperimentRun]:
        stage = Stage(stage)
        return [run for run in self._runs.values() if run.stage is stage]

    def names(self) -> tuple[str, ...]:
        return tuple(self._runs)

    @classmethod
    def default(cls) -> ExperimentRegistry:
        fp = VariantConfig(name="B26-FP", basis=BasisKind.FP, use_ste=False)
        p0 = VariantConfig(name="P0", basis=BasisKind.FP, use_ste=False)
        identity = VariantConfig(name="I-SCR", basis=BasisKind.IDENTITY)
        hadamard = VariantConfig(name="H-SCR", basis=BasisKind.HADAMARD)
        t5 = VariantConfig(name="T5-SCR", basis=BasisKind.T5)
        direct = VariantConfig(name="W-DIR", basis=BasisKind.IDENTITY)
        progressive = VariantConfig(name="W-PROG", basis=BasisKind.IDENTITY, progressive=True)
        dynamic = VariantConfig(name="V1-DYN", basis=BasisKind.IDENTITY)
        fixed = VariantConfig(name="V1-SHEAD", basis=BasisKind.IDENTITY, scale_mode=ScaleMode.FIXED_HEAD)
        power = VariantConfig(name="V1-P2", basis=BasisKind.IDENTITY, scale_mode=ScaleMode.POWER_OF_TWO)
        b0 = VariantConfig(name="V1-B0", basis=BasisKind.IDENTITY)
        bd = VariantConfig(name="V1-BD", basis=BasisKind.IDENTITY, bias=BiasKind.DENSE_2D)
        br = VariantConfig(name="V1-BR", basis=BasisKind.IDENTITY, bias=BiasKind.DECOMPOSED_2D)
        q0 = VariantConfig(
            name="Q0",
            basis=BasisKind.IDENTITY,
            p_bits=8,
            v_bits=8,
            projection_weight_bits=8,
            projection_activation_bits=8,
        )
        q1 = VariantConfig(
            name="Q1-L3A",
            basis=BasisKind.IDENTITY,
            normalization=NormalizationKind.INTEGER_LUT,
            p_bits=8,
            v_bits=8,
            projection_weight_bits=8,
            projection_activation_bits=8,
        )
        q2 = VariantConfig(
            name="Q2",
            basis=BasisKind.IDENTITY,
            normalization=NormalizationKind.INTEGER_LUT,
            row_correction=RowCorrection.NONE,
            p_bits=8,
            v_bits=8,
            projection_weight_bits=8,
            projection_activation_bits=8,
        )
        return cls(
            [
                ExperimentRun(fp, Stage.BASELINE, 0, "official-yolo26m"),
                ExperimentRun(p0, Stage.VALIDATION, 0, "B26-FP"),
                ExperimentRun(identity, Stage.SCREENING, 10, "P0", trainable_scope="attention"),
                ExperimentRun(hadamard, Stage.SCREENING, 10, "P0", trainable_scope="attention"),
                ExperimentRun(t5, Stage.SCREENING, 10, "P0", trainable_scope="attention"),
                ExperimentRun(direct, Stage.RECOVERY, 40, "architecture-winner", trainable_scope="full"),
                ExperimentRun(progressive, Stage.RECOVERY, 40, "architecture-winner", trainable_scope="full"),
                ExperimentRun(dynamic, Stage.SCALE, 0, "formal-V1"),
                ExperimentRun(
                    fixed, Stage.SCALE, 3, "formal-V1", conditional=True, trainable_scope="attention"
                ),
                ExperimentRun(
                    power, Stage.SCALE, 3, "V1-SHEAD", conditional=True, trainable_scope="attention"
                ),
                ExperimentRun(b0, Stage.BIAS, 5, "formal-V1", trainable_scope="attention"),
                ExperimentRun(bd, Stage.BIAS, 5, "formal-V1", trainable_scope="attention"),
                ExperimentRun(br, Stage.BIAS, 5, "formal-V1", trainable_scope="attention"),
                ExperimentRun(q0, Stage.PTQ, 0, "V2-parent"),
                ExperimentRun(q1, Stage.LUT, 0, "Q0"),
                ExperimentRun(q2, Stage.QAT, 5, "Q1-L3A", conditional=True, trainable_scope="full"),
            ]
        )
