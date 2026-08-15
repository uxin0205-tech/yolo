"""Declarative research funnel used by CLI, documentation, and future runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WorkflowStep:
    key: str
    runs: tuple[str, ...]
    epochs: int | str
    purpose: str
    selection: str | None = None
    optional: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["runs"] = list(self.runs)
        return payload


@dataclass(frozen=True)
class ResearchWorkflow:
    main: tuple[WorkflowStep, ...]
    optional: tuple[WorkflowStep, ...]

    @classmethod
    def default(cls) -> ResearchWorkflow:
        return cls(
            main=(
                WorkflowStep("baseline", ("B26-FP",), 0, "Official YOLO26m COCO2017 baseline"),
                WorkflowStep("p0", ("P0",), 0, "Split/fold implementation equivalence"),
                WorkflowStep(
                    "architecture-screening",
                    ("I-SCR", "H-SCR", "T5-SCR"),
                    10,
                    "Binary QK basis screening on both Attention sites",
                    "choose one architecture winner",
                ),
                WorkflowStep(
                    "architecture-recovery",
                    ("W-DIR", "W-PROG"),
                    "20-40",
                    "Direct versus progressive full-model recovery for the winner",
                    "choose formal A0",
                ),
                WorkflowStep(
                    "normalization-screening",
                    (
                        "N0-EXACT",
                        "N0-LUT",
                        "N0-PWL",
                        "N0-SHIFT",
                        "N0-HSIG",
                        "N0-RELU",
                        "N0-MK1",
                        "N0-MK3",
                        "N0-MK5",
                    ),
                    0,
                    "Zero-train normalization comparison on the same A0 checkpoint",
                    "retain at most two by accuracy and estimated cost",
                ),
                WorkflowStep(
                    "normalization-recovery",
                    ("N1-WINNER-1", "N1-WINNER-2"),
                    5,
                    "Attention-only PMP recovery for selected normalization candidates",
                    "choose one normalization winner",
                ),
                WorkflowStep("bdcn-reference", ("D0-IDX",), 0, "Fixed distance-LUT control"),
                WorkflowStep(
                    "bdcn-learning",
                    (
                        "D1-SHARED-10",
                        "D1-PATTN-10",
                        "D1-PHEAD-10",
                        "D1-SEED1 (conditional)",
                    ),
                    "5+5; conditional seed 1 uses 10",
                    "Staged codebook-only convergence and sharing ablation",
                    "choose inside 0.001 mAP, then confirm an uncertain winner with seed 1",
                ),
                WorkflowStep(
                    "bdcn-projection",
                    ("D2-FP", "D2-1P", "D2-2P"),
                    "0 + at most one 5-epoch recovery",
                    "Project one learned codebook to hardware formats",
                    "retain only candidates within 0.01 mAP of A0",
                ),
                WorkflowStep(
                    "bdcn-denominator",
                    ("R0-DIV", "R1-RLUT", "R2-PSHIFT"),
                    0,
                    "Exact, reciprocal-LUT, and division-free denominator comparison",
                    "R1 is primary; Newton correction only if R1 misses the gate",
                ),
                WorkflowStep(
                    "final",
                    ("A-FINAL",),
                    0,
                    "Freeze the formal pre-quantization model",
                    "compare A0, N1 winner, and BDCN winner",
                ),
            ),
            optional=(
                WorkflowStep(
                    "deferred-quantization",
                    ("Q0", "Q1-L3A", "Q2"),
                    "0/0/5",
                    "PTQ sensitivity, integer LUT, then conditional QAT",
                    optional=True,
                ),
            ),
        )

    def to_dict(self) -> dict[str, list[dict[str, object]]]:
        return {
            "main": [step.to_dict() for step in self.main],
            "optional": [step.to_dict() for step in self.optional],
        }
