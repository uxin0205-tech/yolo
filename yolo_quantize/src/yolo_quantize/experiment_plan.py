"""Plan activation-first screens without breaking later weight coupling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar

from .intake import ActivationIntakeReport


@dataclass(frozen=True)
class ActivationExperimentCell:
    """One immutable activation function and output-quantizer policy."""

    policy_id: str
    activation: str
    activation_bits: int | None
    quantizer: str
    weight_format: str
    stage: str
    role: str
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationScreenPlan:
    """Activation-only screen whose result remains provisional for weight work."""

    candidate_activations: tuple[str, ...]
    preferred_seed: str | None
    cells: tuple[ActivationExperimentCell, ...]
    max_promotions: int
    is_final_selection: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "candidate_activations": list(self.candidate_activations),
            "preferred_seed": self.preferred_seed,
            "max_promotions": self.max_promotions,
            "is_final_selection": self.is_final_selection,
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True)
class WeightExperimentCell:
    """One weight family paired with one indivisible activation policy."""

    joint_policy_id: str
    activation_policy_id: str
    weight_family: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CoupledWeightPlan:
    """Weight matrix that never reconstructs activation policies from parts."""

    activation_policy_ids: tuple[str, ...]
    weight_families: tuple[str, ...]
    cells: tuple[WeightExperimentCell, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "activation_policy_ids": list(self.activation_policy_ids),
            "weight_families": list(self.weight_families),
            "cells": [cell.to_dict() for cell in self.cells],
        }


class CoupledExperimentPlanner:
    """Keep full activation policies intact before crossing weight families."""

    _DEFAULT_ACTIVE_CANDIDATES: ClassVar[tuple[str, ...]] = (
        "qsilu_pq",
        "hardswish",
        "poly_shift",
    )
    _WEIGHT_FAMILIES: ClassVar[tuple[str, ...]] = (
        "int8",
        "int4",
        "fixed_sd4",
        "ls_sd4",
        "paper_twn",
        "channel_twn_ttq",
    )

    def __init__(
        self,
        *,
        activation_bits: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
        max_activation_finalists: int = 3,
        active_candidate_activations: tuple[str, ...] = _DEFAULT_ACTIVE_CANDIDATES,
    ) -> None:
        if not activation_bits or len(set(activation_bits)) != len(activation_bits):
            raise ValueError("activation_bits must be non-empty and unique")
        if any(
            isinstance(bits, bool) or not 2 <= bits <= 16 for bits in activation_bits
        ):
            raise ValueError("activation bits must be integers between 2 and 16")
        if not 1 <= max_activation_finalists <= 3:
            raise ValueError("max_activation_finalists must be between one and three")
        if (
            not active_candidate_activations
            or len(set(active_candidate_activations))
            != len(active_candidate_activations)
            or "silu" in active_candidate_activations
        ):
            raise ValueError(
                "active_candidate_activations must be non-empty, unique, and exclude silu"
            )
        self.activation_bits = tuple(sorted(activation_bits))
        self.max_activation_finalists = max_activation_finalists
        self.active_candidate_activations = active_candidate_activations

    def activation_screen(
        self,
        intake: ActivationIntakeReport,
    ) -> ActivationScreenPlan:
        admitted = {
            job.activation
            for job in intake.jobs
            if job.status == "completed"
            and job.kind == "train"
            and job.phase == "short_recovery"
            and job.quantization_eligibility == "screen_candidate"
            and job.activation != "silu"
        }
        ordered = tuple(
            name for name in self.active_candidate_activations if name in admitted
        )
        activations = ("silu", *ordered)
        preferred_seed = (
            "qsilu_pq" if "qsilu_pq" in admitted else (ordered[0] if ordered else None)
        )

        cells: list[ActivationExperimentCell] = []
        for activation in activations:
            role = (
                "matched_control"
                if activation == "silu"
                else "preferred_seed"
                if activation == preferred_seed
                else "screen_candidate"
            )
            matched_policy = f"{activation}--matched-fp"
            cells.append(
                ActivationExperimentCell(
                    policy_id=matched_policy,
                    activation=activation,
                    activation_bits=None,
                    quantizer="none",
                    weight_format="fp32",
                    stage="matched_fp",
                    role=role,
                )
            )
            anchor_policy = f"{activation}--lsq-plus-a8"
            for bits in self.activation_bits:
                policy_id = f"{activation}--lsq-plus-a{bits}"
                dependencies = (matched_policy,)
                stage = "a8_anchor" if bits == 8 else "bit_sweep"
                if bits != 8 and 8 in self.activation_bits:
                    dependencies = (matched_policy, anchor_policy)
                cells.append(
                    ActivationExperimentCell(
                        policy_id=policy_id,
                        activation=activation,
                        activation_bits=bits,
                        quantizer="lsq_plus",
                        weight_format="fp32",
                        stage=stage,
                        role=role,
                        depends_on=dependencies,
                    )
                )
        return ActivationScreenPlan(
            candidate_activations=activations,
            preferred_seed=preferred_seed,
            cells=tuple(cells),
            max_promotions=self.max_activation_finalists,
        )

    def weight_matrix(
        self,
        activation_plan: ActivationScreenPlan,
        finalist_policy_ids: tuple[str, ...],
    ) -> CoupledWeightPlan:
        if not finalist_policy_ids:
            raise ValueError("at least one activation policy finalist is required")
        if len(set(finalist_policy_ids)) != len(finalist_policy_ids):
            raise ValueError("activation policy finalists must be unique")
        if len(finalist_policy_ids) > activation_plan.max_promotions:
            raise ValueError("too many activation policies were promoted")
        quantized_policy_ids = {
            cell.policy_id
            for cell in activation_plan.cells
            if cell.activation_bits is not None
        }
        unknown = tuple(
            policy_id
            for policy_id in finalist_policy_ids
            if policy_id not in quantized_policy_ids
        )
        if unknown:
            raise ValueError(
                "weight work requires complete quantized activation policy IDs: "
                + ", ".join(unknown)
            )
        cells = tuple(
            WeightExperimentCell(
                joint_policy_id=f"{policy_id}--{weight_family}",
                activation_policy_id=policy_id,
                weight_family=weight_family,
            )
            for policy_id in finalist_policy_ids
            for weight_family in self._WEIGHT_FAMILIES
        )
        return CoupledWeightPlan(
            activation_policy_ids=finalist_policy_ids,
            weight_families=self._WEIGHT_FAMILIES,
            cells=cells,
        )
