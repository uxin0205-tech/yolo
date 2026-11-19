"""Hash-pinned epoch-boundary patience migration for an active QAT pilot."""

from __future__ import annotations

import gc
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _verified_file(record: object, label: str) -> tuple[Path, str]:
    payload = _mapping(record, label)
    if set(payload) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    configured = Path(str(payload["path"])).expanduser()
    path = (
        configured.resolve()
        if configured.is_absolute()
        else (PROJECT_ROOT / configured).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(payload["sha256"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 drifted: {actual} != {expected}")
    return path, actual


@dataclass(frozen=True)
class QATPatienceContinuation:
    """Reviewed permission and history for changing only resume patience."""

    config_path: Path
    config_sha256: str
    authorization_id: str
    base_plan: Path
    base_plan_sha256: str
    resume_checkpoint: Path
    resume_checkpoint_sha256: str
    from_patience: int
    to_patience: int
    boundary_next_epoch: int
    score_history: tuple[tuple[int, float], ...]

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        base_plan: Path,
        base_plan_sha256: str,
    ) -> QATPatienceContinuation:
        config_path = Path(path).expanduser().resolve()
        payload = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
            "QAT patience continuation",
        )
        if payload.get("schema_version") != 1:
            raise ValueError("QAT patience continuation schema_version must be 1")
        if payload.get("kind") != "patience_only_epoch_boundary":
            raise ValueError("unsupported QAT continuation kind")
        authorization_id = str(payload.get("authorization_id", "")).strip()
        if not authorization_id:
            raise ValueError("QAT patience continuation authorization_id is empty")
        pinned_plan, pinned_plan_sha256 = _verified_file(
            payload.get("base_plan"), "continuation base plan"
        )
        if (pinned_plan, pinned_plan_sha256) != (
            base_plan.resolve(),
            base_plan_sha256,
        ):
            raise ValueError("continuation base plan differs from the active QAT plan")
        checkpoint, checkpoint_sha256 = _verified_file(
            payload.get("resume_checkpoint"), "continuation resume checkpoint"
        )
        transition = _mapping(payload.get("transition"), "patience transition")
        if set(transition) != {"from", "to"}:
            raise ValueError("patience transition must contain exactly from and to")
        from_patience = int(transition["from"])
        to_patience = int(transition["to"])
        boundary_next_epoch = int(payload.get("boundary_next_epoch", -1))
        if from_patience < 0 or to_patience < 1:
            raise ValueError("patience transition requires from >= 0 and to >= 1")
        if boundary_next_epoch < 1:
            raise ValueError("continuation boundary_next_epoch must be positive")
        raw_history = payload.get("joint_score_history")
        if not isinstance(raw_history, list):
            raise TypeError("joint_score_history must be a list")
        history: list[tuple[int, float]] = []
        for record in raw_history:
            item = _mapping(record, "joint score history record")
            if set(item) != {"epoch", "best_joint_score"}:
                raise ValueError(
                    "joint score history records require epoch and best_joint_score"
                )
            epoch = int(item["epoch"])
            score = float(item["best_joint_score"])
            if not math.isfinite(score):
                raise ValueError("joint score history must be finite")
            history.append((epoch, score))
        expected_epochs = tuple(range(boundary_next_epoch))
        if tuple(epoch for epoch, _ in history) != expected_epochs:
            raise ValueError("joint score history must cover every completed epoch")
        instance = cls(
            config_path=config_path,
            config_sha256=_sha256(config_path),
            authorization_id=authorization_id,
            base_plan=pinned_plan,
            base_plan_sha256=pinned_plan_sha256,
            resume_checkpoint=checkpoint,
            resume_checkpoint_sha256=checkpoint_sha256,
            from_patience=from_patience,
            to_patience=to_patience,
            boundary_next_epoch=boundary_next_epoch,
            score_history=tuple(history),
        )
        state = instance.early_stop_state()
        if int(state["stale_epochs"]) >= to_patience:
            raise ValueError("continuation history had already exhausted new patience")
        return instance

    def early_stop_state(self) -> dict[str, Any]:
        best_score = -math.inf
        stale_epochs = 0
        for _, score in self.score_history:
            if score > best_score:
                best_score = score
                stale_epochs = 0
            else:
                stale_epochs += 1
        if not math.isfinite(best_score):
            raise ValueError("continuation has no finite joint score")
        return {
            "schema_version": 1,
            "stage": "j3",
            "patience": self.to_patience,
            "min_delta": 0.0,
            "best_score": best_score,
            "stale_epochs": stale_epochs,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "patience_only_epoch_boundary",
            "config": str(self.config_path),
            "config_sha256": self.config_sha256,
            "authorization_id": self.authorization_id,
            "base_plan": str(self.base_plan),
            "base_plan_sha256": self.base_plan_sha256,
            "resume_checkpoint": str(self.resume_checkpoint),
            "resume_checkpoint_sha256": self.resume_checkpoint_sha256,
            "from_patience": self.from_patience,
            "to_patience": self.to_patience,
            "boundary_next_epoch": self.boundary_next_epoch,
            "early_stop_state": self.early_stop_state(),
        }

    def prepare_resume_checkpoint(
        self,
        destination: str | Path,
        *,
        expected_resolved_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Copy one exact snapshot while changing only its patience contract."""

        actual_source_sha256 = _sha256(self.resume_checkpoint)
        if actual_source_sha256 != self.resume_checkpoint_sha256:
            raise ValueError("continuation resume checkpoint changed before migration")
        payload = torch.load(
            self.resume_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(payload, dict)
            or payload.get("checkpoint_kind") != "full_resume"
        ):
            raise ValueError(
                "continuation source is not a Full35 full-resume checkpoint"
            )
        progress = _mapping(payload.get("progress"), "checkpoint progress")
        if (
            str(progress.get("stage")) != "j3"
            or int(progress.get("next_epoch", -1)) != self.boundary_next_epoch
            or int(progress.get("joint_epochs_completed", -1))
            != self.boundary_next_epoch
        ):
            raise ValueError(
                "continuation checkpoint is not at the declared epoch boundary"
            )
        loader_state = dict(_mapping(payload.get("loader_state"), "loader state"))
        if (
            loader_state.get("snapshot_boundary") != "formal_epoch_end"
            or int(loader_state.get("epoch", -1)) != self.boundary_next_epoch - 1
        ):
            raise ValueError("continuation checkpoint loader boundary is malformed")
        if self.from_patience == 0 and loader_state.get("early_stop") is not None:
            raise ValueError("patience=0 source unexpectedly contains early-stop state")
        source_resolved = _mapping(payload.get("resolved_config"), "resolved config")
        source_qat = _mapping(source_resolved.get("qat_experiment"), "QAT experiment")
        source_stages = _mapping(
            source_resolved.get("stage_policies"), "stage policies"
        )
        source_j3 = _mapping(source_stages.get("j3"), "j3 stage policy")
        if source_qat.get("plan_sha256") != self.base_plan_sha256:
            raise ValueError("checkpoint was not produced by the pinned base plan")
        if int(source_j3.get("patience", -1)) != self.from_patience:
            raise ValueError("checkpoint patience differs from continuation transition")
        best_state = _mapping(payload.get("best_state"), "best state")
        last = _mapping(best_state.get("last"), "last selector state")
        final_epoch, final_score = self.score_history[-1]
        if (
            int(last.get("epoch", -1)) != final_epoch
            or float(last.get("score", math.nan)) != final_score
        ):
            raise ValueError(
                "checkpoint selector state differs from declared score history"
            )

        continuation = self.to_dict()
        loader_state["early_stop"] = self.early_stop_state()
        provenance = dict(_mapping(payload.get("provenance"), "provenance"))
        provenance["qat_patience_continuation"] = continuation
        payload["loader_state"] = loader_state
        payload["resolved_config"] = dict(expected_resolved_config)
        payload["provenance"] = provenance

        target = Path(destination).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        del payload
        gc.collect()
        result = {
            **continuation,
            "migrated_checkpoint": str(target),
            "migrated_checkpoint_sha256": _sha256(target),
        }
        return result


__all__ = ("QATPatienceContinuation",)
