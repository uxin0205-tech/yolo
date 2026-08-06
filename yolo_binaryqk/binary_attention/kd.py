"""Explicit, frozen-teacher knowledge-distillation loss helpers."""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, kind: str) -> torch.Tensor:
    """Compute one KD term while guaranteeing teacher detachment."""

    teacher = teacher.detach()
    if kind == "output":
        return F.mse_loss(student, teacher)
    if kind == "feature":
        return F.smooth_l1_loss(student, teacher)
    if kind == "positional":
        # T6-O distills the depthwise positional-encoding branch only.
        return F.smooth_l1_loss(student, teacher)
    if kind == "attention":
        student_probability = student.clamp_min(1e-8)
        teacher_probability = teacher.clamp_min(1e-8)
        return F.kl_div(student_probability.log(), teacher_probability, reduction="batchmean")
    if kind == "hard":
        if student.ndim < 2 or teacher.ndim != student.ndim:
            raise ValueError("hard detection distillation expects matching logits with a class dimension")
        # Detection heads expose logits as [B, classes, anchors].  The FP
        # teacher's argmax is the hard pseudo-label; no teacher gradient or
        # soft target is retained.
        target = teacher.detach().argmax(dim=1)
        return F.cross_entropy(student, target)
    raise ValueError(f"unknown KD kind: {kind}")


def _tensor_pairs(student, teacher) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Yield shape-compatible tensors from nested Ultralytics predictions."""

    if isinstance(student, torch.Tensor) and isinstance(teacher, torch.Tensor):
        if student.shape == teacher.shape:
            yield student, teacher
        return
    if isinstance(student, (tuple, list)) and isinstance(teacher, (tuple, list)):
        for left, right in zip(student, teacher):
            yield from _tensor_pairs(left, right)
        return
    if isinstance(student, dict) and isinstance(teacher, dict):
        for key in student.keys() & teacher.keys():
            yield from _tensor_pairs(student[key], teacher[key])


def prediction_distillation_loss(student, teacher, components: tuple[str, ...]) -> torch.Tensor:
    """Apply one or more explicit KD terms to compatible model outputs."""

    if set(components).issubset({"output", "hard"}) and isinstance(student, dict) and isinstance(teacher, dict):
        # Detect.forward_head returns boxes, scores and the raw FPN feature
        # list.  Output/hard KD applies only to detector logits; including
        # ``feats`` here would silently turn output KD into a second feature
        # distillation term.
        pairs = [
            (key, student[key], teacher[key])
            for key in ("boxes", "scores")
            if key in student and key in teacher
            and isinstance(student[key], torch.Tensor)
            and isinstance(teacher[key], torch.Tensor)
            and student[key].shape == teacher[key].shape
        ]
        if not pairs:
            raise ValueError("student and teacher predictions have no compatible detector logits for KD")
        terms = []
        for component in components:
            for key, left, right in pairs:
                if component == "hard" and key == "boxes":
                    # YOLO DFL boxes are [B, 4 * reg_max, anchors].  Each of
                    # the four sides has its own categorical distribution;
                    # treating all 64 channels as one class would be invalid.
                    if left.ndim != 3 or left.shape[1] % 4:
                        raise ValueError("hard box KD expects [B, 4 * reg_max, anchors] logits")
                    batch, channels, anchors = left.shape
                    reg_max = channels // 4
                    student_dfl = left.reshape(batch, 4, reg_max, anchors).permute(0, 2, 1, 3)
                    teacher_target = right.detach().reshape(batch, 4, reg_max, anchors).argmax(dim=2)
                    terms.append(F.cross_entropy(student_dfl, teacher_target))
                else:
                    terms.append(distillation_loss(left, right, component))
        return torch.stack(terms).mean()
    else:
        pairs = list(_tensor_pairs(student, teacher))
    if not pairs:
        # Keep the loss connected to the student graph while failing clearly in
        # diagnostics; a zero constant would hide a broken teacher path.
        raise ValueError("student and teacher predictions have no compatible tensors for KD")
    terms = []
    for component in components:
        terms.extend(distillation_loss(left, right, component) for left, right in pairs)
    return torch.stack(terms).mean()


def add_kd_loss(
    detection_loss: torch.Tensor,
    student_prediction,
    teacher_prediction,
    components: tuple[str, ...],
    weight: float,
    feature_pairs: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    attention_pairs: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    positional_pairs: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``detection + weight * KD`` and the detached diagnostic scalar."""

    if not components or weight <= 0:
        zero = detection_loss.new_zeros(())
        return detection_loss, zero
    terms = []
    feature_pairs = list(feature_pairs or [])
    attention_pairs = list(attention_pairs or [])
    positional_pairs = list(positional_pairs or [])
    for component in components:
        if component == "feature":
            if not feature_pairs:
                raise ValueError("feature KD requested but no compatible attention-module features were captured")
            terms.extend(distillation_loss(left, right, "feature") for left, right in feature_pairs)
        elif component == "positional":
            if not positional_pairs:
                raise ValueError("positional KD requested but no compatible positional-encoding outputs were captured")
            terms.extend(distillation_loss(left, right, "positional") for left, right in positional_pairs)
        elif component == "attention":
            if not attention_pairs:
                raise ValueError("attention KD requested but no compatible attention maps were captured")
            terms.extend(distillation_loss(left, right, "attention") for left, right in attention_pairs)
        else:
            terms.append(prediction_distillation_loss(student_prediction, teacher_prediction, (component,)))
    if not terms:
        raise ValueError("KD components produced no loss terms")
    kd = torch.stack(terms).mean()
    return detection_loss + weight * kd, kd.detach()


def calibrate_weight(detection_loss: float, measured_kd_loss: float, target_fraction: float = 0.1) -> float:
    if not 0.05 <= target_fraction <= 0.2:
        raise ValueError("KD target must be within 5–20%")
    if measured_kd_loss <= 0:
        raise ValueError("measured KD loss must be positive")
    return detection_loss * target_fraction / measured_kd_loss
