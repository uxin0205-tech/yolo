"""Exact scale fitting for signed fixed codebooks."""

from __future__ import annotations

import torch
from torch import Tensor


def optimal_scaled_codebook_scales(
    groups: Tensor,
    *,
    codebook: Tensor,
) -> Tensor:
    """Return a globally MSE-optimal positive scale for every weight group.

    For a fixed ordered codebook, nearest-code assignments change only at a
    finite set of scale boundaries.  The objective is quadratic between those
    boundaries, so this event sweep evaluates the exact minimizer of every
    interval.  This is an implementation of the scaled-codebook objective from
    Idelbayev and Carreira-Perpiñán (CVPR 2021), not a local grid search.
    """

    if groups.ndim != 2:
        raise ValueError("scaled-codebook groups must be a two-dimensional tensor")
    if codebook.ndim != 1 or codebook.numel() < 3:
        raise ValueError(
            "scaled codebook must be one-dimensional with at least 3 values"
        )
    if not bool(torch.all(codebook[1:] > codebook[:-1])):
        raise ValueError("scaled codebook values must be strictly increasing")
    zero_indices = torch.nonzero(codebook == 0).flatten()
    if zero_indices.numel() != 1:
        raise ValueError("scaled codebook must contain exactly one logical zero")
    zero_index = int(zero_indices.item())
    if zero_index == 0 or zero_index == codebook.numel() - 1:
        raise ValueError("scaled codebook must contain negative and positive values")

    calculation_dtype = (
        torch.float64 if groups.dtype == torch.float64 else torch.float32
    )
    values = codebook.to(device=groups.device, dtype=calculation_dtype)
    positive_levels = values[zero_index:]
    negative_levels = -torch.flip(values[: zero_index + 1], dims=(0,))
    scales: list[Tensor] = []

    for group in groups:
        value = group.detach().to(calculation_dtype)
        positive = value[value > 0]
        negative = -value[value < 0]
        if positive.numel() + negative.numel() == 0:
            scales.append(value.new_tensor(1.0))
            continue

        initial_a = value.new_zeros(())
        initial_b = value.new_zeros(())
        event_parts: list[Tensor] = []
        delta_a_parts: list[Tensor] = []
        delta_b_parts: list[Tensor] = []
        for magnitudes, levels in (
            (positive, positive_levels),
            (negative, negative_levels),
        ):
            if magnitudes.numel() == 0:
                continue
            lower = levels[:-1]
            upper = levels[1:]
            midpoints = (lower + upper) * 0.5
            event_parts.append((magnitudes[:, None] / midpoints[None, :]).reshape(-1))
            delta_a_parts.append(
                (magnitudes[:, None] * (lower - upper)[None, :]).reshape(-1)
            )
            delta_b_parts.append(
                (lower.square() - upper.square()).repeat(magnitudes.numel())
            )
            initial_a = initial_a + magnitudes.sum() * levels[-1]
            initial_b = (
                initial_b
                + magnitudes.new_tensor(float(magnitudes.numel())) * levels[-1].square()
            )

        events = torch.cat(event_parts)
        delta_a = torch.cat(delta_a_parts)
        delta_b = torch.cat(delta_b_parts)
        order = torch.argsort(events, stable=True)
        events = events[order]
        delta_a = delta_a[order]
        delta_b = delta_b[order]

        a_states = torch.cat(
            (initial_a.reshape(1), initial_a + torch.cumsum(delta_a, dim=0))
        )
        b_states = torch.cat(
            (initial_b.reshape(1), initial_b + torch.cumsum(delta_b, dim=0))
        )
        lower_bounds = torch.cat((events.new_zeros(1), events))
        upper_bounds = torch.cat((events, events.new_tensor([float("inf")])))
        epsilon = torch.finfo(calculation_dtype).eps
        stationary = torch.where(
            b_states > epsilon,
            a_states / b_states.clamp_min(epsilon),
            lower_bounds,
        )
        candidates = torch.minimum(
            torch.maximum(stationary, lower_bounds),
            upper_bounds,
        )
        energy = positive.square().sum() + negative.square().sum()
        errors = (
            energy - 2.0 * candidates * a_states + candidates.square() * b_states
        ).clamp_min(0.0)
        scales.append(candidates[torch.argmin(errors)])

    return torch.stack(scales).to(dtype=groups.dtype).reshape(-1, 1)
