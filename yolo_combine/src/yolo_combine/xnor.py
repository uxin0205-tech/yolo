"""Memory-bounded XNOR/popcount execution for the immutable attention bundle.

The authoritative Full35/Partial75 bundle intentionally remains byte-for-byte
unchanged. This module installs an execution adapter after the bundle has been
verified and imported. The adapter changes only how the reference XNOR result
is materialized: query and key tokens are tiled while the key dimension is
reduced in one operation, preserving the existing integer reduction semantics.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import torch

XNORBackend = Literal["bool_tiled"]


@dataclass(frozen=True)
class XNORExecutionConfig:
    """Resolved software execution contract for binary Q/K scores."""

    backend: XNORBackend = "bool_tiled"
    token_tile: int = 32

    def __post_init__(self) -> None:
        if self.backend != "bool_tiled":
            raise ValueError(f"unsupported XNOR backend: {self.backend!r}")
        if not isinstance(self.token_tile, int) or isinstance(self.token_tile, bool):
            raise TypeError("token_tile must be an integer")
        if self.token_tile < 1:
            raise ValueError("token_tile must be a positive integer")

    def as_dict(self) -> dict[str, str | int]:
        return {"backend": self.backend, "token_tile": self.token_tile}


@dataclass(frozen=True)
class XNORInstallationReport:
    """Describe the process-global adapter installed into yolo_attention."""

    module: str
    backend: XNORBackend
    token_tile: int
    newly_installed: bool


@dataclass(frozen=True)
class XNORWorkspaceEstimate:
    """Static upper bound for the dominant int64 reduction workspace."""

    reduction_bytes: int
    query_tile: int
    key_tile: int

    @property
    def reduction_gib(self) -> float:
        return self.reduction_bytes / (1024**3)


def estimate_xnor_workspace(
    *,
    batch: int,
    heads: int,
    query_tokens: int,
    key_tokens: int,
    channels: int,
    token_tile: int | None,
) -> XNORWorkspaceEstimate:
    """Estimate the dominant default-int64 reduction allocation.

    This models the workspace that caused the retained Full35 validation OOM:
    B * H * Nq * Nk * D * sizeof(int64). With tiling, Nq and Nk are capped
    independently while the complete channel reduction is retained.
    """

    dimensions = {
        "batch": batch,
        "heads": heads,
        "query_tokens": query_tokens,
        "key_tokens": key_tokens,
        "channels": channels,
    }
    for name, value in dimensions.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if token_tile is not None:
        if not isinstance(token_tile, int) or isinstance(token_tile, bool):
            raise TypeError("token_tile must be an integer or None")
        if token_tile < 1:
            raise ValueError("token_tile must be positive")
    query_tile = query_tokens if token_tile is None else min(query_tokens, token_tile)
    key_tile = key_tokens if token_tile is None else min(key_tokens, token_tile)
    return XNORWorkspaceEstimate(
        reduction_bytes=batch * heads * query_tile * key_tile * channels * 8,
        query_tile=query_tile,
        key_tile=key_tile,
    )


def _validate_inputs(q_sign: torch.Tensor, k_sign: torch.Tensor) -> None:
    if q_sign.ndim != 4 or k_sign.ndim != 4:
        raise ValueError("Q/K must have shape [batch, heads, channels, tokens]")
    if q_sign.shape[:-1] != k_sign.shape[:-1]:
        raise ValueError(
            f"Q/K shape mismatch: {tuple(q_sign.shape)} vs {tuple(k_sign.shape)}"
        )
    if q_sign.device != k_sign.device:
        raise ValueError(f"Q/K device mismatch: {q_sign.device} vs {k_sign.device}")
    if q_sign.dtype != k_sign.dtype:
        raise ValueError(f"Q/K dtype mismatch: {q_sign.dtype} vs {k_sign.dtype}")
    if not q_sign.dtype.is_floating_point:
        raise TypeError(f"Q/K must use a floating dtype, got {q_sign.dtype}")


def broadcast_xnor_popcount_dot(
    q_sign: torch.Tensor,
    k_sign: torch.Tensor,
) -> torch.Tensor:
    """Small-input oracle matching the immutable bundle's broadcast reference."""

    _validate_inputs(q_sign, k_sign)
    q_bits = q_sign >= 0
    k_bits = k_sign >= 0
    matches = (
        q_bits.transpose(-2, -1).unsqueeze(-2)
        == k_bits.transpose(-2, -1).unsqueeze(-3)
    ).sum(dim=-1)
    return (2 * matches - q_sign.shape[-2]).to(q_sign.dtype)


def tiled_xnor_popcount_dot(
    q_sign: torch.Tensor,
    k_sign: torch.Tensor,
    *,
    token_tile: int = 32,
) -> torch.Tensor:
    """Compute exact XNOR/popcount scores without a full Nq*Nk*D workspace.

    Query and key token axes are tiled independently. The channel/key axis is
    deliberately not tiled, so each token pair uses the same integer reduction
    as the retained broadcast implementation. Like the retained implementation,
    the boolean comparison is non-differentiable and therefore introduces no
    Q/K gradient by itself.
    """

    if not isinstance(token_tile, int) or isinstance(token_tile, bool):
        raise TypeError("token_tile must be an integer")
    if token_tile < 1:
        raise ValueError("token_tile must be a positive integer")
    _validate_inputs(q_sign, k_sign)
    q_bits = (q_sign >= 0).transpose(-2, -1)
    k_bits = (k_sign >= 0).transpose(-2, -1)
    query_tokens = q_bits.shape[-2]
    key_tokens = k_bits.shape[-2]
    channel_count = q_bits.shape[-1]

    query_rows: list[torch.Tensor] = []
    for query_start in range(0, query_tokens, token_tile):
        query = q_bits[..., query_start : query_start + token_tile, :]
        key_columns: list[torch.Tensor] = []
        for key_start in range(0, key_tokens, token_tile):
            key = k_bits[..., key_start : key_start + token_tile, :]
            matches = (query.unsqueeze(-2) == key.unsqueeze(-3)).sum(dim=-1)
            key_columns.append((2 * matches - channel_count).to(q_sign.dtype))
        query_rows.append(torch.cat(key_columns, dim=-1))
    return torch.cat(query_rows, dim=-2)


_installed_module: ModuleType | None = None
_installed_config: XNORExecutionConfig | None = None


def install_xnor_backend(
    config: XNORExecutionConfig = XNORExecutionConfig(),
) -> XNORInstallationReport:
    """Install one fail-closed process-global backend after source activation."""

    global _installed_config, _installed_module
    module = importlib.import_module("yolo_attention.binary_basis")
    if _installed_config is not None:
        if module is not _installed_module or config != _installed_config:
            raise RuntimeError(
                "an incompatible XNOR backend is already installed in this process: "
                f"installed={_installed_config}, requested={config}"
            )
        return XNORInstallationReport(
            module=module.__name__,
            backend=config.backend,
            token_tile=config.token_tile,
            newly_installed=False,
        )

    def configured_xnor(q_sign: torch.Tensor, k_sign: torch.Tensor) -> torch.Tensor:
        return tiled_xnor_popcount_dot(
            q_sign,
            k_sign,
            token_tile=config.token_tile,
        )

    configured_xnor.__name__ = "configured_tiled_xnor_popcount_dot"
    configured_xnor.__doc__ = (
        "Process-locked tiled XNOR backend installed by yolo_combine."
    )
    module.xnor_popcount_dot = configured_xnor
    _installed_module = module
    _installed_config = config
    return XNORInstallationReport(
        module=module.__name__,
        backend=config.backend,
        token_tile=config.token_tile,
        newly_installed=True,
    )


def installed_xnor_config() -> XNORExecutionConfig | None:
    """Return the active backend contract without importing or installing it."""

    return _installed_config
