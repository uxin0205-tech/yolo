"""Transactional activation-output quantization at manifest-owned paths."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .quantizers import LSQPlusActivationQuantizer, LSQPlusSpec


@dataclass(frozen=True)
class ActivationOutputPolicy:
    """An indivisible activation function/output-bit policy identifier."""

    policy_id: str
    bits: int
    signed_codes: bool = False

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id cannot be empty")
        LSQPlusSpec(bits=self.bits, signed_codes=self.signed_codes)


class _ObservedQuantizedActivation(nn.Module):
    _OBSERVE = 0
    _FAKE_QUANT = 1
    _DISABLED = 2

    def __init__(self, activation: nn.Module, policy: ActivationOutputPolicy) -> None:
        super().__init__()
        self.policy_id = policy.policy_id
        self.activation = activation
        self.output_quantizer = LSQPlusActivationQuantizer(
            LSQPlusSpec(bits=policy.bits, signed_codes=policy.signed_codes),
            initial_scale=1.0,
            initial_offset=0.0,
        )
        self.register_buffer("observed_min", torch.tensor(float("inf")))
        self.register_buffer("observed_max", torch.tensor(float("-inf")))
        self.register_buffer(
            "_mode_code", torch.tensor(self._OBSERVE, dtype=torch.uint8)
        )

    @property
    def mode(self) -> str:
        code = int(self._mode_code.item())
        return {
            self._OBSERVE: "observe",
            self._FAKE_QUANT: "fake_quant",
            self._DISABLED: "disabled",
        }[code]

    def observed_range(self) -> tuple[float, float] | None:
        lower = float(self.observed_min.item())
        upper = float(self.observed_max.item())
        if not math.isfinite(lower) or not math.isfinite(upper):
            return None
        return lower, upper

    def disable_quantization(self) -> None:
        self._mode_code.fill_(self._DISABLED)

    def freeze_observer(self) -> None:
        observed = self.observed_range()
        if observed is None:
            raise RuntimeError("activation observer has no finite samples")
        lower, upper = observed
        self.output_quantizer.initialize_from_range(lower, upper)
        self._mode_code.fill_(self._FAKE_QUANT)

    def forward(self, value: Tensor) -> Tensor:
        activated = self.activation(value)
        if self.mode == "observe":
            detached = activated.detach()
            if detached.numel() == 0 or not torch.isfinite(detached).all():
                raise RuntimeError(
                    "activation observer received empty or non-finite output"
                )
            lower = detached.amin().to(dtype=self.observed_min.dtype)
            upper = detached.amax().to(dtype=self.observed_max.dtype)
            self.observed_min.copy_(torch.minimum(self.observed_min, lower))
            self.observed_max.copy_(torch.maximum(self.observed_max, upper))
            return activated
        if self.mode == "fake_quant":
            return self.output_quantizer(activated)
        return activated


def _set_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, separator, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = replacement
    elif isinstance(parent, nn.ModuleDict):
        parent[child_name] = replacement
    else:
        setattr(parent, child_name, replacement)


@dataclass(frozen=True)
class AppliedActivationQuantization:
    """Quantized model plus calibration controls through one public interface."""

    model: nn.Module
    policy: ActivationOutputPolicy
    wrapped_paths: tuple[str, ...]

    def _wrappers(self) -> tuple[_ObservedQuantizedActivation, ...]:
        wrappers = tuple(self.model.get_submodule(path) for path in self.wrapped_paths)
        if not all(
            isinstance(module, _ObservedQuantizedActivation) for module in wrappers
        ):
            raise RuntimeError("activation quantization paths became stale")
        return wrappers  # type: ignore[return-value]

    def rebind(self, model: nn.Module) -> AppliedActivationQuantization:
        """Bind calibration controls to an equivalent cloned/fused graph."""

        rebound = AppliedActivationQuantization(
            model=model,
            policy=self.policy,
            wrapped_paths=self.wrapped_paths,
        )
        mismatched = tuple(
            path
            for path, wrapper in zip(
                rebound.wrapped_paths,
                rebound._wrappers(),
                strict=True,
            )
            if wrapper.policy_id != self.policy.policy_id
            or wrapper.output_quantizer.spec.bits != self.policy.bits
            or (wrapper.output_quantizer.spec.signed_codes != self.policy.signed_codes)
        )
        if mismatched:
            raise RuntimeError(
                "rebound activation quantization policy differs at: "
                + ", ".join(mismatched)
            )
        return rebound

    @property
    def quantizer_count(self) -> int:
        return len(self.wrapped_paths)

    @property
    def mode(self) -> str:
        modes = {wrapper.mode for wrapper in self._wrappers()}
        return modes.pop() if len(modes) == 1 else "mixed"

    def observer_ranges(self) -> dict[str, tuple[float, float] | None]:
        return {
            path: wrapper.observed_range()
            for path, wrapper in zip(self.wrapped_paths, self._wrappers(), strict=True)
        }

    def disable_quantization(self) -> None:
        for wrapper in self._wrappers():
            wrapper.disable_quantization()

    def freeze_observers(self) -> None:
        wrappers = self._wrappers()
        invalid = tuple(
            path
            for path, wrapper in zip(self.wrapped_paths, wrappers, strict=True)
            if wrapper.observed_range() is None
            or wrapper.observed_range()[1] <= wrapper.observed_range()[0]  # type: ignore[index]
        )
        if invalid:
            raise RuntimeError(
                "cannot freeze activation observers without non-degenerate ranges: "
                + ", ".join(invalid)
            )
        for wrapper in wrappers:
            wrapper.freeze_observer()


class ActivationOutputAdapter:
    """Wrap reviewed leaf activations after a fail-closed preflight."""

    def apply(
        self,
        model: nn.Module,
        *,
        module_paths: tuple[str, ...],
        policy: ActivationOutputPolicy,
        clone_model: bool = True,
    ) -> AppliedActivationQuantization:
        if not module_paths or len(set(module_paths)) != len(module_paths):
            raise ValueError("module_paths must be non-empty and unique")
        stale: list[str] = []
        non_leaf: list[str] = []
        for path in module_paths:
            if not path:
                stale.append(path)
                continue
            try:
                current = model.get_submodule(path)
            except AttributeError:
                stale.append(path)
                continue
            if isinstance(current, _ObservedQuantizedActivation):
                stale.append(path)
            elif tuple(current.children()):
                non_leaf.append(path)
        if stale or non_leaf:
            details = []
            if stale:
                details.append("missing or already wrapped paths: " + ", ".join(stale))
            if non_leaf:
                details.append("non-leaf activation paths: " + ", ".join(non_leaf))
            raise ValueError(
                "activation-output preflight failed: " + "; ".join(details)
            )

        actual_model = copy.deepcopy(model) if clone_model else model
        for path in module_paths:
            activation = copy.deepcopy(actual_model.get_submodule(path))
            _set_submodule(
                actual_model,
                path,
                _ObservedQuantizedActivation(activation, policy),
            )
        return AppliedActivationQuantization(
            model=actual_model,
            policy=policy,
            wrapped_paths=module_paths,
        )
