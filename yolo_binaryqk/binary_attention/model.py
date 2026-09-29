"""Parser-first YOLO11 model construction for every experiment variant."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.tasks import DetectionModel

from .attention.base import (
    BinaryC2PSA,
    MagnitudeSideChannelAttention,
    ParallelDualBinaryAttention,
    ResidualDualFullBasisAttention,
    ResidualDualMatchedBasisAttention,
    ScaledUltralyticsBinaryAttention,
    SignOnlyUltralyticsBinaryAttention,
)
from .variants.definitions import VariantDefinition


ATTENTION_CLASSES: dict[str, type[nn.Module]] = {
    "SignOnlyBinaryAttention": SignOnlyUltralyticsBinaryAttention,
    "ScaledBinaryAttention": ScaledUltralyticsBinaryAttention,
    "ParallelDualBinaryAttention": ParallelDualBinaryAttention,
    "ResidualDualFullBasisAttention": ResidualDualFullBasisAttention,
    "ResidualDualMatchedBasisAttention": ResidualDualMatchedBasisAttention,
    "MagnitudeSideChannelAttention": MagnitudeSideChannelAttention,
}


class RegisteredBinaryC2PSA(BinaryC2PSA):
    """Class installed into Ultralytics' parser while one model is built.

    The selected attention class is passed into ``BinaryC2PSA``'s constructor,
    which creates it directly.  There is deliberately no old FP module to
    replace and no stateful module surgery after parsing.
    """

    attention_cls: type[nn.Module] = Attention
    attention_kwargs: dict = {}

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__(
            c1,
            c2,
            n,
            e,
            attention_cls=self.attention_cls,
            attention_kwargs=self.attention_kwargs,
        )


@contextmanager
def binary_parser_registry(variant: VariantDefinition) -> Iterator[None]:
    """Install the concrete C2PSA class before ``DetectionModel`` parses YAML."""

    import ultralytics.nn.tasks as tasks

    original = tasks.C2PSA
    RegisteredBinaryC2PSA.attention_cls = (
        Attention if variant.qk_mode == "fp" else ATTENTION_CLASSES[variant.attention_type]
    )
    RegisteredBinaryC2PSA.attention_kwargs = {} if variant.qk_mode == "fp" else {
        "qk_mode": variant.qk_mode,
        "use_qat": variant.use_qat,
        "p_bits": variant.p_bits,
        "v_bits": variant.v_bits,
        "bias_type": variant.bias_type,
        "magnitude_bits": variant.magnitude_bits,
    }
    # parse_model resolves C2PSA from this module's globals each invocation;
    # assigning it here also makes its base/repeat registration concrete.
    tasks.C2PSA = RegisteredBinaryC2PSA
    try:
        yield
    finally:
        tasks.C2PSA = original


def _source_state(weights: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    source = payload.get("model") if isinstance(payload, dict) else None
    if hasattr(source, "state_dict"):
        source = source.float().state_dict()
    elif isinstance(source, dict):
        source = source
    elif isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        source = payload["state_dict"]
    elif isinstance(payload, dict):
        source = payload
    else:
        raise TypeError(f"unsupported weight payload: {weights}")
    return {key.removeprefix("module."): value for key, value in source.items() if isinstance(value, torch.Tensor)}


def build_student(
    yaml: Path,
    variant: VariantDefinition,
    weights: Path | None = None,
    nc: int = 80,
) -> DetectionModel:
    """Build a variant model and optionally transfer matching FP source weights."""

    yaml = Path(yaml)
    with binary_parser_registry(variant):
        model = DetectionModel(str(yaml), ch=3, nc=nc, verbose=False)

    if weights:
        source = _source_state(Path(weights))
        target = model.state_dict()
        matched = {
            key: value
            for key, value in source.items()
            if key in target and target[key].shape == value.shape
        }
        # New binary basis/threshold/bias tensors intentionally have no FP
        # counterpart.  All matching source tensors, including qkv/proj/pe and
        # detector weights, are still copied deterministically.
        model.load_state_dict(matched, strict=False)
        model._binaryqk_source_transfer = {
            "source": str(Path(weights).resolve()),
            "matched_tensors": len(matched),
            "target_tensors": len(target),
        }
    return model
