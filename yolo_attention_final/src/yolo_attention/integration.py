"""同時處理兩個官方 YOLO26m Attention sites 的 fail-closed adapters。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from ultralytics.nn.modules.block import C2PSA, Attention

from .attention import HardwareFriendlyAttention
from .bdcn import BDCNCodebookBank
from .config import BDCNSharing, NormalizationKind, VariantConfig
from .normalization import build_normalizer

YOLO26M_ATTENTION_PATHS = (
    "model.10.m.0.attn",
    "model.22.m.0.1.attn",
)


@dataclass(frozen=True)
class TrainableSummary:
    stage: str
    trainable_parameters: int
    total_parameters: int
    trainable_names: tuple[str, ...]


def bdcn_table_assignment(
    sharing: BDCNSharing,
    *,
    sites: int,
    heads: int,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """為 global、per-site 或 per-head sharing 指派 codebook table IDs。"""

    if sites < 1 or heads < 1:
        raise ValueError("sites and heads must be positive")
    sharing = BDCNSharing(sharing)
    if sharing is BDCNSharing.GLOBAL:
        return tuple((0,) * heads for _ in range(sites)), 1
    if sharing is BDCNSharing.PER_ATTENTION:
        return tuple((site,) * heads for site in range(sites)), sites
    return tuple(tuple(site * heads + head for head in range(heads)) for site in range(sites)), sites * heads


def convert_c2psa(module: C2PSA, config: VariantConfig) -> list[str]:
    """只替換單一 C2PSA 內的 Attention modules，並回傳 relative paths。"""

    converted: list[str] = []
    for block_index, block in enumerate(module.m):
        attention = getattr(block, "attn", None)
        if isinstance(attention, HardwareFriendlyAttention):
            continue
        if not isinstance(attention, Attention):
            raise TypeError(f"C2PSA block {block_index} has unsupported attention {type(attention).__name__}")
        block.attn = HardwareFriendlyAttention.from_ultralytics(attention, config)
        converted.append(f"m.{block_index}.attn")
    return converted


def _set_child_module(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, child_name = path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    setattr(parent, child_name, replacement)


def convert_yolo26_model(
    model: nn.Module,
    config: VariantConfig,
    *,
    expected_paths: tuple[str, ...] | None = YOLO26M_ATTENTION_PATHS,
) -> list[str]:
    """替換全部官方 Attention modules，並拒絕不完整的 YOLO26m conversion。"""

    official = {name: module for name, module in model.named_modules() if isinstance(module, Attention)}
    existing = {
        name for name, module in model.named_modules() if isinstance(module, HardwareFriendlyAttention)
    }
    available = set(official) | existing
    if expected_paths is not None and available != set(expected_paths):
        raise ValueError(f"expected Attention paths {list(expected_paths)}, found {sorted(available)}")
    if not available:
        raise ValueError("model contains no convertible Attention modules")
    paths = sorted(
        available,
        key=lambda item: tuple(int(part) if part.isdigit() else part for part in item.split(".")),
    )
    normalizers: list[nn.Module | None] = [None] * len(paths)
    if config.normalization is NormalizationKind.BDCN:
        modules = {**official, **{path: model.get_submodule(path) for path in existing}}
        head_counts = [modules[path].num_heads for path in paths]
        if len(set(head_counts)) != 1:
            raise ValueError("BDCN sharing currently requires equal head counts at all Attention sites")
        assignments, num_tables = bdcn_table_assignment(
            config.bdcn_sharing,
            sites=len(paths),
            heads=head_counts[0],
        )
        bank = BDCNCodebookBank(
            num_tables=num_tables,
            levels=config.bdcn_levels,
            step=config.resolved_bdcn_step,
            kind=config.bdcn_codebook,
            projection=config.bdcn_projection,
            log_ratio_bound=config.bdcn_log_ratio_bound,
        )
        source_attentions = [modules[path] for path in paths]
        source_bdcn = [
            attention
            for attention in source_attentions
            if isinstance(attention, HardwareFriendlyAttention)
            and attention.config.normalization is NormalizationKind.BDCN
        ]
        if source_bdcn:
            if len(source_bdcn) != len(source_attentions):
                raise ValueError("cannot preserve BDCN state from a partially converted model")
            compatible = all(
                attention.config.bdcn_codebook is config.bdcn_codebook
                and attention.config.bdcn_sharing is config.bdcn_sharing
                and attention.config.bdcn_levels == config.bdcn_levels
                and attention.config.resolved_bdcn_step == config.resolved_bdcn_step
                and attention.config.bdcn_log_ratio_bound == config.bdcn_log_ratio_bound
                for attention in source_bdcn
            )
            if compatible:
                source_bank = source_bdcn[0].normalize.bank
                if any(attention.normalize.bank is not source_bank for attention in source_bdcn[1:]):
                    raise ValueError("existing BDCN Attention sites do not share one codebook bank")
                bank.load_state_dict(source_bank.state_dict(), strict=True)
        normalizers = [
            build_normalizer(
                config,
                bdcn_bank=bank,
                bdcn_table_indices=torch.tensor(indices, dtype=torch.long),
            )
            for indices in assignments
        ]
    converted: list[str] = []
    for path, normalizer in zip(paths, normalizers, strict=True):
        if path in official:
            replacement = HardwareFriendlyAttention.from_ultralytics(
                official[path], config, normalizer=normalizer
            )
        else:
            replacement = HardwareFriendlyAttention.from_hardware(
                model.get_submodule(path), config, normalizer=normalizer
            )
        _set_child_module(model, path, replacement)
        converted.append(path)
    return converted


def fixed_scale_modules(model: nn.Module) -> tuple[HardwareFriendlyAttention, ...]:
    """回傳仍需 zero-shot calibration 的 fixed-scale Attention sites。"""

    return tuple(
        module
        for module in model.modules()
        if isinstance(module, HardwareFriendlyAttention) and module.score.needs_calibration
    )


def freeze_for_stage(model: nn.Module, stage: str) -> TrainableSummary:
    """套用計畫中的 screening／add-on 或 full-model trainable scopes。"""

    normalized = stage.lower().replace("-", "_")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if normalized in {"screening", "normalization", "scale", "bias", "addon"}:
        for module in model.modules():
            if isinstance(module, HardwareFriendlyAttention):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    elif normalized == "bdcn_codebook":
        for module in model.modules():
            if isinstance(module, BDCNCodebookBank):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    elif normalized in {"recovery", "q2", "full"}:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif normalized not in {"baseline", "p0", "ptq", "q0", "q1"}:
        raise ValueError(f"unknown training stage: {stage}")
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    return TrainableSummary(
        stage=normalized,
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_names=names,
    )


def set_progressive_epoch(model: nn.Module, epoch: int) -> None:
    for module in model.modules():
        if isinstance(module, HardwareFriendlyAttention):
            module.set_epoch(epoch)
