"""Strict checkpoint manifests and reload validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn

from ..attention.base import BinaryAttention, UltralyticsBinaryAttention
from ..model import ATTENTION_CLASSES
from ..variants.definitions import VariantDefinition, quantization_contract


@dataclass(frozen=True)
class CheckpointManifest:
    experiment_schema_version: int
    variant_id: str
    attention_class: str
    attention_module_paths: list[str]
    attention_module_count: int
    variant_config_hash: str
    model_yaml_hash: str
    source_checkpoint_hash: str
    binary_forward_count: int
    quantizer_types: dict[str, str]
    distillation_enabled: bool
    distillation_type: str | None
    bias_type: str
    p_bits: int | None
    v_bits: int | None
    magnitude_bits: int | None = None
    quantization_contract: dict[str, str] = field(default_factory=dict)


def sha256_file(path: Path | None) -> str:
    if path is None or not path.exists():
        return "unavailable"
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (BinaryAttention, UltralyticsBinaryAttention))
    ]


def concrete_attention_class(variant: VariantDefinition) -> str:
    if variant.qk_mode == "fp":
        return "Attention"
    return ATTENTION_CLASSES[variant.attention_type].__name__


def build_manifest(
    model: nn.Module,
    variant: VariantDefinition,
    model_yaml: Path | None = None,
    source_checkpoint: Path | None = None,
) -> CheckpointManifest:
    modules = binary_modules(model)
    actual_classes = {type(module).__name__ for _, module in modules}
    expected_class = concrete_attention_class(variant)
    compatible_classes = {expected_class, variant.attention_type, "UltralyticsBinaryAttention"}
    if variant.qk_mode != "fp" and not actual_classes.issubset(compatible_classes):
        raise RuntimeError(f"expected concrete class compatible with {expected_class}, found {sorted(actual_classes)}")
    if variant.qk_mode == "fp" and modules:
        raise RuntimeError("FP control must not contain a binary attention module")
    quantizers = {
        name: ",".join(
            part
            for part in (
                f"P{module.p_bits}" if getattr(module, "p_bits", None) else None,
                f"V{module.v_bits}" if getattr(module, "v_bits", None) else None,
                f"M{module.magnitude_bits}" if getattr(module, "magnitude_bits", None) else None,
            )
            if part is not None
        )
        or "none"
        for name, module in modules
    }
    return CheckpointManifest(
        experiment_schema_version=3,
        variant_id=variant.id,
        attention_class=next(iter(actual_classes)) if actual_classes else expected_class,
        attention_module_paths=[name for name, _ in modules],
        attention_module_count=len(modules),
        variant_config_hash=variant.config_hash,
        model_yaml_hash=sha256_file(model_yaml),
        source_checkpoint_hash=sha256_file(source_checkpoint),
        binary_forward_count=sum(int(module.binary_forward_count.item()) for _, module in modules),
        quantizer_types=quantizers,
        distillation_enabled=variant.use_distillation,
        distillation_type=variant.distillation_type,
        bias_type=variant.bias_type,
        p_bits=variant.p_bits,
        v_bits=variant.v_bits,
        magnitude_bits=variant.magnitude_bits,
        quantization_contract=quantization_contract(variant),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    variant: VariantDefinition,
    **manifest_kwargs,
) -> CheckpointManifest:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(model, variant, **manifest_kwargs)
    torch.save({"state_dict": model.state_dict(), "manifest": asdict(manifest)}, path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def strict_reload(path: Path, model: nn.Module, variant: VariantDefinition) -> CheckpointManifest:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("manifest"), dict):
        raise RuntimeError("checkpoint has no strict BinaryAttention manifest")
    manifest = CheckpointManifest(**payload["manifest"])
    if manifest.variant_id != variant.id or manifest.variant_config_hash != variant.config_hash:
        raise RuntimeError("checkpoint variant/config hash mismatch")
    # strict=True is intentional: missing or unexpected architecture keys are
    # a hard error, never a warning hidden by a partial reload.
    model.load_state_dict(payload["state_dict"], strict=True)
    current = build_manifest(model, variant)
    if current.attention_module_paths != manifest.attention_module_paths:
        raise RuntimeError("checkpoint attention module path mismatch")
    if current.attention_class != manifest.attention_class:
        raise RuntimeError("checkpoint concrete attention class mismatch")
    if current.attention_module_count != manifest.attention_module_count:
        raise RuntimeError("checkpoint attention module count mismatch")
    if current.quantizer_types != manifest.quantizer_types:
        raise RuntimeError("checkpoint quantizer/class fingerprint mismatch")
    if current.quantization_contract != manifest.quantization_contract:
        raise RuntimeError("checkpoint quantization contract mismatch")
    if current.variant_config_hash != manifest.variant_config_hash:
        raise RuntimeError("checkpoint config hash mismatch after reload")
    return manifest
