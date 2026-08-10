from __future__ import annotations
import json
from pathlib import Path
import yaml
from .variants.definitions import VariantDefinition

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "yolo11m-binaryqk-template.yaml"

def materialize_variant_yaml(variant: VariantDefinition, destination: Path) -> Path:
    spec = yaml.safe_load(TEMPLATE.read_text())
    spec["binaryqk"] = {
        "schema_version": 2,
        "variant": variant.id,
        "config_hash": variant.config_hash,
        "attention": variant.attention_type,
        "qk_mode": variant.qk_mode,
        "base_variant": variant.base_variant,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(spec, sort_keys=False))
    return destination
