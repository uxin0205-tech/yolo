"""最終 Bit-True winner 的 checkpoint export contracts。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .attention import HardwareFriendlyAttention
from .integration import YOLO26M_ATTENTION_PATHS


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes())
    return digest.hexdigest()


def export_attention_state(
    model: nn.Module, destination: str | Path, *, parent_sha256: str, final_sha256: str
) -> Path:
    sites = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, HardwareFriendlyAttention)
    }
    if set(sites) != set(YOLO26M_ATTENTION_PATHS):
        raise ValueError("final model does not contain exactly the two required Attention sites")
    payload = {
        "schema_version": 1,
        "sites": {name: module.state_dict() for name, module in sites.items()},
        "site_paths": YOLO26M_ATTENTION_PATHS,
        "pwl": {
            "score_format": "Q8.8",
            "range": [-10.0, 0.0],
            "segments": 20,
            "delta": 0.5,
            "endpoint_format": "UQ1.15",
            "endpoints": 21,
            "bits": 336,
        },
        "parent_sha256": parent_sha256,
        "final_sha256": final_sha256,
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
