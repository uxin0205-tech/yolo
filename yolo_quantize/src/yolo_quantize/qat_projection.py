"""Replay materialized parent weights from checkpoint EMA on an explicit device."""

import torch

from .qat_weights import TrainableWeightFakeQuantizer
from .weight_quantization import FixedSD4WeightSpec, UniformWeightSpec


def replay_parent_state(export_state, ema_state, sites, *, device="cpu"):
    result = {key: ema_state[key].detach().cpu().clone() for key in export_state}
    for site in sites:
        path = site["path"]
        if site["format_id"] == "fixed-sd4":
            spec = FixedSD4WeightSpec(scale_method="optimal_scaled_codebook")
        elif site["format_id"] in {"w4", "w5", "w6", "w7", "w8"}:
            spec = UniformWeightSpec(site["encoded_bits"], "optimal_scaled_codebook")
            if site["format_id"] != f"w{spec.bits}":
                raise ValueError("parent format bit declaration differs")
        else:
            raise ValueError(f"unsupported parent replay format: {site['format_id']}")
        raw = ema_state[f"{path}.weight_quantizer._scale_unconstrained"].to(device)
        quantizer = TrainableWeightFakeQuantizer(
            spec, initial_scales=torch.ones_like(raw)
        ).to(device)
        with torch.no_grad():
            quantizer.scale_parameter.copy_(raw)
            quantizer.set_blend_ratio(1.0)
            result[f"{path}.weight"] = quantizer(
                ema_state[f"{path}.weight"].to(device)
            ).cpu()
    return result
