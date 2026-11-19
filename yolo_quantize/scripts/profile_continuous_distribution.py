"""讀取鎖定parent的原EMA及export，產生148路尺度正規化分布；不建圖、不用GPU。"""

import json
from pathlib import Path

import torch

from yolo_quantize.progressive_preparation import LockedQATParentSpec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/queues/full-model-continuous-0907"


def main():
    torch.set_num_threads(4)
    parent = LockedQATParentSpec.from_yaml(OUT / "parent-manifest.json")
    ema = torch.load(
        parent.full_resume_checkpoint, map_location="cpu", weights_only=True
    )["ema_state"]
    export = torch.load(
        parent.inference_checkpoint, map_location="cpu", weights_only=True
    )["state_dict"]
    graph = json.loads((parent.completion_path.parent / "qat-graph.json").read_text())
    if graph["plan_sha256"] != parent.plan_sha256:
        raise ValueError("distribution graph parent mismatch")
    rows = []
    boundaries = torch.tensor([0, 0.1, 0.25, 0.5, 1, 2, 4, 8, float("inf")])
    for site in graph["weight_sites"]:
        for view, state in (
            ("ema_shadow_before_projection", ema),
            ("deployed_export", export),
        ):
            weight = (
                state[site["path"] + ".weight"]
                .float()
                .reshape(state[site["path"] + ".weight"].shape[0], -1)
            )
            assert torch.isfinite(weight).all()
            energy = weight.square()
            rms = energy.mean(dim=1, keepdim=True).sqrt()
            normalized = weight.abs() / rms.clamp_min(1e-12)
            bins = torch.bucketize(normalized.flatten(), boundaries[1:-1], right=True)
            histogram = torch.bincount(bins, minlength=8)
            threshold = 0.7 * weight.abs().mean()
            discarded = weight.abs() <= threshold
            retained = weight.abs()[~discarded]
            alpha = retained.mean() if retained.numel() else torch.tensor(0.0)
            residual = (
                (retained - alpha).square().sum()
                if retained.numel()
                else torch.tensor(0.0)
            )
            total = energy.sum().clamp_min(1e-24)
            rows.append(
                {
                    "path": site["path"],
                    "region": site["region"],
                    "view": view,
                    "elements": weight.numel(),
                    "current_format": site["format_id"],
                    "exact_zero_ratio": float((weight == 0).float().mean()),
                    "near_zero_0p1_channel_rms": float(
                        (normalized <= 0.1).float().mean()
                    ),
                    "near_zero_0p25_channel_rms": float(
                        (normalized <= 0.25).float().mean()
                    ),
                    "normalized_abs_p50_p90_p99": torch.quantile(
                        normalized.flatten(), torch.tensor([0.5, 0.9, 0.99])
                    ).tolist(),
                    "normalized_abs_max": float(normalized.max()),
                    "normalized_abs_histogram": histogram.tolist(),
                    "paper_threshold_0p7_mean_abs": float(threshold),
                    "paper_discarded_ratio": float(discarded.float().mean()),
                    "paper_discarded_energy_fraction": float(
                        energy[discarded].sum() / total
                    ),
                    "paper_retained_amplitude_error_fraction": float(residual / total),
                    "diagnostic_only_not_map": True,
                }
            )
    assert len(rows) == 296 and len({r["path"] for r in rows}) == 148
    payload = {
        "schema_version": 1,
        "parent_manifest_sha256": parent.config_sha256,
        "inference_sha256": parent.inference_sha256,
        "full_resume_sha256": parent.full_resume_sha256,
        "gpu_used": False,
        "histogram_interval": "[low,high), abs(weight)/per_output_channel_RMS",
        "histogram_boundaries": [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, "inf"],
        "rows": rows,
    }
    destination = OUT / "weight-distribution.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps({"status": "completed", "paths": 148, "views": 2, "rows": len(rows)})
    )


if __name__ == "__main__":
    main()
