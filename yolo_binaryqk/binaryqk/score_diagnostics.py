"""沿用既有 128 張校準影像，分開觀察固定 scale 的 score／P／PV 誤差。"""

import argparse
import json
import math
from pathlib import Path

import torch

from .basis import BASES, BranchScore, calibration_moments
from .calibration import COCO, image_tensor, read_calibration
from .model import ROOT, SITES, load_parent, sha256
from .protocol import verify_reference_review
from .run import require_idle_gpu, seed, write_json


def run(output):
    verify_reference_review()
    calibration_path = ROOT / "yolo_binaryqk/artifacts/stage1/calibration.json"
    calibration = read_calibration(calibration_path)
    if output.exists():
        raise FileExistsError(output)
    require_idle_gpu()
    seed()
    model = load_parent().to("cuda:0")
    indices = {"sign": 0, "residual": 1, "hadamard": 2}
    totals, calls, shapes = {}, {}, {}

    def observer(site):
        def hook(module, args, value):
            height, width = value.shape[-2:]
            q, k, v = value.reshape(1, 4, 128, height * width).split((32, 32, 64), 2)
            native_score = (q * 32**-.5).transpose(-2, -1) @ k
            native_p = native_score.softmax(-1)
            native_pv = v @ native_p.transpose(-2, -1)
            moments = calibration_moments(q, k)
            calls[site] = calls.get(site, 0) + 1
            shapes[site] = [height, width]
            for variant, bases in BASES.items():
                dynamic = moments[:, [indices[b] for b in bases]] / len(bases)
                for label, coefficients in (("fixed", calibration["coefficients"][variant][site]),
                                             ("per_image_magnitude_reference", dynamic)):
                    score = BranchScore(variant, coefficients).cuda().eval()(q, k)
                    p = score.softmax(-1)
                    pv = v @ p.transpose(-2, -1)
                    values = {"score_rmse": float((score - native_score).square().mean().sqrt()),
                              "probability_mae": float((p - native_p).abs().mean()),
                              "pv_relative_rmse": float((pv - native_pv).square().mean().sqrt()
                                                        / native_pv.square().mean().sqrt().clamp_min(1e-12)),
                              "score_std_ratio": float(score.std() / native_score.std().clamp_min(1e-12)),
                              "entropy": float(-(p * p.clamp_min(1e-30).log()).sum(-1).mean()),
                              "row_sum_max_error": float((p.sum(-1) - 1).abs().max())}
                    if not all(math.isfinite(n) for n in values.values()):
                        raise FloatingPointError("nonfinite feature diagnosis")
                    key = f"{site}/{variant}/{label}"
                    previous = totals.setdefault(key, {k: 0. for k in values})
                    for name, number in values.items():
                        previous[name] += number
        return hook

    handles = [model.get_submodule(site).qkv.register_forward_hook(observer(site)) for site in SITES]
    try:
        with torch.inference_mode():
            for index, entry in enumerate(calibration["images"]):
                path = COCO / entry["path"]
                if sha256(path) != entry["sha256"]:
                    raise ValueError("calibration image changed")
                model(image_tensor(path, 640).cuda())
                if (index + 1) % 32 == 0:
                    print(f"score diagnostic {index + 1}/{calibration['count']}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if calls != {site: calibration["count"] for site in SITES}:
        raise AssertionError("not every site observed on every image")
    write_json(output, {"kind": "real-feature-numerical-diagnosis-not-task-AP", "images": calibration["count"],
               "calibration_sha256": sha256(calibration_path), "shapes": shapes,
               "means_over_images": {k: {name: value / calibration["count"] for name, value in row.items()}
                                     for k, row in totals.items()},
               "interpretation": "逐影像幅度僅作固定係數診斷；不是新候選，亦非舊 T5 token-wise scale 的完整重現。沒有更新權重。"})
    print(json.dumps({"images": calibration["count"], "sites": calls, "status": "finite"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    torch.set_num_threads(2)
    run(parser.parse_args().output)
