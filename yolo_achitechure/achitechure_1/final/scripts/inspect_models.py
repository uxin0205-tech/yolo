#!/usr/bin/env python3
"""在 CPU 上驗證所有交付權重的雜湊、架構與 attention backend。"""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path

import torch

from _bundle import BUNDLE_ROOT, atomic_json, file_sha256, load_models


def state_dict_sha256(model: torch.nn.Module) -> str:
    """計算排除 checkpoint metadata 的模型參數 fingerprint。"""

    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        value = tensor.detach().contiguous().cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def inspect_checkpoint(
    record: dict[str, object],
    *,
    path_key: str,
    sha_key: str,
    expected_backends: list[str],
) -> dict[str, object]:
    """載入並稽核 registry 中的一個 checkpoint。"""

    from ultralytics import YOLO

    from achitechure_1.masf import P3MASFFull35, P3MASFPartial75
    from achitechure_1.model import inspect_yolo26_graph

    checkpoint = BUNDLE_ROOT / str(record[path_key])
    actual_file_sha = file_sha256(checkpoint)
    if actual_file_sha != record[sha_key]:
        raise RuntimeError(f"{record['id']} {path_key} 檔案 SHA256 不符")
    raw_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(raw_checkpoint, dict):
        state_model = raw_checkpoint.get("ema") or raw_checkpoint.get("model")
    else:
        state_model = raw_checkpoint
    if state_model is None:
        raise RuntimeError(f"{record['id']} {path_key} checkpoint 缺少 model state")
    actual_state_sha = state_dict_sha256(state_model)
    if path_key == "bittrue" and actual_state_sha != record["state_dict_sha256"]:
        raise RuntimeError(f"{record['id']} state-dict SHA256 不符")
    del state_model, raw_checkpoint
    gc.collect()
    yolo = YOLO(str(checkpoint))
    model = yolo.model
    graph = inspect_yolo26_graph(model)
    masf = getattr(model.model[graph.p3_index], "p3_masf", None)
    if isinstance(masf, P3MASFFull35):
        architecture = "full35"
    elif isinstance(masf, P3MASFPartial75):
        architecture = "partial75"
    elif masf is None:
        architecture = "baseline-attention"
    else:
        raise TypeError(f"{record['id']} 含未知 P3 MASF：{type(masf).__name__}")
    if architecture != record["architecture"]:
        raise RuntimeError(f"{record['id']} 架構不符：{architecture}")
    backends = [
        module.config.normalization.value
        for module in model.modules()
        if module.__class__.__name__ == "HardwareFriendlyAttention"
    ]
    if backends != expected_backends:
        raise RuntimeError(f"{record['id']} {path_key} attention backend 不符：{backends}")
    return {
        "kind": path_key,
        "architecture": architecture,
        "checkpoint": str(checkpoint),
        "file_sha256": actual_file_sha,
        "state_dict_sha256": actual_state_sha,
        "attention_backends": backends,
        "p3_index": graph.p3_index,
        "alpha": None if masf is None else float(masf.alpha.detach().cpu()),
    }


def inspect_one(record: dict[str, object]) -> dict[str, object]:
    """稽核候選的 Bit-True 與可用 Float checkpoint。"""

    bittrue = inspect_checkpoint(
        record,
        path_key="bittrue",
        sha_key="bittrue_sha256",
        expected_backends=["bit_true_pwl", "bit_true_pwl"],
    )
    float_checkpoint = None
    if record["float"] is not None:
        float_checkpoint = inspect_checkpoint(
            record,
            path_key="float",
            sha_key="float_sha256",
            expected_backends=["piecewise_linear", "piecewise_linear"],
        )
    return {"id": record["id"], "bittrue": bittrue, "float": float_checkpoint}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [inspect_one(record) for record in load_models()]
    payload = {"status": "passed", "models": reports}
    if args.output:
        atomic_json(args.output, payload)
    for report in reports:
        print(
            f"PASS {report['id']}: {report['bittrue']['architecture']} "
            f"Bit-True {str(report['bittrue']['file_sha256'])[:12]}"
            + (" + Float" if report["float"] else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
