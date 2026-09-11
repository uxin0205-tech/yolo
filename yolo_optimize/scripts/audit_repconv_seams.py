#!/usr/bin/env python3
"""驗證 YOLO26 Neck RepConv 候選層與結構重參數化等價性。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

YOLO_ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_ROOT = YOLO_ROOT / "yolo_p2"
MODEL_YAML = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml"
EXPECTED_SEAMS = {
    17: [-1, 1, "Conv", [256, 3, 2]],
    20: [-1, 1, "Conv", [512, 3, 2]],
}
MAX_ABS_TOLERANCE = 1e-5


def load_graph() -> list[list[object]]:
    if not MODEL_YAML.is_file():
        raise FileNotFoundError(f"找不到 YOLO26 YAML：{MODEL_YAML}")
    with MODEL_YAML.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return [*config["backbone"], *config["head"]]


def audit_yaml_seams() -> None:
    graph = load_graph()
    for index, expected in EXPECTED_SEAMS.items():
        actual = graph[index]
        if actual != expected:
            raise ValueError(f"model.{index} 不符合預期：{actual!r} != {expected!r}")
        print(f"[PASS] model.{index}：{actual}")


def audit_identity_safe_transfer() -> None:
    sys.path.insert(0, str(ULTRALYTICS_ROOT))
    from ultralytics.nn.modules.conv import Conv, RepConv

    torch.manual_seed(0)
    baseline = Conv(16, 16, 3, 2).eval()
    candidate = RepConv(16, 16, 3, 2, bn=False).eval()
    candidate.conv1.load_state_dict(baseline.state_dict())
    candidate.conv2.bn.weight.data.zero_()
    candidate.conv2.bn.bias.data.zero_()

    sample = torch.randn(2, 16, 17, 17)
    reference = baseline(sample)
    transferred = candidate(sample)
    transfer_error = (reference - transferred).abs().max().detach().item()
    if transfer_error > MAX_ABS_TOLERANCE:
        raise ValueError(f"identity-safe transfer 誤差過大：{transfer_error}")

    candidate.fuse_convs()
    fused = candidate.forward_fuse(sample)
    fusion_error = (reference - fused).abs().max().detach().item()
    if fusion_error > MAX_ABS_TOLERANCE:
        raise ValueError(f"RepConv 融合誤差過大：{fusion_error}")

    print(f"[PASS] identity-safe transfer max_abs={transfer_error:.3e}")
    print(f"[PASS] fused output max_abs={fusion_error:.3e}")

    total_extra = 0
    for channels in (256, 512):
        original = Conv(channels, channels, 3, 2)
        repconv = RepConv(channels, channels, 3, 2, bn=False)
        original_params = sum(parameter.numel() for parameter in original.parameters())
        repconv_params = sum(parameter.numel() for parameter in repconv.parameters())
        extra = repconv_params - original_params
        total_extra += extra
        if repconv.bn is not None:
            raise ValueError(
                f"stride-2 model seam 不應有 identity branch：channels={channels}"
            )
        print(
            f"[INFO] channels={channels}：訓練期相對原 Conv 增加 {extra:,} parameters，"
            "identity branch=off"
        )
    print(f"[INFO] model.17 + model.20 訓練期合計增加 {total_extra:,} parameters")


def main() -> int:
    audit_yaml_seams()
    audit_identity_safe_transfer()
    print("[GREEN] RepConv 局部候選層與等價初始化／融合檢查通過")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"[RED] {error}", file=sys.stderr)
        raise SystemExit(1) from error
