"""只讀小型 JSON 重畫 Stage6 曲線，不需要 GPU 或 checkpoint。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.history.read_text())
    ref = json.loads(args.reference.read_text())
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    epoch = [r["epoch"] for r in rows]
    series = [
        ("coco_AP", "COCO Detect", ref["detect_expected"]["AP50_95"]),
        ("box_AP50_95", "BBAT box", ref["pose_expected"]["box_AP50_95"]),
        ("pose_AP50_95", "BBAT pose", ref["pose_expected"]["pose_AP50_95"]),
    ]
    for ax, (key, title, baseline) in zip(axes, series, strict=True):
        ax.plot(
            epoch,
            [r["metrics"][key] for r in rows],
            marker="o",
            markersize=3,
            label="融合適應",
        )
        ax.axhline(baseline, color="#6b7280", linestyle="--", label="原獨立模型")
        if key != "coco_AP":
            ax.axhline(
                baseline - 0.02, color="#c97d0b", linestyle=":", label="最終允許下限"
            )
        best = max(rows, key=lambda r: r["score"])
        ax.scatter(
            [best["epoch"]],
            [best["metrics"][key]],
            color="#b91c1c",
            zorder=3,
            label=f"綜合選定 E{best['epoch']}",
        )
        ax.set(title=title, xlabel="Epoch", ylabel="AP50–95")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("第六階段：J0 僅訓練 Pose head（單一 seed）")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
