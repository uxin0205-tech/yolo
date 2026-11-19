"""僅用小型 CSV 重畫整合研究；不匯入 PyTorch、不讀 checkpoint。"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})


def read(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(HERE / f"{name}.png", dpi=180)
    fig.savefig(HERE / f"{name}.svg")
    plt.close(fig)


def main():
    stages = [r for r in read("stages.csv") if r["stage"] != "5"]
    labels = ["B2", "S1 Q10", "D1 + KD", "Fusion", "qSiLU", "Rep17+20"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.arange(len(stages))
    for key, label in [("coco_AP", "COCO overall"), ("sports_ball_AP", "COCO ball"),
                       ("baseball_bat_AP", "COCO bat")]:
        axes[0].plot(x, [100 * float(r[key]) for r in stages], "o-", label=label)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="AP50-95 (%)",
                title="Selected lineage (not a controlled ablation)")
    axes[0].legend(loc="lower left")
    keys = ["coco_AP", "person_AP", "sports_ball_AP", "baseball_bat_AP", "box_AP50_95",
            "pose_AP50_95", "ball_box_AP50_95", "ball_pose_AP50_95", "bat_box_AP50_95", "bat_pose_AP50_95"]
    names = ["COCO", "person", "COCO ball", "COCO bat", "BBAT box", "BBAT pose",
             "ball box", "ball pose", "bat box", "bat pose"]
    deltas = [100 * (float(stages[-1][k]) - float(stages[-2][k])) for k in keys]
    colors = ["#287b8e" if v >= 0 else "#ba5b3b" for v in deltas]
    axes[1].barh(names, deltas, color=colors)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="AP change (percentage points)", title="Final qSiLU Rep vs qSiLU parent")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, "mainline-accuracy")

    rows = read("shapes.csv")
    parent = next(r for r in rows if r["candidate"] == "reference_rect640")
    adapted = [r for r in rows if r["phase"] == "adapted_best"]
    labels = [r["candidate"].removeprefix("rect").replace("_", "x") for r in adapted]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    x = np.arange(len(adapted))
    metrics = [("coco_AP", "COCO"), ("baseball_bat_AP", "COCO bat"),
               ("box_AP50_95", "BBAT box"), ("pose_AP50_95", "BBAT pose")]
    for i, (key, label) in enumerate(metrics):
        delta = [100 * (float(r[key]) - float(parent[key])) for r in adapted]
        axes[0].bar(x + (i - 1.5) * 0.2, delta, width=0.19, label=label)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="AP change vs original rect640 (pp)",
                title="Each fixed shape: 10E + half-LR 10E", xlabel="Canvas H x W")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].legend(ncol=2, fontsize=9)
    fixed = next(r for r in rows if r["candidate"] == "fixed640")
    mac_ratio = [100 * (float(r["conv_pv_GMAC"]) / float(fixed["conv_pv_GMAC"]) - 1)
                 for r in adapted]
    axes[1].barh(labels, mac_ratio, color="#4e7897")
    for i, value in enumerate(mac_ratio):
        axes[1].text(value + (1 if value >= 0 else -1), i, f"{value:+.2f}%",
                     va="center", ha="left" if value >= 0 else "right", fontsize=9)
    axes[1].set(xlim=(-84, 47), xlabel="Conv + P.V MAC change vs fixed 640x640 (%)",
                title="Analytical subtotal; not measured speedup")
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", lw=0.8)
    for ax in axes:
        ax.grid(axis="y" if ax is axes[0] else "x", alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, "shape-tradeoffs")


if __name__ == "__main__":
    main()
