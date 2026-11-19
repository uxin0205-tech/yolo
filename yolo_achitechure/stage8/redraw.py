"""只讀小型 CSV／JSON，無 GPU 或 checkpoint 可重畫結果。"""

import csv
import json
from pathlib import Path


def draw(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    with (root / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    keys = (
        "coco_AP",
        "person_AP",
        "sports_ball_AP",
        "baseball_bat_AP",
        "box_AP50_95",
        "pose_AP50_95",
        "ball_box_AP50_95",
        "ball_pose_AP50_95",
        "bat_box_AP50_95",
        "bat_pose_AP50_95",
    )
    fig, axes = plt.subplots(2, 5, figsize=(17, 6), constrained_layout=True)
    for key, ax in zip(keys, axes.flat, strict=True):
        ax.bar([r["candidate"] for r in rows], [float(r[key]) for r in rows])
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(bottom=max(0, min(float(r[key]) for r in rows) - 0.02))
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(root / "accuracy.png", dpi=150)
    fig.savefig(root / "accuracy.svg")
    plt.close(fig)
    curves = json.loads((root / "curves.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for name, segments in curves.items():
        ys = [r["balanced_score"] for segment in segments for r in segment]
        ax.plot(range(1, len(ys) + 1), ys, marker="o", label=name)
    ax.set(xlabel="Cumulative training epochs", ylabel="Balanced validation score")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(root / "curves.svg")
    plt.close(fig)


if __name__ == "__main__":
    draw(Path(__file__).resolve().parent)
