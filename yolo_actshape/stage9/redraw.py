"""只讀 metrics.csv 即可重畫，不需要 torch、GPU 或 checkpoint。"""

import csv
from pathlib import Path


def redraw(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    rows = list(csv.DictReader((folder / "metrics.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for row in rows:
        if not row["conv_pv_GMAC"]:
            continue
        x = float(row["conv_pv_GMAC"])
        label = row["candidate"] + "/" + row["phase"]
        for ax, metric in zip(axes, ("coco_AP", "pose_AP50_95")):
            ax.scatter(x, float(row[metric]) * 100)
            ax.annotate(label, (x, float(row[metric]) * 100), fontsize=7)
            ax.set(xlabel="Conv + P.V estimated GMAC (both routing)", ylabel=metric + " (%)")
            ax.grid(alpha=.2)
    fig.savefig(folder / "accuracy-cost.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    redraw(Path(__file__).resolve().parent)
