"""只讀小型數據重畫；無 torch、checkpoint 或 GPU 依賴。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def redraw(root):
    root = Path(root)
    rows = json.loads((root / "comparison.json").read_text())
    baseline = json.loads((root / "baseline.json").read_text())
    keys = list(baseline)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for ax, key in zip(axes.flat, keys):
        labels = [r["candidate"] + "\n" + r["phase"] for r in rows]
        ax.bar(range(len(rows)), [r[key] for r in rows])
        ax.axhline(baseline[key], color="black", linestyle="--", linewidth=1)
        ax.set_title(key)
        ax.set_xticks(range(len(rows)), labels, rotation=60, fontsize=6)
        ax.set_ylim(
            max(0, min(r[key] for r in rows) - 0.02),
            min(1, max(baseline[key], max(r[key] for r in rows)) + 0.02),
        )
    fig.savefig(root / "comparison.png", dpi=150)
    plt.close(fig)
    sensitivity = json.loads((root / "sensitivity.json").read_text())
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    matrix = [[r["metrics"][k] - baseline[k] for k in keys] for r in sensitivity]
    im = ax.imshow(matrix, cmap="RdBu", vmin=-0.05, vmax=0.05, aspect="auto")
    ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(
        range(len(sensitivity)),
        [r["candidate"] + "/" + r["region"] for r in sensitivity],
        fontsize=8,
    )
    fig.colorbar(im, ax=ax, label="AP delta from E1 (full validation)")
    fig.savefig(root / "sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    redraw(Path(__file__).resolve().parent)
