"""只讀 metrics.csv 重畫最終比較；不需要 torch、GPU 或 checkpoint。

交付版改用共同尺寸列，避免密集散點的文字重疊；實驗數據與選型不變。
"""

import csv
from pathlib import Path


def redraw(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    rows = list(csv.DictReader((folder / "metrics.csv").open()))
    original = next(r for r in rows if r["candidate"] == "reference_rect640")
    direct = {r["candidate"]: r for r in rows if r["phase"] == "e0" and r["conv_pv_GMAC"]}
    adapted = {r["candidate"]: r for r in rows if r["phase"] == "adapted_best"}
    names = sorted(direct, key=lambda n: float(direct[n]["conv_pv_GMAC"]))
    labels = {"fixed640": "640 × 640", "square512": "512 × 512", "square1024": "1024 × 1024"}
    labels.update({n: n.removeprefix("rect").replace("_", " × ") for n in names if n.startswith("rect")})
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    fig.subplots_adjust(left=.12, right=.97, top=.84, bottom=.22, wspace=.24)
    for ax, metric, title in zip(axes[:2], ("coco_AP", "pose_AP50_95"), ("COCO AP50–95 (%)", "BBAT Pose AP50–95 (%)")):
        ax.axvline(100 * float(original[metric]), color="#555555", linestyle="--", linewidth=1.4,
                   label="Original rect640")
        for index, name in enumerate(names):
            start = 100 * float(direct[name][metric])
            ax.scatter(start, index, facecolors="white", edgecolors="#246b94", s=55, zorder=3,
                       label="Direct shape replacement (E0)" if index == 0 else None)
            if name in adapted:
                end = 100 * float(adapted[name][metric])
                ax.plot([start, end], [index, index], color="#9ca3af", linewidth=1.2)
                ax.scatter(end, index, color="#d97706", marker="s", s=38, zorder=4,
                           label="Selected checkpoint after 20 epochs" if index == 0 else None)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=.2)
    costs = [float(direct[n]["conv_pv_GMAC"]) for n in names]
    axes[2].barh(range(len(names)), costs, height=.45, color="#557c96")
    for index, value in enumerate(costs):
        axes[2].text(value + 1.6, index, f"{value:.2f}", va="center", fontsize=9)
    axes[2].set(title="Analytical Conv + P.V GMAC", xlim=(0, max(costs) * 1.19))
    axes[2].grid(axis="x", alpha=.2)
    axes[0].set_yticks(range(len(names)), [labels[n] for n in names])
    axes[0].set_ylabel("Canvas H × W")
    axes[0].invert_yaxis()
    fig.suptitle("Stage9: shape accuracy and analytical cost", fontsize=14, y=.96)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .07), frameon=False)
    fig.text(.5, .025, "Same selected checkpoint for both AP columns. MACs cover both task routes; no RTL / board speedup measured.",
             ha="center", fontsize=9, color="#555555")
    fig.savefig(folder / "accuracy-cost.png", dpi=180)
    fig.savefig(folder / "accuracy-cost.svg")
    plt.close(fig)


if __name__ == "__main__":
    redraw(Path(__file__).resolve().parent)
