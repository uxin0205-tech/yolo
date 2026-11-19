"""只由 CSV 重畫第六階段圖表；不依賴 GPU、checkpoint 或 YOLO。"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(figure, output, name):
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output or root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Serif CJK JP"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})
    history = rows(root / "history.csv")
    phases = list(dict.fromkeys(row["phase"] for row in history))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = (
        ("coco_AP", "官方 COCO 整體"),
        ("baseball_bat_AP", "官方 COCO 球棒"),
        ("box_AP50_95", "BBAT 框"),
        ("pose_AP50_95", "BBAT 關鍵點"),
    )
    colors = plt.get_cmap("tab10").colors
    for ax, (metric, label) in zip(axes.flat, metrics):
        offset = 0
        for index, phase in enumerate(phases):
            group = [row for row in history if row["phase"] == phase]
            xs = [offset + int(row["epoch"]) for row in group]
            ys = [100 * float(row[metric]) for row in group]
            ax.plot(xs, ys, color=colors[index], marker=".", ms=3, label=phase)
            if phase == "P":
                ax.axvspan(offset + 0.5, max(xs) + 0.5, color="gray", alpha=0.08)
            offset = max(xs)
            ax.axvline(offset + 0.5, color="gray", alpha=0.2, lw=0.8)
        ax.set_title(label)
        ax.set_ylabel("AP50–95（%）")
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("各段已完成 epoch 的順序（每段資料／步數不同）")
    axes[0, 0].legend(ncol=4, loc="lower left", fontsize=9)
    fig.suptitle("第 6 階段：各段訓練結果，分段顯示")
    fig.text(
        0.5,
        0.015,
        "P 灰底為獨立副本診斷；段落由所選 parent 接續，並非前段最後一輪。不同段落不連線。",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    save(fig, output, "history")

    comparison = rows(root / "comparison.csv")
    labels = {
        "COCO internal": "COCO 整體（內部）",
        "person internal": "COCO person（內部）",
        "coco_AP": "COCO 整體（官方）",
        "person_AP": "COCO person（官方）",
        "sports_ball_AP": "COCO 球（官方）",
        "baseball_bat_AP": "COCO 球棒（官方）",
        "box_AP50_95": "BBAT 框",
        "pose_AP50_95": "BBAT 關鍵點",
        "ball_box_AP50_95": "球框",
        "ball_pose_AP50_95": "球關鍵點",
        "bat_box_AP50_95": "球棒框",
        "bat_pose_AP50_95": "球棒關鍵點",
    }
    values = [100 * float(row["delta"]) for row in comparison]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [labels[row["metric"]] for row in comparison],
        values,
        color=["#287c8e" if value >= 0 else "#b26832" for value in values],
    )
    for index, value in enumerate(values):
        ax.text(
            value + (0.025 if value >= 0 else -0.025),
            index,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
        )
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(min(values) - 0.22, max(values) + 0.22)
    ax.set_xlabel("本輪選定 − optimize 最終 Rep17＋20 E5（AP 百分點）")
    ax.set_title("相同 evaluator／batch 8／資料順序；架構仍不同，單 seed")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save(fig, output, "comparison")
    print(f"已輸出 history／comparison 的 PNG 與 SVG：{output}")


if __name__ == "__main__":
    main()
