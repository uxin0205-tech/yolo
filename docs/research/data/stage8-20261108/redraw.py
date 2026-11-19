"""從本資料夾 snapshot.json 重畫；不需 GPU、模型或研究程式。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "snapshot.json").read_text())
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK TC", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    labels = ["COCO\n整體", "COCO\nperson", "COCO\n球", "COCO\n球棒",
              "BBAT\n整體 box", "BBAT\n整體 Pose", "BBAT\n球 box", "BBAT\n球 Pose",
              "BBAT\n球棒 box", "BBAT\n球棒 Pose"]
    rows = [row for row in data["records"] if row["metrics"]]
    values = np.array([[row["delta_parent_pp"][key] for key in data["metric_keys"]] for row in rows])
    limit = max(.1, float(np.abs(values).max()))
    fig, ax = plt.subplots(figsize=(14, max(6, .65 * len(rows))), layout="constrained")
    im = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    for i in range(len(rows)):
        for j in range(len(labels)):
            ax.text(j, i, f"{values[i,j]:+.3f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(values[i,j]) > .75 * limit else "black")
    pending = "、".join(row["label"] for row in rows if row["group"] == "new" and not row["deploy_checkpoint_available"])
    ax.set_title("相對無 Rep qSiLU E8 的 AP 差異（百分點；PolyQuality 為不同 activation）\n" +
                 (f"{pending} 尚待最終驗證" if pending else "候選部署驗證已完成"))
    fig.colorbar(im, ax=ax, label="AP 差異（百分點；正值較高）", shrink=.8)
    for ext in ["png", "svg"]:
        fig.savefig(root / f"accuracy-delta.{ext}", dpi=170)
    plt.close(fig)

    adaptations = [row for row in rows if row["group"] == "adaptation"]
    if adaptations:
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        for row in adaptations:
            parent = next(r for r in rows if r["id"] == row["parent_id"])
            epochs = [e for e in data["epochs"] if e["candidate"] == row["id"]]
            ax.plot([0] + [e["epoch"] for e in epochs],
                    [100*parent["balanced_score"]] + [100*e["balanced_score"] for e in epochs],
                    marker="o", label=row["label"])
            ax.axhline(100*parent["balanced_score"], linestyle="--", color="gray", label="適應前 SiLU E1 部署")
        ax.set(xlabel="硬體適應 epoch（0 為接續起點）", ylabel="平衡分數 × 100",
               title="硬體適應由 SiLU E1 接續，與獨立候選分開比較", xticks=[0,1,2,3])
        ax.grid(alpha=.2)
        ax.legend()
        for ext in ["png", "svg"]:
            fig.savefig(root / f"hardware-adaptation.{ext}", dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for row in data["records"]:
        if row["group"] != "new" or not row["metrics"]:
            continue
        epochs = [e for e in data["epochs"] if e["candidate"] == row["id"]]
        ax.plot([e["epoch"] for e in epochs], [100*e["balanced_score"] for e in epochs],
                marker="o", label=row["label"])
    for ident, style in [("parent", ":"), ("old_rep17_20", "--")]:
        row = next(r for r in data["records"] if r["id"] == ident)
        ax.axhline(100*row["balanced_score"], linestyle=style, color="gray", label=row["label"])
    ax.set(xlabel="已完成 epoch", ylabel="事前定義的平衡分數 × 100",
           title="新輪逐輪結果：分數之外仍須逐項 AP 過門檻", xticks=[1,2,3,4,5])
    ax.grid(alpha=.2)
    ax.legend(ncol=2, fontsize=9)
    for ext in ["png", "svg"]:
        fig.savefig(root / f"curves.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
