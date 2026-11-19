"""完整重載驗證及可離線重畫的小型成果；不宣稱硬體實測加速。"""

import csv
import os
from pathlib import Path

from . import checkpoint
from .common import PROJECT, read, sha256, write_json
from .evaluate import accepted, expected, pair
from .models import manifest_for
from .verification import routing


def plot(csv_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(csv_path).open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, key in zip(axes, ("coco_AP", "box_AP50_95", "pose_AP50_95")):
        for phase in dict.fromkeys(r["phase"] for r in rows):
            selected = [r for r in rows if r["phase"] == phase]
            axis.plot(
                [int(r["epoch"]) for r in selected],
                [float(r[key]) for r in selected],
                label=phase,
            )
        axis.set(title=key, xlabel="Epoch within phase", ylabel="AP")
        axis.grid(alpha=0.3)
    if rows:
        axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def finalize(output, parent, root):
    output = Path(output)
    if sha256(parent["checkpoint"]) != parent["sha256"]:
        raise ValueError("selected checksum changed")
    model = checkpoint.load(parent["checkpoint"]).cuda()
    route = routing(model)
    metrics = pair(model, output / "reload-evaluation", progress_root=output)["metrics"]
    if metrics != parent["metrics"]:
        raise ValueError("selected model AP differs after independent reload")
    manifest = manifest_for(model)
    total = sum(p.numel() for p in model.parameters())
    result = {
        "status": "complete",
        "accepted": accepted(metrics),
        "selected": parent,
        "metrics": metrics,
        "independent_references": expected(),
        "manifest": manifest,
        "routing": route,
        "checkpoint_bytes": Path(parent["checkpoint"]).stat().st_size,
        "parameters": total,
        "independent_parameters": manifest["independent_parameters"],
        "hardware_synthesis": "未執行",
        "board_measurement": "未執行",
        "QKV_PV_projection": "浮點；只有 QK scores、PWL exp 與 reciprocal 沿用既定部署契約",
        "next_stage_started": False,
    }
    result_dir = PROJECT / "result/stage6"
    result_dir.mkdir(parents=True, exist_ok=True)
    # A single selected weight, no copy of every candidate or second archive.
    source = Path(parent["checkpoint"])
    for suffix in (".pt", ".json"):
        target = result_dir / ("selected" + suffix)
        if target.exists() or target.is_symlink():
            if (
                not target.is_symlink()
                or target.resolve() != source.with_suffix(suffix).resolve()
            ):
                raise ValueError("refusing to overwrite a different handoff")
        else:
            target.symlink_to(
                os.path.relpath(source.with_suffix(suffix), target.parent)
            )
    checkpoint.load(result_dir / "selected.pt")
    write_json(result_dir / "handoff.json", result)
    write_json(result_dir / "manifest.json", manifest)
    rows = []
    for path in sorted(root.glob("*/history.json")):
        for row in read(path):
            rows.append(
                {"phase": path.parent.name, "epoch": row["epoch"], **row["metrics"]}
            )
    csv_path = result_dir / "history.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["phase", "epoch", *metrics])
        writer.writeheader()
        writer.writerows(rows)
    plot(csv_path, result_dir / "curves.png")
    (
        result_dir / "redraw.py"
    ).write_text('''"""只讀 CSV；不需要 GPU、checkpoint 或 YOLO。"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p = Path(__file__).resolve().parent
with (p / "history.csv").open() as f:
    rows = list(csv.DictReader(f))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, key in zip(axes, ("coco_AP", "box_AP50_95", "pose_AP50_95")):
    for phase in dict.fromkeys(r["phase"] for r in rows):
        part = [r for r in rows if r["phase"] == phase]
        ax.plot([int(r["epoch"]) for r in part], [float(r[key]) for r in part], label=phase)
    ax.set(title=key, xlabel="Epoch within phase", ylabel="AP")
    ax.grid(alpha=.3)
if rows:
    axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(p / "curves.png", dpi=160)
''')
    lines = [
        "# 第 6 階段重建成果",
        "",
        "完整重新載入 AP 與選定 epoch 相同，單任務、雙任務路由一致。",
        "",
        f"最終門檻：{'通過' if result['accepted'] else '未通過；不得升格主線'}。第 7 階段未啟動。",
        "",
        "| 指標 | 融合 | 獨立來源 | 差異 |",
        "|---|---:|---:|---:|",
    ]
    for key, value in metrics.items():
        ref = expected()[key]
        lines.append(f"| {key} | {value:.6f} | {ref:.6f} | {value - ref:+.6f} |")
    lines += [
        "",
        f"參數 {total:,}；獨立模型合計 {manifest['independent_parameters']:,}。",
        "沒有進行 RTL／HLS 或板端量測；參數減少不等於已量測加速。單 seed，未新增原樣 control 加訓。",
        "",
        "[曲線](curves.png) · [CSV](history.csv) · [離線重畫](redraw.py) · [完整交接](handoff.json)",
        "",
        "困難與解法：依各 run 的安全線、停止原因及校準結果記錄；完整執行鏈保留於 artifacts/stage6/rebuild-20261029。",
    ]
    (result_dir / "README.md").write_text("\n".join(lines) + "\n")
    write_json(output / "result.json", result)
    return result
