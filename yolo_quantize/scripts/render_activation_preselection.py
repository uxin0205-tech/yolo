from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/reports/activation-smoke-v2.json"
DEFAULT_OUTPUT_DIR = ROOT / "deliverables"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
ARTIFACT_TIMESTAMP = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)

COLORS = {
    "poly_quality": "#2563EB",
    "poly_shift": "#16A34A",
    "qsilu_pq": "#F97316",
    "hardswish": "#64748B",
}
LABELS = {
    "poly_quality": "poly_quality",
    "poly_shift": "poly_shift",
    "qsilu_pq": "qSiLU",
    "hardswish": "Hardswish",
}
MARKERS = {
    "poly_quality": "o",
    "poly_shift": "s",
    "qsilu_pq": "^",
    "hardswish": "D",
}


def load_rows(path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, dict[int, tuple[float, float]]] = {key: {} for key in COLORS}
    for result in payload["results"].values():
        activation = result["activation"]
        if activation not in rows:
            continue
        raw = []
        overlaps = []
        for task in ("detect", "pose"):
            deployment = result["tasks"][task]["deployment"]
            raw.append(deployment["one2one_raw"]["normalized_rmse"])
            overlaps.append(deployment["topk_overlap"]["selected_pair_overlap"])
        rows[activation][int(result["activation_bits"])] = (
            max(raw),
            min(overlaps),
        )
    expected_bits = set(range(3, 9))
    for activation, values in rows.items():
        if set(values) != expected_bits:
            raise RuntimeError(
                f"{activation} is missing A3-A8 values: {sorted(values)}"
            )
    return rows


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#CBD5E1", alpha=0.55, linewidth=0.8)
    axis.set_xlim(2.75, 8.25)
    axis.set_xticks(range(3, 9), [f"A{bits}" for bits in range(3, 9)])
    axis.axvspan(2.75, 5.35, color="#FEE2E2", alpha=0.42, zorder=0)
    axis.axvspan(5.65, 7.35, color="#FEF3C7", alpha=0.44, zorder=0)
    axis.axvspan(7.65, 8.25, color="#DCFCE7", alpha=0.58, zorder=0)


def add_box(
    axis: plt.Axes,
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    lines: tuple[str, ...],
    face: str,
    edge: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        transform=axis.transAxes,
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        x + 0.035,
        y + height - 0.055,
        title,
        transform=axis.transAxes,
        fontproperties=font_manager.FontProperties(fname=FONT_BOLD, size=13),
        color="#0F172A",
        va="top",
    )
    axis.text(
        x + 0.035,
        y + height - 0.125,
        "\n".join(lines),
        transform=axis.transAxes,
        fontproperties=font_manager.FontProperties(fname=FONT_REGULAR, size=10.3),
        color="#334155",
        va="top",
        linespacing=1.45,
    )


def render(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    rows = load_rows(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    bold = font_manager.FontProperties(fname=FONT_BOLD)

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#F8FAFC",
            "svg.hashsalt": "full35-activation-preselection-v1",
        }
    )
    figure = plt.figure(figsize=(16, 9), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(2.15, 1.0),
        height_ratios=(1, 1),
        left=0.055,
        right=0.975,
        top=0.80,
        bottom=0.105,
        wspace=0.15,
        hspace=0.24,
    )
    raw_axis = figure.add_subplot(grid[0, 0])
    overlap_axis = figure.add_subplot(grid[1, 0])
    decision_axis = figure.add_subplot(grid[:, 1])
    decision_axis.axis("off")

    figure.text(
        0.055,
        0.955,
        "Full35 Activation 預先選擇：為什麼先用 A8？",
        fontproperties=font_manager.FontProperties(fname=FONT_BOLD, size=24),
        color="#0F172A",
        va="top",
    )
    figure.text(
        0.055,
        0.895,
        "FP32 weights、無訓練 smoke；比較量化前後的 one-to-one raw 輸出與 Top-300 候選集合",
        fontproperties=font_manager.FontProperties(fname=FONT_REGULAR, size=12),
        color="#475569",
        va="top",
    )

    for activation, color in COLORS.items():
        bits = sorted(rows[activation])
        raw_values = [rows[activation][bit][0] for bit in bits]
        overlap_values = [rows[activation][bit][1] for bit in bits]
        alpha = 0.95 if activation != "hardswish" else 0.68
        line_style = "-" if activation != "hardswish" else "--"
        kwargs = {
            "label": LABELS[activation],
            "color": color,
            "marker": MARKERS[activation],
            "markersize": 6.5,
            "linewidth": 2.2,
            "linestyle": line_style,
            "alpha": alpha,
        }
        raw_axis.plot(bits, raw_values, **kwargs)
        overlap_axis.plot(bits, overlap_values, **kwargs)

    style_axis(raw_axis)
    style_axis(overlap_axis)
    raw_axis.set_ylim(0.0, 0.88)
    overlap_axis.set_ylim(-0.03, 0.78)
    raw_axis.set_ylabel("Worst raw NRMSE（越低越好）", fontproperties=bold)
    overlap_axis.set_ylabel("Minimum TopK overlap（越高越好）", fontproperties=bold)
    overlap_axis.set_xlabel("Activation output 位寬", fontproperties=bold)
    raw_axis.set_title(
        "① 原始輸出被量化扭曲多少？",
        loc="left",
        fontproperties=bold,
        fontsize=14,
        color="#0F172A",
    )
    overlap_axis.set_title(
        "② Top-300 class–anchor 候選保留多少？",
        loc="left",
        fontproperties=bold,
        fontsize=14,
        color="#0F172A",
    )
    raw_axis.annotate(
        "A5 poly_quality = 0.3797\n誤差RMS約為參考輸出RMS的0.38倍",
        xy=(5, rows["poly_quality"][5][0]),
        xytext=(3.05, 0.49),
        arrowprops={"arrowstyle": "->", "color": "#2563EB"},
        bbox={"boxstyle": "round,pad=0.35", "fc": "#EFF6FF", "ec": "#93C5FD"},
        fontproperties=font_manager.FontProperties(fname=FONT_REGULAR, size=9.4),
        color="#1E3A8A",
    )
    overlap_axis.annotate(
        "A5 poly_quality = 0.097\n較差任務約只保留 29 / 300 候選",
        xy=(5, rows["poly_quality"][5][1]),
        xytext=(3.0, 0.25),
        arrowprops={"arrowstyle": "->", "color": "#2563EB"},
        bbox={"boxstyle": "round,pad=0.35", "fc": "#EFF6FF", "ec": "#93C5FD"},
        fontproperties=font_manager.FontProperties(fname=FONT_REGULAR, size=9.4),
        color="#1E3A8A",
    )
    overlap_axis.annotate(
        "A8 poly_quality = 0.693\n較差任務約保留 208 / 300 候選",
        xy=(8, rows["poly_quality"][8][1]),
        xytext=(6.15, 0.55),
        arrowprops={"arrowstyle": "->", "color": "#2563EB"},
        bbox={"boxstyle": "round,pad=0.35", "fc": "#ECFDF5", "ec": "#86EFAC"},
        fontproperties=font_manager.FontProperties(fname=FONT_REGULAR, size=9.4),
        color="#14532D",
    )
    raw_axis.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        prop=font_manager.FontProperties(fname=FONT_REGULAR, size=9.5),
    )

    add_box(
        decision_axis,
        xy=(0.02, 0.69),
        width=0.96,
        height=0.28,
        title="預選結論",
        lines=(
            "主線 A8：qSiLU、poly_quality、poly_shift",
            "後續研究：qSiLU A6；poly_quality／poly_shift A7",
            "停止新增：SiLU、Hardswish、ReLU",
            "",
            "理由：先用A8隔離weight誤差，再降低A-bit。",
        ),
        face="#FFFFFF",
        edge="#94A3B8",
    )
    add_box(
        decision_axis,
        xy=(0.02, 0.34),
        width=0.96,
        height=0.29,
        title="SD4如何納入",
        lines=(
            "標準主線是 W-SD4（weight codebook）：",
            "每條A8 policy都配 W8／W4／Fixed SD4／LS-SD4。",
            "",
            "A-SD4（activation使用SD4 codebook）另列探索支線，",
            "必須先和同為4-bit的LSQ+ A4公平比較。",
        ),
        face="#F0FDF4",
        edge="#4ADE80",
    )
    add_box(
        decision_axis,
        xy=(0.02, 0.04),
        width=0.96,
        height=0.23,
        title="老師看圖時要注意",
        lines=(
            "NRMSE／TopK overlap是工程proxy，不是mAP。",
            "0.3797不代表準確率下降37.97%。",
            "最後仍以COCO box、BBAT box、BBAT pose",
            "三項mAP下降各自不超過0.04作gate。",
        ),
        face="#FFF7ED",
        edge="#FB923C",
    )
    decision_axis.add_patch(
        FancyArrowPatch(
            (0.50, 0.685),
            (0.50, 0.64),
            transform=decision_axis.transAxes,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color="#64748B",
        )
    )

    figure.text(
        0.055,
        0.035,
        "資料來源：activation-smoke-v2.json｜每點取Detect／Pose較差值｜selection band是預選判讀，不是正式精度門檻",
        fontproperties=font_manager.FontProperties(fname=FONT_REGULAR, size=9.5),
        color="#64748B",
    )

    stem = "activation-preselection-teacher-v1"
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=160, bbox_inches="tight")
    figure.savefig(
        svg,
        bbox_inches="tight",
        metadata={"Creator": "yolo_quantize", "Date": "2026-08-29"},
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "yolo_quantize",
            "CreationDate": ARTIFACT_TIMESTAMP,
            "ModDate": ARTIFACT_TIMESTAMP,
        },
    )
    plt.close(figure)
    return png, svg, pdf


def main() -> int:
    parser = argparse.ArgumentParser(
        description="繪製Full35 activation預先選擇的老師版圖表"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    outputs = render(
        args.input.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
