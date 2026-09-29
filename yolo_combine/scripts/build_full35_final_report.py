#!/usr/bin/env python3
"""由正式 append-only artifacts 產生 Full35 最終中文分析、CSV、JSON 與圖表。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "reports/full35"
OLD_RUN = (
    REPO
    / "variants/full35/artifacts/fusion/formal/full35-joint-adamw-seed0"
)
J2_RUN = (
    REPO
    / "variants/full35/artifacts/fusion/formal/"
    "full35-joint-adamw-v2-j0e8-j1e20-j2e80-seed0"
)
J3_RUN = (
    REPO
    / "variants/full35/artifacts/fusion/formal/"
    "full35-joint-adamw-v2-j3-b32-challenger-seed0"
)
J2_EVALUATION = (
    REPO
    / "variants/full35/artifacts/fusion/formal/evaluations/"
    "full35-v2-best-joint-seed0/epoch-0000"
)
J3_EVALUATION = (
    REPO
    / "variants/full35/artifacts/fusion/formal/evaluations/"
    "full35-v2-j3-best-joint-seed0/epoch-0000"
)
BASELINE = REPO / "variants/full35/baselines/formal-gate-p3-seed0.json"
STANDALONE_EVALUATION = (
    REPO
    / "variants/full35/artifacts/standalone-baseline/p3-seed0/validation/bittrue/metrics.json"
)

GATE_METRICS = (
    "coco/box/map50_95",
    "coco/person/box/map50_95",
    "bbat/box/map50_95",
    "bbat/pose/map50_95",
    "bbat/ball/box/map50_95",
    "bbat/bat/box/map50_95",
    "bbat/ball/pose/map50_95",
    "bbat/bat/pose/map50_95",
)
LABELS = {
    "coco/box/map50_95": "COCO overall box",
    "coco/person/box/map50_95": "COCO person box",
    "bbat/box/map50_95": "BBAT box overall",
    "bbat/pose/map50_95": "BBAT pose overall",
    "bbat/ball/box/map50_95": "ball box",
    "bbat/bat/box/map50_95": "bat box",
    "bbat/ball/pose/map50_95": "ball pose",
    "bbat/bat/pose/map50_95": "bat pose",
}
STAGES = (
    ("J0", 0, 7, "#d9eaf7"),
    ("J1", 8, 27, "#e8f4dc"),
    ("J2", 28, 52, "#fff0cc"),
    ("J3", 53, 63, "#f3dfef"),
)
PERSON_METRICS = {
    "person_ap50_95": "coco/person/box/map50_95",
    "person_ap50": "coco/person/box/map50",
    "person_ap75": "coco/person/box/map75",
    "person_precision": "coco/person/box/precision",
    "person_recall": "coco/person/box/recall",
}
PERSON_CANDIDATES = (
    {
        "candidate_id": "standalone_detect",
        "label": "Standalone Detect",
        "decision_role": "person-only accuracy baseline",
        "checkpoint": "final/full35/weights/standalone/detect-full35-a2-bittrue.pt",
        "global_epoch": None,
        "metrics_path": STANDALONE_EVALUATION,
        "validation_source": "independent formal Bit-True validation",
        "all_eight_gates_passed": None,
    },
    {
        "candidate_id": "shared_best_detect",
        "label": "Shared best_detect",
        "decision_role": "person-first shared checkpoint; Pose not ready",
        "checkpoint": "final/full35/weights/combined/inference/best_detect.pt",
        "global_epoch": 0,
        "metrics_path": J2_RUN / "validation/epoch-0000/bittrue/metrics.json",
        "validation_source": "full per-epoch Bit-True validation",
        "all_eight_gates_passed": False,
    },
    {
        "candidate_id": "j2_best_joint",
        "label": "J2 best_joint",
        "decision_role": "balanced rollback",
        "checkpoint": "final/full35/weights/rollback/j2/inference/best_joint.pt",
        "global_epoch": 35,
        "metrics_path": J2_EVALUATION / "bittrue/metrics.json",
        "validation_source": "independent final Bit-True revalidation",
        "all_eight_gates_passed": True,
    },
    {
        "candidate_id": "j3_best_joint",
        "label": "J3 best_joint",
        "decision_role": "current joint-score default",
        "checkpoint": "final/full35/weights/combined/inference/best_joint.pt",
        "global_epoch": 58,
        "metrics_path": J3_EVALUATION / "bittrue/metrics.json",
        "validation_source": "independent final Bit-True revalidation",
        "all_eight_gates_passed": True,
    },
    {
        "candidate_id": "j3_best_pose",
        "label": "J3 best_pose",
        "decision_role": "pose-first balanced alternative",
        "checkpoint": "final/full35/weights/combined/inference/best_pose.pt",
        "global_epoch": 59,
        "metrics_path": J3_RUN / "validation/epoch-0059/bittrue/metrics.json",
        "validation_source": "full per-epoch Bit-True validation",
        "all_eight_gates_passed": True,
    },
    {
        "candidate_id": "j3_last",
        "label": "J3 last",
        "decision_role": "stage-complete endpoint",
        "checkpoint": "final/full35/weights/combined/inference/last.pt",
        "global_epoch": 63,
        "metrics_path": J3_RUN / "validation/epoch-0063/bittrue/metrics.json",
        "validation_source": "full per-epoch Bit-True validation",
        "all_eight_gates_passed": True,
    },
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be a mapping: {path}")
    return payload


def _long_csv(path: Path) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records[int(row["step"])][str(row["metric"])] = float(row["value"])
    return dict(records)


def _merge_long(paths: Iterable[Path]) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for path in paths:
        for step, values in _long_csv(path).items():
            if step in merged:
                overlap = set(values) & set(merged[step])
                if overlap:
                    raise ValueError(f"duplicate step/metrics at {path}: {step}, {overlap}")
                merged[step].update(values)
            else:
                merged[step] = dict(values)
    return dict(sorted(merged.items()))


def _events(paths: Iterable[Path], kind: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"malformed event at {path}:{line_number}")
            if record.get("kind") == kind:
                output.append(record)
    return output


def _stage(step: int) -> str:
    for name, start, end, _color in STAGES:
        if start <= step <= end:
            return name
    return "unknown"


def _joint_score(metrics: Mapping[str, float]) -> float:
    return (
        0.2 * metrics["coco/box/map50_95"]
        + 0.2 * metrics["coco/person/box/map50_95"]
        + 0.2 * metrics["bbat/box/map50_95"]
        + 0.4 * metrics["bbat/pose/map50_95"]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stage_background(axis: Any) -> None:
    for name, start, end, color in STAGES:
        axis.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.42, zorder=0)
        axis.text(
            (start + end) / 2,
            1.01,
            name,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(plt: Any, figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _metric_rows(
    baseline: Mapping[str, float],
    j2: Mapping[str, float],
    j3: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for metric in GATE_METRICS:
        base = float(baseline[metric])
        j2_value = float(j2[metric])
        j3_value = float(j3[metric])
        rows.append(
            {
                "metric": metric,
                "label": LABELS[metric],
                "standalone": base,
                "j2": j2_value,
                "j3": j3_value,
                "j2_delta_vs_standalone": j2_value - base,
                "j3_delta_vs_standalone": j3_value - base,
                "j3_delta_vs_j2": j3_value - j2_value,
                "j3_gate_passed": j3_value - base >= -0.08,
            }
        )
    return rows


def _metrics_payload(path: Path) -> dict[str, float]:
    payload = _json(path).get("metrics")
    if not isinstance(payload, dict):
        raise TypeError(f"metrics must be a mapping: {path}")
    output = {str(name): float(value) for name, value in payload.items()}
    required = {"coco/box/map50_95", *PERSON_METRICS.values()}
    missing = sorted(required - output.keys())
    if missing:
        raise KeyError(f"missing COCO person metrics at {path}: {missing}")
    if any(not math.isfinite(output[name]) for name in required):
        raise ValueError(f"non-finite COCO person metrics at {path}")
    return output


def _person_candidate_rows(
    gates: Mapping[int, Mapping[str, float]],
    validation: Mapping[int, Mapping[str, float]],
) -> list[dict[str, Any]]:
    standalone = _metrics_payload(Path(PERSON_CANDIDATES[0]["metrics_path"]))
    rows: list[dict[str, Any]] = []
    for specification in PERSON_CANDIDATES:
        metrics = _metrics_payload(Path(specification["metrics_path"]))
        epoch = specification["global_epoch"]
        if epoch is None:
            joint_score: float | None = None
        else:
            epoch = int(epoch)
            if epoch not in gates or epoch not in validation:
                raise KeyError(f"candidate epoch is absent from formal logs: {epoch}")
            joint_score = float(gates[epoch]["score/best_joint"])
            expected = float(validation[epoch]["coco/person/box/map50_95"])
            actual = float(metrics["coco/person/box/map50_95"])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"candidate person AP does not match epoch {epoch}: "
                    f"metrics={actual}, validation.csv={expected}"
                )
        row = {
            "candidate_id": specification["candidate_id"],
            "label": specification["label"],
            "decision_role": specification["decision_role"],
            "checkpoint": specification["checkpoint"],
            "global_epoch": epoch,
            "validation_source": specification["validation_source"],
            "all_eight_gates_passed": specification["all_eight_gates_passed"],
            "joint_score": joint_score,
            "coco_overall_ap50_95": metrics["coco/box/map50_95"],
            **{field: metrics[metric] for field, metric in PERSON_METRICS.items()},
        }
        row.update(
            {
                "person_ap50_95_delta_vs_standalone": (
                    row["person_ap50_95"]
                    - standalone[PERSON_METRICS["person_ap50_95"]]
                ),
                "person_ap50_delta_vs_standalone": (
                    row["person_ap50"]
                    - standalone[PERSON_METRICS["person_ap50"]]
                ),
                "person_ap75_delta_vs_standalone": (
                    row["person_ap75"]
                    - standalone[PERSON_METRICS["person_ap75"]]
                ),
            }
        )
        rows.append(row)
    return rows


def _person_epoch_rows(
    validation: Mapping[int, Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in sorted(validation):
        run = J2_RUN if epoch <= 52 else J3_RUN
        path = run / f"validation/epoch-{epoch:04d}/bittrue/metrics.json"
        metrics = _metrics_payload(path)
        expected = float(validation[epoch]["coco/person/box/map50_95"])
        actual = float(metrics["coco/person/box/map50_95"])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"person epoch metric mismatch at {epoch}: "
                f"metrics={actual}, validation.csv={expected}"
            )
        rows.append(
            {
                "global_epoch": epoch,
                "stage": _stage(epoch),
                "coco_overall_ap50_95": metrics["coco/box/map50_95"],
                **{field: metrics[metric] for field, metric in PERSON_METRICS.items()},
            }
        )
    return rows


def _gradient_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    ratios: list[float] = []
    cosines: list[float] = []
    by_epoch: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"ratio": [], "cosine": []}
    )
    for record in events:
        values = record.get("values", {})
        context = record.get("context", {})
        detect = values.get("gradient/detect_shared_norm")
        pose = values.get("gradient/pose_shared_norm")
        cosine = values.get("gradient/shared_cosine")
        if detect is None or pose is None or float(detect) <= 0:
            continue
        ratio = float(pose) / float(detect)
        epoch = int(context["epoch"])
        ratios.append(ratio)
        by_epoch[epoch]["ratio"].append(ratio)
        if cosine is not None and math.isfinite(float(cosine)):
            cosines.append(float(cosine))
            by_epoch[epoch]["cosine"].append(float(cosine))
    if not ratios:
        return {"samples": 0, "by_epoch": []}
    epoch_rows = [
        {
            "epoch": epoch,
            "stage": _stage(epoch),
            "pose_detect_norm_ratio_mean": mean(values["ratio"]),
            "shared_cosine_mean": (
                mean(values["cosine"]) if values["cosine"] else math.nan
            ),
            "samples": len(values["ratio"]),
        }
        for epoch, values in sorted(by_epoch.items())
    ]
    return {
        "samples": len(ratios),
        "ratio_mean": mean(ratios),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "cosine_mean": mean(cosines) if cosines else math.nan,
        "negative_cosine_samples": sum(value < 0 for value in cosines),
        "by_epoch": epoch_rows,
    }


def _render_figures(
    output: Path,
    *,
    validation: Mapping[int, Mapping[str, float]],
    gates: Mapping[int, Mapping[str, float]],
    epochs: Mapping[int, Mapping[str, float]],
    metric_rows: list[dict[str, Any]],
    gradients: Mapping[str, Any],
    person_epochs: list[dict[str, Any]],
    person_candidates: list[dict[str, Any]],
) -> list[str]:
    plt = _setup_matplotlib()
    figures = output / "figures"
    made: list[str] = []

    steps = sorted(gates)
    scores = [gates[step]["score/best_joint"] for step in steps]
    fig, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    _stage_background(axis)
    axis.plot(steps, scores, color="#1f5c99", marker="o", markersize=3, linewidth=1.6)
    axis.scatter([35, 58], [gates[35]["score/best_joint"], gates[58]["score/best_joint"]],
                 color=["#e08800", "#bd1f2d"], s=55, zorder=4)
    axis.annotate("J2 best", (35, gates[35]["score/best_joint"]), xytext=(30, 12),
                  textcoords="offset points", fontsize=9)
    axis.annotate("Final J3 best", (58, gates[58]["score/best_joint"]), xytext=(-70, 14),
                  textcoords="offset points", fontsize=9)
    axis.set(title="Bit-True joint score across formal training", xlabel="global epoch", ylabel="joint score")
    target = figures / "01-joint-score-by-epoch.png"
    _save(plt, fig, target)
    made.append(target.relative_to(output).as_posix())

    fig, axis = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    _stage_background(axis)
    core = (
        "coco/box/map50_95",
        "coco/person/box/map50_95",
        "bbat/box/map50_95",
        "bbat/pose/map50_95",
    )
    for metric in core:
        points = [(step, values[metric]) for step, values in validation.items()]
        axis.plot([x for x, _ in points], [y for _, y in points], marker="o", markersize=2.5,
                  linewidth=1.4, label=LABELS[metric])
    axis.set(title="Bit-True validation mAP50-95", xlabel="global epoch", ylabel="mAP50-95")
    axis.legend(fontsize=8, ncol=2)
    target = figures / "02-validation-map-by-epoch.png"
    _save(plt, fig, target)
    made.append(target.relative_to(output).as_posix())

    fig, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    _stage_background(axis)
    for metric, label in (
        ("loss/detect_mean", "Detect mean loss"),
        ("loss/pose_mean", "Pose mean loss"),
        ("loss/joint_mean", "Weighted joint mean loss"),
    ):
        points = [
            (step, values[metric])
            for step, values in epochs.items()
            if metric in values and values[metric] > 0
        ]
        axis.plot([x for x, _ in points], [y for _, y in points], marker="o", markersize=2.5,
                  linewidth=1.4, label=label)
    axis.set_yscale("log")
    axis.set(title="Batch-normalized training losses", xlabel="global epoch", ylabel="loss (log scale)")
    axis.legend(fontsize=8)
    target = figures / "03-training-loss-by-epoch.png"
    _save(plt, fig, target)
    made.append(target.relative_to(output).as_posix())

    labels = [row["label"] for row in metric_rows]
    y = list(range(len(labels)))
    height = 0.35
    fig, axis = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    axis.barh([value + height / 2 for value in y],
              [row["j2_delta_vs_standalone"] for row in metric_rows], height=height,
              label="J2 - standalone", color="#e6a23c")
    axis.barh([value - height / 2 for value in y],
              [row["j3_delta_vs_standalone"] for row in metric_rows], height=height,
              label="J3 - standalone", color="#4b78a8")
    axis.axvline(-0.08, color="#bd1f2d", linestyle="--", linewidth=1.4, label="hard gate (-0.08)")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set(title="Accuracy deltas against standalone Bit-True baselines", xlabel="mAP50-95 delta")
    axis.legend(fontsize=8)
    target = figures / "04-gate-deltas-j2-j3.png"
    _save(plt, fig, target)
    made.append(target.relative_to(output).as_posix())

    x = list(range(len(labels)))
    width = 0.26
    fig, axis = plt.subplots(figsize=(12, 6.1), constrained_layout=True)
    axis.bar([value - width for value in x], [row["standalone"] for row in metric_rows],
             width, label="standalone", color="#929292")
    axis.bar(x, [row["j2"] for row in metric_rows], width, label="J2", color="#e6a23c")
    axis.bar([value + width for value in x], [row["j3"] for row in metric_rows],
             width, label="J3 final", color="#4b78a8")
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylim(0.45, 0.98)
    axis.set(title="Standalone vs shared-trunk final metrics", ylabel="mAP50-95")
    axis.legend(fontsize=8, ncol=3)
    target = figures / "05-final-metric-comparison.png"
    _save(plt, fig, target)
    made.append(target.relative_to(output).as_posix())

    gradient_rows = list(gradients.get("by_epoch", []))
    if gradient_rows:
        fig, left = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
        _stage_background(left)
        right = left.twinx()
        xs = [row["epoch"] for row in gradient_rows]
        left.plot(xs, [row["pose_detect_norm_ratio_mean"] for row in gradient_rows],
                  color="#bd1f2d", marker="o", linewidth=1.4, label="Pose/Detect norm ratio")
        right.plot(xs, [row["shared_cosine_mean"] for row in gradient_rows],
                   color="#1f77b4", marker="s", linewidth=1.2, label="shared cosine")
        left.axhline(1.0, color="#bd1f2d", linestyle=":", alpha=0.55)
        right.axhline(0.0, color="#1f77b4", linestyle=":", alpha=0.55)
        left.set(xlabel="global epoch", ylabel="Pose/Detect shared-gradient norm ratio",
                 title="Sampled shared-gradient balance")
        right.set_ylabel("cosine similarity")
        handles1, labels1 = left.get_legend_handles_labels()
        handles2, labels2 = right.get_legend_handles_labels()
        left.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper right")
        target = figures / "06-shared-gradient-by-epoch.png"
        _save(plt, fig, target)
        made.append(target.relative_to(output).as_posix())

    standalone_person = person_candidates[0]
    fig, (ap_axis, pr_axis) = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.4),
        sharex=True,
        constrained_layout=True,
    )
    _stage_background(ap_axis)
    for _name, start, end, color in STAGES:
        pr_axis.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.42, zorder=0)
    person_x = [int(row["global_epoch"]) for row in person_epochs]
    for field, label, color in (
        ("person_ap50_95", "person AP50-95", "#1f5c99"),
        ("person_ap50", "person AP50", "#2f8f5b"),
        ("person_ap75", "person AP75", "#bd6b19"),
    ):
        ap_axis.plot(
            person_x,
            [float(row[field]) for row in person_epochs],
            color=color,
            linewidth=1.5,
            label=label,
        )
        ap_axis.axhline(
            float(standalone_person[field]),
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.48,
        )
    for candidate in person_candidates[1:]:
        epoch = int(candidate["global_epoch"])
        ap_axis.scatter(
            [epoch],
            [float(candidate["person_ap50_95"])],
            color="#222222",
            s=25,
            zorder=5,
        )
    ap_axis.set(
        title="COCO person AP across formal training (dashed = standalone Detect)",
        ylabel="AP",
    )
    ap_axis.legend(fontsize=8, ncol=3)
    for field, label, color in (
        ("person_precision", "person precision", "#7a3e9d"),
        ("person_recall", "person recall", "#b73544"),
    ):
        pr_axis.plot(
            person_x,
            [float(row[field]) for row in person_epochs],
            color=color,
            linewidth=1.5,
            label=label,
        )
        pr_axis.axhline(
            float(standalone_person[field]),
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.48,
        )
    pr_axis.set(xlabel="global epoch", ylabel="score")
    pr_axis.legend(fontsize=8, ncol=2)
    target = figures / "07-coco-person-ap-by-epoch.png"
    _save(plt, fig, target)
    made.append(target.relative_to(output).as_posix())

    candidate_labels = [str(row["label"]) for row in person_candidates]
    x = list(range(len(person_candidates)))
    width = 0.25
    fig, axis = plt.subplots(figsize=(12, 6.3), constrained_layout=True)
    for offset, field, label, color in (
        (-width, "person_ap50_95", "AP50-95", "#4b78a8"),
        (0.0, "person_ap50", "AP50", "#54a26f"),
        (width, "person_ap75", "AP75", "#e29a3f"),
    ):
        axis.bar(
            [value + offset for value in x],
            [float(row[field]) for row in person_candidates],
            width,
            label=label,
            color=color,
        )
    all_ap_values = [
        float(row[field])
        for row in person_candidates
        for field in ("person_ap50_95", "person_ap50", "person_ap75")
    ]
    axis.set_xticks(x, candidate_labels, rotation=22, ha="right")
    axis.set_ylim(max(0.0, min(all_ap_values) - 0.025), min(1.0, max(all_ap_values) + 0.02))
    axis.set(
        title="COCO person AP by available checkpoint",
        ylabel="AP",
    )
    axis.legend(fontsize=8, ncol=3)
    target = figures / "08-coco-person-candidate-comparison.png"
    _save(plt, fig, target)
    made.append(target.relative_to(output).as_posix())
    return made


def build(output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = {name: float(value) for name, value in _json(BASELINE)["metrics"].items()}
    j2_bittrue = _json(J2_EVALUATION / "bittrue/metrics.json")["metrics"]
    j2_float = _json(J2_EVALUATION / "float/metrics.json")["metrics"]
    j3_bittrue = _json(J3_EVALUATION / "bittrue/metrics.json")["metrics"]
    j3_float = _json(J3_EVALUATION / "float/metrics.json")["metrics"]

    logs = (J2_RUN / "logs", J3_RUN / "logs")
    validation = _merge_long(path / "validation.csv" for path in logs)
    gates = _merge_long(path / "gate.csv" for path in logs)
    epochs = _merge_long(path / "epoch.csv" for path in logs)
    expected = set(range(64))
    for label, records in (("validation", validation), ("gate", gates), ("epoch", epochs)):
        if set(records) != expected:
            raise ValueError(f"{label} epochs are incomplete: {sorted(expected - set(records))}")

    metric_rows = _metric_rows(baseline, j2_bittrue, j3_bittrue)
    epoch_rows = [
        {"global_epoch": step, "stage": _stage(step), **values}
        for step, values in epochs.items()
    ]
    validation_rows = [
        {
            "global_epoch": step,
            "stage": _stage(step),
            "joint_score": gates[step]["score/best_joint"],
            "detect_score": gates[step]["score/best_detect"],
            "pose_score": gates[step]["score/best_pose"],
            "all_gates_passed": bool(gates[step]["passed"]),
            **{metric: values[metric] for metric in GATE_METRICS},
        }
        for step, values in validation.items()
    ]
    gradient_events = _events(
        (J2_RUN / "logs/events.jsonl", J3_RUN / "logs/events.jsonl"), "macro"
    )
    gradients = _gradient_summary(gradient_events)
    old_events = _events((OLD_RUN / "logs/events.jsonl",), "macro")
    old_valid_events = [
        record
        for record in old_events
        if record.get("values", {}).get("gradient/detect_shared_norm") is not None
        and record.get("values", {}).get("gradient/pose_shared_norm") is not None
        and float(record["values"]["gradient/detect_shared_norm"]) > 0
    ]
    old_initial_gradients = _gradient_summary(old_valid_events[:24])
    old_full_gradients = _gradient_summary(old_valid_events)
    person_candidates = _person_candidate_rows(gates, validation)
    person_epochs = _person_epoch_rows(validation)

    data = output / "data"
    _write_csv(data / "epoch-metrics.csv", epoch_rows)
    _write_csv(data / "validation-metrics.csv", validation_rows)
    _write_csv(data / "final-metric-comparison.csv", metric_rows)
    _write_csv(data / "coco-person-candidate-comparison.csv", person_candidates)
    _write_csv(data / "coco-person-by-epoch.csv", person_epochs)
    if gradients["by_epoch"]:
        _write_csv(data / "gradient-by-epoch.csv", gradients["by_epoch"])

    figures = _render_figures(
        output,
        validation=validation,
        gates=gates,
        epochs=epochs,
        metric_rows=metric_rows,
        gradients=gradients,
        person_epochs=person_epochs,
        person_candidates=person_candidates,
    )

    j2_score = _joint_score(j2_bittrue)
    j3_score = _joint_score(j3_bittrue)
    bittrue_float_deltas = {
        metric: float(j3_bittrue[metric]) - float(j3_float[metric])
        for metric in GATE_METRICS
    }
    stage_counts = {
        stage: sum(_stage(step) == stage for step in epochs) for stage in ("J0", "J1", "J2", "J3")
    }
    totals = {
        "epochs": len(epochs),
        "optimizer_macro_steps": int(sum(values["macros"] for values in epochs.values())),
        "detect_images": int(sum(values["images/detect"] for values in epochs.values())),
        "pose_images": int(sum(values["images/pose"] for values in epochs.values())),
    }
    summary = {
        "schema_version": 2,
        "status": "completed_and_j3_promoted",
        "deployment_selection": {
            "state": "pending_user_choice",
            "current_default": "j3_best_joint",
            "note": (
                "J3 best_joint remains the experiment-selector default; no deployment "
                "checkpoint is locked until the user reviews the person AP comparison."
            ),
            "candidates": person_candidates,
        },
        "accepted": {
            "stage": "J3",
            "global_epoch": 58,
            "j3_local_epoch_zero_based": 5,
            "joint_score": j3_score,
            "all_eight_gates_passed": all(row["j3_gate_passed"] for row in metric_rows),
            "worst_gate_delta": min(row["j3_delta_vs_standalone"] for row in metric_rows),
        },
        "parent_j2": {"global_epoch": 35, "joint_score": j2_score},
        "j3_improvement_over_j2": j3_score - j2_score,
        "actual_stage_epochs": stage_counts,
        "training_totals": totals,
        "metrics": metric_rows,
        "j3_float_minus_bittrue": {
            metric: -delta for metric, delta in bittrue_float_deltas.items()
        },
        "old_strategy_gradient_evidence": {
            "initial_24_samples": {
                key: value
                for key, value in old_initial_gradients.items()
                if key != "by_epoch"
            },
            "all_durable_samples": {
                key: value
                for key, value in old_full_gradients.items()
                if key != "by_epoch"
            },
        },
        "final_strategy_gradient_evidence": {
            key: value for key, value in gradients.items() if key != "by_epoch"
        },
        "figures": figures,
        "limitations": [
            "只有 seed0；未執行 seed1。",
            "logical Detect batch 128 由 physical microbatch 累積，不宣稱等同單次 physical128。",
            "41.796% 是參數/weight storage 減少，尚非 latency、energy 或 FPGA 實測。",
            "Q/K binary sign path與score.gamma保持硬體契約凍結；J3只微調attention可微部分。",
        ],
    }
    (output / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def metric_table(rows: list[dict[str, Any]]) -> str:
        lines = [
            "| 指標 | 獨立模型 | J2 | 最終 J3 | J3−獨立 | J3−J2 | gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['label']} | {row['standalone']:.6f} | {row['j2']:.6f} | "
                f"{row['j3']:.6f} | {row['j3_delta_vs_standalone']:+.6f} | "
                f"{row['j3_delta_vs_j2']:+.6f} | {'通過' if row['j3_gate_passed'] else '失敗'} |"
            )
        return "\n".join(lines)

    def person_candidate_table(rows: list[dict[str, Any]]) -> str:
        lines = [
            "| 候選 | epoch | person AP50-95 | AP50 | AP75 | precision | recall | "
            "AP50-95差於獨立Detect | joint score | 八項gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
        for row in rows:
            epoch = "—" if row["global_epoch"] is None else str(row["global_epoch"])
            joint = "—" if row["joint_score"] is None else f"{row['joint_score']:.6f}"
            gate = (
                "不適用"
                if row["all_eight_gates_passed"] is None
                else ("通過" if row["all_eight_gates_passed"] else "未通過")
            )
            lines.append(
                f"| {row['label']} | {epoch} | {row['person_ap50_95']:.6f} | "
                f"{row['person_ap50']:.6f} | {row['person_ap75']:.6f} | "
                f"{row['person_precision']:.6f} | {row['person_recall']:.6f} | "
                f"{row['person_ap50_95_delta_vs_standalone']:+.6f} | {joint} | {gate} |"
            )
        return "\n".join(lines)

    person_by_id = {row["candidate_id"]: row for row in person_candidates}

    report = f"""# Full35 Detect＋Pose 融合訓練最終分析

## 結論先行

Full35 seed0 的正式 J0→J3 訓練與獨立 Float／Bit-True 驗證已完成。依固定實驗selector接受的是
**J3 global epoch 58**（J3 內第6個epoch，zero-based 5）的 `best_joint.pt`：Bit-True
joint score 為 `{j3_score:.12f}`，高於 J2 best 的 `{j2_score:.12f}`，絕對改善
`{j3_score - j2_score:+.12f}`。八項 mAP50-95 hard gate 全部通過；最差是 ball pose
相對獨立模型 `{min(row['j3_delta_vs_standalone'] for row in metric_rows):+.6f}`，仍遠高於
`-0.08` 下限。

這代表共享模型在本次 seed0 可以把兩個 standalone trunk 合成一份，參數由
45,580,762 降為 26,529,701（少 19,051,061，`41.796%`），同時將八項精度下降控制在
允許範圍內。它不代表每項都比 J2 或獨立模型更好，也不代表 latency、energy 或 FPGA
資源已改善；這些邊界在後文分開說明。

這個「接受」只代表實驗的joint-score預設。依使用者最新要求，實際部署要用
`best_detect`、`best_joint`、`best_pose`、`last`或獨立Detect，仍保留到看完下方COCO
person AP後再決定；本次沒有更換或刪除任何權重。

![Joint score](figures/01-joint-score-by-epoch.png)

## 實際使用的資料與模型

- Detect：`/home/uxin/yolo/coco2017.yaml`，COCO80 train/val 118,287/5,000；應用層取 person，
  正式驗證同時保留 COCO overall 與 person 指標。
- Pose：canonical `bbat5-v1` 的 `configs/pose.yaml`，formal train/val 5,964/683，類別
  ball/bat，`kpt_shape=[2,3]`，visibility 0/2 由原生 Pose26 loss處理。
- 本次正式 run 全程 `fraction=1`，兩個資料集都使用；沒有使用歷史 basic split，沒有把
  Detect/Pose label粗暴合成同一loader。
- 架構：Full35 shared YOLO26m layers 0–22只執行一次，P3/P4/P5同時餵 COCO80 Detect head
  與 ball/bat Pose26 head。不是兩個完整模型串行forward，也不是權重平均。

## 最終八項結果

正式選模只用 Bit-True backend；Float 用來檢查部署數值一致性。`0.08` 是「絕對
mAP50-95 下降上限」，不是百分比。

{metric_table(metric_rows)}

![Final metric comparison](figures/05-final-metric-comparison.png)

![Gate deltas](figures/04-gate-deltas-j2-j3.png)

J3並非全項勝過J2：ball box `{next(row for row in metric_rows if row['metric'] == 'bbat/ball/box/map50_95')['j3_delta_vs_j2']:+.6f}`、
ball pose `{next(row for row in metric_rows if row['metric'] == 'bbat/ball/pose/map50_95')['j3_delta_vs_j2']:+.6f}`，但 person、BBAT overall pose、
bat box與bat pose改善，固定joint score因此選J3。不能用joint平均掩蓋任一任務失敗，所以
先逐項套八項gate，再比較joint score。

## COCO person AP與可選權重

下表全部使用同一個COCO2017 val 5,000張、同一套Bit-True evaluator。`AP50-95`是person
單類別在IoU 0.50:0.95的平均AP；`AP50`與`AP75`分別固定IoU 0.50／0.75。precision與
recall是validator選定操作點的值，不是AP，不能和AP欄直接相加。

{person_candidate_table(person_candidates)}

![COCO person AP by checkpoint](figures/08-coco-person-candidate-comparison.png)

![COCO person AP by epoch](figures/07-coco-person-ap-by-epoch.png)

這些數字對選擇的意義如下：

- **只在意person、可接受不融合**：獨立Detect的person AP50-95最高，為
  `{person_by_id['standalone_detect']['person_ap50_95']:.6f}`；但它沒有ball/bat Pose head，
  也不享有shared trunk的一次feature extraction。
- **只跑shared模型的Detect head**：`best_detect.pt`為global epoch0，person AP50-95
  `{person_by_id['shared_best_detect']['person_ap50_95']:.6f}`，只比獨立Detect低
  `{abs(person_by_id['shared_best_detect']['person_ap50_95_delta_vs_standalone']):.6f}`。
  但當時Pose仍未適應完成，八項gate未通過，所以不能拿它作`task=both`正式答案。
- **兩個head都要、以固定joint score為準**：目前預設仍是J3 `best_joint.pt`。它的person
  AP50-95為`{person_by_id['j3_best_joint']['person_ap50_95']:.6f}`，比J2 best_joint高
  `{person_by_id['j3_best_joint']['person_ap50_95'] - person_by_id['j2_best_joint']['person_ap50_95']:+.6f}`，
  且八項gate全部通過。
- **兩個head都要、略偏Pose與person recall**：J3 `best_pose.pt`的person AP50-95為
  `{person_by_id['j3_best_pose']['person_ap50_95']:.6f}`，比best_joint高
  `{person_by_id['j3_best_pose']['person_ap50_95'] - person_by_id['j3_best_joint']['person_ap50_95']:+.6f}`，
  recall也高
  `{person_by_id['j3_best_pose']['person_recall'] - person_by_id['j3_best_joint']['person_recall']:+.6f}`；
  但precision低
  `{person_by_id['j3_best_pose']['person_precision'] - person_by_id['j3_best_joint']['person_precision']:+.6f}`，
  joint score低`{person_by_id['j3_best_pose']['joint_score'] - person_by_id['j3_best_joint']['joint_score']:+.6f}`。
  差距很小，只有seed0時不應宣稱具有統計顯著性。
- **要stage-complete端點**：`last.pt`的person precision最高，為
  `{person_by_id['j3_last']['person_precision']:.6f}`，但joint score已回落到
  `{person_by_id['j3_last']['joint_score']:.6f}`，因此只適合作續訓／端點比較，不是目前首選。

`Standalone Detect`、J2/J3 `best_joint`另有獨立final revalidation；`best_detect`、
`best_pose`與`last`使用訓練當下保存的完整COCO Bit-True逐epoch validation。兩者資料、
evaluator與backend一致，但若要在極小差距上做不可逆部署決策，仍建議對最後兩個候選
各做一次同指令重驗或跑seed1。目前交付預設不變，等使用者選定後再改正式入口。

## 為什麼更換訓練方式

第一版融合一開始就解凍neck、MASF與兩個heads。Pose head原本配合獨立Pose P3 trunk，
融合時shared trunk卻由Detect checkpoint初始化，兩者初始feature分布不完全匹配。舊版
shared-gradient初期24次抽樣中，Pose/Detect norm平均比為
`{old_initial_gradients.get('ratio_mean', math.nan):.2f}×`
（範圍`{old_initial_gradients.get('ratio_min', math.nan):.2f}–{old_initial_gradients.get('ratio_max', math.nan):.2f}×`），
平均cosine `{old_initial_gradients.get('cosine_mean', math.nan):.4f}`，其中
`{old_initial_gradients.get('negative_cosine_samples', 0)}/{old_initial_gradients.get('samples', 0)}`次為負。
把舊run全部`{old_full_gradients.get('samples', 0)}`次durable抽樣都納入後，ratio平均降為
`{old_full_gradients.get('ratio_mean', math.nan):.2f}×`，表示後段有所緩和，但初期失衡確實存在。
這比較像Pose梯度量級壓過Detect，而不是兩個任務始終完全反向。

舊J1的COCO overall delta在warmup附近由epoch1 `-0.0813`惡化到epoch3 `-0.1855`，LR下降後
又恢復到epoch5 `-0.0848`，呈現learning-rate overshoot。若neck一解凍就不可逆破壞，後段
通常不會這樣恢復。因此新版沒有只增加epoch，而是先解決初始化失配、梯度量級與更新範圍。

新版採下列順序：

| stage | 實跑 | 目的 | trainable scope | peak LR |
| --- | ---: | --- | --- | --- |
| J0 | 8/8 | Pose head適應Detect shared trunk | 只開Pose head | Pose head `2e-4` |
| J1 | 20/20 | 保守開始雙任務融合 | neck＋兩heads | neck `7.5e-5`、heads `2e-4` |
| J2 | 25/80 | 再讓後段shared feature協調 | backbone layer9+、neck、MASF、兩heads | `1.5e-5/7.5e-5/1.5e-4/2e-4` |
| J3 | 11/20 | 全backbone與attention可微部分低LR refinement | full backbone、neck、MASF、attention可微部分、兩heads | `3.8e-6/1.9e-5/3.8e-5/5e-7/5e-5` |

J2在8個stale epochs後只做一次LR×0.5，之後以patience17停止；J3在global epoch58創新高後
連續5次沒有超越，以patience5於epoch63正常停止。這就是J2只跑25而非80、J3只跑11而非20
的原因，不是訓練中斷。

![Validation curves](figures/02-validation-map-by-epoch.png)

![Training losses](figures/03-training-loss-by-epoch.png)

## 正式訓練設定

- optimizer：AdamW；`weight_decay=0.00027`、`betas=(0.948,0.999)`。本run不在中途改成
  MuSGD；MuSGD保留為未執行challenger，避免同時改optimizer與stage策略而無法歸因。
- scheduler：各stage cosine；J0/J1/J2 warmup 1 epoch，J3 warmup 3 epochs。
- loss：Ultralytics 8.4.90原生Detect/Pose26 E2E、progressive、RLE與visibility loss；
  criterion schedule每joint epoch更新一次。
- task權重：Detect/Pose=`1.0/0.25`；J0只有Pose active時重新正規化，不會再乘0.25。
- macro：2個logical Detect128＋1個Pose16，影像exposure 256:16；各microbatch順序
  forward/backward，最後才clip(max norm10)、optimizer step、scaler與EMA update。
- augmentation：Detect/Pose都mosaic0；Detect fliplr0.5，Pose fliplr0，避免尚未證明的
  bat endpoint交換語意。
- BN：shared backbone/neck running statistics固定；head BN維持train。
- backend：Float訓練；每epoch Float與Bit-True各驗一次，Bit-True負責gate與checkpoint selector。
- XNOR：`tiled_exact`、token tile32、`qk_ste=false`；bit-true XNOR是必須契約。

本輪共完成`{totals['epochs']}`個validation epochs、`{totals['optimizer_macro_steps']:,}`次optimizer
macro-step，訓練看過`{totals['detect_images']:,}`張Detect exposure與
`{totals['pose_images']:,}`張Pose exposure（含Pose loader cycle，不是unique image數）。

![Shared gradient](figures/06-shared-gradient-by-epoch.png)

## 遇到的問題、根因與解法

1. **舊策略初期精度大跌**：Pose head與Detect trunk失配，加上Pose shared-gradient較大、
   neck/head LR偏高。解法是J0 head-only適應、延後解凍、task weight降為0.25並降低LR；
   不是單純無限增加epoch。
2. **原始XNOR validation OOM 5.94GiB**：untiled boolean equality的int64 reduction workspace
   為`O(B·H·N²·D·8)`；val batch32、rect672時單一workspace就5.935GiB。改成query/key token
   tile32的exact reduction，逐值與既有gradient semantics測試一致，保留bit-true XNOR。
3. **physical Detect batch128 OOM**：空卡RTX5090第一個正式forward已用30.58GiB，仍需
   400MiB。保留使用者要求的logical128，但J1/J2用physical64×2；這維持每次update exposure，
   不宣稱head BN與microbatch loss逐值等同physical128。
4. **J3 physical64再次OOM**：full unfreeze第一個forward程序已用30.74GiB，只剩209MiB。
   J3改physical32×4，logical128不變；實跑約21GiB並完成全程。
5. **起始AMP overflow**：初始scale65536造成多組參數Inf/NaN。新增參數級診斷及同macro
   replay，GradScaler先skip step、降scale、清gradient；成功前不前進EMA/scheduler/loader。
   第一個正式macro重算12次後於scale16成功，沒有silent bad step。
6. **正式trainer漏`import math`**：啟動即NameError。保留失敗artifact，先加CPU regression
   test，再補最小import；後續完整suite通過。
7. **J2平台期**：不是直接改momentum。依事前策略，stale8只降一次LR×0.5，stale17停止；
   best epoch35與last epoch52 joint score只差`0.0000226`，顯示已在平台附近。
8. **J3 checkpoint銜接**：J2 best位於stage中途，不帶`stage_complete=true`；用它resume會
   繼續J2。故J3從分數近似且stage已完成的J2 last exact-resume，另以J2 best作必須超越的
   acceptance reference，兩個run完全分開。

## Float／Bit-True一致性

J3 final八項Float與Bit-True的最大絕對差為
`{max(abs(value) for value in bittrue_float_deltas.values()):.8f}`。這支持目前Float訓練結果在
Bit-True attention backend下沒有明顯數值落差；它仍不是FPGA bitstream或HLS實測。

## 交付、復現與證據

- 正式推論權重：`final/full35/weights/combined/inference/best_joint.pt`。
- exact-resume最佳權重：`final/full35/weights/combined/full-resume/best_joint.pt`。
- J3終點：`final/full35/weights/combined/full-resume/last.pt`。
- J2 rollback：`final/full35/weights/rollback/j2/`。
- 原始CSV/JSONL/PNG：`final/full35/outputs/training/`。
- 最終Float/Bit-True完整metrics與predictions：
  `final/full35/outputs/validation/accepted-best-joint/`。
- 本報告機器可讀資料：`SUMMARY.json`及`data/*.csv`。

```bash
# 驗證交付包每個檔案SHA256
.venv/bin/python final/full35/verify.py

# 正式雙backend重驗
.venv/bin/python final/full35/run.py validate \\
  --checkpoint final/full35/weights/combined/inference/best_joint.pt \\
  --backend both --device 0 --name final-recheck

# shared feature只抽一次的both推論
.venv/bin/python final/full35/run.py infer \\
  --checkpoint final/full35/weights/combined/inference/best_joint.pt \\
  --source /path/to/image.jpg --task both --device 0 \\
  --output-json final/full35/outputs/inference.json
```

## 限制與下一步

- 目前只有seed0，正式跨seed穩定性仍屬provisional；若時間允許，下一個最有價值的精度
  實驗是鎖定本recipe跑seed1，不是再用seed0挑超參數。
- Q/K binary sign與`score.gamma`因bit-true硬體契約保持凍結；J3所謂attention tuning是
  V、PE、projection與relative bias等可微部分，不可宣稱訓練了Q/K sign decision。
- 41.796%只是參數與weight storage減少；尚需在相同GPU設定量測latency、peak VRAM、
  throughput，再進入HLS/FPGA latency、energy、BRAM/DSP驗收。
- final不內含COCO/BBAT5影像與labels；移機時必須提供同一canonical資料與更新路徑，
  不能重新切BBAT5。
- Partial75已有隔離環境但未執行；依使用者要求，只有再次明確下令才啟動。
- 未執行任何刪除。失敗startup、physical64 J3 OOM與舊run保留作診斷證據；若要清理，應
  先另列清單與可回復性再取得授權。
"""
    (output / "FINAL_ANALYSIS.md").write_text(report, encoding="utf-8")
    return {
        "generated": True,
        "output": str(output),
        "accepted_stage": "J3",
        "best_global_epoch": 58,
        "joint_score": j3_score,
        "j3_improvement_over_j2": j3_score - j2_score,
        "all_eight_gates_passed": all(row["j3_gate_passed"] for row in metric_rows),
        "figures": figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
