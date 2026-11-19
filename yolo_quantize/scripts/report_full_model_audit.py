"""以 CPU 讀取既有產物，重建全模型量化盤點表；不執行訓練。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/full-model-audit-2026-09-07"
QUEUE = ROOT / "artifacts/queues/v36-qsilu-full-coverage-progressive-v1"
RUN = (
    ROOT
    / "artifacts/runs/qat/v36-qsilu-full-coverage-short-v2/v36-qsilu-full-coverage-short-qat-v1-qat-seed1"
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {}

    def read(path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        provenance[str(path)] = digest(path)
        return yaml.safe_load(path.read_text())

    completion = read(RUN / "qat-experiment.json")
    plan = read(completion["plan"])
    assert digest(Path(completion["plan"])) == completion["plan_sha256"]
    for record in plan["sources"].values():
        path = Path(record["path"])
        path = path if path.is_absolute() else ROOT / path
        assert digest(path) == record["sha256"], path
        provenance[str(path)] = record["sha256"]
    for name, path in completion["checkpoint_paths"].items():
        assert digest(Path(path)) == completion["checkpoint_sha256"][name]
        provenance[path] = completion["checkpoint_sha256"][name]
    accepted = read(plan["sources"]["accepted_metrics"]["path"])["metrics"]
    keys = sorted(k for k in accepted if k.endswith(("/map50", "/map50_95")))
    assert len(keys) == 16
    epoch_rows, metric_rows = [], []
    for path in sorted(RUN.glob("validation/epoch-*/bittrue/metrics.json")):
        data = read(path)
        metrics = data["metrics"]
        epoch = data["epoch"]
        deltas = {}
        for key in keys:
            assert math.isfinite(metrics[key]) and math.isfinite(accepted[key])
            deltas[key] = metrics[key] - accepted[key]
            metric_rows.append(
                {
                    "epoch": epoch,
                    "metric": key,
                    "accepted": accepted[key],
                    "value": metrics[key],
                    "total_delta": deltas[key],
                }
            )
        worst50 = min(v for k, v in deltas.items() if k.endswith("/map50"))
        worst95 = min(v for k, v in deltas.items() if k.endswith("/map50_95"))
        epoch_rows.append(
            {
                "epoch": epoch,
                "worst_total_map50_delta": worst50,
                "worst_total_map50_95_delta": worst95,
                "deployment_gate": worst50 >= -0.015 and worst95 >= -0.04,
            }
        )
    assert len(epoch_rows) == completion["epochs_completed"] == 3
    graph = read(RUN / "qat-graph.json")
    sites = graph["weight_sites"]
    assert len(sites) == len({s["path"] for s in sites}) == 148
    assert sum(s["elements"] for s in sites) == 22571840
    profile = read(
        "artifacts/reports/v36-qsilu-v35-parent-148-layer-9format-cpu-v1.json"
    )
    cpu_rows = profile["summary"]["path_rankings"]
    assert {r["path"] for r in cpu_rows} == {s["path"] for s in sites}
    assert all(len(r["formats"]) == 9 for r in cpu_rows)
    assert all(
        math.isfinite(f["normalized_rmse"])
        for r in cpu_rows
        for f in r["formats"].values()
    )
    ptq_rows = []
    for phase in ("special", "uniform", "final"):
        data = read(f"artifacts/reports/v36-qsilu-full-coverage-{phase}-dual-v1.json")
        for cid, row in data["candidates"].items():
            ptq_rows.append(
                {
                    "phase": phase,
                    "candidate": cid,
                    "decision": row["decision"],
                    "worst_total_map50_delta": row["worst_map50_delta"],
                    "worst_total_map50_95_delta": row["worst_map50_95_delta"],
                    "worst_incremental_map50_delta": row[
                        "worst_incremental_map50_delta"
                    ],
                    "worst_incremental_map50_95_delta": row[
                        "worst_incremental_map50_95_delta"
                    ],
                }
            )
    ptq_plan = read(QUEUE / "generated/special-plan.yaml")
    regions = defaultdict(Counter)
    for s in sites:
        regions[s["region"]][s["format_id"]] += 1

    def write_csv(name, rows):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write_csv("qat-epochs.csv", epoch_rows)
    write_csv("qat-metrics.csv", metric_rows)
    write_csv("ptq-candidates.csv", ptq_rows)
    write_csv("weight-sites.csv", sites)
    folder_inventory = []
    for folder in sorted(
        p for p in ROOT.iterdir() if p.is_dir() and not p.is_symlink()
    ):
        count, size = 0, 0
        for path in folder.rglob("*"):
            if path.is_file() and not path.is_symlink():
                count += 1
                size += path.stat().st_size
        folder_inventory.append({"folder": folder.name, "files": count, "bytes": size})
    write_csv("folder-inventory.csv", folder_inventory)
    summary = {
        "snapshot_date": "2026-09-07",
        "queue": read(QUEUE / "execution-status.json"),
        "epochs": epoch_rows,
        "regions": dict(regions),
        "formats": dict(Counter(s["format_id"] for s in sites)),
        "activation_quantizers": graph["activation_quantizers"],
        "graph_totals": graph["master_catalog"]["totals"],
        "cpu_coverage": profile["summary"]["coverage"],
        "cpu_parent": profile["parent"],
        "ptq_parent": ptq_plan["activation"]["checkpoint"],
        "qat_warm_start": graph["warm_start"],
        "checkpoint_hashes_verified": True,
        "sources": provenance,
        "claims": {
            "full_weight_fake_quant": True,
            "integer_only_deployment": False,
            "exhaustive_layer_accuracy_sensitivity": False,
            "paired_ptq_qat_improvement": False,
            "formal_validation": False,
        },
    }
    (OUT / "audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    # 每一個點皆由原始 metric 減 accepted 計算，門檻使用百分點。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in [
        ("worst_total_map50_delta", "Worst total mAP50"),
        ("worst_total_map50_95_delta", "Worst total mAP50-95"),
    ]:
        ax.plot(
            [r["epoch"] for r in epoch_rows],
            [100 * r[key] for r in epoch_rows],
            "o-",
            label=label,
        )
    ax.axhline(-1.5, linestyle="--", color="C0", label="mAP50 limit (-1.5 pp)")
    ax.axhline(-4, linestyle="--", color="C1", label="mAP50-95 limit (-4 pp)")
    ax.set(
        xlabel="Epoch (zero-based)",
        ylabel="Delta vs accepted (percentage points)",
        title="V36: 148 weight sites, qSiLU + A8, search validation",
        xticks=[0, 1, 2],
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"qat-total-deltas.{ext}", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "epochs": epoch_rows,
                "formats": summary["formats"],
                "regions": summary["regions"],
                "output": str(OUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
