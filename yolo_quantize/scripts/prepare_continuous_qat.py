"""由已完成雙gate結果建立三組5epoch QAT；固定/學習SD4公平配對。"""

import copy
import hashlib
import json
from pathlib import Path

import yaml

from yolo_quantize.qat_plan import Full35QATPlan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/queues/full-model-continuous-0907"


def record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_new(path, payload):
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    parent_path = OUT / "parent-manifest.json"
    parent = json.loads(parent_path.read_text())
    base = yaml.safe_load(Path(parent["plan"]["path"]).read_text())
    report_path = OUT / "isolated-special-report.json"
    summary = json.loads((OUT / "special-dual-summary.json").read_text())
    records = {r["candidate"]: r for r in summary["candidates"]}
    dual = {
        "status": "completed",
        "source": {"report_sha256": record(report_path)["sha256"]},
        "candidates": {
            name: {**r, "decision": "green" if r.get("passes_total_gate") else "reject"}
            for name, r in records.items()
        },
    }
    dual_path = OUT / "special-qat-dual-gate.json"
    write_new(dual_path, dual)
    sites = json.loads((OUT / "parent-preflight.json").read_text())["weight_sites"]
    jobs = []
    choices = [
        ("pose-sd4-fixed", "pose_one2one_tower", "fixed-sd4", True),
        ("pose-ls-sd4", "pose_one2one_tower", "fixed-sd4", False),
        ("masf-exact-ternary", "masf", "exact-ternary", False),
    ]
    for name, region, fmt, fixed in choices:
        candidate = f"isolated-{region}-{fmt}"
        if not records[candidate].get("passes_total_gate"):
            raise ValueError(f"candidate not dual-green: {candidate}")
        payload = copy.deepcopy(base)
        payload.update(
            plan_id=f"continuous-0907-{name}-5ep-v1",
            date="2026-09-07",
            run_root="artifacts/runs/qat/continuous-0907-special-v1",
        )
        payload["execution_authorization"]["authorization_id"] = (
            "user-0907-adaptive-five-epoch-qat"
        )
        payload["execution_authorization"]["scope"] = (
            "same_v36_parent_special_formats_no_new_sham"
        )
        payload["sources"].update(
            weight_plan=record(OUT / "generated/isolated-special-plan.json"),
            candidate_evidence=record(report_path),
            candidate_dual_regate=record(dual_path),
        )
        assignments = [
            copy.deepcopy(a)
            for a in base["weight_policy"]["assignments"]
            if not a.get("paths")
        ]
        changed = []
        for site in sites:
            if site["region"] == region:
                spec = {
                    "family": "ls_sd4"
                    if fmt == "fixed-sd4"
                    else "exact_scaled_ternary",
                    "scale_method": "optimal_scaled_codebook",
                }
                changed.append(site["path"])
            elif site["format_id"] == "fixed-sd4":
                spec = {"family": "ls_sd4", "scale_method": "optimal_scaled_codebook"}
            else:
                spec = {
                    "family": "uniform",
                    "bits": site["encoded_bits"],
                    "scale_method": "optimal_scaled_codebook",
                }
            assignments.append(
                {"region": site["region"], "paths": [site["path"]], "format": spec}
            )
        payload["weight_policy"] = {
            "candidate_id": candidate,
            "float_regions": [],
            "assignments": assignments,
        }
        payload["warm_start"].update(
            locked_parent=record(parent_path), reset_paths=changed
        )
        payload["frozen_weight_scale_paths"] = changed if fixed else []
        payload["training"].update(
            epochs=5,
            patience=5,
            warmup_epochs=1,
            scale_only_epochs=1,
            progressive_start_epoch=0,
            progressive_full_epoch=1,
        )
        payload["quick_recovery_provenance"] = {
            "parent_manifest": record(parent_path),
            "short_qat_epochs": 5,
            "selection": "same_parent_dual_green",
            "fixed_selected_weight_scales": fixed,
            "reset_only_paths": changed,
            "no_new_sham": True,
        }
        payload["successor_provenance"] = {
            "parent": parent["parent_id"],
            "scope": "148_weight_paths_quantized",
            "pair_id": "pose-sd4-scale-learning"
            if region == "pose_one2one_tower"
            else None,
        }
        path = OUT / "generated" / f"{name}-qat-plan.json"
        write_new(path, payload)
        plan = Full35QATPlan.from_yaml(path)
        assert plan.training.epochs == 5 and plan.warm_start.reset_paths == tuple(
            changed
        )
        jobs.append(
            {
                "job_id": name,
                "candidate_id": candidate,
                "plan": record(path),
                "epochs": 5,
                "reset_weight_scales": len(changed),
                "frozen_weight_scales": len(plan.frozen_weight_scale_paths),
                "status": "pending_preflight",
            }
        )
    write_new(
        OUT / "selected-qat-jobs.json",
        {
            "status": "prepared_not_started",
            "parent": parent["parent_id"],
            "prerequisite": "592_single_layer_probes_and_qat_preflight",
            "jobs": jobs,
            "new_sham_training": False,
        },
    )
    print(json.dumps(jobs))


if __name__ == "__main__":
    main()
