"""只重新評估鎖定export的搜尋精度，不重訓baseline，不使用formal split。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import yaml

from yolo_quantize.mixed_policy_search import (
    Full35MixedPolicySearchPlan,
    build_locked_parent,
)
from yolo_quantize.progressive_preparation import LockedQATParentSpec
from yolo_quantize.search_evidence import read_dual_search_metrics
from yolo_quantize.search_validation import _prepare_contract, _validate_role

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/queues/full-model-continuous-0907"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover-completed-report", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    plan = Full35MixedPolicySearchPlan.from_yaml(
        OUT / "generated/isolated-special-plan.json"
    )
    parent = LockedQATParentSpec.from_yaml(OUT / "parent-manifest.json")
    if args.recover_completed_report:
        raw_path = (
            ROOT
            / "artifacts/runs/continuous-0907-parent-revalidation-v1/validation/parent/epoch-0000/bittrue/metrics.json"
        )
        record = {
            "metrics_report": str(raw_path),
            "metrics_report_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        }
        contract = {
            "recovered_observed_completed_attempt": True,
            "plan_sha256": plan.config_sha256,
            **record,
        }
        build = {
            "inference_sha256": parent.inference_sha256,
            "parent_manifest_sha256": parent.config_sha256,
        }
    else:
        torch.cuda.set_device(0)
        prepared, contract = _prepare_contract(plan, device_index=0)
        model, source, _, build = build_locked_parent(plan)
        model.cuda(0).eval()
        record = _validate_role(
            role="parent",
            model=model,
            source=source,
            plan=plan,
            pose_yaml=prepared.yaml,
            output_root=ROOT
            / "artifacts/runs/continuous-0907-parent-revalidation-v1/validation",
            device_index=0,
        )
    reference = json.loads(parent.metrics_path.read_text())["metrics"]
    old_plan = yaml.safe_load(parent.plan_path.read_text())
    accepted = json.loads(
        Path(old_plan["sources"]["accepted_metrics"]["path"]).read_text()
    )["metrics"]
    keys = sorted(k for k in reference if k.endswith(("/map50", "/map50_95")))
    assert len(keys) == 16
    actual = read_dual_search_metrics(record)
    assert all(math.isfinite(actual[k]) for k in keys)
    drift = {k: actual[k] - reference[k] for k in keys}
    total = {k: actual[k] - accepted[k] for k in keys}
    passed = max(abs(v) for v in drift.values()) <= 1e-4 and all(
        v >= (-0.015 if k.endswith("/map50") else -0.04) for k, v in total.items()
    )
    payload = {
        "status": "passed_search_parent_revalidation"
        if passed
        else "requires_parent_drift_analysis",
        "parent_manifest_sha256": parent.config_sha256,
        "metrics": actual,
        "drift_from_recorded_parent": drift,
        "total_deltas": total,
        "reproduction_tolerance": 1e-4,
        "contract": contract,
        "build": build,
        "formal_validation": False,
        "training_run": False,
    }
    (OUT / "parent-revalidation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    if not passed:
        raise RuntimeError("parent重載accuracy有漂移或超過總門檻，先分析而非開40格矩陣")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "maximum_drift": max(abs(v) for v in drift.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
