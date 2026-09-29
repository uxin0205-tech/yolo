"""共同parent第一批PTQ supervisor；600秒狀態更新，完成後要求分析再選QAT。"""

import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from yolo_quantize.mixed_policy_search import Full35MixedPolicySearchPlan
from yolo_quantize.progressive_preparation import LockedQATParentSpec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/queues/full-model-continuous-0907"
PLAN = OUT / "generated/isolated-special-plan.json"
REPORT = OUT / "isolated-special-report.json"


def write_status(**updates):
    path = OUT / "execution-status.json"
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload.update(schema_version=1, **updates)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = None
        try:
            parent = LockedQATParentSpec.from_yaml(OUT / "parent-manifest.json")
            preflight = json.loads((OUT / "parent-preflight.json").read_text())
            trace = json.loads((OUT / "operator-precision.json").read_text())
            if (
                preflight["forward_parity"]["status"] != "passed"
                or preflight["inference"]["sha256"] != parent.inference_sha256
            ):
                raise ValueError("parent/export preflight mismatch")
            if (
                trace["status"] != "passed_diagnostic_precision_inventory"
                or trace["parent_manifest_sha256"] != parent.config_sha256
            ):
                raise ValueError("operator trace parent mismatch")
            plan = Full35MixedPolicySearchPlan.from_yaml(PLAN)
            assert (
                len(plan.candidates) == 40
                and plan.locked_qat_parent_sha256 == parent.config_sha256
            )
            write_status(
                status="waiting_gpu",
                current_index=0,
                current_candidate="isolated-special",
                current_arm="ptq",
                completed_jobs=0,
                error=None,
            )
            while True:
                busy = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                if not busy:
                    break
                time.sleep(600)
            environment = dict(os.environ)
            environment.update(
                PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"
            )
            parent_validation = OUT / "parent-revalidation.json"
            if not parent_validation.exists():
                with (OUT / "parent-validation.log").open("a") as stream:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            str(ROOT / "scripts/validate_continuous_parent.py"),
                        ],
                        cwd=ROOT,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    write_status(
                        status="running_parent_validation",
                        current_candidate="v36-export-search-revalidation",
                        child_pid=child.pid,
                        supervisor_pid=os.getpid(),
                    )
                    while True:
                        try:
                            code = child.wait(timeout=600)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                    if code:
                        raise RuntimeError(
                            f"parent validation exit {code}; inspect parent-validation.log"
                        )
            verified = json.loads(parent_validation.read_text())
            if (
                verified.get("status") != "passed_search_parent_revalidation"
                or verified.get("parent_manifest_sha256") != parent.config_sha256
            ):
                raise ValueError("parent accuracy revalidation has not passed")
            command = [
                sys.executable,
                "-m",
                "yolo_quantize.mixed_policy_search",
                "--plan",
                str(PLAN),
                "--output",
                str(REPORT),
                "--device",
                "0",
                "--execute-reviewed-plan",
                "--resume",
            ]
            with (OUT / "special-ptq.log").open("a") as stream:
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                write_status(
                    status="running_ptq",
                    child_pid=child.pid,
                    supervisor_pid=os.getpid(),
                )
                while True:
                    try:
                        code = child.wait(timeout=600)
                        break
                    except subprocess.TimeoutExpired:
                        if REPORT.exists():
                            progress = json.loads(REPORT.read_text())
                            completed = len(progress.get("results", {}))
                            write_status(
                                current_candidate=progress.get(
                                    "current_candidate", "isolated-special"
                                ),
                                current_index=completed,
                                completed_jobs=completed,
                            )
                if code:
                    raise RuntimeError(
                        f"PTQ child exit {code}; inspect special-ptq.log on this error event"
                    )
            report = json.loads(REPORT.read_text())
            results = report.get("results", {})
            if len(results) != 40 or report.get("status") != "completed":
                raise ValueError("PTQ terminal artifact incomplete")
            source_plan = yaml.safe_load(parent.plan_path.read_text())
            accepted = json.loads(
                Path(source_plan["sources"]["accepted_metrics"]["path"]).read_text()
            )["metrics"]
            baseline = json.loads(parent.metrics_path.read_text())["metrics"]
            keys = sorted(k for k in accepted if k.endswith(("/map50", "/map50_95")))
            assert len(keys) == 16
            summary = []
            for candidate in plan.candidates:
                record = results[candidate.candidate_id]
                if record["status"] == "skipped_diagnostic_failed":
                    summary.append(
                        {
                            "candidate": candidate.candidate_id,
                            "status": record["status"],
                        }
                    )
                    continue
                metrics = record["all_search_metrics"]
                assert all(k in metrics and math.isfinite(metrics[k]) for k in keys)
                total = {k: metrics[k] - accepted[k] for k in keys}
                incremental = {k: metrics[k] - baseline[k] for k in keys}
                w50 = min(total[k] for k in keys if k.endswith("/map50"))
                w95 = min(total[k] for k in keys if k.endswith("/map50_95"))
                summary.append(
                    {
                        "candidate": candidate.candidate_id,
                        "status": "evaluated",
                        "total_deltas": total,
                        "incremental_deltas": incremental,
                        "passes_total_gate": w50 >= -0.015 and w95 >= -0.04,
                        "worst_map50_delta": w50,
                        "worst_map50_95_delta": w95,
                    }
                )
            (OUT / "special-dual-summary.json").write_text(
                json.dumps(
                    {
                        "parent_sha256": parent.config_sha256,
                        "candidates": summary,
                        "formal_validation": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            write_status(
                status="decision_required",
                current_index=40,
                current_candidate="select_same_parent_short_qat",
                current_arm="analysis",
                completed_jobs=40,
                error=None,
            )
        except Exception as error:
            write_status(
                status="error",
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "child_pid": child.pid if child else None,
                },
            )
            raise


if __name__ == "__main__":
    main()
