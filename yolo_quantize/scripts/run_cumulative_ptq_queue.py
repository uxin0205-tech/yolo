"""有限的十區累積 PTQ；先 CPU 準備，明確授權後才使用 GPU。"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/queues/full-model-continuous-0907"
OUT = ROOT / "artifacts/queues/full-model-cumulative-0908"
REGIONS = (
    "backbone_early",
    "backbone_deep",
    "backbone_attention_safe",
    "neck",
    "neck_attention_safe",
    "masf",
    "detect_one2one_tower",
    "pose_one2one_tower",
    "detect_one2one_predictor",
    "pose_one2one_predictor",
)
FORMATS = frozenset(
    {"fixed-sd4", "exact-scaled-ternary", "paper-twn", "twn-v3-0.75-filterwise"}
)


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    return {
        "path": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def write(path, payload, *, immutable=False):
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"拒絕覆寫不同的已固定契約：{path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def dual(metrics, accepted):
    keys = sorted(k for k in accepted if k.endswith(("/map50", "/map50_95")))
    if len(keys) != 16 or sum(k.endswith("/map50") for k in keys) != 8:
        raise ValueError("需要完整 8+8 項指標")
    for values in (metrics, accepted):
        if any(
            k not in values or not math.isfinite(values[k]) or not 0 <= values[k] <= 1
            for k in keys
        ):
            raise ValueError("指標缺漏、非有限或超出範圍")
    deltas = {k: metrics[k] - accepted[k] for k in keys}
    w50 = min(v for k, v in deltas.items() if k.endswith("/map50"))
    w95 = min(v for k, v in deltas.items() if k.endswith("/map50_95"))
    return {
        "total_deltas": deltas,
        "worst_map50_delta": w50,
        "worst_map50_95_delta": w95,
        "passes_total_gate": w50 >= -0.015 and w95 >= -0.04,
    }


def raw_risk(row):
    """只用 raw boxes/scores/kpts，不混入排序後 Top-300 或特徵尺度。"""
    if set(row["tasks"]) != {"detect", "pose"}:
        raise ValueError("probe 任務缺漏")
    values = []
    for task in row["tasks"].values():
        if not task["all_finite"] or not task["same_structure"]:
            raise ValueError("probe 結構／有限性錯誤")
        raw = [
            t["normalized_rmse"]
            for t in task["tensors"]
            if t["path"].endswith(
                (".one2one.boxes", ".one2one.scores", ".one2one.kpts")
            )
        ]
        if not raw or any(not math.isfinite(v) or v < 0 for v in raw):
            raise ValueError("raw probe 缺漏或非有限")
        values.extend(raw)
    return max(values)


def nominate(region, rows, sites):
    choices = []
    for row in rows:
        if row["region"] != region or row["format"] not in FORMATS:
            continue
        site = sites[row["path"]]
        bits = 4 if row["format"] == "fixed-sd4" else 2
        if site["encoded_bits"] <= bits:
            continue  # 排除相同／更高 bit 的無容量收益重投影。
        choices.append(
            {
                "path": row["path"],
                "format_id": row["format"],
                "raw_risk": raw_risk(row),
                "nominal_saved_bits": site["elements"] * (site["encoded_bits"] - bits),
            }
        )
    if not choices:
        raise ValueError(f"此區沒有新且有容量收益的 probe 候選：{region}")
    choices.sort(key=lambda r: (r["raw_risk"], r["path"], r["format_id"]))
    shortlist = choices[: max(1, math.ceil(len(choices) / 4))]
    return min(
        shortlist,
        key=lambda r: (
            -r["nominal_saved_bits"],
            r["raw_risk"],
            r["path"],
            r["format_id"],
        ),
    )


def extend(prefix, additions):
    if set(prefix) & set(additions):
        raise ValueError("後階段不得覆寫已接受路徑")
    return {**prefix, **additions}


def choose(records, previous_cost):
    valid = [
        r
        for r in records
        if r.get("passes_total_gate") and r["packed_weight_bytes"] < previous_cost
    ]
    return (
        min(
            valid,
            key=lambda r: (
                r["packed_weight_bytes"],
                -r["worst_map50_delta"],
                -r["worst_map50_95_delta"],
                r["candidate_id"],
            ),
        )
        if valid
        else None
    )


def prepare():
    from yolo_quantize.progressive_preparation import LockedQATParentSpec

    parent = LockedQATParentSpec.from_yaml(SOURCE / "parent-manifest.json")
    summary = read(SOURCE / "qat-recovery-summary.json")
    if summary["status"] != "completed" or len(summary["jobs"]) != 6:
        raise ValueError("六組 QAT 尚未完成")
    for job in summary["jobs"]:
        if (
            pin(Path(job["completion"]["path"])) != job["completion"]
            or len(job["epochs"]) != 5
        ):
            raise ValueError("QAT completion／epoch 證據不一致")
    evidence = read(SOURCE / "evidence-consistency-audit.json")
    if evidence["parent"] != pin(parent.config_path) or not all(
        j["executed_graph_verified"] for j in evidence["qat_jobs"]
    ):
        raise ValueError("六組實際 warm-start 尚未驗證")
    probe = read(SOURCE / "single-layer-probe.json")
    if (
        probe["parent"]["locked_qat_parent"] != pin(parent.config_path)
        or probe["status"] != "completed"
    ):
        raise ValueError("probe 不是同一 parent")
    sites = {
        s["path"]: s for s in read(SOURCE / "parent-preflight.json")["weight_sites"]
    }
    if len(sites) != 148 or set(REGIONS) != {s["region"] for s in sites.values()}:
        raise ValueError("十區／148 路徑覆蓋漂移")
    # 以已解析 PTQ 格式定義為準，不另推定論文／量化器參數。
    from yolo_quantize.mixed_policy_search import Full35MixedPolicySearchPlan

    template_path = SOURCE / "generated/isolated-special-plan.json"
    template = read(template_path)
    parsed = Full35MixedPolicySearchPlan.from_yaml(template_path)
    # to_dict 僅描述 format_id；機讀 family 契約仍沿用原始 JSON。
    formats = {
        c.region_defaults[0].spec.format_id: raw["region_defaults"][0]["format"]
        for c, raw in zip(parsed.candidates, template["candidates"], strict=True)
    }
    stages = []
    forced = {
        "pose_one2one_tower": "fixed-sd4",
        "detect_one2one_predictor": "fixed-sd4",
        "neck_attention_safe": "fixed-sd4",
        "masf": "twn-v3-0.75-filterwise",
    }
    for region in REGIONS:
        if region in forced:
            fmt = forced[region]
            additions = {p: fmt for p, s in sites.items() if s["region"] == region}
            reason = "已有同區 PTQ／短 QAT 證據；需重驗累積交互作用"
        else:
            nomination = nominate(region, probe["rows"], sites)
            additions = {nomination["path"]: nomination["format_id"]}
            reason = {
                "rule": "最低 raw 輸出誤差前 25% 中選 nominal 位元節省最大；不是 mAP 保證",
                **nomination,
            }
        stages.append({"stage_id": region, "additions": additions, "reason": reason})
    for bits in (6, 5):
        formats[f"w{bits}"] = {
            "family": "uniform",
            "bits": bits,
            "scale_method": "optimal_scaled_codebook",
        }
    payload = {
        "schema_version": 1,
        "plan_id": "full-model-cumulative-0908-v1",
        "parent": pin(parent.config_path),
        "sources": [
            pin(SOURCE / n)
            for n in (
                "single-layer-probe.json",
                "qat-recovery-summary.json",
                "evidence-consistency-audit.json",
                "parent-revalidation.json",
            )
        ],
        "runner": pin(Path(__file__).resolve()),
        "template": pin(template_path),
        "stages": stages,
        "formats": formats,
        "max_ptq_candidates": 12,
        "max_additional_qat_jobs": 2,
        "qat_epochs": 5,
        "new_training_started": False,
        "uniform_alternatives": [6, 5],
        "activation": "unchanged_qsilu_lsq_plus_a8",
        "formal_validation": False,
        "selection": "all_16_total_gates_then_smaller_codes_plus_scales_estimate",
        "lineage": "reuse_V36_export; QAT nominates formats only; never splice independently trained states",
    }
    write(OUT / "plan.json", payload, immutable=True)
    return payload, sites


def make_plan(blueprint, stage_id, candidates):
    from yolo_quantize.mixed_policy_search import Full35MixedPolicySearchPlan

    payload = read(Path(blueprint["template"]["path"]))
    payload["plan_id"] = f"cumulative-0908-{stage_id}-v1"
    payload["date"] = "2026-09-08"
    payload["routes"] = {}
    payload["candidates"] = []
    for name, assignments in candidates.items():
        routes = []
        for fmt in sorted(set(assignments.values())):
            route_id = f"{name}-{fmt}"
            route_path = OUT / "generated" / f"{route_id}-paths.json"
            write(
                route_path,
                {
                    "paths": sorted(p for p, f in assignments.items() if f == fmt),
                    "execution_authorized": True,
                    "map_validation_run": False,
                },
                immutable=True,
            )
            payload["routes"][route_id] = {
                "manifest": pin(route_path),
                "list_key": "paths",
            }
            routes.append({"route_id": route_id, "format": blueprint["formats"][fmt]})
        payload["candidates"].append(
            {"candidate_id": name, "region_defaults": [], "path_routes": routes}
        )
    payload["execution_authorization"].update(
        authorization_id="user-0908-continue-bounded-cumulative",
        candidate_ids=list(candidates),
        route_ids=list(payload["routes"]),
        stop_after="cumulative_dual_gate",
    )
    payload["continuous_contract"] = {
        "blueprint": pin(OUT / "plan.json"),
        "unmodified_paths": "inherit_locked_quantized_parent_without_reprojection",
        "stage": stage_id,
    }
    path = OUT / "generated" / f"{stage_id}-plan.json"
    write(path, payload, immutable=True)
    Full35MixedPolicySearchPlan.from_yaml(path)
    return path


def execute(blueprint, sites, *, resume):
    import yaml

    from yolo_quantize.mixed_policy_search import inherited_parent_costs

    status_path = OUT / "execution-status.json"
    if status_path.exists() and not resume:
        raise FileExistsError("已有執行狀態；明確 --resume 才可續跑")
    if status_path.exists() and resume:
        previous_status = read(status_path)
        if previous_status.get("child_pid"):
            try:
                os.kill(int(previous_status["child_pid"]), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("原 child PID 仍存在，先診斷，不啟動重複 GPU 工作")
        ledger_path = OUT / "cumulative-selection.json"
        if ledger_path.exists():
            previous_ledger = read(ledger_path)
            if previous_ledger["blueprint"] != pin(OUT / "plan.json"):
                raise ValueError("續跑的累積契約漂移")
            for step in previous_ledger["steps"]:
                if pin(Path(step["report"]["path"])) != step["report"]:
                    raise ValueError("已完成累積報告雜湊漂移")
    parent = read(Path(blueprint["parent"]["path"]))
    accepted_path = Path(
        yaml.safe_load(Path(parent["plan"]["path"]).read_text())["sources"][
            "accepted_metrics"
        ]["path"]
    )
    accepted = read(accepted_path)["metrics"]
    revalidation = read(SOURCE / "parent-revalidation.json")
    if (
        revalidation["status"] != "passed_search_parent_revalidation"
        or revalidation["parent_manifest_sha256"] != blueprint["parent"]["sha256"]
    ):
        raise ValueError("parent export 搜尋重驗無效")
    dual(revalidation["metrics"], accepted)
    current = {}
    cost = sum(inherited_parent_costs(Path(blueprint["parent"]["path"])).values())
    ledger = {
        "blueprint": pin(OUT / "plan.json"),
        "accepted_reference": pin(accepted_path),
        "parent_cost_bytes": cost,
        "steps": [],
        "formal_validation": False,
    }
    count = 0

    def status(state, stage, **extra):
        write(
            status_path,
            {
                "status": state,
                "current_index": len(ledger["steps"]),
                "current_candidate": stage,
                "current_arm": "ptq",
                "completed_jobs": count,
                "error": None,
                **extra,
            },
        )

    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONDONTWRITEBYTECODE="1",
    )
    try:
        for stage in [*blueprint["stages"], {"stage_id": "uniform-w6-w5"}]:
            name = stage["stage_id"]
            if name == "uniform-w6-w5":
                variants = {
                    f"{name}-{bits}": extend(
                        current,
                        {
                            p: f"w{bits}"
                            for p, s in sites.items()
                            if p not in current
                            and s["region"]
                            in {"backbone_early", "backbone_deep", "neck"}
                            and s["encoded_bits"] > bits
                        },
                    )
                    for bits in (6, 5)
                }
            else:
                variants = {name: extend(current, stage["additions"])}
            plan_path = make_plan(blueprint, name, variants)
            report_path = OUT / f"{name}-report.json"
            if (
                not report_path.exists()
                or read(report_path).get("status") != "completed"
            ):
                status("waiting_gpu", name)
                while subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip():
                    time.sleep(600)
                with (OUT / f"{name}.log").open("a") as stream:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "yolo_quantize.mixed_policy_search",
                            "--plan",
                            str(plan_path),
                            "--output",
                            str(report_path),
                            "--device",
                            "0",
                            "--execute-reviewed-plan",
                            "--resume",
                        ],
                        cwd=ROOT,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    status("running_cumulative_ptq", name, child_pid=child.pid)
                    while True:
                        try:
                            code = child.wait(timeout=600)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                if code:
                    raise RuntimeError(
                        f"{name} exit {code}；此錯誤事件才可檢查對應 log"
                    )
            report = read(report_path)
            if report["status"] != "completed" or set(report["results"]) != set(
                variants
            ):
                raise ValueError("累積候選終態不完整")
            records = []
            for candidate_id, result in report["results"].items():
                count += 1
                if result["status"] == "skipped_diagnostic_failed":
                    records.append(
                        {
                            "candidate_id": candidate_id,
                            "passes_total_gate": False,
                            "status": result["status"],
                        }
                    )
                    continue
                if result["build"]["locked_qat_parent"] != blueprint["parent"]:
                    raise ValueError("累積實際 build 起點漂移")
                weight = result["weight_quantization"]
                if weight["effective_quantized_modules"] != 148:
                    raise ValueError("累積配置未覆蓋 148 部署路徑")
                packed = (
                    weight["inherited_unmodified_bytes"]
                    + weight["weight_code_bytes"]
                    + weight["scale_bytes"]
                )
                records.append(
                    {
                        "candidate_id": candidate_id,
                        **dual(result["all_search_metrics"], accepted),
                        "packed_weight_bytes": packed,
                        "metrics": result["all_search_metrics"],
                    }
                )
            chosen = choose(records, cost)
            previous = dict(current)
            if chosen:
                current = variants[chosen["candidate_id"]]
                cost = chosen["packed_weight_bytes"]
            ledger["steps"].append(
                {
                    "stage": name,
                    "report": pin(report_path),
                    "previous_assignments": previous,
                    "candidate_assignments": variants,
                    "candidates": records,
                    "accepted_candidate": chosen["candidate_id"] if chosen else None,
                    "accepted_assignments": dict(current),
                    "packed_weight_bytes": cost,
                }
            )
            write(OUT / "cumulative-selection.json", ledger)
            status("stage_complete", name)
        if count > 12:
            raise ValueError("超出 PTQ 候選上限")
        status(
            "decision_required", "select_max_two_cumulative_qat", current_arm="analysis"
        )
    except Exception as error:
        status(
            "error",
            "cumulative_ptq",
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-reviewed-plan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    blueprint, sites = prepare()
    if not args.execute_reviewed_plan:
        # 真正 schema smoke：生成第一階段，不使用 CUDA 或執行驗證。
        first = blueprint["stages"][0]
        make_plan(blueprint, first["stage_id"], {first["stage_id"]: first["additions"]})
        print(
            json.dumps(
                {
                    "status": "prepared_not_gpu_started",
                    "stages": 11,
                    "max_candidates": 12,
                    "max_additional_qat": 2,
                }
            )
        )
        return
    with (
        (SOURCE / "supervisor.lock").open("a") as parent_lock,
        (OUT / "supervisor.lock").open("a") as lock,
    ):
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(blueprint, sites, resume=args.resume)


if __name__ == "__main__":
    main()
