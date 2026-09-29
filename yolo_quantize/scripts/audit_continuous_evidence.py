"""CPU 稽核本輪證據血緣；不重新訓練、不把 proxy 當作 mAP。"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/queues/full-model-continuous-0907"


def load(path):
    return json.loads(path.read_text())


def record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_build(build, parent, inference):
    if (
        build["locked_qat_parent"] != parent
        or build["checkpoint_sha256"] != inference["sha256"]
    ):
        raise ValueError("實際 build 的 parent/export 不一致")
    if (
        build.get("activation_recalibrated") is not False
        or build.get("activation_quantizers") != 124
    ):
        raise ValueError("activation 重校準或覆蓋不一致")


def main():
    parent_path = OUT / "parent-manifest.json"
    parent_record = record(parent_path)
    parent = load(parent_path)
    for source in parent["checkpoints"].values():
        if record(Path(source["path"])) != source:
            raise ValueError("parent checkpoint hash 漂移")
    inference = parent["checkpoints"]["inference"]
    ptq_path = OUT / "isolated-special-report.json"
    ptq = load(ptq_path)
    if ptq["status"] != "completed" or len(ptq["results"]) != 40:
        raise ValueError("十區 PTQ 不完整")
    for result in ptq["results"].values():
        verify_build(result["build"], parent_record, inference)
    if ptq["incremental_reference"]["sha256"] != parent["metrics"]["sha256"]:
        raise ValueError("PTQ 增量基準不是鎖定 parent")
    probe_path = OUT / "single-layer-probe.json"
    probe = load(probe_path)
    verify_build(probe["parent"], parent_record, inference)
    paths = defaultdict(set)
    for row in probe["rows"]:
        paths[row["path"]].add(row["format"])
    if (
        probe["status"] != "completed"
        or len(probe["rows"]) != 592
        or len(paths) != 148
        or any(len(v) != 4 for v in paths.values())
    ):
        raise ValueError("單層輸出探測覆蓋不完整")
    if probe["map_validation"] is not False:
        raise ValueError("probe 不得冒稱 mAP")
    diagnostic = ptq["contract"]["diagnostic"]
    if probe["manifest_sha256"] != diagnostic["manifest_sha256"]:
        raise ValueError("probe manifest 不一致")
    distribution = load(OUT / "weight-distribution.json")
    if (
        distribution["parent_manifest_sha256"] != parent_record["sha256"]
        or distribution["inference_sha256"] != inference["sha256"]
    ):
        raise ValueError("CPU 分布 parent 不一致")
    revalidation = load(OUT / "parent-revalidation.json")
    if (
        revalidation["parent_manifest_sha256"] != parent_record["sha256"]
        or revalidation["status"] != "passed_search_parent_revalidation"
    ):
        raise ValueError("parent 精度重驗紀錄不一致")
    jobs = []
    for job in load(OUT / "selected-qat-jobs-v2.json")["jobs"]:
        plan_path = Path(job["plan"]["path"])
        if record(plan_path) != job["plan"]:
            raise ValueError("QAT plan hash 不一致")
        plan = load(plan_path)
        warm = plan["warm_start"]
        if (
            warm["locked_parent"] != parent_record
            or warm["state_key"] != "ema_state"
            or warm["checkpoint_role"] != "full_resume"
        ):
            raise ValueError("QAT 起點不同")
        run = (
            ROOT
            / plan["run_root"]
            / f"{plan['plan_id']}-qat-seed{plan['training']['seed']}"
        )
        graph_path = run / "qat-graph.json"
        executed = graph_path.exists()
        legacy_label = None
        if executed:
            graph = load(graph_path)
            warm = graph["warm_start"]
            if (
                graph["plan_sha256"] != job["plan"]["sha256"]
                or warm["parent_manifest_sha256"] != parent_record["sha256"]
                or warm["checkpoint_sha256"]
                != parent["checkpoints"]["full_resume"]["sha256"]
                or warm["state_key"] != "ema_state"
            ):
                raise ValueError("QAT 實際 warm-start 與計畫不一致")
            legacy_label = warm.get("activation_calibration")
        jobs.append(
            {
                "job_id": job["job_id"],
                "plan_verified": True,
                "executed_graph_verified": executed,
                "legacy_calibration_label": legacy_label,
            }
        )
    result = {
        "status": "passed_identity_audit_not_new_accuracy_validation",
        "parent": parent_record,
        "parent_id": parent["parent_id"],
        "inference": inference,
        "ptq_source": record(ptq_path),
        "ptq_same_parent_results": 40,
        "probe_source": record(probe_path),
        "single_layer_output_probe_paths": 148,
        "single_layer_output_probe_rows": 592,
        "single_layer_map_matrix_completed": False,
        "qat_jobs": jobs,
        "warnings": [
            "歷史 1332 筆屬 V35 CPU 權重投影，不是本輪 V36 的逐層 mAP。",
            "PTQ reference 保留歷史參照；本輪 incremental_reference 與各 result.build 才記錄實際鎖定 parent。",
            "舊 selection 僅 mAP50；本輪晉級必須讀 special-dual-summary 的 16 項雙門檻。",
            "warm_start.activation_calibration 的 v19 字樣是沿用標籤；實際 parent/checkpoint 雜湊已核對為 V36。",
            "同 parent 不代表 QAT 只更新該層，QAT 允許模型及量化參數適應；不得視為 PTQ 單層隔離實驗。",
        ],
        "gpu_used": False,
    }
    (OUT / "evidence-consistency-audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "ptq": 40,
                "probe": 592,
                "qat_plans": len(jobs),
                "qat_executed_graphs": sum(j["executed_graph_verified"] for j in jobs),
            }
        )
    )


if __name__ == "__main__":
    main()
