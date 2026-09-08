"""保留原三組計畫，新增三元配對與未達標 PTQ 的有限 QAT 恢復實驗。"""

import copy
import json

from prepare_continuous_qat import OUT, record, write_new

from yolo_quantize.qat_plan import Full35QATPlan


def main():
    probe_path = OUT / "single-layer-probe.json"
    probe = json.loads(probe_path.read_text())
    if probe["status"] != "completed" or len(probe["rows"]) != 592:
        raise ValueError("逐層探測尚未完整")
    parent = record(OUT / "parent-manifest.json")
    if probe["parent"]["locked_qat_parent"] != parent:
        raise ValueError("逐層探測 parent 不一致")
    selected = json.loads((OUT / "selected-qat-jobs.json").read_text())
    dual = json.loads((OUT / "special-qat-dual-gate.json").read_text())
    recovery = "isolated-detect_one2one_predictor-fixed-sd4"
    dual["candidates"][recovery].update(
        decision="recover",
        selection_reason="使用者核准有限 QAT 恢復試驗；PTQ 未達部署門檻，不改寫實測結果",
    )
    dual_path = OUT / "special-qat-recovery-admission.json"
    write_new(dual_path, dual)
    sites = json.loads((OUT / "parent-preflight.json").read_text())["weight_sites"]
    region_paths = {}
    for site in sites:
        region_paths.setdefault(site["region"], []).append(site["path"])
    template = json.loads((OUT / "generated/pose-ls-sd4-qat-plan.json").read_text())
    masf = json.loads((OUT / "generated/masf-exact-ternary-qat-plan.json").read_text())
    choices = [
        (
            "masf-paper-twn",
            "masf",
            "paper-twn-v2",
            {"family": "paper_twn", "threshold_multiplier": 0.7},
        ),
        (
            "masf-twn",
            "masf",
            "twn-v3",
            {"family": "twn_filterwise", "threshold_multiplier": 0.75},
        ),
        (
            "detect-predictor-ls-sd4-recovery",
            "detect_one2one_predictor",
            "fixed-sd4",
            {"family": "ls_sd4", "scale_method": "optimal_scaled_codebook"},
        ),
    ]
    # 先還原為 inherited parent assignments，再替換單一 region。
    inherited = copy.deepcopy(template["weight_policy"]["assignments"])
    original = {
        a["paths"][0]: a for a in masf["weight_policy"]["assignments"] if a.get("paths")
    }
    for i, assignment in enumerate(inherited):
        if assignment.get("paths") and assignment["region"] == "pose_one2one_tower":
            inherited[i] = copy.deepcopy(original[assignment["paths"][0]])
    for name, region, fmt, spec in choices:
        payload = copy.deepcopy(template)
        payload["plan_id"] = f"continuous-0907-{name}-5ep-v1"
        payload["sources"]["candidate_dual_regate"] = record(dual_path)
        candidate = f"isolated-{region}-{fmt}"
        assignments = copy.deepcopy(inherited)
        for assignment in assignments:
            if assignment.get("paths") and assignment["region"] == region:
                assignment["format"] = spec
        payload["weight_policy"].update(candidate_id=candidate, assignments=assignments)
        payload["warm_start"]["reset_paths"] = region_paths[region]
        payload["frozen_weight_scale_paths"] = []
        payload["quick_recovery_provenance"].update(
            selection="bounded_ptq_recovery"
            if candidate == recovery
            else "same_region_ternary_comparison",
            reset_only_paths=region_paths[region],
            single_layer_probe=record(probe_path),
        )
        payload["successor_provenance"]["pair_id"] = (
            "masf-ternary-formats" if region == "masf" else None
        )
        path = OUT / "generated" / f"{name}-qat-plan.json"
        write_new(path, payload)
        plan = Full35QATPlan.from_yaml(path)
        assert plan.training.epochs == 5
        selected["jobs"].append(
            {
                "job_id": name,
                "candidate_id": candidate,
                "plan": record(path),
                "epochs": 5,
                "reset_weight_scales": len(region_paths[region]),
                "frozen_weight_scales": 0,
                "status": "pending_preflight",
            }
        )
    selected.update(
        status="prepared_not_started",
        single_layer_probe=record(probe_path),
        prerequisite="per_job_cpu_graph_preflight",
        maximum_jobs=6,
    )
    write_new(OUT / "selected-qat-jobs-v2.json", selected)
    print(json.dumps({"jobs": len(selected["jobs"]), "status": selected["status"]}))


if __name__ == "__main__":
    main()
