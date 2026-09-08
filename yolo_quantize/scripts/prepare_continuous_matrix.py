"""建立同V36 parent的十區特殊格式PTQ計畫；只做CPU schema驗證。"""

import copy
import hashlib
import json
from pathlib import Path

import yaml

from yolo_quantize.mixed_policy_search import Full35MixedPolicySearchPlan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/queues/full-model-continuous-0907"


def main():
    report = json.loads((OUT / "parent-preflight.json").read_text())
    if report.get("forward_parity", {}).get("status") != "passed":
        raise RuntimeError("export forward parity尚未通過，不能生成可執行矩陣")
    parent_path = OUT / "parent-manifest.json"
    parent = json.loads(parent_path.read_text())
    base = yaml.safe_load(
        (
            ROOT
            / "artifacts/queues/v36-qsilu-full-coverage-progressive-v1/generated/special-plan.yaml"
        ).read_text()
    )
    continuous = json.loads(
        (ROOT / "configs/experiments/full-model-continuous-0907.json").read_text()
    )
    regions = sorted({s["region"] for s in report["weight_sites"]})
    assert len(regions) == 10
    # 沿用歷史已解析的格式定義，並只使用格式，不沿用其選層routes。
    formats = {}
    for candidate in base["candidates"]:
        for route in candidate["path_routes"]:
            route_id = route["route_id"]
            if route_id.startswith("special-"):
                key = route_id.split("-", 2)[2]
                formats[key] = route["format"]
    assert set(formats) == {"fixed-sd4", "exact-ternary", "twn-v3", "paper-twn-v2"}
    candidates = [
        {
            "candidate_id": f"isolated-{region}-{fmt}",
            "region_defaults": [{"region": region, "format": spec}],
            "path_routes": [],
        }
        for region in regions
        for fmt, spec in formats.items()
    ]
    payload = copy.deepcopy(base)
    payload.update(
        plan_id="continuous-0907-isolated-special-v1",
        date="2026-09-07",
        routes={},
        candidates=candidates,
    )
    payload["activation"]["checkpoint"] = parent["checkpoints"]["inference"]
    payload["locked_qat_parent"] = {
        "path": str(parent_path),
        "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
    }
    payload["execution_authorization"].update(
        authorization_id="user-0907-continuous-implementation",
        candidate_ids=[c["candidate_id"] for c in candidates],
        route_ids=[],
        stop_after="special_dual_regate",
    )
    payload["continuous_contract"] = {
        "config": str(ROOT / "configs/experiments/full-model-continuous-0907.json"),
        "epochs_for_selected_qat": continuous["training"]["epochs"],
        "unmodified_paths": "inherit_locked_quantized_parent_without_reprojection",
        "old_comparison_candidates": "historical_sources_only_excluded_from_selection",
    }
    destination = OUT / "generated/isolated-special-plan.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if destination.exists() and destination.read_text() != encoded:
        raise FileExistsError("新計畫與既有檔案不同，請用新版本而非覆蓋")
    destination.write_text(encoded)
    plan = Full35MixedPolicySearchPlan.from_yaml(destination)
    assert len(plan.candidates) == 40 and plan.locked_qat_parent == parent_path
    print(
        json.dumps(
            {
                "status": "schema_validated_not_gpu_enqueued",
                "candidates": len(plan.candidates),
                "plan": str(destination),
                "parent": parent["parent_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
