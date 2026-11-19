"""CPU嚴格重載V36 parent，建立後續共同血緣及覆蓋證據；不啟動GPU。"""

import argparse
import copy
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
import yaml

from yolo_quantize.qat_projection import replay_parent_state
from yolo_quantize.qat_runtime import Full35QATRuntime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/queues/full-model-continuous-0907"
RUN = (
    ROOT
    / "artifacts/runs/qat/v36-qsilu-full-coverage-short-v2/v36-qsilu-full-coverage-short-qat-v1-qat-seed1"
)


def record(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = datetime.now(UTC)
    ledger_path = OUT / "implementation-start.json"
    if not ledger_path.exists():
        write(
            ledger_path,
            {
                "started_utc": started.isoformat(),
                "soft_target_utc": (started + timedelta(hours=96)).isoformat(),
                "gpu_started": False,
            },
        )
    status = {
        "schema_version": 1,
        "status": "running_cpu_preflight",
        "current_index": 0,
        "current_candidate": "v36-best-joint-export",
        "current_arm": "cpu",
        "completed_jobs": 0,
        "error": None,
    }
    write(OUT / "execution-status.json", status)
    try:
        completion = json.loads((RUN / "qat-experiment.json").read_text())
        plan_path = Path(completion["plan"])
        assert record(plan_path)["sha256"] == completion["plan_sha256"]
        inference = RUN / "inference/best_joint.pt"
        resume = RUN / "checkpoints/best_joint.pt"
        resume_record = record(resume)
        assert resume_record["sha256"] == completion["checkpoint_sha256"]["best_joint"]
        payload = torch.load(inference, map_location="cpu", weights_only=True)
        epoch = payload["metadata"]["epoch"]
        metrics_path = RUN / f"validation/epoch-{epoch:04d}/bittrue/metrics.json"
        metrics = json.loads(metrics_path.read_text())["metrics"]
        plan = yaml.safe_load(plan_path.read_text())
        accepted_path = Path(plan["sources"]["accepted_metrics"]["path"])
        assert (
            record(accepted_path)["sha256"]
            == plan["sources"]["accepted_metrics"]["sha256"]
        )
        accepted = json.loads(accepted_path.read_text())["metrics"]
        keys = sorted(k for k in accepted if k.endswith(("/map50", "/map50_95")))
        assert len(keys) == 16
        for k in keys:
            assert (
                math.isfinite(metrics[k])
                and metrics[k] == payload["metadata"]["metrics"][k]
            )
        worst50 = min(metrics[k] - accepted[k] for k in keys if k.endswith("/map50"))
        worst95 = min(metrics[k] - accepted[k] for k in keys if k.endswith("/map50_95"))
        assert worst50 >= -0.015 and worst95 >= -0.04
        runtime = Full35QATRuntime.from_yaml(plan_path)
        loaded = runtime.load_deployment_parent(
            inference,
            checkpoint_sha256=record(inference)["sha256"],
            full_resume_sha256=resume_record["sha256"],
            epoch=epoch,
        )
        state = loaded.model.state_dict()
        assert state.keys() == payload["state_dict"].keys()
        assert all(
            t.device.type == "cpu" and torch.equal(t, payload["state_dict"][k])
            for k, t in state.items()
        )
        graph = json.loads((RUN / "qat-graph.json").read_text())
        sites = graph["weight_sites"]
        assert len(sites) == len({s["path"] for s in sites}) == 148
        assert {s["path"] for s in sites} == {
            s.path for s in loaded.catalog.deployment_sites
        }
        # 由full-resume EMA獨立重建量化部署圖，不能只驗證export自己重載自己。
        resume_payload = torch.load(resume, map_location="cpu", weights_only=True)
        reference_state = replay_parent_state(
            state, resume_payload["ema_state"], sites, device=args.projection_device
        )
        mismatches = [
            key for key in state if not torch.equal(state[key], reference_state[key])
        ]
        write(
            OUT / "projection-parity.json",
            {
                "device": args.projection_device,
                "checked_weight_paths": len(sites),
                "mismatched_state_keys": mismatches,
                "inference": record(inference),
                "resume": resume_record,
                "status": "passed" if not mismatches else "failed",
            },
        )
        assert reference_state.keys() == state.keys()
        assert not mismatches, "EMA投影與export不同；CPU/GPU須分開驗證，不放寬容差掩蓋"
        reference = copy.deepcopy(loaded.model).eval()
        reference.load_state_dict(reference_state, strict=True)

        def compare(a, b):
            if isinstance(a, torch.Tensor):
                assert isinstance(b, torch.Tensor) and a.shape == b.shape
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                return 1
            if isinstance(a, dict):
                assert isinstance(b, dict) and a.keys() == b.keys()
                return sum(compare(a[k], b[k]) for k in a)
            if isinstance(a, (tuple, list)):
                assert type(a) is type(b) and len(a) == len(b)
                return sum(compare(x, y) for x, y in zip(a, b, strict=True))
            assert a == b
            return 0

        parity = {}
        loaded.model.eval()
        with torch.inference_mode():
            probe = torch.rand(
                (1, 3, 64, 64), generator=torch.Generator().manual_seed(0)
            )
            for task in ("detect", "pose"):
                parity[task] = compare(
                    reference(probe, task=task), loaded.model(probe, task=task)
                )
                assert parity[task] > 0
        inventory = []
        for name, module in loaded.model.named_modules():
            parameters = list(module.parameters(recurse=False))
            inventory.append(
                {
                    "path": name,
                    "class": type(module).__name__,
                    "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
                    "direct_parameter_elements": sum(p.numel() for p in parameters),
                    "scope": "module_inventory_not_full_operator_trace",
                }
            )
        report = {
            "status": "cpu_export_reload_passed",
            "gpu_used": args.projection_device.startswith("cuda"),
            "gpu_training_started": False,
            "projection_device": args.projection_device,
            "epoch": epoch,
            "plan": record(plan_path),
            "inference": record(inference),
            "full_resume": resume_record,
            "metrics": record(metrics_path),
            "accepted": record(accepted_path),
            "worst_total_map50_delta": worst50,
            "worst_total_map50_95_delta": worst95,
            "strict_state_reload_equal": True,
            "forward_parity": {
                "status": "passed",
                "reference": "independently_materialized_full_resume_ema",
                "input": "seed0_uniform_1x3x64x64_cpu",
                "tensor_leaves": parity,
                "accuracy_validation": False,
            },
            "catalog": loaded.catalog.summary(),
            "activation_quantizers": loaded.activation.quantizer_count,
            "weight_sites": sites,
            "modules": inventory,
            "remaining": [
                "functional_operator_precision_trace",
                "shared_parent_ptq_qat_integration",
                "five_epoch_schedule_and_live_queue",
            ],
        }
        write(OUT / "parent-preflight.json", report)
        manifest = {
            "schema_version": 1,
            "parent_id": "v36-qsilu-a8-full-coverage-epoch1",
            "status": "locked_search_parent",
            "formal_validation": False,
            "selected_epoch": epoch,
            "activation": {"name": "qsilu_pq", "bits": 8, "quantizer": "lsq_plus_a8"},
            "weight_policy": {
                "format_id": "mixed-w8-ls-sd4-w6-w4-148-paths",
                "deployment_modules": 148,
                "deployment_weight_elements": sum(s["elements"] for s in sites),
            },
            "plan": record(plan_path),
            "completion": record(RUN / "qat-experiment.json"),
            "metrics": record(metrics_path),
            "checkpoints": {
                "full_resume": resume_record,
                "inference": record(inference),
            },
            "gate": {
                "decision": "green",
                "map50_max_drop": 0.015,
                "map50_95_max_drop": 0.04,
                "worst_map50_delta": worst50,
                "worst_map50_95_delta": worst95,
            },
        }
        manifest_path = OUT / "parent-manifest.json"
        write(manifest_path, manifest)
        from yolo_quantize.progressive_preparation import LockedQATParentSpec

        LockedQATParentSpec.from_yaml(manifest_path)
        status.update(
            status="cpu_preflight_complete_pending_integration", completed_jobs=1
        )
        write(OUT / "execution-status.json", status)
        print(
            json.dumps(
                {
                    "status": status["status"],
                    "epoch": epoch,
                    "weights": len(sites),
                    "activation": loaded.activation.quantizer_count,
                    "worst50": worst50,
                    "worst95": worst95,
                }
            )
        )
    except Exception as error:
        status.update(
            status="error", error={"type": type(error).__name__, "message": str(error)}
        )
        write(OUT / "execution-status.json", status)
        raise


if __name__ == "__main__":
    main()
