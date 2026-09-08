"""同parent 148×4單層輸出敏感度；不是592次mAP驗證。"""

import fcntl
import json
import subprocess
import time
from pathlib import Path

import torch

from yolo_quantize.activation_smoke import _compare_outputs, _letterbox
from yolo_quantize.mixed_policy_search import (
    Full35MixedPolicySearchPlan,
    build_locked_parent,
)
from yolo_quantize.weight_quantization import (
    Full35WeightRegionCatalog,
    WeightQuantizationAdapter,
    WeightRegionAssignment,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/queues/full-model-continuous-0907"


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    status = {
        "schema_version": 1,
        "status": "waiting_gpu",
        "current_index": 0,
        "current_candidate": "single_layer_probe",
        "current_arm": "probe",
        "completed_jobs": 40,
        "error": None,
    }
    with (OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            atomic(OUT / "execution-status.json", status)
            while subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip():
                time.sleep(600)
            plan = Full35MixedPolicySearchPlan.from_yaml(
                OUT / "generated/isolated-special-plan.json"
            )
            model, _, _, build = build_locked_parent(plan)
            model.cuda(0).eval()
            catalog = Full35WeightRegionCatalog.inspect(model)
            formats = {
                c.region_defaults[0].spec.format_id: c.region_defaults[0].spec
                for c in plan.candidates
            }
            assert len(formats) == 4 and len(catalog.deployment_sites) == 148
            manifest = plan.weight_study.load_diagnostic_manifest(verify_files=True)
            probes = {
                task: _letterbox(
                    manifest.paths("probe", source)[0], plan.image_size
                ).cuda(0)
                for task, source in (("detect", "coco_detect"), ("pose", "bbat_pose"))
            }
            with torch.inference_mode():
                references = {
                    task: model(image, task=task) for task, image in probes.items()
                }
            destination = OUT / "single-layer-probe.json"
            result = {
                "status": "running",
                "parent": build,
                "manifest_sha256": manifest.sha256,
                "image_size": plan.image_size,
                "map_validation": False,
                "rows": [],
            }
            if destination.exists():
                previous = json.loads(destination.read_text())
                if (
                    previous["parent"] != build
                    or previous["manifest_sha256"] != manifest.sha256
                ):
                    raise ValueError("probe resume parent or data changed")
                result["rows"] = previous["rows"]
            done = {(r["path"], r["format"]) for r in result["rows"]}
            adapter = WeightQuantizationAdapter()
            for site in catalog.deployment_sites:
                for fmt, spec in formats.items():
                    if (site.path, fmt) in done:
                        continue
                    with adapter.quantized_policy(
                        model,
                        catalog=catalog,
                        assignments=(
                            WeightRegionAssignment(
                                site.region, spec, paths=(site.path,)
                            ),
                        ),
                    ), torch.inference_mode():
                        comparisons = {
                            task: _compare_outputs(
                                references[task], model(image, task=task)
                            )
                            for task, image in probes.items()
                        }
                    result["rows"].append(
                        {
                            "path": site.path,
                            "region": site.region,
                            "format": fmt,
                            "tasks": comparisons,
                        }
                    )
                    atomic(destination, result)
                    status.update(
                        status="running_single_layer_probe",
                        current_index=len(result["rows"]),
                        current_candidate=f"{site.path}:{fmt}",
                        completed_jobs=40 + len(result["rows"]),
                    )
                    atomic(OUT / "execution-status.json", status)
            assert len(result["rows"]) == 592
            result["status"] = "completed"
            atomic(destination, result)
            status.update(
                status="decision_required",
                current_candidate="single_layer_probe_complete",
                completed_jobs=632,
            )
            atomic(OUT / "execution-status.json", status)
            print(
                json.dumps(
                    {"status": "completed", "probe_cells": 592, "map_validation": False}
                )
            )
        except Exception as error:
            status.update(
                status="error",
                error={"type": type(error).__name__, "message": str(error)},
            )
            atomic(OUT / "execution-status.json", status)
            raise


if __name__ == "__main__":
    main()
