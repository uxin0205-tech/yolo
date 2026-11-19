"""Stage6 固定 checkpoint 分組回退；完整 AP，不加訓或修改來源。"""

import copy
import gc
from pathlib import Path

import torch
from binaryqk.run import require_idle_gpu, seed
from stage6 import checkpoint
from stage6.common import ROOT, event, read, sha256, write_json
from stage6.evaluate import COCO_KEYS, pair
from stage6.models import manifest_for


def predicate(case, name, parameters):
    head = ".detect_head." in name
    bn = name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    return {
        "reproduce": False,
        "restore_detect_bn": head and bn,
        "restore_detect_parameters": head and name in parameters,
        "restore_detect_head": head,
        "restore_neck": name.startswith("graph.model.")
        and 11 <= int(name.split(".")[2]) <= 22,
    }[case]


def main():
    require_idle_gpu()
    torch.set_num_threads(2)
    output = ROOT / "yolo_combine/artifacts/stage6/j1-review-20261029"
    audit = read(output / "reference-review.json")
    for path, digest in audit["references_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("reference source changed")
    if sha256(audit["resume"]) != audit["resume_sha256"]:
        raise ValueError("resume changed")
    saved = torch.load(audit["resume"], map_location="cpu", weights_only=True)
    if saved["next_epoch"] != audit["epoch"] + 1:
        raise ValueError("epoch mismatch")
    initial = checkpoint.load(audit["parent"]["checkpoint"])
    initial_state = initial.state_dict()
    parameters = set(dict(initial.named_parameters()))
    trained = saved["ema"]
    manifest = copy.deepcopy(initial.stage6_manifest)
    config = saved["config"]
    manifest.setdefault("training_lineage", []).append(
        {
            "parent": config["parent"]["checkpoint"],
            "parent_sha256": config["parent"]["sha256"],
            "recipe": config["recipe"],
            "pose_weight": config["calibration"]["selected_pose_weight"],
            "initial_ema_updates": 0,
            "completed_epochs": audit["epoch"],
            "ema_updates": saved["ema_updates"],
            "resume_sha256": audit["resume_sha256"],
            "execution_source_lock_sha256": sha256(
                Path(audit["resume"]).parent / "source.json"
            ),
        }
    )
    del saved
    results = {}
    for case in audit["cases"]:
        seed(20261003)
        model = checkpoint.load(audit["parent"]["checkpoint"])
        selected = [n for n in trained if predicate(case, n, parameters)]
        state = {
            n: initial_state[n] if n in selected else v for n, v in trained.items()
        }
        model.load_state_dict(state, strict=True)
        if any(not torch.equal(v, model.state_dict()[n]) for n, v in state.items()):
            raise ValueError("rollback state mismatch")
        model.stage6_manifest = copy.deepcopy(manifest)
        model.stage6_manifest["rollback_diagnostic"] = {
            "case": case,
            "reference_review_sha256": sha256(output / "reference-review.json"),
            "restored_names": selected,
            "not_accepted_automatically": True,
        }
        manifest_for(model)
        case_output = output / case
        case_output.mkdir(exist_ok=True)
        # Keep the actual latest trained candidate and one whole-head candidate for review.
        if case in ("reproduce", "restore_detect_head"):
            checkpoint.save(case_output / "model.pt", model)
        event(output, kind="rollback-evaluation", case=case, restored=len(selected))
        metrics = pair(model.cuda(), case_output / "evaluation", progress_root=output)[
            "metrics"
        ]
        if case == "reproduce" and metrics != audit["expected_metrics"]:
            raise ValueError("E6 full AP reproduction differs")
        if case.startswith("restore_detect") and any(
            metrics[k] != audit["expected_metrics"][k]
            for k in metrics
            if k not in COCO_KEYS
        ):
            raise ValueError("Detect-only rollback changed Pose")
        results[case] = {"metrics": metrics, "restored_tensors": len(selected)}
        write_json(output / "results.json", results)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        output / "result.json",
        {
            "status": "complete",
            "results": results,
            "interpretation": "固定checkpoint反事實診斷；不是恢復訓練已有效的證據",
        },
    )
    event(output, kind="diagnosis-complete")


if __name__ == "__main__":
    main()
