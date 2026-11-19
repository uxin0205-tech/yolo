"""覆核已停止的 H2；不改原 queue、不續訓。plot 僅讀小型 JSON。"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def plot(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((folder / "audit.json").read_text())
    keys = list(data["parent_metrics"])
    values = [100 * data["delta"][k] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(keys, values)
    ax.axvline(-2, color="red", linestyle="--", label="safety stop: -2 pp")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set(xlabel="Epoch 1 minus parent (percentage points)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "delta.svg")
    plt.close(fig)


def verify(output):
    import time

    import torch
    from binaryqk.data import dataset
    from binaryqk.evaluate import evaluate
    from binaryqk.run import require_idle_gpu, seed
    from hog_yolo import checkpoint
    from hog_yolo.common import read, review, sha256, source_guard, write_json
    from hog_yolo.models import configure, strip, strip_proof
    from hog_yolo.training import HOGStudy
    from yolo_attention.stage3_kd_training import metric_vector

    torch.set_num_threads(2)
    source_guard(output)
    data = review()
    folder = output / "safety-stop-review"
    folder.mkdir(exist_ok=False)
    status = read(output / "training/status.json")
    if (
        status.get("error")
        != "ValueError: HOG AP safety stop; completed epoch retained"
    ):
        raise ValueError("only the recorded AP safety stop can be closed")
    resume_path = output / "training/resume.pt"
    digest = sha256(resume_path)
    if digest != read(resume_path.with_suffix(".json"))["sha256"]:
        raise ValueError("resume SHA mismatch")
    payload = torch.load(resume_path, map_location="cpu", weights_only=True)
    if payload["next_epoch"] != 1 or payload["best"]["epoch"] != 0:
        raise ValueError("this audit requires the recorded one-epoch negative result")
    baseline = read(output / "training/metrics-00.json")
    current = read(output / "training/metrics-01.json")
    if payload["history"][-1]["evaluation"] != current:
        raise ValueError("resume and formal metrics disagree")
    policy = HOGStudy(output / "calibration/calibration.json")
    policy.guard_metrics(baseline, [], False)
    try:
        policy.guard_metrics(current, payload["history"], False)
    except ValueError as error:
        if str(error) != "HOG AP safety stop; completed epoch retained":
            raise
    else:
        raise AssertionError("AP stop not reproduced")
    parent_metrics, metrics = metric_vector(baseline), metric_vector(current)
    delta = {k: metrics[k] - parent_metrics[k] for k in parent_metrics}
    model = checkpoint.build(output / "calibration/calibration.json")
    if checkpoint.manifest_for(model) != payload["manifest"]:
        raise ValueError("resume manifest differs from the fixed parent contract")
    configure(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    initial = model.state_dict()
    drift = {
        name: [
            key
            for key in initial
            if key not in trainable and not torch.equal(initial[key], state[key])
        ]
        for name, state in (
            ("live", payload["live_state"]),
            ("ema", payload["ema_state"]),
        )
    }
    if any(drift.values()):
        raise ValueError(f"frozen tensor drift: {drift}")
    updates = {
        name: float((payload["ema_state"][name] - initial[name]).abs().max())
        for name in (*policy.gradient_names(model), "hog_aux.projection.weight")
    }
    if any(value <= 0 for value in updates.values()):
        raise ValueError("must verify actually trained epoch 1, not epoch 0")
    model.load_state_dict(payload["ema_state"], strict=True)
    model.eval()
    proof = strip_proof(model)
    deployment = strip(model)
    checkpoint.save(
        folder / "epoch01-stripped.pt", deployment, checkpoint.manifest_for(deployment)
    )
    restored, _ = checkpoint.load(folder / "epoch01-stripped.pt")
    if any(
        not torch.equal(value, restored.state_dict()[key])
        for key, value in deployment.state_dict().items()
    ):
        raise ValueError("deployment reload state mismatch")
    proof.update(
        {"trained_epoch": 1, "ema_update_abs_max": updates, "reload_state_exact": True}
    )
    write_json(folder / "strip-proof.json", proof)
    require_idle_gpu()
    seed(data["protocol"]["seed"])
    ds, data_yaml, _ = dataset("val", 8)
    last_monitor = 0

    def progress(event):
        nonlocal last_monitor
        now = time.time()
        write_json(folder / "progress.json", {"time_unix": now, **event})
        if now - last_monitor >= 600:
            print(json.dumps({"time_unix": now, **event}), flush=True)
            last_monitor = now

    measured = evaluate(
        restored.to("cuda:0"), ds, data_yaml, folder / "evaluation", progress=progress
    )
    if (
        metric_vector(measured) != metrics
        or measured["official"] != current["official"]
    ):
        raise ValueError("epoch 1 stripped reload AP mismatch")
    write_json(folder / "metrics.json", measured)
    audit = {
        "status": "completed-with-negative-result",
        "stage": 5,
        "next_stage_started": False,
        "original_queue_status": "failed at AP guard; not rewritten",
        "finished_unix": time.time(),
        "source_guard_passed": True,
        "closure_script_sha256": sha256(__file__),
        "resume_sha256": digest,
        "resume": str(resume_path),
        "parent": data["parent"],
        "parent_sha256": data["parent_sha256"],
        "parent_metrics": parent_metrics,
        "candidate_metrics": metrics,
        "delta": delta,
        "triggered_metrics": [k for k, value in delta.items() if value < -0.02],
        "history": payload["history"],
        "best": payload["best"],
        "completed_epochs": 1,
        "candidate_epoch": 1,
        "candidate": str(folder / "epoch01-stripped.pt"),
        "candidate_sha256": sha256(folder / "epoch01-stripped.pt"),
        "stop_reason": "AP safety stop",
        "frozen_tensor_drift": drift,
        "strip_proof": proof,
        "formal_reload_images": 5000,
        "formal_reload_AP_exact": True,
        "adopt_hog_recommendation": False,
        "limitation": "single seed; one epoch; no unchanged control; HOG-specific causal attribution unresolved",
    }
    source_guard(output)
    write_json(folder / "audit.json", audit)
    with (folder / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["metric", "parent", "epoch1", "delta"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "metric": k,
                "parent": parent_metrics[k],
                "epoch1": metrics[k],
                "delta": delta[k],
            }
            for k in delta
        )
    plot(folder)
    print(
        json.dumps(
            {"status": audit["status"], "formal_reload_AP_exact": True, "delta": delta}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "plot"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "yolo_hog/artifacts/stage5"):
        parser.error("Stage5 run directory required")
    if args.action == "plot":
        plot(output / "safety-stop-review")
    else:
        verify(output)
