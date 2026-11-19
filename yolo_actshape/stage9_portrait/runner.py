"""單一新增方向候選：E0 → replay → 校準 → smoke → 10E → 半 LR 10E。"""

import argparse
import copy
import fcntl
import os
import time
import traceback
from pathlib import Path

from stage9.runner import outsiders
from .common import ROOT, HERE, CONFIG, RUN, NAME, protocol, read, write, digest, guard, stopped, training_run, queue_engine


def state(status, **extra):
    value = dict(status=status, supervisor_pid=os.getpid(), time_unix=time.time(),
                 run=str(RUN), current_epoch_cap=20, prior_completed_epochs=100,
                 total_epoch_cap=120, **extra)
    write(RUN / "status.json", value)
    write(HERE / "queue-status.json", value)
    return value


def recovered_calibration(spec):
    """只接住原規則已明確失敗且還原 parent 的結果，其他例外原樣上拋。"""
    try:
        with training_run() as engine:
            return engine.calibrate(spec)
    except ValueError:
        path = Path(spec["output"]) / "result.json"
        failed = read(path) if path.exists() else {}
        if not (failed.get("status") == "needs_review" and failed.get("parent_restored") is True
                and failed.get("validation_used") is False
                and failed.get("parent_sha256") == spec["parent"]["sha256"]):
            raise
    extra = Path(spec["output"]) / "bounded-recovery"
    attempt = extra / "attempt.json"
    if attempt.exists():
        raise ValueError("此校準已執行過一次補充，停止供診斷")
    write(attempt, {"failed_result_sha256": digest(path), "policy": protocol()["calibration_recovery"],
                    "source_sha256": digest(RUN / "source.json"), "time_unix": time.time()})
    with training_run(calibration_recovery=True) as engine:
        result = engine.calibrate({**spec, "output": str(extra)})
    result = {**result, "calibration_recovery": {
        "original_result": str(path), "original_result_sha256": digest(path),
        "supplemental_result": str(extra / "result.json"),
        "supplemental_result_sha256": digest(extra / "result.json"),
        "policy": protocol()["calibration_recovery"], "extra_epochs": 0}}
    write(Path(spec["output"]) / "effective-result.json", result)
    return result


def worker(spec_path):
    spec = read(spec_path)
    if spec["candidate"] != NAME or spec["shape"] != [768, 512]:
        raise ValueError("worker 的方向或候選不符")
    guard()
    if outsiders(os.getpid()):
        raise RuntimeError("GPU 有其他工作，等待資源")
    if digest(ROOT / spec["parent"]["checkpoint"]) != spec["parent"]["sha256"]:
        raise ValueError("parent SHA 不符")
    if spec["action"] == "calibrate":
        result = recovered_calibration(spec)
        guard()
        write(Path(spec["output"]) / "done.json", {"result": result, "job_sha256": digest(spec_path),
              "source_sha256": digest(RUN / "source.json"), "time_unix": time.time()})
    else:
        with queue_engine(state) as engine:
            engine.worker(spec_path)


def run_queue():
    guard()
    validation = read(RUN / "preflight.json")
    if validation["status"] != "passed" or validation["source_sha256"] != digest(RUN / "source.json"):
        raise ValueError("CPU preflight 尚未通過")
    write(RUN / "validation-lock.json", {"preflight.json": digest(RUN / "preflight.json")})
    p = protocol()
    with (ROOT / p["dependency"]["exclusive_queue_lock"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with queue_engine(state) as engine:
            spec = engine.make_spec
            zero = engine.job(spec("evaluate", NAME, p["parent"], "e0"))
            layout = engine.job(spec("layout", NAME, p["parent"]))
            recipe = copy.deepcopy(p["training"])
            parent = {**p["parent"], "metrics": zero["metrics"], "candidate": NAME,
                      "shape": [768, 512], "epoch": 0}
            calibration = engine.job(spec("calibrate", NAME, parent, recipe=recipe))
            smoke = engine.job(spec("smoke", NAME, parent, recipe=recipe, calibration=calibration))
            write(RUN / "budget-estimate.json", {
                "measured_smoke_seconds_per_macro": smoke["seconds_per_macro"], "macro_per_epoch": 463,
                "train_seconds_per_epoch_estimate": 463 * smoke["seconds_per_macro"],
                "eval_seconds_per_epoch_measured": zero["detect"]["seconds"] + zero["pose"]["seconds"],
                "initial_epochs": 10, "low_lr_epochs": 10, "peak_reserved_GiB": smoke["peak_reserved_GiB"]})
            initial = engine.job(spec("train", NAME, parent, "adapt10", recipe=recipe, calibration=calibration))
            best, low = initial["best"], None
            if initial["status"] == "complete":
                lower = {**recipe, "epochs": 10, "loss_horizon": 10,
                         "lrs": {key: value / 2 for key, value in recipe["lrs"].items()}}
                low_calibration = engine.job(spec("calibrate", NAME, best, "low-lr-calibrate", recipe=lower))
                low = engine.job(spec("train", NAME, best, "low-lr10", recipe=lower, calibration=low_calibration))
                best = low["best"]
            verified = engine.job(spec("verify", NAME, best))
        trained = {"initial": initial, "low_lr": low, "best": best, "verification": verified}
        write(RUN / "trained.json", {NAME: trained})
        state("finalizing", job="直向候選比較與 CPU 可攜交接")
        from .report import finalize
        result = finalize(zero, trained, layout)
        state("complete", result=result, next_stage_started=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run", "worker", "status"))
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        import json
        print(json.dumps(read(RUN / "status.json") if (RUN / "status.json").exists()
                         else {"status": "not_started"}, ensure_ascii=False, indent=2))
    elif args.action == "preflight":
        from .validation import run
        run()
    elif args.action == "worker":
        if args.spec is None:
            raise ValueError("worker 缺少 spec")
        worker(args.spec)
    else:
        with (RUN / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (RUN / "status.json").exists() and read(RUN / "status.json")["status"] == "complete":
                return
            write(RUN / "supervisor.json", {"pid": os.getpid(), "started_unix": time.time()})
            try:
                run_queue()
            except BaseException as exc:
                state("stopped" if isinstance(exc, InterruptedError) else "failed",
                      error=str(exc), traceback=traceback.format_exc())
                raise


if __name__ == "__main__":
    main()
