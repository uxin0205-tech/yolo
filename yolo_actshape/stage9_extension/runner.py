"""串接初輪之後的擴充 queue；原始配方與來源鎖完全不改。"""

import argparse
import copy
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from stage9.runner import outsiders, gpu_snapshot, stop_worker, retry_allowed
from .common import ROOT, HERE, RUN, BASE_RUN, protocol, read, write, digest, guard, training_run, stopped


def state(status, **extra):
    value = {"status": status, "supervisor_pid": os.getpid(), "time_unix": time.time(),
             "run": str(RUN), "initial_run": str(BASE_RUN), "total_epoch_cap": 100, **extra}
    write(RUN / "status.json", value)
    write(HERE / "queue-status.json", value)
    return value


def monitor(value):
    with (RUN / "monitor.jsonl").open("a") as stream:
        stream.write(json.dumps({**value, "gpu": gpu_snapshot()}, ensure_ascii=False) + "\n")


def acquire_initial(lock):
    """Return only after initial publication/cleanup releases its actual flock."""
    last_monitor = 0
    p = protocol()["queue"]
    while True:
        if stopped():
            raise InterruptedError("STOP；尚未啟動擴充 GPU")
        initial = read(BASE_RUN / "status.json")
        if initial["status"] in ("failed", "stopped"):
            raise RuntimeError(f"初輪尚未完成：{initial['status']}，請先診斷初輪")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            acquired = False
        else:
            acquired = True
        if acquired:
            # Re-read under the exclusive lock; avoid the status/lock race.
            initial = read(BASE_RUN / "status.json")
            if initial["status"] != "complete":
                fcntl.flock(lock, fcntl.LOCK_UN)
                raise RuntimeError("初輪 queue 已無持鎖程序，但未完成；不繞過未完成訓練")
            return
        value = state("waiting_initial_queue", initial=initial, job="等待初輪完成與釋放同一GPU鎖")
        if time.time() - last_monitor >= p["monitor_seconds"]:
            monitor(value)
            last_monitor = time.time()
        time.sleep(p["resource_check_seconds"])


def resume_valid(output):
    try:
        value = read(output / "resume.json")
        return (value["next_epoch"] >= 2 and value["sha256"] == digest(output / "resume.pt")
                and value["config_sha256"] == digest(output / "config.json")
                and value["source_sha256"] == digest(RUN / "source.json"))
    except (OSError, KeyError, ValueError):
        return False


def completed(spec_path):
    result = read(spec_path.parent / "done.json")
    if result["job_sha256"] != digest(spec_path) or result["source_sha256"] != digest(RUN / "source.json"):
        raise ValueError("完成標記來源不符")
    best = result["result"].get("best")
    if best and digest(ROOT / best["checkpoint"]) != best["sha256"]:
        raise ValueError("best 權重遺失或改變")
    return result["result"]


def job(spec):
    guard()
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    spec_path, done = output / "job.json", output / "done.json"
    if spec_path.exists() and read(spec_path) != spec:
        raise ValueError(f"不能更改已排定 job：{spec['id']}")
    write(spec_path, spec)
    if done.exists():
        return completed(spec_path)
    policy = protocol()["queue"]
    attempts_path = output / "attempts.json"
    attempts = read(attempts_path) if attempts_path.exists() else {
        "sigsegv_retries": 0, "resource_interruptions": 0, "starts": 0}
    last_wait_monitor = 0
    while True:
        if stopped():
            raise InterruptedError("STOP；保留完整 epoch")
        external = outsiders()
        if external:
            value = state("waiting_gpu", job=spec["id"], external=external)
            if time.time() - last_wait_monitor >= policy["monitor_seconds"]:
                monitor(value)
                last_wait_monitor = time.time()
            time.sleep(policy["resource_check_seconds"])
            continue
        guard()
        attempts["starts"] += 1
        write(attempts_path, attempts)
        with (output / "worker.log").open("a", buffering=1) as log:
            child = subprocess.Popen([str(HERE / "run_extension.sh"), "worker", "--spec", str(spec_path)],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
            start, last_monitor, resource_pause = time.time(), 0, False
            try:
                while child.poll() is None:
                    value = state("running", job=spec["id"], worker_pid=child.pid, started_unix=start,
                                  output=str(output), attempts=attempts)
                    external = outsiders(child.pid)
                    if external or stopped():
                        stop_worker(child)
                        if stopped():
                            raise InterruptedError("STOP；保留完整 epoch")
                        attempts["resource_interruptions"] += 1
                        write(attempts_path, attempts)
                        resource_pause = True
                        break
                    if time.time() - last_monitor >= policy["monitor_seconds"]:
                        paths = list(output.rglob("progress.json"))
                        newest = max(paths, key=lambda x: x.stat().st_mtime) if paths else None
                        monitor({**value, "progress": read(newest) if newest else None})
                        last_monitor = time.time()
                        if time.time() - (newest.stat().st_mtime if newest else start) > 7200:
                            raise TimeoutError("兩小時無進度，停止供診斷")
                    try:
                        child.wait(timeout=policy["resource_check_seconds"])
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                stop_worker(child)
        if resource_pause:
            continue
        if child.returncode == 0 and done.exists():
            return completed(spec_path)
        if retry_allowed(child.returncode, attempts["sigsegv_retries"], policy["max_sigsegv_retries_per_job"],
                         spec["action"] == "train" and resume_valid(output)):
            attempts["sigsegv_retries"] += 1
            write(attempts_path, attempts)
            continue
        raise RuntimeError(f"{spec['id']} 失敗 rc={child.returncode}，詳見 {output / 'worker.log'}")


def make_spec(action, candidate, parent, name=None, **extra):
    name = name or action
    shapes = {**protocol()["layout"]["canvas_shapes"],
              **{n: c["shape"] for n, c in protocol()["candidates"].items()}}
    return {"id": f"{candidate}/{name}", "name": name, "candidate": candidate, "action": action,
            "shape": shapes[candidate], "parent": parent, "output": str(RUN / candidate / name), **extra}


def run_queue():
    guard()
    tests = read(RUN / "tests.json")
    if tests["status"] != "passed" or tests["source_sha256"] != digest(RUN / "source.json"):
        raise ValueError("CPU 驗證未通過或程式在驗證後改變")
    write(RUN / "validation-lock.json", {"tests.json": digest(RUN / "tests.json")})
    # Holding this through finalization also prevents the initial queue being restarted concurrently.
    with (BASE_RUN / "queue.lock").open("a") as lock:
        acquire_initial(lock)
        from .report import mark_running, finalize
        mark_running()
        p, zero, trained, layouts = protocol(), {}, {}, {}
        for name in p["candidates"]:
            zero[name] = job(make_spec("evaluate", name, p["parent"], "e0"))
        for name in p["layout"]["canvas_shapes"]:
            layouts[name] = job(make_spec("layout", name, p["parent"]))
            write(RUN / "layout-results.json", layouts)
        for name, candidate in p["candidates"].items():
            recipe = copy.deepcopy(p["training"])
            parent = {**p["parent"], "metrics": zero[name]["metrics"], "candidate": name,
                      "shape": candidate["shape"], "epoch": 0}
            cal = job(make_spec("calibrate", name, parent, recipe=recipe))
            smoke = job(make_spec("smoke", name, parent, recipe=recipe, calibration=cal))
            write(RUN / name / "budget-estimate.json", {
                "measured_smoke_seconds_per_macro": smoke["seconds_per_macro"], "macro_per_epoch": 463,
                "train_seconds_per_epoch_estimate": 463 * smoke["seconds_per_macro"],
                "eval_seconds_per_epoch_measured": zero[name]["detect"]["seconds"] + zero[name]["pose"]["seconds"],
                "initial_epochs": 10, "low_lr_epochs": 10, "peak_reserved_GiB": smoke["peak_reserved_GiB"]})
            initial = job(make_spec("train", name, parent, "adapt10", recipe=recipe, calibration=cal))
            best, low = initial["best"], None
            if initial["status"] == "complete":
                lower = {**recipe, "epochs": recipe["low_lr_epochs"], "loss_horizon": recipe["low_lr_epochs"],
                         "lrs": {k: v / 2 for k, v in recipe["lrs"].items()}}
                low_cal = job(make_spec("calibrate", name, best, "low-lr-calibrate", recipe=lower))
                low = job(make_spec("train", name, best, "low-lr10", recipe=lower, calibration=low_cal))
                best = low["best"]
            verified = job(make_spec("verify", name, best))
            trained[name] = {"initial": initial, "low_lr": low, "best": best, "verification": verified}
            write(RUN / "trained.json", trained)
        state("finalizing", job="整合初輪與擴充結果、交接與清理")
        result = finalize(zero, trained, layouts)
        state("complete", result=result, next_stage_started=False)


def worker(spec_path):
    from binaryqk.run import seed
    from stage9.model import load, verify
    from stage9.evaluate import pair
    guard()
    spec = read(spec_path)
    if outsiders(os.getpid()):
        raise RuntimeError("GPU 有其他運算，worker 不啟動")
    if digest(ROOT / spec["parent"]["checkpoint"]) != spec["parent"]["sha256"]:
        raise ValueError("parent SHA 改變")
    seed(protocol()["seed"])
    if spec["action"] in ("evaluate", "verify"):
        model = load(spec["parent"]["checkpoint"]).cuda().eval()
        result = pair(model, spec["candidate"], spec["shape"], spec["output"], diagnostics=True)
        if spec["action"] == "verify":
            drift = max(abs(result["metrics"][k] - v) for k, v in spec["parent"]["metrics"].items())
            if drift > 1e-5:
                raise ValueError(f"重載 AP 漂移 {drift}")
            result.update(reload_ap_max_error=drift, operators_verified=bool(verify(model)))
    elif spec["action"] == "layout":
        from .layout import run
        result = run(spec)
    else:
        with training_run() as engine:
            result = {"calibrate": engine.calibrate, "smoke": engine.smoke, "train": engine.train}[spec["action"]](spec)
    guard()
    write(Path(spec["output"]) / "done.json", {"result": result, "job_sha256": digest(spec_path),
                                              "source_sha256": digest(RUN / "source.json"), "time_unix": time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "worker", "status", "test", "preflight"))
    parser.add_argument("--spec")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        print(json.dumps(read(RUN / "status.json") if (RUN / "status.json").exists() else {"status": "not_started"}, ensure_ascii=False, indent=2))
    elif args.action == "worker":
        worker(Path(args.spec))
    elif args.action == "preflight":
        guard(create=True)
        print("PASS 初輪鎖不變；擴充來源鎖已固定。")
    elif args.action == "test":
        import unittest
        from . import tests
        guard(create=True)
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        guard()
        write(RUN / "tests.json", {"status": "passed" if result.wasSuccessful() else "failed",
                                  "tests": result.testsRun, "source_sha256": digest(RUN / "source.json")})
        if not result.wasSuccessful():
            raise SystemExit(1)
    else:
        with (RUN / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (RUN / "status.json").exists() and read(RUN / "status.json")["status"] == "complete":
                return
            write(RUN / "supervisor.json", {"pid": os.getpid(), "started_unix": time.time()})
            try:
                run_queue()
            except BaseException as exc:
                state("stopped" if isinstance(exc, InterruptedError) else "failed", error=str(exc), traceback=traceback.format_exc())
                raise


if __name__ == "__main__":
    main()
