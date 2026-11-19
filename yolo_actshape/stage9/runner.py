"""單 GPU 條件式 queue；每 600 秒監測，資源衝突只停止自己的 worker。"""

import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .common import ROOT, HERE, RUN, protocol, read, write, digest, guard, score
from .policy import choose_training


def gpu_processes():
    output = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
    ], text=True)
    return [{"pid": int(row.split(",")[0]), "MiB": row.split(",")[1].strip()}
            for row in output.splitlines() if row.strip()]


def gpu_snapshot():
    return subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits"
    ], text=True).strip()


def outsiders(worker_pid=None):
    result = []
    for row in gpu_processes():
        try:
            own = worker_pid is not None and os.getpgid(row["pid"]) == worker_pid
        except ProcessLookupError:
            continue
        if not own:
            result.append(row)
    return result


def stop_worker(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def resume_valid(output):
    """Only a committed complete epoch with matching code/config may retry a SIGSEGV."""
    output = Path(output)
    try:
        value = read(output / "resume.json")
        return (value["next_epoch"] >= 2
                and value["sha256"] == digest(output / "resume.pt")
                and value["config_sha256"] == digest(output / "config.json")
                and value["source_sha256"] == digest(RUN / "source.json"))
    except (OSError, KeyError, ValueError):
        return False


def retry_allowed(code, attempts, maximum, complete_epoch):
    return code == -signal.SIGSEGV and attempts < maximum and complete_epoch


def job(spec):
    guard()
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / "job.json"
    if spec_path.exists() and read(spec_path) != spec:
        raise ValueError(f"job 配置改變：{output}")
    write(spec_path, spec)
    done = output / "done.json"
    if done.exists():
        saved = read(done)
        if saved["job_sha256"] != digest(spec_path) or saved["source_sha256"] != digest(RUN / "source.json"):
            raise ValueError("完成標記來源不符")
        if saved["result"].get("best"):
            best = saved["result"]["best"]
            if digest(ROOT / best["checkpoint"]) != best["sha256"]:
                raise ValueError("已完成 job 的 best 遺失或被改寫")
        return saved["result"]
    policy = protocol()["queue"]
    attempt_path = output / "attempts.json"
    attempts = read(attempt_path) if attempt_path.exists() else {"sigsegv_retries": 0, "resource_interruptions": 0, "starts": 0}
    while True:
        if (RUN / "STOP").exists():
            raise InterruptedError("使用者 STOP")
        external = outsiders()
        if external:
            write(RUN / "status.json", {"status": "waiting_gpu", "job": spec["id"], "external": external, "time_unix": time.time()})
            time.sleep(policy["resource_check_seconds"])
            continue
        guard()
        attempts["starts"] += 1
        write(attempt_path, attempts)
        with (output / "worker.log").open("a", buffering=1) as log:
            child = subprocess.Popen([str(HERE / "run_stage9.sh"), "worker", "--spec", str(spec_path)],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, start_new_session=True)
            start, last_monitor, resource_pause = time.time(), 0, False
            try:
                while child.poll() is None:
                    now = time.time()
                    state = {"status": "running", "job": spec["id"], "worker_pid": child.pid,
                             "supervisor_pid": os.getpid(), "started_unix": start, "time_unix": now,
                             "output": str(output), "attempts": attempts}
                    write(RUN / "status.json", state)
                    external = outsiders(child.pid)
                    if external or (RUN / "STOP").exists():
                        stop_worker(child)
                        if (RUN / "STOP").exists():
                            raise InterruptedError("STOP；保留完整 epoch")
                        attempts["resource_interruptions"] += 1
                        write(attempt_path, attempts)
                        resource_pause = True
                        break
                    if now - last_monitor >= policy["monitor_seconds"]:
                        progress_paths = list(output.rglob("progress.json"))
                        newest = max(progress_paths, key=lambda p: p.stat().st_mtime) if progress_paths else None
                        report = {**state, "gpu": gpu_snapshot(),
                                  "progress": read(newest) if newest else None}
                        with (RUN / "monitor.jsonl").open("a") as stream:
                            stream.write(json.dumps(report, ensure_ascii=False) + "\n")
                        last_monitor = now
                        if now - (newest.stat().st_mtime if newest else start) > 7200:
                            stop_worker(child)
                            raise TimeoutError("兩小時無進度；停止此 queue 供診斷")
                    try:
                        child.wait(timeout=policy["resource_check_seconds"])
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                stop_worker(child)
        if resource_pause:
            continue
        if child.returncode == 0 and done.exists():
            return read(done)["result"]
        if retry_allowed(child.returncode, attempts["sigsegv_retries"], policy["max_sigsegv_retries_per_job"],
                         spec["action"] == "train" and resume_valid(output)):
            attempts["sigsegv_retries"] += 1
            write(attempt_path, attempts)
            continue
        raise RuntimeError(f"{spec['id']} worker 失敗 rc={child.returncode}；詳見 {output / 'worker.log'}")


def make_spec(action, candidate, parent, name=None, **extra):
    name = name or action
    return {"id": f"{candidate}/{name}", "name": name, "candidate": candidate, "action": action,
            "shape": protocol()["candidates"][candidate]["shape"], "parent": parent,
            "output": str(RUN / candidate / name), **extra}


def run_queue():
    p = protocol()
    if (RUN / "status.json").exists() and read(RUN / "status.json").get("status") == "complete":
        return
    if any(read(RUN / name)["status"] != "passed" for name in
           ("cpu-audit.json", "tests.json", "integration.json", "portable-smoke.json")):
        raise ValueError("CPU contract 尚未通過")
    guard(create=True)
    write(RUN / "validation-lock.json", {name: digest(RUN / name) for name in
          ("cpu-audit.json", "tests.json", "integration.json", "portable-smoke.json", "data-audit.json")})
    zero = {}
    for name in p["candidates"]:
        zero[name] = job(make_spec("evaluate", name, p["parent"], "e0"))
        if name == "reference_rect640":
            drift = max(abs(zero[name]["metrics"][k] - v) for k, v in p["parent"]["metrics"].items())
            if drift > 1e-5:
                raise ValueError(f"原流程重驗漂移 {drift}，先查 evaluator／checkpoint")
    chosen, reasons = choose_training(p["candidates"], zero, p["parent"]["metrics"], p["selection"], score)
    write(RUN / "screening.json", {"selected": chosen, "reasons": reasons,
                                   "max_training_epochs": p["selection"]["max_training_epochs"],
                                   "zero_shot": {n: r["metrics"] for n, r in zero.items()}})
    completed = {}
    for name in chosen:
        recipe = copy.deepcopy(p["training"])
        parent = {**p["parent"], "metrics": zero[name]["metrics"], "candidate": name,
                  "shape": p["candidates"][name]["shape"], "epoch": 0}
        calibration = job(make_spec("calibrate", name, parent, recipe=recipe))
        smoke = job(make_spec("smoke", name, parent, recipe=recipe, calibration=calibration))
        write(RUN / name / "budget-estimate.json", {"measured_smoke_seconds_per_macro": smoke["seconds_per_macro"],
                "initial_epochs": recipe["epochs"], "possible_low_lr_epochs": recipe["low_lr_epochs"],
                "macro_per_epoch": 463, "train_seconds_per_epoch_estimate": 463 * smoke["seconds_per_macro"],
                "eval_seconds_per_epoch_measured": zero[name]["detect"]["seconds"] + zero[name]["pose"]["seconds"],
                "peak_reserved_GiB": smoke["peak_reserved_GiB"]})
        initial = job(make_spec("train", name, parent, f"adapt{recipe['epochs']}", recipe=recipe, calibration=calibration))
        best = initial["best"]
        low_lr = None
        if initial["status"] == "complete":
            low_recipe = {**recipe, "epochs": recipe["low_lr_epochs"], "loss_horizon": recipe["low_lr_epochs"],
                          "lrs": {k: v / 2 for k, v in recipe["lrs"].items()}}
            # New parent must get its own gradient confirmation before a new training segment.
            low_cal = job(make_spec("calibrate", name, best, "low-lr-calibrate", recipe=low_recipe))
            low_lr = job(make_spec("train", name, best, f"low-lr{low_recipe['epochs']}", recipe=low_recipe, calibration=low_cal))
            best = low_lr["best"]
        verified = job(make_spec("verify", name, best, "verify"))
        completed[name] = {"initial": initial, "low_lr": low_lr, "best": best, "verification": verified}
        write(RUN / "trained.json", completed)
    from .report import finalize
    result = finalize(zero, completed)
    write(RUN / "status.json", {"status": "complete", "time_unix": time.time(), "result": result,
                               "next_stage_started": False})


def worker(spec_path):
    import torch
    from binaryqk.run import seed
    from .model import load, verify
    from .evaluate import pair
    from .training import calibrate, smoke, train
    from .diagnostics import finite
    guard()
    spec = read(spec_path)
    if outsiders(os.getpid()):
        raise RuntimeError("GPU 有其他運算，worker 不啟動")
    if digest(ROOT / spec["parent"]["checkpoint"]) != spec["parent"]["sha256"]:
        raise ValueError("job parent SHA 改變")
    seed(protocol()["seed"])
    if spec["action"] in ("evaluate", "verify"):
        model = load(spec["parent"]["checkpoint"]).cuda().eval()
        result = pair(model, spec["candidate"], spec["shape"], spec["output"], diagnostics=True)
        if spec["action"] == "verify":
            drift = max(abs(result["metrics"][k] - v) for k, v in spec["parent"]["metrics"].items())
            if drift > 1e-5:
                raise ValueError(f"shape best 重載 AP 不符 {drift}")
            result["reload_ap_max_error"] = drift
            result["operators_verified"] = bool(verify(model))
    else:
        result = {"calibrate": calibrate, "smoke": smoke, "train": train}[spec["action"]](spec)
    guard()
    write(Path(spec["output"]) / "done.json", {"result": result, "job_sha256": digest(spec_path),
                                              "source_sha256": digest(RUN / "source.json"), "time_unix": time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cpu", "run", "worker", "status", "test"))
    parser.add_argument("--spec")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        print(json.dumps(read(RUN / "status.json") if (RUN / "status.json").exists() else {"status": "not_started"}, ensure_ascii=False, indent=2))
    elif args.action == "test":
        import unittest
        from . import tests
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        write(RUN / "tests.json", {"status": "passed" if result.wasSuccessful() else "failed", "tests": result.testsRun})
        if not result.wasSuccessful():
            raise SystemExit(1)
    elif args.action == "cpu":
        from .diagnostics import cpu_audit, data_audit
        cpu_audit(); data_audit()
        print("PASS CPU shape／D1／cache／完整 val metadata")
    elif args.action == "worker":
        worker(Path(args.spec))
    else:
        with (RUN / "queue.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(RUN / "supervisor.json", {"pid": os.getpid(), "started_unix": time.time()})
            try:
                run_queue()
            except BaseException as exc:
                write(RUN / "status.json", {"status": "stopped" if isinstance(exc, InterruptedError) else "failed",
                        "error": str(exc), "traceback": traceback.format_exc(), "time_unix": time.time()})
                raise


if __name__ == "__main__":
    main()
