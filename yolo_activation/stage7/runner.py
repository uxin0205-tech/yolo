"""第七階段條件式 queue；獨立 worker、600 秒監測與單卡資源協調。"""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .common import (
    ARTIFACTS,
    CANDIDATES,
    HERE,
    REGIONS,
    ROOT,
    event,
    passed,
    protocol,
    rank,
    read,
    recipe,
    sha256,
    source_guard,
    write_json,
)


def gpu_processes():
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    return [line.strip() for line in query.stdout.splitlines() if line.strip()]


def worker(spec):
    import torch
    from binaryqk.run import require_idle_gpu, seed

    from . import model as checkpoint
    from .evaluate import pair

    source_guard(spec["root"])
    torch.set_num_threads(2)
    require_idle_gpu()
    seed(protocol()["seed"])
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    event(output, kind="worker-start", action=spec["action"])
    action = spec["action"]
    if action in ("zero", "region"):
        model = checkpoint.create(spec["candidate"], spec.get("region"))
        path = output / "model.pt"
        if action == "zero":
            if path.exists():
                model = checkpoint.load(path)
                if (
                    model.stage7_manifest["activation"]["candidate"]
                    != spec["candidate"]
                ):
                    raise ValueError("候選 checkpoint 身分不符")
            else:
                checkpoint.save(path, model)
        result = pair(
            model.cuda(),
            output / "evaluation",
            progress_root=output,
            diagnostics=action == "zero",
        )
        result.update(
            status="passed",
            candidate=spec["candidate"],
            region=spec.get("region"),
            phase=action,
        )
        if action == "zero":
            result.update(checkpoint=str(path.resolve()), sha256=sha256(path))
        write_json(output / "result.json", result)
        return result
    if action == "calibrate":
        from .calibration import calibrate

        return calibrate(output, spec["parent"], spec["recipe"])
    if action == "smoke":
        from .training import smoke

        return smoke(output, spec["parent"], spec["recipe"], spec["calibration"])
    if action == "train":
        from .training import train

        return train(
            output, spec["parent"], spec["recipe"], spec["calibration"], spec["root"]
        )
    if action == "verify":
        parent = spec["parent"]
        if sha256(parent["checkpoint"]) != parent["sha256"]:
            raise ValueError("選定模型 SHA 改變")
        model = checkpoint.load(parent["checkpoint"]).cuda()
        result = pair(
            model, output / "evaluation", progress_root=output, diagnostics=True
        )
        difference = {k: result["metrics"][k] - v for k, v in parent["metrics"].items()}
        if any(abs(v) > 1e-9 for v in difference.values()):
            raise ValueError(f"重載完整 AP 不同：{difference}")
        result.update(
            status="passed",
            parent=parent,
            difference=difference,
            manifest=checkpoint.manifest_for(model),
        )
        write_json(output / "result.json", result)
        return result
    raise ValueError(action)


def run(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_guard(root, create=True)
    for name in ("numerics.json", "cpu-contracts.json"):
        if read(HERE / name)["status"] != "passed":
            raise ValueError(f"缺少通過的 CPU 驗證：{name}")
    state = {
        "pid": os.getpid(),
        "status": "running",
        "started_at": time.time(),
        "stage": 7,
        "next_stage_started": False,
        "monitor_seconds": 600,
        "max_training_epochs": 45,
    }

    def update(**values):
        state.update(values, updated_at=time.time())
        write_json(root / "supervisor.json", state)

    def stop_check():
        if (root / "STOP").exists():
            raise InterruptedError("使用者 STOP；未啟動下一階段")

    def terminate(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=20)

    def job(name, action, **args):
        source_guard(root)
        stop_check()
        output = root / name
        output.mkdir(parents=True, exist_ok=True)
        spec = {"root": str(root), "output": str(output), "action": action, **args}
        specfile = output / "job.json"
        if specfile.exists() and read(specfile) != spec:
            raise ValueError(f"既有 job 配置改變：{name}")
        write_json(specfile, spec)
        if (output / "result.json").exists():
            return read(output / "result.json")
        while True:
            while gpu_processes():
                update(
                    status="waiting_gpu",
                    job=name,
                    gpu_processes=gpu_processes(),
                    worker_pid=None,
                )
                stop_check()
                time.sleep(5)
            with (output / "worker.log").open("a") as stream:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "stage7.runner",
                        "worker",
                        "--spec",
                        str(specfile),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    start_new_session=True,
                )
                started = time.time()
                update(
                    status="running",
                    job=name,
                    worker_pid=proc.pid,
                    job_started_at=started,
                )
                monitored, checked_resources, interrupted = 0.0, started, False
                try:
                    while proc.poll() is None:
                        stop_check()
                        now = time.time()
                        progress = (
                            read(output / "progress.json")
                            if (output / "progress.json").exists()
                            else {}
                        )
                        if (
                            now - max(started, progress.get("time_unix", started))
                            > 2400
                        ):
                            raise RuntimeError(f"worker 超過 2400 秒無進度：{name}")
                        if now - checked_resources >= 60:
                            competing = [
                                p
                                for p in gpu_processes()
                                if int(p.split(",")[0]) != proc.pid
                            ]
                            checked_resources = now
                            if competing:
                                update(
                                    status="yielding_gpu",
                                    conflicting_processes=competing,
                                )
                                terminate(proc)
                                interrupted = True
                                break
                        if now - monitored >= 600:
                            source_guard(root)
                            telemetry = subprocess.check_output(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                                    "--format=csv,noheader",
                                ],
                                text=True,
                            )
                            update(progress=progress, gpu=telemetry.strip())
                            event(
                                root,
                                kind="monitor",
                                job=name,
                                worker_pid=proc.pid,
                                progress=progress,
                                gpu=telemetry.strip(),
                            )
                            monitored = now
                        time.sleep(5)
                finally:
                    terminate(proc)
                if interrupted:
                    continue
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"{name} worker exit={proc.returncode}；詳見 {output / 'worker.log'}"
                    )
                if not (output / "result.json").exists():
                    raise RuntimeError(f"{name} 未產生結果")
                result = read(output / "result.json")
                update(
                    job=name, worker_pid=None, last_result_status=result.get("status")
                )
                return result

    try:
        zero = {
            name: job("zero-" + name, "zero", candidate=name) for name in CANDIDATES
        }
        write_json(root / "zero-shot-summary.json", zero)
        # Global diagnosis precedes adaptation. Region analysis runs after training
        # so the first verified long run can be handed off without an idle wait.
        training = {}
        for name in CANDIDATES:
            parent = {
                k: zero[name][k]
                for k in ("checkpoint", "sha256", "metrics", "candidate", "phase")
            }
            phase = recipe(name)
            cal = job("calibrate-" + name, "calibrate", parent=parent, recipe=phase)
            if cal["status"] != "passed":
                training[name] = {
                    "status": "needs_review",
                    "reason": "task-gradient calibration",
                    "best": parent,
                }
                write_json(root / "training-summary.json", training)
                continue
            job("smoke-" + name, "smoke", parent=parent, recipe=phase, calibration=cal)
            training[name] = job(
                "train-" + name, "train", parent=parent, recipe=phase, calibration=cal
            )
            write_json(root / "training-summary.json", training)
        qbest = training["qsilu_pq"]["best"]
        if not passed(qbest["metrics"]) and rank(qbest["metrics"]) > rank(
            zero["qsilu_pq"]["metrics"]
        ):
            phase = recipe("qsilu-low-lr", extension=True)
            cal = job("calibrate-qsilu-low-lr", "calibrate", parent=qbest, recipe=phase)
            if cal["status"] == "passed":
                job(
                    "smoke-qsilu-low-lr",
                    "smoke",
                    parent=qbest,
                    recipe=phase,
                    calibration=cal,
                )
                extension = job(
                    "train-qsilu-low-lr",
                    "train",
                    parent=qbest,
                    recipe=phase,
                    calibration=cal,
                )
                training["qsilu-low-lr"] = extension
                qbest = extension["best"]
                write_json(root / "training-summary.json", training)
        other = max(
            (n for n in CANDIDATES if n != "qsilu_pq"),
            key=lambda n: rank(zero[n]["metrics"]),
        )
        sensitivity = []
        for name in ("qsilu_pq", other):
            for region in REGIONS:
                sensitivity.append(
                    job(
                        f"region-{name}-{region}",
                        "region",
                        candidate=name,
                        region=region,
                    )
                )
        write_json(root / "sensitivity-summary.json", sensitivity)
        selected = {name: training[name]["best"] for name in CANDIDATES}
        selected["qsilu_pq"] = qbest
        verifications = {}
        for name, parent in selected.items():
            verifications[name] = job("verify-" + name, "verify", parent=parent)
        from .report import finalize

        result = finalize(root, zero, training, sensitivity, selected, verifications)
        write_json(root / "result.json", result)
        update(
            status=result["status"],
            result=result,
            worker_pid=None,
            finished_at=time.time(),
        )
    except BaseException as exc:
        update(
            status="stopped" if isinstance(exc, InterruptedError) else "error",
            error=f"{type(exc).__name__}: {exc}",
            worker_pid=None,
            traceback=traceback.format_exc(),
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cpu-check", "run", "worker", "status"))
    parser.add_argument("--root", default=str(ARTIFACTS / "activation-20261102"))
    parser.add_argument("--spec")
    args = parser.parse_args()
    if args.action == "cpu-check":
        from .diagnostics import cpu_contracts, numerics

        numerics()
        print(cpu_contracts(), flush=True)
    elif args.action == "worker":
        worker(read(args.spec))
    elif args.action == "status":
        root = Path(args.root)
        result = read(root / "supervisor.json")
        job = result.get("job")
        if job and (root / job / "progress.json").exists():
            result["live_progress"] = read(root / job / "progress.json")
        print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
    else:
        run(args.root)


if __name__ == "__main__":
    main()
