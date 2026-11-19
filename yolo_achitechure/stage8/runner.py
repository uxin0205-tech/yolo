"""本階段完整 queue、600 秒監測及其他 GPU 工作避讓。"""

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
    ROOT,
    baseline,
    event,
    protocol,
    require_gpu_authorized,
    rank,
    read,
    recipe,
    score,
    sha256,
    source_guard,
    write_json,
)


def gpu_processes():
    result = subprocess.run(
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
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def extend(result):
    history = result["history"]
    best = result["best"]
    return (
        result["status"] == "complete"
        and len(history) >= 5
        and best.get("epoch", 0) >= 3
        and all(best["metrics"][k] >= v - 0.015 for k, v in baseline().items())
        and sum(score(row["metrics"]) for row in history[-2:]) / 2
        > sum(score(row["metrics"]) for row in history[:2]) / 2 + 0.0001
    )


def worker(spec):
    require_gpu_authorized()
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
    action = spec["action"]
    event(output, kind="worker-start", action=action)
    if action == "zero":
        entry = read(HERE / "cpu-contracts.json")["results"][spec["candidate"]]
        path = Path(entry["initial_checkpoint"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("初始 Rep checkpoint 改變")
        model = checkpoint.load(path)
        result = pair(
            model, output / "evaluation", progress_root=output, diagnostics=True
        )
        native = pair(model, output / "training-evaluation", progress_root=output, operators="training")
        result["training_metrics"] = native["metrics"]
        result["deployment_minus_training"] = {k: result["metrics"][k] - native["metrics"][k] for k in baseline()}
        difference = {k: result["metrics"][k] - v for k, v in baseline().items()}
        if any(abs(v) > 0.0001 for v in difference.values()):
            raise ValueError(f"等價初始化 AP 差異超過 .0001：{difference}")
        result.update(
            status="passed",
            checkpoint=str(path),
            sha256=sha256(path),
            epoch=0,
            candidate=spec["candidate"],
            phase="zero",
            difference=difference,
        )
        write_json(output / "result.json", result)
        return result
    if action == "prepare-adaptation":
        from .operators import set_operators
        parent = spec["parent"]
        if sha256(parent["checkpoint"]) != parent["sha256"]:
            raise ValueError("適應來源 SHA 改變")
        model = checkpoint.load(parent["checkpoint"])
        model.stage8_manifest["activation_policy"] = set_operators(model, "hardware")
        model.stage8_manifest["operators"] = "hardware"
        model.stage8_manifest["training_lineage"].append({"conversion": "unfolded -> qSiLU/D1", "parent_sha256": parent["sha256"]})
        path = output / "hardware-parent.pt"
        if not path.exists():
            checkpoint.save(path, model)
        loaded = checkpoint.load(path)
        if any(not torch.equal(v, loaded.state_dict()[n]) for n,v in model.state_dict().items()):
            raise ValueError("適應來源重載不同")
        result = {**parent, "checkpoint": str(path), "sha256": sha256(path), "epoch": 0, "status": "passed", "phase": "hardware-adaptation-parent"}
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
        from .diagnostics import audit, fold_check

        parent = spec["parent"]
        if sha256(parent["checkpoint"]) != parent["sha256"]:
            raise ValueError("選定模型 SHA 改變")
        model = checkpoint.load(parent["checkpoint"])
        folding = {str(size): fold_check(model, size) for size in (160, 640)}
        deploy = checkpoint.folded(model, operators="hardware")
        operators = audit(deploy)
        write_json(output / "operators.json", operators)
        path = output / "deploy.pt"
        if not path.exists():
            checkpoint.save(path, deploy)
        loaded = checkpoint.load(path)
        if any(
            not torch.equal(v, loaded.state_dict()[n])
            for n, v in deploy.state_dict().items()
        ):
            raise ValueError("folded 重載 tensor 不同")
        result = pair(
            loaded, output / "evaluation", progress_root=output, diagnostics=True
        )
        difference = {k: result["metrics"][k] - v for k, v in parent["metrics"].items()}
        if any(abs(v) > 1e-9 for v in difference.values()):
            raise ValueError(f"重載完整 AP 不同：{difference}")
        float_reference = pair(
            loaded,
            output / "float-reference",
            progress_root=output,
            float_reference=True,
        )
        native = pair(model, output / "training-evaluation", progress_root=output, operators="training")
        result.update(
            training_metrics=native["metrics"],
            deployment_minus_training={k: result["metrics"][k] - native["metrics"][k] for k in baseline()},
            status="passed",
            parent=parent,
            difference=difference,
            folding=folding,
            manifest=checkpoint.manifest_for(loaded),
            checkpoint=str(path.resolve()),
            sha256=sha256(path),
            float_reference_metrics=float_reference["metrics"],
            float_reference_is_diagnostic_only=True,
        )
        write_json(output / "result.json", result)
        return result
    raise ValueError(action)


def run(root):
    require_gpu_authorized()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_guard(root, create=True)
    for name in ("cpu-contracts.json",):
        if read(HERE / name)["status"] != "passed":
            raise ValueError(f"缺少通過的 CPU 驗證：{name}")
    state = {
        "pid": os.getpid(),
        "status": "running",
        "started_at": time.time(),
        "stage": 8,
        "next_stage_started": False,
        "monitor_seconds": 600,
        "max_training_epochs": protocol()["max_training_epochs"],
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
                        "stage8.runner",
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
        zeros, trained, selected, verified = {}, {}, {}, {}
        for name in CANDIDATES:
            zeros[name] = job(f"zero-{name}", "zero", candidate=name)
            initial = zeros[name]
            cfg = recipe(name)
            calibration = job(
                f"calibrate-{name}", "calibrate", parent=initial, recipe=cfg
            )
            if calibration["status"] != "passed":
                raise ValueError(f"{name} 梯度校準未通過；需診斷，未直接開始長訓")
            smoke = job(
                f"smoke-{name}",
                "smoke",
                parent=initial,
                recipe=cfg,
                calibration=calibration,
            )
            if smoke["status"] != "passed":
                raise ValueError(f"{name} smoke 未通過")
            result = job(
                f"train-{name}",
                "train",
                parent=initial,
                recipe=cfg,
                calibration=calibration,
            )
            trained[name] = [result]
            selected[name] = result["best"]
            should_extend = extend(result)
            write_json(
                root / f"extension-{name}.json",
                {
                    "extend": should_extend,
                    "policy": protocol()["extension"],
                    "initial_best": result["best"],
                },
            )
            if should_extend:
                cfg = recipe(name, extension=True)
                parent = result["best"]
                calibration = job(
                    f"calibrate-{name}-low-lr", "calibrate", parent=parent, recipe=cfg
                )
                if calibration["status"] != "passed":
                    raise ValueError(f"{name} 降 LR 校準未通過")
                job(
                    f"smoke-{name}-low-lr",
                    "smoke",
                    parent=parent,
                    recipe=cfg,
                    calibration=calibration,
                )
                extra = job(
                    f"train-{name}-low-lr",
                    "train",
                    parent=parent,
                    recipe=cfg,
                    calibration=calibration,
                )
                trained[name].append(extra)
                if rank(extra["best"]["metrics"]) > rank(selected[name]["metrics"]):
                    selected[name] = extra["best"]
            if name == protocol()["hardware_adaptation"]["candidate"]:
                best = selected[name]
                training_metrics = best.get("training_metrics", zeros[name]["training_metrics"])
                gap = {k: training_metrics[k] - best["metrics"][k] for k in baseline()}
                adapt = ((score(training_metrics) - score(best["metrics"]) > .001 or max(gap.values()) > .003)
                         and all(best["metrics"][k] >= v - .015 for k,v in baseline().items()))
                write_json(root / f"adaptation-{name}.json", {"adapt": adapt, "gap": gap, "policy": protocol()["hardware_adaptation"]})
                if adapt:
                    parent = job(f"prepare-adaptation-{name}", "prepare-adaptation", parent=best)
                    cfg = recipe(name)
                    cfg.update(name=name + "-hardware-adapt", operators="hardware", epochs=3, loss_horizon=3,
                               lrs={k:v*.25 for k,v in cfg["lrs"].items()})
                    calibration = job(f"calibrate-{name}-hardware", "calibrate", parent=parent, recipe=cfg)
                    if calibration["status"] != "passed":
                        raise ValueError("硬體適應梯度校準失敗")
                    result_smoke = job(f"smoke-{name}-hardware", "smoke", parent=parent, recipe=cfg, calibration=calibration)
                    if result_smoke["status"] != "passed":
                        raise ValueError("硬體適應 smoke 失敗")
                    extra = job(f"train-{name}-hardware", "train", parent=parent, recipe=cfg, calibration=calibration)
                    trained[name].append(extra)
                    if rank(extra["best"]["metrics"]) > rank(selected[name]["metrics"]):
                        selected[name] = extra["best"]
            verified[name] = job(f"verify-{name}", "verify", parent=selected[name])
        source_guard(root)
        update(status="finalizing", worker_pid=None)
        from .report import finalize

        result = finalize(root, zeros, trained, selected, verified)
        write_json(root / "result.json", result)
        update(
            status="complete",
            completed_at=time.time(),
            result=result,
            next_stage_started=False,
            job=None,
        )
        event(
            root,
            kind="stage-complete",
            recommendation=result["recommendation"],
            next_stage_started=False,
        )
        return result
    except BaseException as exc:
        update(
            status="stopped" if isinstance(exc, InterruptedError) else "error",
            error=repr(exc),
            traceback=traceback.format_exc(),
            worker_pid=None,
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "worker", "status", "cpu-check"))
    parser.add_argument("--root", type=Path, default=ARTIFACTS / protocol()["run_id"])
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        run(args.root)
    elif args.action == "worker":
        worker(read(args.spec))
    elif args.action == "cpu-check":
        from .diagnostics import run as check

        check()
    else:
        import json

        print(
            json.dumps(
                read(args.root / "supervisor.json"), ensure_ascii=False, indent=2
            )
        )


if __name__ == "__main__":
    main()
