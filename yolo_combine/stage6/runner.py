"""有條件的完整 Stage6 queue 與 600 秒背景監測，不跨至 Stage7。"""

import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, event, read, review, sha256, source_guard, write_json
from .evaluate import accepted, score
from .policy import recipe_for


def worker(spec):
    import torch
    from binaryqk.run import require_idle_gpu, seed

    torch.set_num_threads(2)
    require_idle_gpu()
    seed(review()["protocol"]["seed"])
    action, output = spec["action"], Path(spec["output"])
    source_guard(spec["root"])
    if action == "initialize":
        from .verification import initialize

        return initialize(output)
    if action == "train":
        from .training import train

        return train(output, spec["parent"], spec["recipe"], spec.get("calibration"))
    if action == "calibrate":
        from .calibration import calibrate

        return calibrate(output, spec["parent"], spec["recipe"])
    if action == "calibrate-kd":
        from .kd import calibrate_and_smoke

        return calibrate_and_smoke(
            output, spec["parent"], spec["recipe"], spec["kd_spec"]
        )
    if action == "smoke-head-recovery":
        from .verification import smoke

        parent = spec["parent"]
        if sha256(parent["checkpoint"]) != parent["sha256"]:
            raise ValueError("head recovery smoke parent changed")
        records = smoke(
            output, parent["checkpoint"], recipes=[spec["recipe"]], macro_count=2
        )
        result = {
            "status": "passed",
            "records": records,
            "parent_sha256": parent["sha256"],
        }
        write_json(output / "result.json", result)
        return result
    if action == "verify-recovery":
        from .recovery import verify

        return verify(output, spec["parent"])
    if action == "mix":
        from . import checkpoint
        from .evaluate import pair
        from .models import mix
        from .policy import eligible

        anchor = checkpoint.load(spec["anchor"]["checkpoint"])
        pose = checkpoint.load(spec["parent"]["checkpoint"])
        choices = []
        for rho in review()["protocol"]["rho"]:
            model = mix(anchor, pose, rho, parent=spec["parent"])
            target = output / f"rho-{rho:g}"
            path = target / "model.pt"
            if not path.exists():
                checkpoint.save(path, model)
            values = pair(model.cuda(), target / "evaluation", progress_root=output)[
                "metrics"
            ]
            choices.append(
                {
                    "checkpoint": str(path.resolve()),
                    "sha256": sha256(path),
                    "metrics": values,
                    "rho": rho,
                    "eligible": eligible(values),
                    "phase": "mix",
                }
            )
            del model
            torch.cuda.empty_cache()
        candidates = [c for c in choices if c["eligible"]]
        best = (
            max(
                candidates,
                key=lambda c: (accepted(c["metrics"]), score(c["metrics"]), -c["rho"]),
            )
            if candidates
            else None
        )
        result = {
            "status": "passed" if best else "needs_review",
            "choices": choices,
            "best": best,
        }
        write_json(output / "result.json", result)
        return result
    if action == "finalize":
        from .report import finalize

        return finalize(output, spec["parent"], Path(spec["root"]))
    raise ValueError(action)


def gpu_processes():
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return proc.stdout.strip()


def last_activity_time(progress, started):
    return max(started, progress.get("time_unix", started))


def run(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = review()["protocol"]
    if (root / "source.json").exists():
        source_guard(root)
    else:
        source_guard(root, create=True)
    state = {
        "pid": os.getpid(),
        "status": "running",
        "started_at": time.time(),
        "monitor_seconds": protocol["monitor_seconds"],
        "stage": 6,
        "next_stage_started": False,
    }
    write_json(root / "supervisor.json", state)

    def update(**values):
        state.update(values, updated_at=time.time())
        write_json(root / "supervisor.json", state)

    def stop_check():
        if (root / "STOP").exists():
            raise InterruptedError("user STOP")

    def job(name, action, **kwargs):
        stop_check()
        source_guard(root)
        output = root / name
        output.mkdir(parents=True, exist_ok=True)
        spec = {"action": action, "root": str(root), "output": str(output), **kwargs}
        specfile = output / "job.json"
        if specfile.exists() and read(specfile) != spec:
            raise ValueError(f"job identity changed: {name}")
        write_json(specfile, spec)
        if (output / "result.json").exists():
            return read(output / "result.json")
        while gpu_processes():
            update(status="waiting_gpu", job=name, gpu_processes=gpu_processes())
            for _ in range(12):
                stop_check()
                time.sleep(5)
        update(status="running", job=name, worker_pid=None)
        with (output / "worker.log").open("a") as stream:
            proc = subprocess.Popen(
                [sys.executable, "-m", "stage6.cli", "worker", "--spec", str(specfile)],
                stdout=stream,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                start_new_session=True,
            )
            update(worker_pid=proc.pid, job_started_at=time.time())
            monitored = 0.0
            try:
                while proc.poll() is None:
                    stop_check()
                    now = time.time()
                    progress_path = output / "progress.json"
                    progress = read(progress_path) if progress_path.exists() else {}
                    latest = last_activity_time(progress, state["job_started_at"])
                    if now - latest > protocol["max_stall_seconds"]:
                        raise RuntimeError(f"worker stalled: {name}")
                    if now - monitored >= protocol["monitor_seconds"]:
                        update(
                            progress=progress,
                            gpu_processes=gpu_processes(),
                            last_monitor_at=now,
                        )
                        event(
                            root,
                            kind="monitor",
                            job=name,
                            worker_pid=proc.pid,
                            progress=progress,
                            gpu_processes=state["gpu_processes"],
                        )
                        monitored = now
                    time.sleep(5)
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=10)
                raise
        update(worker_pid=None, last_job_exit=proc.returncode)
        if proc.returncode or not (output / "result.json").exists():
            raise RuntimeError(
                f"{name} failed (exit {proc.returncode}); see {output / 'worker.log'}"
            )
        result = read(output / "result.json")
        event(root, kind="job-complete", job=name, status=result["status"])
        return result

    def phase(name, parent, extension=0):
        recipe = recipe_for(name, extension)
        label = name + (f"-extend{extension}" if extension else "")
        calibration = None
        if recipe["task"] == "joint":
            calibration = job(
                label + "-calibration", "calibrate", parent=parent, recipe=recipe
            )
            if calibration["status"] != "passed":
                raise RuntimeError(f"{label} task-gradient calibration requires review")
        result = job(
            label, "train", parent=parent, recipe=recipe, calibration=calibration
        )
        if result["status"] != "complete":
            raise RuntimeError(f"{label}: {result['reason']}")
        return result

    try:
        if (root / "protocol.json").exists():
            from .recovery import run_recovery

            result = run_recovery(job, root, read(root / "protocol.json"))
            update(
                status=result["status"],
                final=result,
                completed_at=time.time(),
                progress=None,
            )
            event(
                root,
                kind="recovery-candidate-complete",
                status=result["status"],
                stage6_complete=False,
                next_stage_started=False,
            )
            return result
        initial = job("initialization", "initialize")
        anchor = parent = initial["anchor"]
        for name in ("H0", "H1"):
            if accepted(parent["metrics"]):
                break
            parent = phase(name, parent)["best"]
        if not accepted(parent["metrics"]):
            trained_pose = phase("P", parent)
            parent = trained_pose["best"]
            if parent.get("phase") != "P":
                raise RuntimeError(
                    "完整 Pose 適應未改善，停止覆核；不混合未適應的 trunk"
                )
            mixed = job("mix", "mix", anchor=anchor, parent=parent)
            if mixed["status"] != "passed":
                raise RuntimeError("no rho candidate passes COCO selection")
            parent = mixed["best"]
            if not accepted(parent["metrics"]):
                parent = phase("M0", parent)["best"]
        for name in ("J1", "J2", "J3"):
            if accepted(parent["metrics"]):
                break
            result = phase(name, parent)
            parent = result["best"]
            for extension in range(1, len(protocol["extensions"].get(name, [])) + 1):
                if accepted(parent["metrics"]) or result["reason"] != "budget":
                    break
                history = result["history"]
                # Extension requires a recent meaningful gain, not only an early best.
                recent = max(score(h["metrics"]) for h in history[-5:])
                earlier = max(
                    [score(result["parent"]["metrics"])]
                    + [score(h["metrics"]) for h in history[:-5]]
                )
                if recent <= earlier + result["recipe"]["min_delta"]:
                    break
                result = phase(name, parent, extension)
                parent = result["best"]
        result = job("verification", "finalize", parent=parent)
        update(status="complete", final=result, completed_at=time.time(), progress=None)
        event(
            root,
            kind="stage6-complete",
            accepted=result["accepted"],
            next_stage_started=False,
        )
    except BaseException as exc:
        update(
            status="stopped" if isinstance(exc, InterruptedError) else "needs_review",
            error=f"{type(exc).__name__}: {exc}",
            worker_pid=None,
        )
        event(
            root, kind="queue-stopped", error=state["error"], next_stage_started=False
        )
        raise
    finally:
        lock.close()
