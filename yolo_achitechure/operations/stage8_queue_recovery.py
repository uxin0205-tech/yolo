"""原 Stage8 queue 的有限恢復；不更改鎖定的模型或實驗程式。"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_guard(root):
    sys.path.insert(0, str(REPO / "yolo_achitechure"))
    from stage8.common import source_guard as check

    check(root)


def validate_recovery(root, source_check=source_guard):
    """只有 SIGSEGV 且有相符完整 epoch 狀態時才允許續跑。"""
    root = Path(root).resolve()
    if (root / "STOP").exists():
        raise RuntimeError("使用者 STOP，禁止恢復")
    state = read(root / "supervisor.json")
    name = state.get("job", "")
    folder = (root / name).resolve()
    if (
        state.get("status") != "error"
        or not name.startswith("train-")
        or folder.parent != root
        or not re.search(r"worker exit=-11(?:；|;|$)", state.get("error", ""))
    ):
        raise RuntimeError("僅處理本輪 train worker SIGSEGV；其他錯誤須診斷")
    if (folder / "STOP").exists():
        raise RuntimeError("job STOP，禁止恢復")
    source_check(root)
    job = read(folder / "job.json")
    config = read(folder / "config.json")
    expected = {k: job[k] for k in ("parent", "recipe", "calibration")}
    expected["source_sha256"] = digest(root / "source.json")
    if job.get("action") != "train" or config != expected:
        raise RuntimeError("config／job／來源鎖不相符")
    parent = config["parent"]
    if digest(parent["checkpoint"]) != parent["sha256"]:
        raise RuntimeError("parent SHA 改變")
    path = folder / "resume.pt"
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise RuntimeError("沒有完整 epoch resume，停止自動重試")
    meta = read(path.with_suffix(".json"))
    if digest(path) != meta["sha256"]:
        raise RuntimeError("resume SHA 不符")
    epoch = meta["next_epoch"]
    if type(epoch) is not int or not 2 <= epoch <= config["recipe"]["epochs"] + 1:
        raise RuntimeError("非法 next_epoch")
    return {"job": name, "next_epoch": epoch, "resume_sha256": meta["sha256"],
            "config_sha256": digest(folder / "config.json"),
            "source_lock_sha256": digest(root / "source.json")}


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (root / "recovery-supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "state.json").exists():
        raise RuntimeError("恢復記錄已存在；不能重設已使用的重試預算")
    state = {"pid": os.getpid(), "started_at": time.time(), "status": "preparing",
             "max_auto_retries": 1, "script_sha256": digest(__file__),
             "next_stage_started": False, "root": str(root),
             "environment_added": {"PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"}}
    stop_requested = False

    def update(**values):
        state.update(values, updated_at=time.time())
        write(output / "state.json", state)

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        (root / "STOP").touch()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        # 第一次為已查核的恢復；之後最多再自動重試一次。
        for attempt in range(2):
            if stop_requested:
                update(status="stopped")
                return 130
            verification = validate_recovery(root)
            with (root / "supervisor.lock").open("a") as queue_lock:
                fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(queue_lock, fcntl.LOCK_UN)
            write(output / f"attempt-{attempt}.json", {
                "time_unix": time.time(), "verification": verification,
                "previous_supervisor": read(root / "supervisor.json"),
                "previous_progress": read(root / verification["job"] / "progress.json"),
                "worker_log_start_bytes": (root / verification["job"] / "worker.log").stat().st_size,
            })
            env = os.environ.copy()
            env.update(PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
            with (root / "supervisor.log").open("a") as log:
                log.write(f"\n[bounded-recovery] attempt={attempt} time={time.time()}\n")
                log.flush()
                proc = subprocess.Popen(
                    ["bash", "yolo_achitechure/run_stage8.sh", "run", "--root", str(root)],
                    cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                update(status="running", attempt=attempt, queue_pid=proc.pid,
                       verification=verification)
                last_monitor = 0.0
                while proc.poll() is None:
                    now = time.time()
                    if now - last_monitor >= 600:
                        current = read(root / "supervisor.json")
                        update(queue_status=current.get("status"), job=current.get("job"),
                               worker_pid=current.get("worker_pid"))
                        last_monitor = now
                    time.sleep(5)
            current = read(root / "supervisor.json")
            update(queue_exit=proc.returncode, last_supervisor=current)
            if stop_requested or (root / "STOP").exists():
                update(status="stopped")
                return 130
            if current.get("pid") != proc.pid:
                raise RuntimeError("queue PID 不相符，禁止重試其他程序的狀態")
            if proc.returncode == 0 and current.get("status") == "complete":
                update(status="complete")
                return 0
            if attempt == 1:
                raise RuntimeError("已用完一次自動重試；保留完整 epoch 狀態並停止")
            update(status="checking_retry")
        return 1
    except Exception as error:
        update(status="error", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.root, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
