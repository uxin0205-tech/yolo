"""Stage9 段落邊界校準補充：保留原鎖，只擴充 train-only 量測數。"""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "yolo_actshape"
RUN = HERE / "artifacts/shape-expanded-20261108-v1"
AUDIT = HERE / "artifacts/calibration-recovery-20261110"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def setup():
    for name in reversed(("yolo_actshape", "yolo_achitechure", "yolo_activation",
                          "yolo_activation/src", "yolo_combine", "yolo_combine/src",
                          "yolo_attention/src", "yolo_binaryqk", "yolo_masf",
                          "yolo_achitechure/achitechure_1/src")):
        sys.path.insert(0, str(ROOT / name))
    for key, value in {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                       "PYTHONFAULTHANDLER": "1", "CUDA_VISIBLE_DEVICES": "0"}.items():
        os.environ[key] = value


def guard():
    from stage9_extension.common import guard as original_guard
    original_guard()
    for name, expected in read(AUDIT / "source.json").items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"校準補充來源改變：{name}")
    return read(AUDIT / "policy.json")


def validate_result(result, cfg):
    """拒絕原少量確認、非有限梯度、未還原 parent 或放寬門檻的結果。"""
    if result.get("status") != "passed" or not result.get("parent_restored") or result.get("validation_used") is not False:
        raise ValueError("校準未通過或使用了不允許的資料／模型狀態")
    rows = result["records"]
    measured = [r for r in rows if r["phase"] == "measure"]
    confirmations = [r for r in rows if r["phase"].startswith("confirm-")]
    if len(measured) != cfg["macros"] or not confirmations:
        raise ValueError("量測數不足")
    phases = list(dict.fromkeys(r["phase"] for r in confirmations))
    if len(phases) > cfg["attempts"] or len(rows) > cfg["macros"] + cfg["confirm"] * cfg["attempts"]:
        raise ValueError("校準超出預算")
    for phase in phases:
        if sum(r["phase"] == phase for r in rows) != cfg["confirm"]:
            raise ValueError("確認數不足")
    for row in rows:
        if not all(math.isfinite(row[k]) for k in ("ratio", "pose_weight", "detect_norm", "pose_norm", "cosine_similarity")):
            raise ValueError("非有限梯度")
        if min(row["detect_norm"], row["pose_norm"], row["ratio"]) <= 0:
            raise ValueError("缺少梯度")
    last = [r for r in confirmations if r["phase"] == phases[-1]]
    weight = result["selected_pose_weight"]
    if not cfg["bounds"][0] <= weight <= cfg["bounds"][1] or abs(weight / cfg["step"] - round(weight / cfg["step"])) > 1e-6:
        raise ValueError("權重超出原範圍或步長")
    if any(r["pose_weight"] != weight for r in last):
        raise ValueError("確認權重不符")
    ratio = statistics.median(r["ratio"] for r in last)
    if not cfg["band"][0] <= ratio <= cfg["band"][1]:
        raise ValueError("原梯度比例門檻未通過")
    return ratio


def recover(spec_path):
    from stage9_extension.common import training_run, BASE_RUN
    from stage9.runner import outsiders
    policy = guard()
    spec_path = Path(spec_path).resolve()
    spec = read(spec_path)
    if spec["id"] not in policy["eligible_jobs"] or spec["action"] != "calibrate" or spec_path != RUN / spec["id"] / "job.json":
        raise ValueError("不是已授權的待執行校準")
    old_result = spec_path.parent / "result.json"
    previous = read(old_result)
    if previous["status"] != "needs_review" or not previous["parent_restored"] or previous["validation_used"]:
        raise ValueError("不允許處理非梯度比例的失敗")
    if digest(ROOT / spec["parent"]["checkpoint"]) != spec["parent"]["sha256"]:
        raise ValueError("parent SHA 不符")
    out = AUDIT / "jobs" / spec["id"]
    if (out / "attempt.json").exists():
        raise ValueError("此校準已使用一次補充預算；不能無限重試")
    with (RUN / "queue.lock").open("a") as extra_lock, (BASE_RUN / "queue.lock").open("a") as base_lock:
        for lock in (extra_lock, base_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if outsiders(os.getpid()):
            raise RuntimeError("GPU 有其他工作，停止校準")
        write(out / "attempt.json", {"started_unix": time.time(), "original_job_sha256": digest(spec_path),
                                    "failed_result_sha256": digest(old_result), "source_sha256": digest(AUDIT / "source.json")})
        amended = {**spec, "output": str(out)}
        write(out / "job.json", amended)
        with training_run() as engine:
            original_protocol = engine.protocol
            def revised_protocol():
                value = copy.deepcopy(original_protocol())
                value["calibration"] = policy["calibration"]
                return value
            engine.protocol = revised_protocol
            try:
                result = engine.calibrate(amended)
            finally:
                engine.protocol = original_protocol
        ratio = validate_result(result, policy["calibration"])
        guard()
        result["calibration_recovery"] = {"policy": str(AUDIT / "policy.json"),
            "policy_sha256": digest(AUDIT / "policy.json"), "source_sha256": digest(AUDIT / "source.json"),
            "original_job_sha256": digest(spec_path), "failed_result_sha256": digest(old_result),
            "base_source_sha256": digest(RUN / "source.json"), "result_path": str(out / "result.json"),
            "confirmation_median": ratio, "calibration": policy["calibration"],
            "reason": "原3+2校準逐批次完全重現；擴大至optimize的16+8，保留原門檻"}
        write(out / "result.json", result)
        if (spec_path.parent / "done.json").exists():
            raise ValueError("不得覆蓋既有完成標記")
        write(spec_path.parent / "done.json", {"result": result, "job_sha256": digest(spec_path),
            "source_sha256": digest(RUN / "source.json"), "time_unix": time.time(),
            "recovery_source_sha256": digest(AUDIT / "source.json"),
            "recovery_result_sha256": digest(out / "result.json")})
        print(f"校準補充通過：{spec['id']} weight={result['selected_pose_weight']} median={ratio}", flush=True)


def check_published():
    policy = guard()
    for job in policy["eligible_jobs"]:
        path = RUN / job / "done.json"
        if not path.exists():
            continue
        done = read(path)
        if "recovery_source_sha256" not in done:
            continue
        rp = AUDIT / "jobs" / job / "result.json"
        if done["recovery_source_sha256"] != digest(AUDIT / "source.json") or done["recovery_result_sha256"] != digest(rp) or done["result"] != read(rp):
            raise ValueError("補充校準完成標記不符")
        validate_result(done["result"], policy["calibration"])


def supervised(command, log_path, *, calibration=False):
    from stage9.runner import outsiders, stop_worker
    from stage9_extension.common import stopped
    with log_path.open("a", buffering=1) as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=os.environ.copy())
        started = time.time()
        try:
            while child.poll() is None:
                if stopped() or (calibration and (outsiders(child.pid) or time.time() - started > 300)):
                    stop_worker(child)
                    raise RuntimeError("STOP、資源衝突或校準逾時；保留原結果並停止")
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            stop_worker(child)
    return child.returncode


def supervise():
    policy = guard()
    with (AUDIT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(AUDIT / "supervisor.json", {"pid": os.getpid(), "started_unix": time.time(), "source_sha256": digest(AUDIT / "source.json")})
        while True:
            check_published()
            state = read(RUN / "status.json")
            if state["status"] == "complete":
                return
            if state["status"] == "failed":
                job = state.get("error", "").split(" 失敗 rc=", 1)[0]
                if job not in policy["eligible_jobs"]:
                    raise RuntimeError("非已授權校準失敗，停止供診斷")
                if not (RUN / job / "done.json").exists():
                    rc = supervised([sys.executable, "-B", str(Path(__file__).resolve()), "calibrate", "--spec", str(RUN / job / "job.json")],
                                    AUDIT / "calibration.log", calibration=True)
                    if rc:
                        raise RuntimeError(f"補充校準失敗 rc={rc}，不跳過門檻")
                    check_published()
            rc = supervised([str(HERE / "run_extension.sh"), "run"], AUDIT / "queue.log")
            if rc == 0:
                if read(RUN / "status.json")["status"] != "complete":
                    raise RuntimeError("queue 提前結束但未完成")
                return
            if read(RUN / "status.json")["status"] != "failed":
                raise RuntimeError("queue 未正常結束；不自動重啟")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("calibrate", "check", "run"))
    parser.add_argument("--spec")
    args = parser.parse_args()
    setup()
    if args.action == "calibrate":
        recover(args.spec)
    elif args.action == "check":
        check_published()
        print("PASS 原來源鎖與校準補充契約")
    else:
        supervise()


if __name__ == "__main__":
    main()
