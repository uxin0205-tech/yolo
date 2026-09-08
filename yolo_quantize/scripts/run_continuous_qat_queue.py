"""單 GPU 串行五回合 QAT；shell supervisor 等待，錯誤即停並留下狀態。"""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_continuous_special_queue import OUT, ROOT, write_status

from yolo_quantize.qat_runtime import Full35QATRuntime


def verify_file(item):
    path = Path(item["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"hash 不一致: {path}")
    return path


def validate_result(result, plan):
    if result.get("plan_sha256") != plan.config_sha256 or result.get("arm") != "qat":
        raise ValueError("QAT 完成證據 plan/arm 不一致")
    if result.get("epochs_completed") != 5 or result.get("epochs_planned") != 5:
        raise ValueError("QAT 未完成指定五回合，需分析而非自動晉級")
    paths = result.get("checkpoint_paths", {})
    if not paths or set(paths) != set(result.get("checkpoint_sha256", {})):
        raise ValueError("QAT checkpoint 證據不完整")
    for name, path in paths.items():
        verify_file({"path": path, "sha256": result["checkpoint_sha256"][name]})


def wait_child(command, log, environment):
    with log.open("a") as stream:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_status(child_pid=child.pid, supervisor_pid=os.getpid())
        while True:
            try:
                code = child.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                pass
    if code:
        raise RuntimeError(f"子工作退出 {code}；事件診斷檔案 {log}")


def main():
    with (OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            queue = json.loads((OUT / "selected-qat-jobs-v2.json").read_text())
            probe = json.loads(verify_file(queue["single_layer_probe"]).read_text())
            if probe["status"] != "completed" or len(probe["rows"]) != 592:
                raise ValueError("逐層探測前置條件不完整")
            if not 1 <= len(queue["jobs"]) <= 6:
                raise ValueError("超出本批次六組上限")
            environment = dict(
                os.environ,
                PYTHONPATH=str(ROOT / "src"),
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
            )
            completed = 632
            for index, job in enumerate(queue["jobs"]):
                plan_path = verify_file(job["plan"])
                runtime = Full35QATRuntime.from_yaml(plan_path)
                if runtime.plan.training.epochs != 5:
                    raise ValueError("訓練超出五回合")
                run_dir = runtime.plan.run_root / runtime.run_name("qat")
                result_path = run_dir / "qat-experiment.json"
                if result_path.exists():
                    validate_result(json.loads(result_path.read_text()), runtime.plan)
                    completed += 1
                    continue
                if run_dir.exists():
                    raise ValueError(
                        f"發現未完成 run；需檢查 checkpoint 後明確 resume: {run_dir}"
                    )
                write_status(
                    status="running_qat_preflight",
                    current_index=index,
                    current_candidate=job["job_id"],
                    current_arm="cpu_preflight",
                    completed_jobs=completed,
                    error=None,
                )
                command = [
                    sys.executable,
                    "-m",
                    "yolo_quantize.qat_runtime",
                    "--plan",
                    str(plan_path),
                ]
                preflight = OUT / f"{job['job_id']}-preflight.json"
                wait_child(
                    command
                    + ["--preflight-only", "--preflight-output", str(preflight)],
                    OUT / f"{job['job_id']}-preflight.log",
                    environment,
                )
                checked = json.loads(preflight.read_text())
                if (
                    not checked.get("ready")
                    or checked["resolved"]["plan_sha256"] != runtime.plan.config_sha256
                ):
                    raise ValueError("CPU QAT preflight 未通過")
                write_status(status="waiting_gpu", current_arm="qat")
                while subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip():
                    time.sleep(600)
                write_status(status="running_qat", current_arm="qat")
                wait_child(
                    command
                    + ["--arm", "qat", "--device", "0", "--execute-reviewed-plan"],
                    OUT / f"{job['job_id']}-qat.log",
                    environment,
                )
                validate_result(json.loads(result_path.read_text()), runtime.plan)
                completed += 1
                write_status(status="qat_job_completed", completed_jobs=completed)
            write_status(
                status="decision_required",
                current_index=len(queue["jobs"]),
                current_candidate="six_qat_results_ready_for_cumulative_selection",
                current_arm="analysis",
                completed_jobs=completed,
                error=None,
            )
        except Exception as error:
            write_status(
                status="error",
                error={"type": type(error).__name__, "message": str(error)},
            )
            raise


if __name__ == "__main__":
    main()
