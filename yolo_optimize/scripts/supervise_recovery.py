#!/usr/bin/env python3
"""監測本機 recovery training child；不自動分析、不重跑、不管理其他程序。

這個 wrapper 只會呼叫同一專案的 recovery train 或 EMA age 診斷入口，並把
child 的 stdout/stderr、progress 與硬體／磁碟快照保存到獨立的監測檔案。它
不是 agent，也不會依 loss 或 AP 自行決定下一個 arm。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


sys.dont_write_bytecode = True

WORKSPACE = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().with_name("run_recovery.py")
EMA_RUNNER = Path(__file__).resolve().with_name("run_ema_diagnostic.py")
POLL_SECONDS = 10.0
GPU_INTERVAL_SECONDS = 400.0
PROGRESS_PRINT_INTERVAL_SECONDS = GPU_INTERVAL_SECONDS
PROGRESS_TAIL_BYTES = 256 * 1024


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    """把 monitoring payload 轉成嚴格 JSON 可序列化的值。"""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_jsonl(path: Path, event: dict[str, Any]) -> None:
    payload = _json_safe(event)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _resolve_run_dir(raw: str | Path) -> Path:
    candidate = Path(raw).expanduser()
    unresolved = candidate if candidate.is_absolute() else Path.cwd() / candidate
    resolved = unresolved.resolve()
    if resolved == WORKSPACE or not resolved.is_relative_to(WORKSPACE):
        raise ValueError(f"--output 必須是 workspace 子目錄：{resolved}")
    # lexists 也會拒絕 dangling symlink，避免將既有路徑當成新 run。
    if os.path.lexists(unresolved):
        raise FileExistsError(f"run_dir 已存在或是 symlink，禁止覆寫：{resolved}")
    return resolved


def _prepare_log_paths(run_dir: Path) -> tuple[Path, Path]:
    log_dir = (run_dir.parent / "logs").resolve()
    if log_dir == WORKSPACE or not log_dir.is_relative_to(WORKSPACE):
        raise ValueError(f"監測 log 目錄必須位於 workspace：{log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    child_log = log_dir / f"{run_dir.name}.log"
    monitor_log = log_dir / f"{run_dir.name}.monitor.jsonl"
    if os.path.lexists(child_log) or os.path.lexists(monitor_log):
        raise FileExistsError(f"監測 log 已存在，禁止覆寫：{child_log}／{monitor_log}")
    return child_log, monitor_log


def _parse_number(value: str) -> int | float | str:
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _query_gpu() -> dict[str, Any]:
    """以唯讀 nvidia-smi query 取得當下 GPU 快照。"""
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        return {"status": "unavailable", "error": "nvidia-smi query 沒有回傳 GPU row"}
    fields = ["temperature_c", "utilization_percent", "memory_used_mib", "memory_total_mib", "power_w"]
    values = [item.strip() for item in rows[0].split(",")]
    snapshot: dict[str, Any] = {"status": "ok", "device_rows": len(rows)}
    for key, value in zip(fields, values):
        snapshot[key] = _parse_number(value)
    if len(values) != len(fields):
        snapshot["raw_first_row"] = rows[0]
        snapshot["parse_warning"] = "GPU query 欄位數量與預期不同"
    return snapshot


def _disk_snapshot() -> dict[str, int]:
    usage = shutil.disk_usage(WORKSPACE)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _last_progress_event(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"exists": False, "path": str(path)}
    try:
        stat = path.stat()
        with path.open("rb") as handle:
            handle.seek(max(0, stat.st_size - PROGRESS_TAIL_BYTES))
            tail = handle.read().splitlines()
        last_event = None
        for raw in reversed(tail):
            if not raw.strip():
                continue
            try:
                last_event = json.loads(raw.decode("utf-8"))
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        return {
            "exists": True,
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_unix": stat.st_mtime,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "last_complete_event": last_event,
        }
    except OSError as error:
        return {"exists": False, "path": str(path), "error": f"{type(error).__name__}: {error}"}


def _event_progress_fields(progress: dict[str, Any]) -> dict[str, Any]:
    event = progress.get("last_complete_event")
    if not isinstance(event, dict):
        return {}
    report = event.get("report")
    fields: dict[str, Any] = {}
    for key in ("epoch", "macro", "kind", "loss", "time_unix"):
        if key in event:
            fields[key] = event[key]
    if isinstance(report, dict):
        for key in ("loss", "total_loss", "detect_loss", "pose_loss", "next_global_macro_step"):
            if key in report and key not in fields:
                fields[key] = report[key]
    return fields


def _short_print(kind: str, *, pid: int, returncode: int | None, progress: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "event": kind,
        "time": _timestamp(),
        "pid": pid,
        "returncode": returncode,
        "progress": _event_progress_fields(progress),
    }
    if extra:
        payload.update(extra)
    print(json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":")), flush=True)


def _summary_status(run_dir: Path) -> tuple[str | None, str | None]:
    summary = run_dir / "summary.json"
    if not summary.is_file():
        return None, None
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"summary.json read error: {type(error).__name__}: {error}"
    status = payload.get("status") if isinstance(payload, dict) else None
    return (str(status) if status is not None else None), None


def _monitor_event(
    *,
    kind: str,
    pid: int,
    started: float,
    run_dir: Path,
    gpu: dict[str, Any] | None,
    progress: dict[str, Any],
    returncode: int | None,
) -> dict[str, Any]:
    return {
        "event": kind,
        "timestamp": _timestamp(),
        "pid": pid,
        "elapsed_seconds": time.monotonic() - started,
        "run_dir": str(run_dir),
        "returncode": returncode,
        "gpu": gpu,
        "disk": _disk_snapshot(),
        "progress": progress,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, metavar="RUN_DIR",
                        help="workspace 內不存在的 training run 目錄")
    parser.add_argument("--variant", required=True, choices=("native", "hog", "ema-age"),
                        help="native/HOG 正式臂，或固定 1 epoch 的 EMA age 配對診斷")
    parser.add_argument("--ema-age-mode", choices=("fresh", "parent"), default="fresh")
    parser.add_argument("--validate-live", action="store_true",
                        help="native/HOG 每 epoch 額外驗證 live BitTrue")
    parser.add_argument("--adopt-ema-prefix", type=Path,
                        help="採用已完成 EMA 診斷 E1，僅限 native/parent/32/live")
    parser.add_argument("--base-lr-scale", type=float, choices=(1.0, 0.25), default=1.0)
    parser.add_argument("--pause-on-live-regression", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--physical-batch", type=int, choices=(32, 64, 128),
                       help="轉發給 train 的 physical batch")
    # 接受使用者常用的簡寫，但實際轉發一律使用 run_recovery.py 的正式參數。
    group.add_argument("--physical-batch32", dest="physical_batch", action="store_const", const=32,
                       help=argparse.SUPPRESS)
    group.add_argument("--physical-batch64", dest="physical_batch", action="store_const", const=64,
                       help=argparse.SUPPRESS)
    group.add_argument("--physical-batch128", dest="physical_batch", action="store_const", const=128,
                       help=argparse.SUPPRESS)
    return parser


def supervise(args: argparse.Namespace) -> int:
    if args.pause_on_live_regression and not args.validate_live:
        raise ValueError("live 安全暫停需開啟 live 驗證")
    if args.variant == "ema-age" and (args.base_lr_scale != 1.0 or args.pause_on_live_regression):
        raise ValueError("EMA age 診斷不可改動 LR 或 live 暫停策略")
    if args.adopt_ema_prefix is not None:
        if (args.variant, args.ema_age_mode, args.physical_batch, args.validate_live) != ("native", "parent", 32, True):
            raise ValueError("EMA 前綴採用僅限 native/parent/32 並開啟 live 驗證")
        prefix = args.adopt_ema_prefix.expanduser().resolve()
        if prefix == WORKSPACE or not prefix.is_relative_to(WORKSPACE) or not prefix.is_dir():
            raise ValueError("EMA 診斷來源必須是 workspace 內既有子目錄")
    if args.variant == "ema-age" and args.physical_batch != 32:
        raise ValueError("EMA age 單變因診斷固定 physical batch 32")
    if args.variant == "ema-age" and (args.ema_age_mode != "fresh" or args.validate_live):
        raise ValueError("EMA age 診斷固定比較三版本，不接受單臂 EMA/validation 覆寫")
    run_dir = _resolve_run_dir(args.output)
    child_log_path, monitor_log_path = _prepare_log_paths(run_dir)
    command = [
        sys.executable,
        str(RUNNER),
        "train",
        "--variant",
        args.variant,
        "--physical-batch",
        str(args.physical_batch),
        "--output",
        str(run_dir),
    ]
    if args.variant == "ema-age":
        command = [sys.executable, str(EMA_RUNNER), "--physical-batch", "32",
                   "--output", str(run_dir)]
    else:
        command.extend(["--ema-age-mode", args.ema_age_mode])
        command.extend(["--base-lr-scale", str(args.base_lr_scale)])
        if args.pause_on_live_regression:
            command.append("--pause-on-live-regression")
        if args.validate_live:
            command.append("--validate-live")
        if args.adopt_ema_prefix is not None:
            command.extend(["--adopt-ema-prefix", str(prefix)])
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    progress_path = run_dir / "progress.jsonl"
    gpu = _query_gpu()
    progress = _last_progress_event(progress_path)
    child_log = child_log_path.open("x", encoding="utf-8", buffering=1)
    monitor_event = _monitor_event(
        kind="start", pid=-1, started=started, run_dir=run_dir, gpu=gpu,
        progress=progress, returncode=None,
    )
    monitor_event["command"] = command
    _write_jsonl(monitor_log_path, monitor_event)
    try:
        child = subprocess.Popen(
            command,
            cwd=WORKSPACE,
            env=environment,
            stdout=child_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except BaseException:
        child_log.close()
        raise
    # 以真正的 child PID 補寫一筆 startup event；不把 -1 當成實際 PID。
    _write_jsonl(monitor_log_path, _monitor_event(
        kind="child_started", pid=child.pid, started=started, run_dir=run_dir,
        gpu=gpu, progress=_last_progress_event(progress_path), returncode=None,
    ))
    _short_print("child_started", pid=child.pid, returncode=None,
                 progress=_last_progress_event(progress_path), extra={"variant": args.variant, "physical_batch": args.physical_batch})
    next_gpu_probe = started + GPU_INTERVAL_SECONDS
    next_progress_print = started + PROGRESS_PRINT_INTERVAL_SECONDS
    try:
        while True:
            now = time.monotonic()
            returncode = child.poll()
            if returncode is None and now < next_gpu_probe:
                time.sleep(POLL_SECONDS)
                continue
            gpu_sampled = False
            if returncode is not None or now >= next_gpu_probe:
                gpu = _query_gpu()
                gpu_sampled = True
                while next_gpu_probe <= now:
                    next_gpu_probe += GPU_INTERVAL_SECONDS
            progress = _last_progress_event(progress_path)
            event = _monitor_event(
                kind="monitor", pid=child.pid, started=started, run_dir=run_dir,
                gpu=gpu,
                progress=progress, returncode=returncode,
            )
            event["gpu_sampled"] = gpu_sampled
            _write_jsonl(monitor_log_path, event)
            if returncode is not None:
                summary_status, summary_error = _summary_status(run_dir)
                exit_event = {
                    "event": "child_exit",
                    "timestamp": _timestamp(),
                    "pid": child.pid,
                    "elapsed_seconds": time.monotonic() - started,
                    "run_dir": str(run_dir),
                    "returncode": returncode,
                    "training_summary_status": summary_status,
                    "summary_error": summary_error,
                    "early_exit": returncode != 0 or summary_status not in ("complete", "paused_for_analysis"),
                    "gpu": gpu,
                    "disk": _disk_snapshot(),
                    "progress": progress,
                }
                _write_jsonl(monitor_log_path, exit_event)
                _short_print("child_exit", pid=child.pid, returncode=returncode,
                             progress=progress, extra={"training_summary_status": summary_status, "early_exit": exit_event["early_exit"]})
                return int(returncode)
            if now >= next_progress_print:
                _short_print("progress", pid=child.pid, returncode=None, progress=progress)
                while next_progress_print <= now:
                    next_progress_print += PROGRESS_PRINT_INTERVAL_SECONDS
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        signal_sent = False
        if child.poll() is None:
            try:
                child.send_signal(signal.SIGINT)
                signal_sent = True
            except ProcessLookupError:
                pass
            # 只等待本 wrapper 自己建立的 child；不使用 pkill／killpg。
            while child.poll() is None:
                try:
                    child.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    continue
                except KeyboardInterrupt:
                    continue
        returncode = child.returncode
        progress = _last_progress_event(progress_path)
        summary_status, summary_error = _summary_status(run_dir)
        interrupt_event = {
            "event": "keyboard_interrupt",
            "timestamp": _timestamp(),
            "pid": child.pid,
            "elapsed_seconds": time.monotonic() - started,
            "run_dir": str(run_dir),
            "signal": "SIGINT" if signal_sent else None,
            "returncode": returncode,
            "training_summary_status": summary_status,
            "summary_error": summary_error,
            "gpu": gpu,
            "disk": _disk_snapshot(),
            "progress": progress,
        }
        _write_jsonl(monitor_log_path, interrupt_event)
        _short_print("keyboard_interrupt", pid=child.pid, returncode=returncode,
                     progress=progress, extra={"signal_sent": signal_sent, "training_summary_status": summary_status})
        return 130 if returncode is None else int(returncode)
    finally:
        child_log.close()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
