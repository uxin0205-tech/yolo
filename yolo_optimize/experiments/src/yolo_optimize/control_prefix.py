"""明示採用已完成 EMA 診斷的原生 E1 prefix；不是失敗 run 的 exact-resume 證明。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from . import runtime
from .ema_diagnostic import parent_ema_updates
import torch
from yolo_combine.metrics import GATE_METRICS, joint_score
from yolo_combine.resume import load_training_snapshot


_MATCHED_SETTINGS = (
    "scope", "optimizer", "optimizer_hyperparameters", "loaders", "amp",
    "gradient_clip_norm", "scheduler", "physical_batch", "detect_logical",
    "detect_per_macro", "pose_per_macro", "reference_batch", "seed",
    "scheduler_horizon", "warmup_epochs", "initial_native_criterion",
    "base_state_sha256", "data", "deployment_auxiliary",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"E1 prefix 拒絕採用：{message}")


def _same(actual, expected, label: str) -> None:
    _require(actual == expected, f"{label} 不一致")


def _digest_state(values: Mapping, *, prefix: str = "") -> str:
    digest = hashlib.sha256()
    matched = 0
    for original_name, tensor in values.items():
        if not original_name.startswith(prefix):
            continue
        name = original_name[len(prefix):]
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        matched += 1
    _require(matched > 0, "來源 state 為空")
    return digest.hexdigest()


def _metrics(values: Mapping, label: str) -> dict[str, float]:
    _require(isinstance(values, Mapping), f"{label} 必須是 metrics mapping")
    _require(all(key in values for key in GATE_METRICS), f"{label} 缺少 selector 必要八 AP")
    _require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                 for value in values.values()), f"{label} 含無效 AP")
    return {key: float(value) for key, value in values.items()}


def _check_criterion(saved: Mapping, metadata: Mapping) -> None:
    _same(saved.get("schema_version"), 1, "criterion schema")
    _same(saved.get("mu"), 0.0, "HOG μ")
    _same(saved.get("calibration"), None, "HOG calibration")
    for task, initial in metadata["initial_native_criterion"].items():
        actual = saved["native"][task]
        expected = dict(initial)
        _require(initial.get("end2end") is True, f"{task} 不是 progressive E2E criterion")
        expected["updates"] = initial["updates"] + 1
        horizon = metadata["criteria_continuation"]["horizons"][task]
        expected["o2m"] = max(1 - expected["updates"] / max(horizon - 1, 1), 0) * (
            expected["o2m_copy"] - expected["final_o2m"]) + expected["final_o2m"]
        expected["o2o"] = expected["total"] - expected["o2m"]
        _same(actual, expected, f"{task} E1 結束 criterion（不能重複 advance）")


def adopt_ema_prefix(state, diagnostic_root: Path, expected_parent_metrics: Mapping) -> dict:
    """校驗後恢復完整 continued snapshot，回傳供 E1 selector 與 E2 loop 的資料。

``state`` 必須是尚未訓練的 native／parent-age／physical32 建構結果。
``expected_parent_metrics`` 是扁平八項 parent BitTrue AP。函式不建立 selector，
不 reseed、不 advance criterion／warmup，不改原始診斷或 checkpoint。
檢查完成後才呼叫真實 load_training_snapshot(..., restore_rng=True)。
"""
    root = Path(diagnostic_root).expanduser().resolve()
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _same(summary.get("status"), "complete", "診斷完成狀態")
    _require(summary.get("shared_live_trajectory") is True, "診斷沒有 shared live trajectory")
    _require(summary.get("independent_replicates") is False, "診斷被錯標成獨立重複")
    _require(summary.get("hog_enabled") is False, "診斷包含 HOG")
    _require(summary.get("automatic_promotion") is False, "診斷已被自動升格")
    source = summary["metadata"]
    loader_paths = [str(state.detect.data_yaml), str(state.pose.data_yaml)]
    if "data" in state.metadata:
        _same(state.metadata["data"], loader_paths, "目標 metadata 與實際 loader 路徑")
    target = {**state.metadata, "data": loader_paths}
    _same(source.get("experiment"), "paired_ema_age", "診斷種類")
    _same(source.get("max_epochs"), 1, "診斷長度")
    _require(source.get("calibration") is None, "診斷包含 HOG calibration")
    for label, metadata in (("來源", source), ("目標", target)):
        _same(metadata.get("variant"), "native", f"{label} variant")
        _same(metadata.get("physical_batch"), 32, f"{label} physical batch")
    _require(getattr(state.model, "aux", None) is None, "目標模型有 auxiliary")
    _same(target["ema"].get("age_mode"), "parent", "目標 EMA age mode")
    _same(type(state.optimizer).__name__, "AdamW", "目標 optimizer")
    _require(not state.optimizer.state, "目標 optimizer 已有訓練狀態，不能套用 prefix")
    for key in _MATCHED_SETTINGS:
        _require(key in source and key in target, f"缺少設定 {key}")
        _same(source[key], target[key], key)
    for key in ("path", "horizons", "state", "parent_metrics", "parent_progress"):
        _same(source["criteria_continuation"][key], target["criteria_continuation"][key], f"parent criterion {key}")
    parent_metrics = _metrics(expected_parent_metrics, "expected parent")
    for key in GATE_METRICS:
        _same(target["criteria_continuation"]["parent_metrics"][key], parent_metrics[key], f"parent AP {key}")
    parent_path = Path(target["parent"]).expanduser().resolve()
    _same(Path(source["parent"]).expanduser().resolve(), parent_path, "parent 路徑")
    _same(source["parent_sha256"], runtime.sha256(parent_path), "parent SHA-256")
    _same(target["loaded"]["state_source"], "ema", "目標 parent state source")
    _same(Path(target["loaded"]["path"]).expanduser().resolve(), parent_path, "實際 loaded parent 路徑")
    _same(_digest_state(state.model.base.state_dict()), target["base_state_sha256"], "目標初始 live state hash")
    _same(_digest_state(state.ema.ema.base.state_dict()), target["base_state_sha256"], "目標初始 EMA state hash")
    age = parent_ema_updates({"ema_updates": source["parent_ema_updates"]})
    _same(target["criteria_continuation"]["ema_updates"], age, "paired parent EMA age")
    _same(target["ema"]["updates_at_start"], age, "目標初始 EMA age")
    _same(state.ema.updates, age, "目標目前 EMA age")
    fixed_names = tuple(state.ema.fixed_state_names)
    for label, ema_settings in (("來源", source["ema"]), ("目標", target["ema"])):
        _same(ema_settings["decay"], 0.9999, f"{label} EMA decay")
        _same(ema_settings["tau"], 2000, f"{label} EMA tau")
        _require(ema_settings["fixed_state_exact_copy"] is True, f"{label} 缺少 fixed-state EMA")
        _same(tuple(ema_settings["fixed_state_names"]), fixed_names, f"{label} BN/frozen-state 清單")
    steps = state.macros
    _require(type(steps) is int and steps > 0, "macro 數量無效")
    current_scheduler = state.scheduler.state_dict()
    _same(current_scheduler["current_step"], 0, "目標 scheduler 必須尚未開始")
    for key, value in {"stage": "recovery", "epochs": 10, "steps_per_epoch": steps,
                       "warmup_epochs": 1, "warmup_start_factor": 0.1, "final_lr_factor": 0.5}.items():
        _same(current_scheduler[key], value, f"目標 scheduler {key}")
    _same(target["scheduler"]["steps_per_epoch"], steps, "設定與 scheduler macro 數")
    initial_criterion = state.router.state_dict()
    _same(initial_criterion["native"], target["initial_native_criterion"], "目標初始 criterion")
    _same(initial_criterion["mu"], 0.0, "目標 μ")
    _same(initial_criterion["calibration"], None, "目標 calibration")

    snapshot_path = root / "checkpoints" / "continued-epoch-0001.pt"
    snapshot_info = summary["snapshots"]["continued"]
    _same(Path(snapshot_info["path"]).expanduser().resolve(), snapshot_path, "snapshot 路徑")
    snapshot_sha = runtime.sha256(snapshot_path)
    _same(snapshot_sha, snapshot_info["sha256"], "snapshot SHA-256")
    _same(snapshot_path.stat().st_size, snapshot_info["bytes"], "snapshot 大小")
    payload = torch.load(snapshot_path, map_location="cpu", weights_only=True, mmap=True)
    _same(payload.get("schema_version"), 2, "snapshot schema")
    _same(payload.get("checkpoint_kind"), "full_resume", "snapshot 必須完整 resume，不是 inference EMA")
    # .pt 保留整數 mapping keys／tuple；summary JSON 則會轉成字串 keys／list。
    # 只正規化 metadata 的表示，不改 tensor、optimizer state 或正式設定。
    saved_metadata = json.loads(json.dumps(payload["resolved_config"], allow_nan=False))
    _same(saved_metadata, {**source, "ema_variant": "continued"}, "snapshot resolved config")
    _same(payload["ema_updates"], age + steps, "snapshot EMA age")
    expected_progress = {"stage": "ema_age_diagnostic", "next_epoch": 1,
                         "global_macro_step": steps, "joint_epochs_completed": 1}
    _same(payload["progress"], expected_progress, "snapshot progress")
    _same(payload["scheduler_state"], {**current_scheduler, "current_step": steps}, "snapshot scheduler horizon/step")
    _check_criterion(payload["criteria_state"], target)
    _same(payload["loader_state"]["snapshot_boundary"], "epoch_end_before_validation", "snapshot RNG 邊界")
    _same(payload["loader_state"]["seeds"], {"detect": 6148914691236517205, "pose": 6148914691236517206}, "E1 loader seeds")
    _same(payload["best_state"], {"automatic_promotion": False}, "診斷 best_state（必須另外重建 selector）")
    _same(payload["environment"]["torch"], str(torch.__version__), "Torch 版本")
    for name in fixed_names:
        _require(torch.equal(payload["model_state"][name], state.model.state_dict()[name].detach().cpu()), f"固定 live state 改變：{name}")
        _require(torch.equal(payload["ema_state"][name], payload["model_state"][name]), f"固定 EMA state 不一致：{name}")
    end = summary["ema_observers_at_end"]
    _same(end["current_updates"], {"fresh": steps, "continued": age + steps}, "summary EMA age")
    _same(end["observations"], steps, "summary EMA observation 數")
    _same(tuple(end["fixed_state_names"]), fixed_names, "summary 固定 state 清單")
    _same(len(summary["epochs"]), 1, "prefix epoch 數")
    epoch_record = summary["epochs"][0]
    _same(epoch_record["epoch"], 1, "prefix epoch 編號")
    training = epoch_record["training"]
    for key, value in {"epoch": 1, "stage": "ema_age_diagnostic", "macros": steps,
                       "next_global_macro_step": steps}.items():
        _same(training[key], value, f"E1 training report {key}")
    versions = summary["versions"]
    _same(versions["continued"]["source"], "ema", "continued metric source")
    _same(versions["live"]["source"], "live", "live metric source")
    _same(_digest_state(payload["model_state"], prefix="base."), versions["live"]["state_sha256"], "validated live state SHA")
    _same(_digest_state(payload["ema_state"], prefix="base."), versions["continued"]["state_sha256"], "validated EMA state SHA")
    metrics = {backend: _metrics(versions["continued"]["metrics"][backend], f"E1 EMA {backend}")
               for backend in ("float", "bittrue")}
    live_metrics = _metrics(versions["live"]["metrics"]["bittrue"], "E1 live BitTrue")
    for backend, values in metrics.items():
        _same(joint_score(values), versions["continued"]["joint_scores"][backend], f"E1 {backend} selector score")
    result = {"reused": True, "source": str(snapshot_path), "start_epoch": 1,
              "global_macro_step": steps, "training": deepcopy(training),
              "metrics": metrics, "live_metrics": live_metrics,
              "provenance": {"kind": "explicit_diagnostic_e1_prefix_adoption",
                  "diagnostic_root": str(root), "snapshot_sha256": snapshot_sha,
                  "summary_sha256": runtime.sha256(summary_path),
                  "parent_sha256": source["parent_sha256"],
                  "source_progress_stage": "ema_age_diagnostic", "continuation_stage": "recovery",
                  "scheduler_stage_unchanged": "recovery", "restored_ema_updates": age + steps,
                  "failed_run_exact_state_equivalence_claimed": False,
                  "selector_action_required": "observe E1 continued BitTrue metrics with the configured AccuracyGate"}}
    # 原生 API 仍負責真正的 contract／optimizer manifest 與完整 state restore。
    # 不把診斷 metadata.max_epochs=1 寫回目前 control 的設定。
    restored = load_training_snapshot(snapshot_path, model=state.model, ema=state.ema,
        optimizer=state.optimizer, scheduler=state.scheduler, scaler=state.scaler,
        criteria=state.router, restore_rng=True)
    _same(restored.progress.next_epoch, 1, "restored next_epoch")
    _same(restored.progress.global_macro_step, steps, "restored global macro")
    result["loader_state"] = deepcopy(restored.loader_state)
    return result
