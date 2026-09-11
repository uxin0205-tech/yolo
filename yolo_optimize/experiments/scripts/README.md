# 操作工具

> 本目錄已實際移入 `experiments/`。內部 run 名與模型內容保留；操作請見[整理後指南](<../../docs/OPERATIONS.md>)。

## 維護／報告（不訓練）

[maintenance](<../../tools/README.md>)：封存、GitHub 報告準備、目錄整理。維護工具已移到 optimize 根層的 tools；舊命令需更新路徑。

## 早期融合後研究工具（已移入 experiments/scripts）

| 類型 | 主要檔案 |
| --- | --- |
| 訓練與監測 | run_recovery.py、supervise_recovery.py、start_quiet_recovery.py、quiet_recovery_supervisor.py、blocking_job_monitor.py、monitor_existing_recovery.py |
| EMA／BN／LR | run_ema_diagnostic.py、report_ema_diagnostic.py、diagnose_native_recovery.py、analyze_native_control.py |
| 模型與安全稽核 | audit_recovery_states.py、audit_head_updates.py、verify_recovery_safety.py、audit_accuracy_regressions.py |
| Pose／視覺診斷 | audit_pose_errors.py、analyze_pose_ranking.py、analyze_pose_localization.py、probe_pose_classifier.py、render_pose_error_report.py |
| HOG／RepConv／scale | analyze_hog_and_repconv.py、audit_repconv_seams.py、verify_rep17_integration.py、verify_fixed_scale.py、validate_fixed_scale.py |
| 優化器與結果表 | probe_musgd_updates.py、build_round1_evidence.py、report_recovery.py |

這些不是目前應直接執行的 queue。後來融合前／combine／activation／KD 程式留在各階段，避免跨研究誤用入口。沒有為整理而重跑訓練。
