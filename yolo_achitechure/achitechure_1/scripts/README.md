# 操作腳本

本目錄提供可重現的薄包裝入口；正式設定與完整命令仍以 `../RUNBOOK.md` 為準。

| 腳本 | 用途 |
|---|---|
| `import_fraction03_source.py` | 匯入並雜湊驗證 accepted Phase A2 Float checkpoint |
| `run_fraction03_continuation_queue.py` | 30% 資料的 B batch16 → C batch8×2 對稱 queue |
| `run_fraction10_phase_b_control_queue.py` | 完整資料、batch16 的 Phase B 對照 queue |
| `run_fraction10_phase_c_queue.py` | 從完整資料 Phase B gate 接續 C batch8×2，支援 patience override |
| `profile_dataloader_workers.py` | 量測 workers／batch／fraction 的 DataLoader RAM 與吞吐 |
| `validate_ball_bat.py` | 建立 ball／bat detection view 並驗證 Bit-True checkpoint |

最終交付包另提供 `../final/scripts/build_report.py` 與 `build_training_report.py`，可從已保存 raw evidence
重建 11 個候選的完整 AP 表、8 次訓練摘要、權重索引與三組比較圖；`inspect_models.py` 與
`verify_bundle.py` 負責 checkpoint／manifest 稽核。

所有 queue 都以獨立子程序執行 GPU job，並將 runtime state、checkpoints、telemetry 與 validation 留在
`../artifacts/`。這些大型／生成檔由 `.gitignore` 排除；完成結論另存於 `../results/`。
