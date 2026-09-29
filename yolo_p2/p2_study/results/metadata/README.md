# 實驗 Metadata

| 檔案 | 用途 |
| --- | --- |
| `archive_manifest.json` | 四權重路徑、大小、來源、SHA-256 與 seed 0 |
| `config.yaml` | 正式訓練當時的設定快照 |
| `preflight.json` | Ultralytics/PyTorch/CUDA/GPU、資料數量與 evaluator |
| `state.json` | 排程階段、attempt、時間與完成狀態 |
| `early_stop.json` | A2 Stage 2 提前停止原因與 best epoch |
| `batch_manifest.json` | batch 9 校準結果 |
| `model_info.json` | Params、GFLOPs、Detect scales 與 stride |
| `initial_checkpoints.json` | 初始權重歷史來源 |
| `a1_weight_transfer.json` | 官方 P3/P4/P5 Detect 權重映射驗證 |
| `cleanup_summary.md` | 整理前後容量與刪除範圍 |

部分歷史 JSON 包含當時 worktree 的絕對路徑，作為原始 provenance 保留；目前可用路徑以根 README、主 `p2_study/config.yaml` 與 `archive_manifest.json` 為準。
