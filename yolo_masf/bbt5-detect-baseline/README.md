# BBT5 Detect Baseline

本資料夾依使用者要求完整保留，提供 BBT5 兩類別 detection view、舊 pose-derived initializer 與資料轉換工具；任何清理不得刪除或搬動本目錄。

## 結構

| 路徑 | 用途 |
|---|---|
| `dataset/` | BBT5 images/labels 與來源 split view；正式 Clean 實驗只讀 |
| `data.yaml` | 原資料入口 |
| `prepare_dataset.py` | 從 pose dataset 建立 detection view |
| `weights/yolo11m_bat_detect_init.pt` | 已接觸 BBT5 的舊 initializer，只保留相容性與歷史用途 |
| `transfer_pose_to_detect.py` | pose-to-detect 轉換工具 |

## Clean 實驗如何使用

- 訓練資料仍來自本目錄，但使用 [`../artifacts/locked-bbt5-dataset/`](../artifacts/locked-bbt5-dataset/) 鎖定的 train/validation 清單。
- Clean initializer 改用 `../../original/weight/yolo11m.pt`，不使用本目錄的 data-exposed initializer。
- Trainer 取得的 YAML 不含 historical test。
- 目前沒有有效正式結果；完整規劃見 [`../codex_plan.md`](../codex_plan.md)。

舊 initializer 本身不是「錯誤檔案」，但不得再被標示為 clean、unseen 或新公平實驗的起始權重。
