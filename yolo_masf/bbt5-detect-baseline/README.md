# BBT5 Detect Baseline

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../docs/cleanup-v1/README.md)。


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
