# 設定檔

目前唯一可用於新正式結論的入口是 [`clean/clean_ablation.yaml`](clean/clean_ablation.yaml)。它鎖定官方 COCO80 `yolo11m.pt` 的 SHA-256、`artifacts/locked-bbt5-dataset/`、direct 100 epochs、seeds 42/43，以及 Clean B0／P2／P3 模型矩陣。

| 路徑 | 用途 |
|---|---|
| `clean/` | Clean initializer 公平實驗的正式設定、說明與 CPU feasibility |
| `models/` | 共用 YOLO11m P2 架構及 slot 定義 |
| `retest/` | 舊第二輪流程的歷史實作；不可作為新排名或 unseen 證據 |
| `static-phase1.yaml`、`b0-reference.yaml`、`m7-priority.yaml` | 舊 data-exposed 流程的歷史設定，不再是正式入口 |

`bbt5-detect-baseline/` 會完整保留；其中 `yolo11m_bat_detect_init.pt` 已看過 BBT5，只能作歷史對照，不能用於 Clean 正式初始化。
