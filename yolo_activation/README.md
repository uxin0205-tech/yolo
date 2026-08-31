# YOLO Activation 研究交付

這個子專案整理 YOLO26m Full35 共用模型的 activation 研究、正式實驗、數學候選、量化交接與
可重建權重。Activation-only 階段已停止，目前不需要再延長 epoch 或繼續 activation 調參；下一步應在
相同量化規則下比較 accepted SiLU 與完成 10-epoch recovery 的 `qsilu_pq`。

## 一分鐘結論

- 正式父模型來自 `yolo_combine/final/full35`，不是隨機初始化。
- 使用完整 COCO2017 與不可變的 Canonical BBAT5 v1；沒有 30% 資料夾、重新抽樣或重新切分。
- 實際納入六種 activation：SiLU、Hardswish、ReLU、qSiLU、`poly_quality`、`poly_shift`。
- 唯一通過 10-epoch 八項 accuracy gate 的非 SiLU 候選是 `qsilu_pq`。
- 全非 SiLU Q3 已完成 22/22 純 CPU 逐區評估；Hardswish 保留 3 個部署候選，LeakyReLU 1/8 只保留 `masf`。
- SIPA 已完成數學、實作與 Full35 實驗；BCSP 只完成支援架構，沒有完成 mixed-policy 搜尋。
- qSiLU 20-epoch finalist 是人工中止狀態，不能當成正式 finalist；量化應使用已完成的 10-epoch 權重。
- 尚無 FPGA／ASIC 的 latency、power、LUT、DSP 或 BRAM 實測。

## 主要入口

| 想知道的內容 | 入口 |
| --- | --- |
| 完成了什麼、結果如何、接下來做什麼 | [Activation 權威整合報告](reports/completed-activation-integrated-report.md) |
| qSiLU 哪些 region 可換成更簡單 activation | [Full35 Q3 純 CPU 最終報告](reports/full35-q3-cpu-final-report.md) |
| 每種 activation 的公式與係數怎麼來 | [可讀版完整數學推導](docs/research/full35-activation-mathematical-derivations.md) |
| 所有文件如何區分目前與歷史狀態 | [文件導覽](docs/README.md) |
| 正式訓練配置、epoch、batch、queue 與 gate | [Full35 實驗配置](training/full35/README.md) |
| 可供程式核對的結果 | [JSON](reports/full35-activation-results.json)／[CSV](reports/full35-activation-results.csv) |
| SiLU 與 qSiLU 權重 | [可重建權重](release/weights/README.md) |
| 每次修改與驗證紀錄 | [中文工作紀錄](docs/worklogs/README.md) |
| BBAT5 全域規範 | [Canonical BBAT5 v1 規範](../docs/agents/bbat5-datasets.md) |

## 資料夾用途

| 路徑 | 用途 | GitHub 狀態 |
| --- | --- | --- |
| `src/activation_lab/` | activation、fixed-point、manifest、training 與 search 支援程式 | 發布 |
| `scripts/` | 驗證、profiling、sensitivity、queue 與結果匯出工具 | 發布 |
| `training/` | 正式設定、schema、manifest、recipe 與 queue | 發布 |
| `reports/` | 權威人類報告與 machine-readable 結果 | 發布 |
| `docs/` | 文件導覽、研究、數學與工作紀錄 | 發布 |
| `tests/` | 介面、資料契約、queue、結果與權重回歸測試 | 發布 |
| `release/weights/` | 兩組必要 inference 權重的分片、hash 與重建工具 | 發布 |
| `artifacts/` | 本機約 29 GB 原始 run、queue、OOM 與稽核證據 | 不發布、不得當雜物刪除 |

## 資料契約

| 任務 | 正式入口 | 固定規則 |
| --- | --- | --- |
| BBAT5 v1 | `/home/uxin/yolo/original/pose/derived/bbat5-v1/` | 使用 formal assignment，不得重切、抽樣或改標註 |
| COCO2017 COCO80 Detect | `/home/uxin/yolo/coco2017.yaml` | 完整 train／val，不得接到 BBAT5 二類 head |

COCO 與 BBAT5 分別驗證並對各自 accepted SiLU baseline 計算差值，不把兩個資料集的 raw mAP 直接平均。

## 本地驗證

```bash
/home/uxin/yolo/.venv/bin/python -m pytest
/home/uxin/yolo/.venv/bin/python -m ruff check .
/home/uxin/yolo/.venv/bin/python -m ruff format --check .
/home/uxin/yolo/.venv/bin/python scripts/validate_phase0.py
/home/uxin/yolo/.venv/bin/python scripts/activation_training.py toy-dry-run
```

原始結果由本機 `artifacts/` 產生，但 GitHub 上的 JSON／CSV、測試、SHA-256 與權重分片足以核對主要
結論。歷史失敗權重、中止 qSiLU finalist 的 `last.pt` 與 optimizer checkpoints 不發布。
