# 依用途選擇部署模型

| 用途 | 本機入口 | 權重與輸入 | 交接文件 |
|---|---|---|---|
| 硬體 | `result/hardware/` | 第 9 階段 rect512_768，adapt10 E2，固定高 512、寬 768 | [載入與前處理](hardware/README.md) |
| 精度 | `result/accuracy/` | 第 8 階段 qSiLU Rep17＋20 E1，原 rect640 批次補邊 | [原可攜模型](../yolo_actshape/result/stage9/handoff/README.md) |

`result/model/` 與 `result/accuracy/` 都連到同一份精度版。兩個部署版本各有 26,453,168 個參數，各自以相同資料與 evaluator 完整驗證。下表是 AP50–95 百分比，COCO 與 BBAT 的球、棒分別計算。[完整精度與 checkpoint SHA-256](manifest.json)

| 驗證指標 | 硬體 · 512×768 E2 | 精度 · 原 rect640 E1 |
|---|---:|---:|
| COCO 整體 | 51.9632% | 51.4347% |
| COCO person | 64.5865% | 64.2796% |
| COCO sports ball | 52.1912% | 51.5782% |
| COCO baseball bat | 45.3351% | 46.9576% |
| BBAT 整體框 | 62.4501% | 61.4371% |
| BBAT 整體 Pose | 89.9026% | 90.1389% |
| BBAT 球框 | 51.6028% | 49.9764% |
| BBAT 球 Pose | 86.2889% | 86.4546% |
| BBAT 棒框 | 73.2974% | 72.8977% |
| BBAT 棒 Pose | 93.5162% | 93.8233% |

## 使用方式

在有完整本機模型與 Python 相依套件的工作區，從根層執行任一命令。每個命令會載入該版本的凍結 runtime，核對 qSiLU、D1、權重及參考輸出。

```bash
.venv/bin/python -B result/hardware/load_shape.py
.venv/bin/python -B result/accuracy/load_shape.py
```

硬體版的 `prepare(bgr, (512, 768))` 將影像保持比例後補邊。精度版的 `shape` 為 `null`，需沿用原 validator 的批次 letterbox 才能重現表中 AP。[第 9 階段 evaluator](../yolo_actshape/stage9/evaluate.py) · [形狀研究導讀](../yolo_reshape/README.md)

搬移模型時請複製所選入口指向的完整內容。Git 的研究提交含程式、架構範本、sidecar、manifest 與小型結果；實際推論另需放入本機可信的同名 `rep17_20.pt`。兩個版本的檔名相同，請依下列 SHA-256 配對。

| 用途 | 檔案大小（bytes） | SHA-256 |
|---|---:|---|
| 硬體 | 106,728,476 | `a7526f060612fdada757679b6b32d408b760d4c53f99e6adcb9923f16636b457` |
| 精度 | 106,717,343 | `2454548104ed64768eb9ecdb99d44804ee9d8e1e371e2a508cf052c0911fdd35` |

第 8 階段的最佳多分支 EMA 另在本機 `result/training/rep17_20-best-ema.pt`，可供新訓練段初始化。它對應原 qSiLU Rep17＋20，訓練使用 Float PWL，部署使用 D1。該 EMA 未包含精確接續所需的 optimizer state。[第 8 階段訓練與折疊](../yolo_achitechure/result/stage8-retrain/README.md)

## 驗證與追溯

- [兩個模型獨立 CPU 驗證與清理數據](validation-1119.json) · [必要舊來源指紋](../docs/research/data/1119/reference-index.json)
- [整輪總報告](../docs/research/yolo-research-summary-2026-11-17.md) · [無 GPU 重畫數據](../docs/research/data/integrated-20261117/README.md) · [11 月 17 日歷史驗證](validation.json)
- [11 月 19 日工作紀錄](../docs/worklogs/2026-11-19-selections-and-cleanup.md) · [先前清理清單](cleanup-20261117.json)
