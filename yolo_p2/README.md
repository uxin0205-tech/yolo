# YOLO11m P2 COCO2017 實驗專案

本專案基於 Ultralytics 8.4.90 與 YOLO11m，加入 stride 4 的 P2 Detect 輸出，以研究小物件偵測效果。正式實驗使用完整 COCO2017、單張 NVIDIA GeForce RTX 5090、imgsz 640、batch 9、seed 0 與 deterministic 訓練。

## 模型版本

| 版本 | 架構 | Detect 尺度 | 訓練方式 |
| --- | --- | --- | --- |
| A0 | 官方 YOLO11m | P3/P4/P5，stride 8/16/32 | 官方權重，本機不重訓 |
| A1 | YOLO11m-P2 Direct | P2/P3/P4/P5，stride 4/8/16/32 | 全模型 fine-tune 100 epochs |
| A2 | YOLO11m-P2 Direct Staged | 與 A1 相同 | P2-only 20 epochs，再全模型微調 73/80 epochs |

A2 不是額外的 feature-fusion 架構，而是與 A1 共用相同推論架構、採兩階段訓練。正式 A2 使用 Stage 2 epoch 39 的最佳權重。

## 主要結果

| 版本 | COCO mAP50-95 | AP_small | Params | GFLOPs | RTX 5090 FP16 median |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 51.53 | 33.43 | 20.11M | 68.5 | 6.66 ms |
| A1 | 51.18 | 33.41 | 20.49M | 87.9 | 7.19 ms |
| A2 | **51.78** | **33.83** | 20.49M | 87.9 | 7.18 ms |

A2 相對 A0 的 COCO mAP50-95 提升 0.25 AP、AP_small 提升 0.40 AP，但 GFLOPs 增加 28.27%，大物件 AP 降低 0.80。完整分析見 [正式報告](p2_study/results/REPORT.md) 與 [比較圖](p2_study/results/comparison.png)。

## 環境

```bash
cd yolo_p2
source ../.venv/bin/activate
python -m pip install --no-deps -e .
```

已驗證環境：

- Python 3.12
- Ultralytics 8.4.90（此專案 editable install）
- PyTorch 2.11.0+cu128
- CUDA runtime 12.8
- NVIDIA GeForce RTX 5090
- faster-coco-eval 1.7.2

## 資料

COCO2017 保存在 `p2_study/data`：

- train2017：118,287 張影像、860,001 個物件標註
- val2017：5,000 張影像、36,781 個物件標註
- 80 類別
- `images` 是指向本機 COCO2017 `images/` 目錄的符號連結
- labels 與官方 annotation JSON 保留在專案中，但由 Git ignore

詳細結構見 [資料說明](p2_study/data/README.md)。

## 重新訓練

共同設定在 `p2_study/config.yaml`，其中 seed 固定為 `0`、`deterministic: true`。正式封存成果位於 `p2_study/results`；新訓練只寫入可重建的 `p2_study/artifacts`，不會覆寫正式成果。

```bash
./p2_study/ctl.sh start
./p2_study/ctl.sh status
./p2_study/ctl.sh logs
./p2_study/ctl.sh stop
./p2_study/ctl.sh resume
```

前景執行：

```bash
python -m p2_study.run --config p2_study/config.yaml
python -m p2_study.run --config p2_study/config.yaml --resume
```

操作與失敗恢復方式見 [P2 Study 說明](p2_study/README.md)。

## 驗證與推論

```bash
yolo detect val model=p2_study/results/weights/A2_best.pt data=p2_study/coco2017.yaml imgsz=640 batch=1 device=0
yolo detect predict model=p2_study/results/weights/A2_best.pt source=/path/to/image.jpg imgsz=640 device=0
```

四個有效權重及 SHA-256 見 [權重說明](p2_study/results/weights/README.md)。

- `p2_study/results/weights/A0_yolo11m.pt`
- `p2_study/results/weights/A1_best.pt`
- `p2_study/results/weights/A2_stage1_best.pt`
- `p2_study/results/weights/A2_best.pt`

## 測試

```bash
python -m pytest tests/test_p2_study.py -q
ruff check tests/test_p2_study.py p2_study
```

詳細測試範圍見 [tests/README.md](tests/README.md)。

## 資料夾結構

```text
yolo_p2/
├── README.md                    # 本文件
├── LICENSE                      # AGPL-3.0 授權
├── pyproject.toml               # Python 套件與依賴設定
├── P2_PLAN.md                   # 實驗設計與排程
├── p2_study/
│   ├── README.md                # 訓練操作
│   ├── config.yaml              # seed 0 與共同訓練設定
│   ├── coco2017.yaml            # Ultralytics dataset YAML
│   ├── ctl.sh                   # 背景啟停控制器
│   ├── run.py                   # 頂層排程器
│   ├── worker.py                # 各階段 worker
│   ├── models.py                # P2 建模與權重轉移
│   ├── analyze.py               # 結果分析
│   ├── data/                    # Git 僅含說明；COCO payload 保留本機
│   ├── results/                 # 已接受且不可覆寫的正式成果
│   └── artifacts/               # 重新訓練時建立，不提交 Git
├── ultralytics/                 # 上游原始碼及 P2 修改
├── tests/                       # P2 focused tests
└── docs/                        # P2 整理設計與實作計畫
```

## 重要連結

- [完整正式報告](p2_study/results/REPORT.md)
- [成果索引](p2_study/results/README.md)
- [比較圖](p2_study/results/comparison.png)
- [機器可讀摘要](p2_study/results/summary.json)
- [實驗計畫](P2_PLAN.md)
- [整理設計](docs/superpowers/specs/2026-08-10-yolo-p2-cleanup-design.md)
- [整理實作計畫](docs/superpowers/plans/2026-08-10-yolo-p2-cleanup.md)

本專案沿用 Ultralytics 的 AGPL-3.0 授權，詳見 `LICENSE`。
