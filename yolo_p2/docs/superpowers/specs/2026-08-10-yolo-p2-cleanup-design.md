# YOLO11m P2 專案整理設計

## 目標

將隱藏 worktree 中已完成的 YOLO11m P2 實驗整合回 `yolo_p2` 主工作目錄，保留完整重新訓練能力、四個有效權重、seed、正式報告、指標與圖表，同時移除可重建、重複或已失效的大型產物。

整理完成後，使用者不需要進入 `.worktrees/p2-study`，即可閱讀專案、重新訓練、驗證模型及找到所有正式結果。

## 採用方案

採用「可重新訓練的精簡成果版」：

- 保留完整 Ultralytics P2 修改、排程程式、設定與測試。
- 保留 COCO2017 labels、annotations 與既有影像連結。
- 將正式成果與未來重新訓練產生的暫時 artifacts 分離。
- 只保存四個有效 checkpoint，不保存 gate、health、取消或 invalidated 權重。
- 將專案擁有的功能目錄加上 README；不在 Ultralytics 數百個 Python 套件子目錄重複建立 README。

## Git 與 worktree 整合

1. 在刪除前驗證兩個 worktree、分支、正式權重與結果檔。
2. 將被 ignore 的必要成果暫存到工作區內明確的 staging 目錄，並建立檔名、大小與 SHA-256 清單。
3. 將 `feature/yolo11m-p2-study` fast-forward 整合到 `p2-detect-head`。
4. 在主工作目錄建立整理後的 `p2_study/results`，搬入已驗證成果。
5. 更新設定與 README 中的路徑，讓重新訓練使用主工作目錄。
6. 執行模型載入、資料集、測試、報告與檔案清單驗證。
7. 使用 `git worktree remove` 正規移除 `.worktrees/p2-study`，再刪除已整合的本機 feature branch。

若任何 checkpoint 雜湊、模型載入或資料集檢查失敗，停止清理，不移除 worktree。

## 最終目錄結構

```text
yolo_p2/
├── README.md
├── P2_PLAN.md
├── p2_study/
│   ├── README.md
│   ├── config.yaml
│   ├── coco2017.yaml
│   ├── ctl.sh
│   ├── run.py
│   ├── worker.py
│   ├── models.py
│   ├── analyze.py
│   ├── data/
│   │   ├── README.md
│   │   ├── coco2017/
│   │   └── annotations/
│   └── results/
│       ├── README.md
│       ├── REPORT.md
│       ├── comparison.png
│       ├── comparison.csv
│       ├── summary.json
│       ├── weights/
│       │   ├── README.md
│       │   ├── A0_yolo11m.pt
│       │   ├── A1_best.pt
│       │   ├── A2_stage1_best.pt
│       │   └── A2_best.pt
│       ├── metrics/
│       │   ├── README.md
│       │   ├── A0/
│       │   ├── A1/
│       │   └── A2/
│       └── metadata/
│           ├── README.md
│           ├── config.yaml
│           ├── preflight.json
│           ├── state.json
│           ├── early_stop.json
│           ├── batch_manifest.json
│           ├── model_info.json
│           ├── initial_checkpoints.json
│           └── a1_weight_transfer.json
├── ultralytics/
├── tests/
├── docs/
├── examples/
└── docker/
```

`p2_study/artifacts` 不作為成果封存位置。重新訓練時仍可由排程器重新建立該目錄；整理後的正式成果固定放在 `p2_study/results`。

## 必須保留

### 程式與重現設定

- P2 Detect Head YAML、Ultralytics parser/head 修改、排程、分析程式與測試。
- `config.yaml`、`coco2017.yaml`、實際正式 `args.yaml` 與 `results.csv`。
- seed 0、deterministic、batch、imgsz、optimizer、LR、Ultralytics/PyTorch/CUDA/GPU 版本。
- 權重轉移與初始 checkpoint metadata。

seed 0 必須同時出現在主設定、正式訓練參數、機器可讀摘要與 README/報告中。

### 資料

- COCO train2017/val2017 labels。
- 官方 train/val annotation JSON。
- 既有 images 連結。
- 資料集 README 中記錄 118,287 train、5,000 val、80 類別及標註數。

### 四個有效權重

1. A0 官方 `yolo11m.pt`。
2. A1 正式 100 epochs 的 `best.pt`。
3. A2 Stage 1 P2-only 20 epochs 的 `best.pt`。
4. A2 Stage 2 最終正式評估使用的 `best.pt`。

每個權重在移動前後都要比對 SHA-256，並以 Ultralytics 實際載入。A0 必須是 stride 8/16/32；A1、A2 Stage 1、A2 必須是 stride 4/8/16/32。

### 報告、指標與圖

- 更新後的 `REPORT.md`。
- `summary.json`、`comparison.csv`、`comparison.png`。
- A0/A1/A2 的 `coco_metrics.json`。
- A0/A1/A2 的 benchmark JSON。
- 正式訓練歷史 `results.csv` 與 `args.yaml`。

目前正式實驗只有 `comparison.png` 一張圖；該圖必須保留並可由 README 連結。

## 必須刪除

- `artifacts/invalidated` 全部內容。
- gate 與 health runs，包括其中的 `best.pt` 和 `last.pt`。
- 所有正式與 staged `last.pt`。
- 初始化中間權重 `a1_initial.pt`。
- 已下載並解壓完成的 COCO ZIP。
- 大量 stage/controller logs。
- train/val `*.cache`、所有 `__pycache__`、`.pytest_cache`、`.ruff_cache`。
- 大型 validation `predictions.json`，只保留正式 COCO metrics。
- 非正式 gate/health predictions 與 plots。
- 過期的頂層 `coco2017.yaml`。
- 成功整合後的 `.worktrees/p2-study`。

刪除範圍不包含四個有效權重、正式結果圖、seed 或任何正式指標。

## README 覆蓋範圍

- 根目錄 `README.md`：專案目的、A0/A1/A2 定義、快速開始、訓練、驗證、結果摘要與整體目錄樹。
- `p2_study/README.md`：排程器、訓練階段、停止/續訓與 artifacts/results 差異。
- `p2_study/data/README.md`：資料來源、數量、路徑與可重建 cache。
- `p2_study/results/README.md`：正式成果索引。
- `weights/README.md`：四個權重的來源、用途、最佳 epoch、stride、參數量與 SHA-256。
- `metrics/README.md`：COCO API、Ultralytics mAP 與 benchmark 方法。
- `metadata/README.md`：seed、環境、設定與狀態檔用途。
- `tests/README.md`：P2 測試入口。
- `ultralytics/README.md`：說明此目錄為上游原始碼及本研究修改點。
- `docs`、`examples` 已有上游 README，保留並由根 README 導覽。

不為 `.git`、快取、生成輸出或每個 Python package 子目錄建立 README。

## 路徑與重訓行為

- 主設定中的 pretrained 路徑改為整理後的 A0 權重位置。
- 資料 YAML 使用主工作目錄下的 `p2_study/data/coco2017`。
- 重新訓練輸出寫入可刪除的 `p2_study/artifacts`。
- 正式封存結果位於唯讀概念的 `p2_study/results`，排程器不得覆寫。
- README 提供從環境啟用、preflight、開始、狀態、續訓、驗證到分析的完整指令。

## 驗證標準

整理完成必須同時滿足：

1. Git 只剩單一 `yolo_p2` 工作目錄，不再有 `.worktrees/p2-study`。
2. 四個有效權重的 SHA-256 與整理前一致，且全部可載入。
3. A1/A2 Detect 包含 stride 4 的 P2 輸出。
4. seed 0 可在設定、args、summary、README 與報告中找到。
5. COCO 影像/labels/annotations 數量檢查通過。
6. `pytest tests/test_p2_study.py` 通過。
7. 分析程式可讀取整理後的正式 metrics，或 README 明確記錄封存結果不可被重訓覆寫。
8. README 連結、結果圖、權重與所有列出的路徑存在。
9. `git diff --check` 通過。
10. 最終容量與刪除清單記錄在整理報告中。

## 預期結果

預計目錄由約 4.0 GiB 降至約 1.3–1.5 GiB。最終仍可從 A0 重新建立 P2 權重並完整執行 A1/A2 訓練，也可直接使用四個保留 checkpoint 進行驗證或推論。
