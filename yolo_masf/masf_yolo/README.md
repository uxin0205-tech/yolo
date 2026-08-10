# `masf_yolo` 資料結構與用途

## 套件定位

`masf_yolo` 是本專案的實驗控制核心，負責把資料稽核、模型建立、權重轉移、訓練、評估、選模、checkpoint 驗證與流程狀態串成可重現的工作流。研究規格以根目錄的 `MFAM_plan.md` 為準，近期實際工作順序以 `codex_plan.md` 為準。

本套件只讀取來源資料與來源權重；產生的資料切分、訓練結果、評估結果及報告都應寫到根目錄的 `artifacts/`。請注意：

- `masf_yolo/artifacts/` 是管理 artifact 的 Python 程式碼。
- 根目錄的 `artifacts/` 才是執行後產生的實驗資料，不是同一個目錄。

## 專案相關目錄

```text
yolo_masf/
├── MFAM_plan.md                    # 完整研究設計與後續 S0／Temporal 規格
├── codex_plan.md                   # 依目前環境整理的實際後續工作清單
├── README.md                       # 根目錄快速入口
├── OPERATIONS.md                   # 服務啟動、恢復與故障處理
├── pyproject.toml                  # Python 套件與依賴版本
├── configs/                        # 實驗、模型與參考權重設定
├── bbt5-detect-baseline/           # BBT5 detect 資料檢視、轉換腳本與 baseline 權重
├── field_check/                    # 獨立的資料／context 探索結果，不屬於正式管線
├── masf_yolo/                      # 本文件說明的主要 Python 套件
├── tests/                          # 與套件結構對應的自動測試
└── artifacts/                      # 執行時產物；由程式建立且不提交 Git
```

重要外部輸入：

- `bbt5-detect-baseline/dataset/`：本階段使用的 BBT5 影像與 YOLO detection labels。
- `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`：pose-derived、已接 detection head 的 baseline 初始化權重；必須標示資料暴露。
- `bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt`：依使用者指定，B0 直接評估此權重，B1 也由此初始化；其後 MFAM 變體由 B1 canonical checkpoint 初始化。

## `masf_yolo` 套件完整結構

```text
masf_yolo/
├── README.md
├── __init__.py
├── cli.py
├── contracts.py
├── pipeline.py
├── reporting.py
├── runtime.py
├── variants.py
├── workflow.py
├── artifacts/
│   ├── __init__.py
│   ├── checkpoints.py
│   ├── finalize.py
│   ├── io.py
│   ├── state.py
│   └── strict_reload.py
├── data/
│   ├── __init__.py
│   ├── audit.py
│   ├── export.py
│   ├── grouping.py
│   ├── labels.py
│   └── split.py
├── evaluation/
│   ├── __init__.py
│   ├── galleries.py
│   ├── metrics.py
│   ├── profiling.py
│   ├── reference.py
│   ├── runner.py
│   └── selection.py
├── models/
│   ├── __init__.py
│   ├── builder.py
│   ├── mfam.py
│   └── transfer.py
└── training/
    ├── __init__.py
    ├── completion.py
    ├── preflight.py
    ├── profiles.py
    ├── resume.py
    ├── runner.py
    └── worker.py
```

## 根層模組

| 檔案 | 用途 |
|---|---|
| `__init__.py` | 宣告 `masf_yolo` 為 Python 套件並提供套件基本資訊。 |
| `cli.py` | 使用者操作入口；提供 `audit`、`verify`、`pipeline start/execute/status`、`report`。 |
| `contracts.py` | 定義設定、環境、資料集、checkpoint、pipeline state 等不可變資料契約，以及 canonical JSON/SHA-256。 |
| `pipeline.py` | 把每個真實動作接到工作流：資料稽核、preflight、訓練、評估、選模、profiling 與 final audit。 |
| `reporting.py` | 從已存在且不可變的 artifacts 重建人類可讀報告，不應自行產生實驗數字。 |
| `runtime.py` | 驗證主機環境、計算 pipeline identity、啟動 systemd user service、查詢狀態與執行正式流程。 |
| `variants.py` | 集中定義 B1、M7、M0、M1、M2、M3 的 kernel、processed ratio、slot 與變體雜湊。 |
| `workflow.py` | 定義唯一的依賴 DAG、stage 狀態、輸入／輸出雜湊、重用條件與失敗狀態。 |

## `artifacts/`：產物安全與 checkpoint 管理程式

| 檔案 | 用途 |
|---|---|
| `io.py` | 原子寫入 JSON、`fsync`、檔案鎖與 pipeline lock，避免中斷時留下半寫入狀態。 |
| `state.py` | 依 input/output hashes 判斷已完成 stage 是否可以安全重用。 |
| `checkpoints.py` | 儲存與載入 CPU float32 canonical state dict，並驗證資料、設定、環境與模型雜湊。 |
| `finalize.py` | 在獨立 process 中把 Ultralytics native best EMA 轉成 repository-owned canonical checkpoint。 |
| `strict_reload.py` | 在全新 process 重建指定變體、嚴格載入全部 tensor，並可接續做 validation gate。 |

## `data/`：資料稽核與固定切分

| 檔案 | 用途 |
|---|---|
| `labels.py` | 嚴格解析 YOLO detection label；拒絕缺欄、非有限值、越界座標與未知類別。 |
| `grouping.py` | 依來源 stem、sequence、Roboflow 名稱與影像內容雜湊做 union grouping，防止相鄰幀或增強 siblings 洩漏。 |
| `split.py` | 以 seed 42 產生 deterministic 80/10/10 group split，並驗證類別、數量與零重疊條件。 |
| `export.py` | 把 YOLO boxes 轉成精確 640-letterbox 座標的 COCO boxes。 |
| `audit.py` | 串接索引、標註、分組、切分、COCO export 與尺寸統計，產生正式 dataset manifest。 |

資料稽核不會修改 `bbt5-detect-baseline/dataset/`；輸出會放在 `artifacts/static-phase1/dataset/`。

## `models/`：P2、MFAM 與權重轉移

| 檔案 | 用途 |
|---|---|
| `mfam.py` | 實作完整 MFAM 與 PartialMFAM。3/5 使用 depthwise square convolution；7/9 使用 `1×k → k×1` depthwise convolution；最後以 1×1 fusion。 |
| `selective.py` | 實作 SP2：hidden 32 的 Ball-only 輕量 P2 towers、Ball target filtering、雙 loss 與共同兩類別 decode。 |
| `builder.py` | 從 repository-owned YOLO11m P2 YAML 建模，在第一次 forward 前安裝 P2/P3 slots，不修改 Ultralytics 全域 registry。 |
| `transfer.py` | 以明確 semantic mapping 轉移官方 YOLO11m、B1 canonical 或 SP2P 雙來源權重；SP2P 僅允許 BEST_PARTIAL 覆寫 `model.20.*`。 |

目前變體：

| ID | P2 slot | channels 處理比例 | 分支 |
|---|---|---:|---|
| B1 | Identity | 0 | 無 MFAM |
| M7 | MFAM | 1 | 3、5、7，其中 7 為 `1×7 → 7×1` |
| M0 | MFAM | 1 | 3、5、7、9 |
| M1 | MFAM | 1 | 3、5 |
| M2 | PartialMFAM | 1/2 | processed branch 使用 M1 |
| M3 | PartialMFAM | 1/4 | processed branch 使用 M1 |
| P3M | P2 Identity、P3 MFAM | 1 | 只在 P3 使用 3、5、7；7 為 `1×7 → 7×1`，不含 9×9 |
| SP2 | Ball-only 輕量 P2 head | 1 | hidden 32；P3–P5 保留 Ball/Bat 主 head |
| SP2P | PartialMFAM＋Ball-only 輕量 P2 head | 1/2 或 1/4 | 由 M2/M3 validation 勝出者決定；只含 3、5 |

SP2 已完成 CPU 實作與驗證。P2 使用 hidden 32 的兩層 depthwise-separable box/class towers，class tower 只輸出 Ball；P3–P5 主 head 保留 Ball/Bat。訓練時 P2 只接收 Ball target；推論時 P2 Ball 與 P3–P5 候選合併成共同兩類別輸出並交由官方 NMS。SP2 P2 head 為 16,161 params，B1 標準 P2 head 為 151,362 params，參數減少 89.32%。它不使用 Top-K、資料相依 Gather、不規則 sparse tile 或事後 dense mask。既有 CPU ONNX 結果僅作歷史技術驗證，正式 pipeline 不執行 ONNX 匯出。

SP2 正式訓練分為 SP2-A（凍結 backbone indices 0–10、10 epochs）與 SP2-B（全解凍、90 epochs）。M2/M3 validation 選模後，SP2P 同時繼承 SP2-B 的共用網路與 Selective head，以及勝出者的 P2 partial-MFAM，再依相同 A/B 10+90 訓練。SP2P 是序列式訓練，報告必須單獨標示其較高訓練預算。

以 640×640 CPU hook 做相同口徑的靜態 profiling：B1 為 20,550,872 params、43,715,353,600 MACs、87.431 GFLOPs；SP2 為 20,415,671 params、40,264,064,000 MACs、80.528 GFLOPs。SP2 總 MACs 下降 7.895%、總 params 下降 0.658%；P2 feature activation仍為 13,107,200 bytes，因為本方案縮減的是高解析 prediction tower，而不是移除 P2 feature map。

## `training/`：訓練前檢查、執行與恢復

| 檔案 | 用途 |
|---|---|
| `profiles.py` | 產生 B1、SP2、SP2P 的 A/B 兩階段 profiles，以及 smoke、formal 的完整 Ultralytics arguments。 |
| `preflight.py` | 執行真實 loss、backward、optimizer step 與共同 batch probe。正式使用時需要 CUDA。 |
| `runner.py` | Repository-owned DetectionTrainer adapter；保留 custom model，並略過 Ultralytics 重複的 final validation。 |
| `worker.py` | 在隔離 subprocess 執行單一訓練 stage，父程序只接收路徑、雜湊與結果 manifest。 |
| `completion.py` | 由 `results.csv`、best/last checkpoint 與 epochs 判斷 run 為 absent、incomplete、complete 或 invalid。 |

## `evaluation/`：指標、選模與硬體分析

| 檔案 | 用途 |
|---|---|
| `runner.py` | 執行 Ultralytics inference，匯出原始預測，並以 faster-coco-eval 計算共同 COCO 指標。 |
| `metrics.py` | 計算 per-class、Ball、tiny/small/larger ball、blur proxy 等應用指標與 GT counts。 |
| `selection.py` | 僅以 M2/M3 validation 指標選出 `BEST_PARTIAL`，原子凍結 `selection.json`，並阻止 test 提前執行。 |
| `profiling.py` | 統計 Params、MACs/GFLOPs、activation、depthwise/1×1 operator 與 feature traffic。 |
| `galleries.py` | 建立可追溯的 false-positive records 與 letterbox contact sheet。 |
| `reference.py` | 嚴格檢查資料夾內 pose-derived B0 權重的 SHA-256、task、classes、strides、forward 與資料暴露資訊。 |

## 正式執行產物

正式輸出預設位於 `artifacts/static-phase1/`，大致如下：

```text
artifacts/static-phase1/
├── dataset/               # 固定 split、image lists、COCO JSON、統計與 manifest
├── preflight/             # 各變體的 CUDA acceptance 證據
├── smoke_runs/            # 工程 smoke 訓練；不作論文結果
├── runs/                  # 正式 Ultralytics 訓練目錄
├── training/              # 各 stage 的 resolved args、canonical checkpoint 與 run manifest
├── references/            # B0 等外部參考權重的檢查結果
├── evaluation/
│   ├── val/               # validation metrics、預測與錯誤案例
│   └── test/              # selection 凍結後才允許產生
├── profiles/              # 全模型硬體成本資料
├── selection.json         # M2/M3 validation-only 選模結果
├── stages/                # 每個 DAG stage 的狀態與 input/output hashes
├── state.json             # 目前 pipeline 狀態
├── final_audit.json       # 終局完整性檢查
└── report.md              # 所有結果完成後才重建的最終報告
```

目錄只會在對應工作真的執行後出現；不能因程式碼或單元測試存在就假設實驗已完成。

## 主要資料流

```text
BBT5 原始 detect view
    → 資料稽核與 leakage-safe split
    → 固定 dataset manifest
    → 環境／模型檢查
    → gpu_wait 等待 GPU 空閒
    → CUDA preflight
    → B1 與各 MFAM 變體訓練
    → fresh-process canonical checkpoint 驗證
    → SP2-A/B 10+90
    → M2/M3 validation 選出 BEST_PARTIAL
    → 凍結 selection.json
    → SP2P smoke 與 SP2P-A/B 10+90
    → validation 全模型評估
    → test 全模型評估與 profiling
    → final audit
    → 最終報告
```

## 常用命令

在專案根目錄執行：

```bash
# CPU：資料稽核
../.venv/bin/python -m masf_yolo.cli audit --config configs/static-phase1.yaml

# CPU：測試與語法檢查
../.venv/bin/python -m pytest -q
../.venv/bin/python -m compileall -q masf_yolo tests

# GPU 空閒後：完整環境驗證
../.venv/bin/python -m masf_yolo.cli verify --config configs/static-phase1.yaml

# GPU 空閒且所有前置 gate 通過後：啟動背景流程
../.venv/bin/python -m masf_yolo.cli pipeline start --config configs/static-phase1.yaml

# 唯讀查詢狀態
../.venv/bin/python -m masf_yolo.cli pipeline status --config configs/static-phase1.yaml

# 全部結果完成後才重建報告
../.venv/bin/python -m masf_yolo.cli report --config configs/static-phase1.yaml
```

`pipeline execute` 是 systemd 背景服務使用的內部入口，平常不要手動與另一個 pipeline 同時執行。

## 操作原則

- GPU 被其他工作占用時，只執行 CPU audit、測試、結構檢查與文件工作。
- 不刪除或覆寫來源 dataset、官方權重及 baseline 權重。
- 不把 `artifacts/`、`runs/`、checkpoint、cache 加入 Git。
- B0 直接使用 pose-derived detect 權重做 baseline 評估，B1 與 MFAM 主線也沿用此初始化來源；全部結果必須標示資料暴露，且 B0 不得參與 M2/M3 選模。
- MASF 只使用固定名稱 `masf-yolo-phase1.service`，不依 pipeline hash 另建多個服務；pipeline identity 另外保存在 metadata。
- 背景流程先等待 `yolo-p2-study.service` 完全結束，再要求 GPU 至少 24 GiB 可用且利用率不超過 10%，連續三次 60 秒輪詢都通過後，才會離開 `gpu_wait`。
- 正式啟動前先審閱專案根目錄的 `GPU_TEST_PLAN.md`；未確認前不啟動 MASF service。
- test 只在 `selection.json` 凍結後執行。
- S0 與 Temporal Motion 不屬於目前 Phase 1；需要另行擴充模型、資料 pairing 與工作流。
