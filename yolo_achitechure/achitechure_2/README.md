# architecture_2：YOLO26m 架構、訓練、Pose 與量化研究

這個目錄是一個可重現的 YOLO26m 研究工作區。它承接 architecture_1 選出的 P3 MASF、BinaryQK 與
PWL attention，主要比較 Lite-C3k2 架構候選，之後可選擇是否執行 Pose、量化與雙來源融合研究。

目前狀態：

- 已完成正式規格、架構 builder、完整 YAML、設定驗證、測試與 Pose grouped 資料分割。
- 已建立 Detect/Pose/量化的可執行流程，但尚未執行正式長訓練。
- Pose 預設停用，是否建立、訓練或驗證 Pose 由使用者決定。
- 雙權重／雙架構融合只有來源範本，尚未選定方法，也不會自動執行。
- 尚未收到 architecture_1 正式 handoff，因此目前沒有正式 C_best 或研究結果。

## 第一次看這個專案，請照這個順序

1. 本 README：先了解整體目錄、流程與目前狀態。
2. [設定導覽](configs/README.md)：選擇架構與訓練 YAML。
3. [設定目錄](configs/catalog.yaml)：所有正式 YAML 的機器可讀索引。
4. [統一實驗規格](EXPERIMENT_SPEC.md)：需要確認正式規則與門檻時閱讀。
5. [執行手冊](RUNBOOK.md)：需要複製實際命令時閱讀。
6. [訓練指南](TRAINING_GUIDE.md)：需要理解 LR、freeze、resume 與失敗處理時閱讀。

如果只想開始一個 Detect D1 dry-run，使用：

```bash
export PYTHONPATH="$PWD/src:$PWD/../achitechure_1/src:$PWD/../../yolo_attention_final/src"
PY=/home/uxin/yolo/.venv/bin/python

$PY -m achitechure_2 train \
  --config configs/training/detect/d1-main.yaml \
  --candidate C0 \
  --checkpoint artifacts/candidates/c0/float-parent.pt \
  --run-id c0-d1
```

沒有 `--execute` 時只預覽，不會開始訓練。

## 規範優先順序

1. [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md)：唯一 authoritative spec。
2. [configs/catalog.yaml](configs/catalog.yaml) 與 [configs/](configs/)：正式 machine-readable 設定。
3. [TRAINING_GUIDE.md](TRAINING_GUIDE.md)：理由與操作原則。
4. [RUNBOOK.md](RUNBOOK.md)：經驗證命令。
5. `plan.pdf`、[YOLO26_Codex_Research_Protocol.md](YOLO26_Codex_Research_Protocol.md)：歷史參考，不是第二份規格。

目前規格版本是 `1.2.0`。正式 YAML 都保存 `spec_version` 與 `spec_sha256`。

## 目錄總覽

```text
achitechure_2/
├── EXPERIMENT_SPEC.md          唯一正式規格
├── README.md                   專案總覽（本文件）
├── RUNBOOK.md                  可複製的執行命令
├── TRAINING_GUIDE.md           訓練策略、轉換與失敗處理
├── configs/
│   ├── README.md               設定專用導覽
│   ├── catalog.yaml            所有正式設定的唯一索引
│   ├── candidates/             C0/C1/C2/C3/C3-P5/R1 架構契約
│   ├── training/
│   │   ├── detect/             D0、D1、D2、Q2；每階段一份完整 YAML
│   │   └── pose/               P0–P4；每階段一份完整 YAML
│   ├── data/                   COCO2017 與 grouped Pose dataset YAML
│   ├── quant/                  W8A8 simulation 設定
│   ├── fusion/                 未啟用的雙來源融合範本
│   └── schema/                 JSON Schema Draft 2020-12
├── src/achitechure_2/          builder、loader、trainer、量化與資料工具
├── tests/                      pytest 自動測試
├── artifacts/                  產生的資料、checkpoint、run 與 manifests
├── results/                    正式結果範本；目前沒有正式結果
└── docs/                       官方訓練來源與對照
```

## 設定系統：現在只需要一份訓練 YAML

舊方式需要自行理解 canonical、candidate overlay 與 stage fragment 如何合併。v1.2 已取消這個外部介面。

現在每個訓練階段都有一份完整 YAML，其中已包含：

- 中文名稱、用途與狀態。
- task 與 stage。
- 可使用的候選政策。
- 是否需要 `--execute`／`--enable-pose`。
- dataset。
- checkpoint transition 與 gate。
- 是否參與架構排名、研究指標與 selection backend。
- 永久凍結模組。
- 完整 Ultralytics training arguments。

正式 training YAML 由 `achitechure_2` launcher 讀取。它不是模型 architecture YAML，因此不能直接傳給
`YOLO(yaml)`；launcher 會驗證 metadata，再只把 `training:` 內容傳給 Ultralytics。

### 架構候選

| ID | 設定 | 唯一變更 | 狀態 |
|---|---|---|---|
| C0 | [c0.yaml](configs/candidates/c0.yaml) | 不修改目標 C3k2 | 必跑 reference |
| C1 | [c1-e0375.yaml](configs/candidates/c1-e0375.yaml) | `e: 0.5 → 0.375` | 必跑 |
| C2 | [c2-n1.yaml](configs/candidates/c2-n1.yaml) | `inner_n: 2 → 1` | 必跑 |
| C3 | [c3-mixed.yaml](configs/candidates/c3-mixed.yaml) | `3x3_3x3 → 1x1_3x3` | 必跑 |
| C3-P5 | [c3-p5-only.yaml](configs/candidates/c3-p5-only.yaml) | C3 kernel 只套用 layer 8 | C3 drop > 0.008 才跑 |
| R1 | [r1-rep.yaml](configs/candidates/r1-rep.yaml) | C2 加 RepBottleneck | C2 conditional 且成本改善 ≥8% 才跑 |

這些檔案是 graft contract，`standalone_loadable=false`。必須用 `build --candidate ...` 建立 checkpoint。

### Detect 正式訓練設定

| 階段 | 完整 YAML | 用途 | 是否排名 | 執行條件 |
|---|---|---|---|---|
| D0 | [d0-smoke.yaml](configs/training/detect/d0-smoke.yaml) | 3–5 epoch 冒煙測試 | 否 | 候選建立後 |
| D1 | [d1-main.yaml](configs/training/detect/d1-main.yaml) | 正式架構比較 | 是 | C0–C3 必跑；條件候選依 trigger |
| D2 | [d2-extension.yaml](configs/training/detect/d2-extension.yaml) | 從 D1 last.pt 延長 | 是 | 通過 late-improvement gate |
| Q2 | [q2-qat.yaml](configs/training/detect/q2-qat.yaml) | W8A8 假量化微調 | 否 | 只允許 C0/C_best |

### Pose 正式訓練設定

Pose 全部預設停用。

| 階段 | 完整 YAML | 用途 | checkpoint |
|---|---|---|---|
| P0 | [p0-smoke.yaml](configs/training/pose/p0-smoke.yaml) | 獨立 3 epoch smoke | grafted Pose26 checkpoint |
| P1 | [p1-head-only.yaml](configs/training/pose/p1-head-only.yaml) | 凍結 0–22，只訓練 head | grafted Pose26 checkpoint |
| P2 | [p2-neck-head.yaml](configs/training/pose/p2-neck-head.yaml) | 凍結 0–10，訓練 neck+head | P1 best.pt |
| P3 | [p3-full.yaml](configs/training/pose/p3-full.yaml) | 正式完整微調 | P2 best.pt |
| P4 | [p4-extension.yaml](configs/training/pose/p4-extension.yaml) | 條件式 resume | P3 last.pt |

真正執行任何 Pose 訓練都必須同時加入 `--enable-pose --execute`。

## 常用參數都能在 YAML 調整

每份完整 training YAML 都已顯式列出常用 Ultralytics 設定，包括 `batch`、`fraction`、`scale`、
`cache`、`device`、`workers`、`epochs`、`patience`、LR、optimizer、freeze、resume、augmentation
與 validation 參數。

- Batch size 的正式鍵名是 `batch`，不是 `batch_size`。
- `fraction=1.0` 代表使用全部訓練資料。
- `scale` 是幾何 augmentation 幅度，不是 YOLO model scale。
- `cache` 可依本機資源改為 false、ram 或 disk。
- 完整欄位、型別與修改注意事項請看 [configs/README.md 的常用可調參數](configs/README.md#常用可調參數)。

正式比較時，共用 learning/augmentation 參數必須在同 task 各階段保持一致；`config-check` 會自動檢查。

## 資料在哪裡

### Detect

- Dataset YAML：[configs/data/coco2017.yaml](configs/data/coco2017.yaml)
- `path: coco2017` 由本機 Ultralytics datasets directory 解析。
- train：`train2017.txt`
- val：`val2017.txt`
- 類別：COCO 80 classes

### Pose 原始資料

- 路徑：`/home/uxin/yolo/original/pose/bbt5.v1i.yolov8`
- 使用方式：唯讀
- 不允許直接修改原始 labels

### Pose 衍生資料

- Dataset YAML：[configs/data/pose-grouped.yaml](configs/data/pose-grouped.yaml)
- 實際目錄：[artifacts/datasets/pose_grouped](artifacts/datasets/pose_grouped)
- 分割方式：依檔名 `.rf.` 前 prefix 分組，seed 0，90/10 train/val
- train：5,988 張／2,320 groups
- val：659 張／258 groups
- leakage：0
- test split：沒有
- images：6,647 個 symlink
- labels：複製後使用
- 負座標：四個，只在衍生 label 裁切為 0

證據：

- [split-manifest.json](artifacts/datasets/pose_grouped/split-manifest.json)
- [patch-manifest.json](artifacts/datasets/pose_grouped/patch-manifest.json)

建立 grouped dataset 不等於執行 Pose，不會建立模型或開始訓練。

## 完整實驗流程

```text
architecture_1 正式 handoff
        │
        ▼
intake 驗證與 accepted.json
        │
        ▼
由同一 Float-PWL parent 獨立建立 C0/C1/C2/C3
        │
        ▼
D0 smoke ──不排名
        │
        ▼
D1 正式比較 ──Bit-True mAP50-95 排名
        │
        ├── 通過晚期改善門檻 → D2 resume
        ├── C3 rejection → 可跑 C3-P5
        └── C2 conditional 且成本改善 ≥8% → 可跑 R1
        │
        ▼
記錄 C_best（也可能沒有）
        │
        ├── 使用者選擇 Pose → C0/C_best 執行 P0–P4
        ├── C0/C_best 執行 Q0/Q1/Q2 simulation
        └── 未來另立雙來源融合研究
```

## 常用命令

### 1. 檢查全部設定

```bash
$PY -m achitechure_2 config-check
```

這會檢查 catalog、全部正式 YAML、候選矩陣、共同欄位公平性、Ultralytics 8.4.90 支援、
spec hash、Pose policy、Q2 lr0、dataset、量化與融合契約。

### 2. 預覽／執行 Detect D1

```bash
$PY -m achitechure_2 train \
  --config configs/training/detect/d1-main.yaml \
  --candidate C0 \
  --checkpoint artifacts/candidates/c0/float-parent.pt \
  --run-id c0-d1

$PY -m achitechure_2 train \
  --config configs/training/detect/d1-main.yaml \
  --candidate C0 \
  --checkpoint artifacts/candidates/c0/float-parent.pt \
  --run-id c0-d1 \
  --execute
```

### 3. 可選 Pose P1

```bash
# 預覽，不執行
$PY -m achitechure_2 train \
  --config configs/training/pose/p1-head-only.yaml \
  --candidate C0 \
  --checkpoint /path/to/c0-pose-graft.pt \
  --run-id pose-c0-p1

# 使用者明確同意後執行
$PY -m achitechure_2 train \
  --config configs/training/pose/p1-head-only.yaml \
  --candidate C0 \
  --checkpoint /path/to/c0-pose-graft.pt \
  --run-id pose-c0-p1 \
  --enable-pose \
  --execute
```

### 4. 測試

```bash
$PY -m pytest
$PY -m ruff check .
```

完整命令與階段順序請看 [RUNBOOK.md](RUNBOOK.md)。

## 量化與融合

量化設定：[w8a8-simulation.yaml](configs/quant/w8a8-simulation.yaml)

- Q0：fused FP32 reference
- Q1：calibrated W8A8 PTQ fake-quant
- Q2：W8A8 fake-quant fine-tuning
- 所有結果都必須標示 `simulation_only=true`

融合範本：[source-pair.template.yaml](configs/fusion/source-pair.template.yaml)

- 目前 `enabled=false`
- `fusion_mode=null`
- `switch_policy=null`
- 可保存來源 A/B checkpoint、architecture、head contract 與 hashes
- 你選定 weight interpolation、ensemble、routed switch 或 staged transfer 前不會實作

## 執行後東西會放哪裡

| 路徑 | 內容 |
|---|---|
| `artifacts/intake/accepted.json` | architecture_1 handoff 驗收 |
| `artifacts/candidates/<id>/` | Detect 候選 checkpoint、transfer report、snapshot、lineage |
| `artifacts/datasets/pose_grouped/` | grouped Pose 資料與 manifests |
| `artifacts/pose/candidates/<id>/` | 可選 Pose26 graft 與 lineage |
| `artifacts/runs/<run-id>/` | effective config、metrics、best/last、completion、profile |
| `artifacts/selection.json` | C0-relative 決策與 C_best |
| `artifacts/quant/<id>/` | Q0/Q1/Q2 simulation 產物 |
| `results/` | 正式結果表與報告範本 |

產生的資料、weights、runs 與 benchmarks 不放進 `src/`。

## 修改設定時的規則

1. 先更新 [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md) 並升版。
2. 修改對應的完整正式 YAML；不要重新引入 canonical/overlay/stage 拼接。
3. 如果是同 task 共用欄位，所有相關完整 YAML 必須同步。
4. 更新 `spec_version` 與 `spec_sha256`。
5. 新檔必須登錄到 [configs/catalog.yaml](configs/catalog.yaml)。
6. 執行 `config-check` 與 pytest。

`config-check` 會比較完整 YAML 的共同欄位，因此單一階段偷偷改 LR、augmentation、batch 或 optimizer 會失敗。

## 目前尚未完成的外部條件

- architecture_1 正式 handoff 尚未提供。
- C0/C1/C2/C3 正式長訓練尚未執行。
- C_best 尚未選出。
- Pose 尚未執行。
- Q0/Q1/Q2 尚未執行。
- 雙來源融合模式尚未由使用者決定。

因此目前不能宣稱任何正式架構 winner、Pose 改善或 INT8 部署結果。
