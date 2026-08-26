# YOLO26m Binary Q/K＋Bit-True PWL 完整可攜 Workspace

`final/` 是可以單獨複製到另一台機器的完整 workspace。它包含：

- 完整 production code，不依賴上一層 `src/`。
- 正式推論 checkpoint 與兩個必要 training parent checkpoints。
- Float-PWL／Bit-True PWL、LR sweep、recovery 與 evaluation configs。
- 可直接執行的 CLI、完整 functional tests 與 CPU smoke script。
- Binary Q/K、Hadamard、power-of-two scale 與 PWL 的創新／演算法文件。
- 已完成實驗的結果摘要與 retained parent provenance。

唯一未包含的是 COCO2017 圖片與標註。收到或掛載資料後，使用
`python run.py configure` 對接即可。

## 文件入口

- [創新與演算法：Binary Q/K＋Bit-True PWL](BINARY_QK_PWL_ALGORITHM.md)
- [訓練方法與分層 LR](TRAINING.md)
- [正式結果、negative results 與限制](RESULTS.md)
- [機器可讀結果](results.json)
- [精簡指標表](metrics.csv)
- [Production package 導覽](yolo_attention/README.md)

## 完整檔案架構

```text
final/
├── README.md
├── BINARY_QK_PWL_ALGORITHM.md
├── TRAINING.md
├── RESULTS.md
├── results.json
├── metrics.csv
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── run.py
│   └── 對外單一入口：check / doctor / configure / predict / recipe / train / status
├── training_recipe.py
│   └── 可攜設定、checkpoint hash、資料對接與 reproduction queue wrapper
├── pwl-final-best.pt
│   └── 正式 YOLO26m Bit-True PWL 推論 checkpoint
│
├── yolo_attention/
│   ├── __init__.py
│   ├── README.md
│   │
│   ├── config.py                 # VariantConfig 與 enum contract
│   ├── attention.py              # HardwareFriendlyAttention 完整資料流
│   ├── binary_basis.py           # sign、XNOR＋popcount、Hadamard Binary Q/K
│   ├── normalization.py          # Exact、Float-PWL、Bit-True PWL
│   ├── projection.py             # fused QKV 拆分與 channel layout
│   ├── relative_bias.py          # decomposed 2D relative-position bias
│   ├── quantization.py           # deterministic fake quant primitives
│   ├── schedule.py               # progressive schedule
│   ├── bdcn.py                   # BDCN reference／checkpoint 相容
│   │
│   ├── integration.py            # fail-closed YOLO26 兩-site 轉換
│   ├── training.py               # Float-PWL Ultralytics training Adapter
│   ├── scopes.py                 # trainable parameter／BN buffer scopes
│   ├── optimizer_scope.py        # layer-wise LR 與 optimizer filtering
│   ├── run_config.py             # immutable TrainingRecipe
│   ├── runner.py                 # training request 與 artifact 建立
│   ├── evaluation.py             # COCO evaluation 與標準結果 contract
│   ├── final_evaluation.py       # Bit-True materialization／evaluation
│   ├── profiling.py              # analytical operation accounting
│   ├── artifacts.py              # run manifest 與環境 provenance
│   ├── export.py                 # winner checkpoint export contract
│   │
│   ├── experiments.py            # immutable experiment registry
│   ├── workflow.py               # declarative research funnel
│   ├── queue_model.py            # typed QueueState／QueueJob／QueueResult
│   ├── queue_policy.py           # pure winner／gate rules
│   ├── queue_store.py            # atomic queue persistence 與 file lock
│   ├── queue_executor.py         # single-worker state machine
│   ├── queue_backend.py          # train／evaluate／select dispatch
│   ├── queue_workflow.py         # queue graph construction
│   ├── final_workflow.py         # PWL final multi-seed graph
│   ├── recovery_workflow.py      # staged recovery graph
│   ├── lr_sweep_workflow.py      # x1／x2／x4 LR sweep graph
│   ├── final_backend.py          # gate、selection、export backend
│   ├── final_selection.py        # deterministic final policy
│   ├── pwl_validation.py         # streaming PWL diagnostics
│   ├── pwl_experiment.py         # PWL report workflow
│   └── cli.py                    # production CLI
│
├── configs/
│   ├── variants/
│   │   ├── float-pwl-final.yaml
│   │   └── bittrue-pwl-final.yaml
│   ├── evaluation/
│   │   └── coco2017.yaml
│   └── training/
│       ├── pilot-1e-5.yaml
│       ├── pilot-5e-6.yaml
│       ├── phase-a.yaml
│       ├── phase-b.yaml
│       ├── phase-c.yaml
│       ├── lr-block-x1.yaml
│       ├── lr-block-x2.yaml
│       ├── lr-block-x4.yaml
│       ├── recovery-block.yaml
│       ├── recovery-neck.yaml
│       ├── recovery-backbone.yaml
│       └── recovery-full.yaml
│
├── data/
│   └── coco2017.yaml              # COCO80 dataset contract；路徑由 configure 寫入
├── weights/
│   └── v1-br-best.pt              # 原始 YOLO26m hardware-attention parent
├── artifacts/
│   └── runs/
│       └── s0-phase-b-bittrue/
│           ├── manifest.json
│           ├── checkpoints/evaluated-variant.pt
│           ├── metrics/queue-result.json
│           └── profiles/analytical.json
│
├── tests/
│   ├── conftest.py
│   ├── test_checkpoint_contract.py
│   ├── test_cli_retry.py
│   ├── test_early_stop_completion.py
│   ├── test_lr_sweep.py
│   ├── test_optimizer_scope.py
│   ├── test_pwl_final.py
│   ├── test_recovery.py
│   ├── test_scopes.py
│   └── test_workflow.py
│
└── scripts/
    └── smoke_cpu.py
```

以下目錄會在執行後產生，不屬於原始交付內容：

```text
artifacts/final-reproduction-queue/
├── queue.json
├── events.jsonl
├── generated/
└── worker.lock

artifacts/runs/<run-id>/
├── manifest.json
├── variant.yaml
├── training.yaml
├── training-complete.json
├── checkpoints/
├── metrics/
├── profiles/
├── exports/
├── logs/
└── ultralytics/
    └── weights/
```

## 整體架構

```text
外部使用者／自動化程式
        │
        ▼
run.py  ─────────────────────────────────────────────── 對外 Interface
  │
  ├── configure ──► data/coco2017.yaml + configs/*     資料與執行設定
  ├── doctor ─────► 檔案／資料路徑／checkpoint hash    Fail-closed 檢查
  ├── check ──────► 正式模型架構與 PWL contract
  ├── predict ────► YOLO + pwl-final-best.pt
  │
  └── train ──────► training_recipe.py
                       │
                       ▼
                yolo_attention.cli
                       │
                       ▼
      Workflow ─► QueueState ─► QueueStore ─► QueueExecutor
                                                  │
                    ┌─────────────────────────────┼─────────────────────┐
                    ▼                             ▼                     ▼
             Training Adapter             Evaluation Adapter       Selection／Gate
                    │                             │                     │
                    ▼                             ▼                     ▼
       HardwareAttentionTrainer      Float／Bit-True metrics     rollback／winner
                    │                             │                     │
                    └──────────────► artifacts/runs/ ◄─────────────────┘
```

### 主要 Module 與 Interface

| Module | 對外 Interface | 隱藏的 implementation／invariant |
| --- | --- | --- |
| `run.py` | 7 個使用者命令 | 路徑定位、hash、模型 contract、queue safety |
| `VariantConfig` | 兩份 variant YAML | Binary basis、bias、scale、normalization 組合與 validation |
| `HardwareFriendlyAttention` | `forward(X) -> Y` | QKV 拆分、Binary Q/K、bias、PWL、V aggregation |
| `TrainingRecipe` | training YAML | scope、LR、warmup、scheduler、determinism invariants |
| `QueueState`／`QueueStore` | serializable queue | dependency graph、atomic save、single-worker lock |
| `QueueExecutor` | dry-run／execute | readiness、result validation、failure state |
| `HardwareAttentionTrainer` | Ultralytics trainer seam | Float-PWL training、parameter scope、BN buffer lock |
| evaluation／selection modules | standardized `QueueResult` | Bit-True materialization、phase gate、rollback、winner policy |

這些都是深 Module：外部使用者只需要學會 `run.py` 的小型 Interface，複雜的模型轉換、
queue、checkpoint lineage 與評估規則保留在 implementation 內，讓訓練與測試走同一個 seam。

## 對接送入資料

### 1. Training dataset contract

本 workspace 固定為 COCO80 Detect。接入資料應符合 Ultralytics COCO2017 layout：

```text
/path/to/coco2017/
├── images/
│   ├── train2017/
│   └── val2017/
├── labels/
│   ├── train2017/
│   └── val2017/
└── annotations/                   # 若來源仍保留 COCO JSON，可一併保存
```

`configure` 目前 fail-closed 檢查 `images/train2017/` 與 `images/val2017/`；
labels／annotations 是否完整會再由 Ultralytics loader 驗證。

```bash
python run.py configure \
  --data-root /path/to/coco2017 \
  --device 0 \
  --batch 16 \
  --workers 8
```

這個命令會：

1. 將資料絕對路徑寫入 `data/coco2017.yaml`。
2. 將 `device`、`batch`、`workers` 同步到 12 份 training recipes。
3. 更新 `configs/evaluation/coco2017.yaml`。
4. 若 reproduction queue 已初始化則拒絕改寫，避免 provenance 漂移。

這不是 BBAT5 二類 Detect 或 Pose workspace。若要改成自訂 class 數、BBAT5 或其他 split，
必須建立新的 dataset／model contract，不能只把 YAML 路徑換掉。

### 2. Inference input contract

`predict` 的 `source` 直接交給 Ultralytics，可接：

| 輸入 | 範例 |
| --- | --- |
| 單張圖片 | `python run.py predict image.jpg --device 0` |
| 圖片目錄 | `python run.py predict images/ --device 0 --save` |
| 影片 | `python run.py predict video.mp4 --device 0 --save` |
| camera／stream | 使用 Ultralytics 支援的 source 字串 |

輸入會 resize／letterbox 到預設 `imgsz=640`，預設 confidence threshold 是 `0.25`。
`--save` 的輸出位置依 Ultralytics prediction run 規則建立。

### 3. Checkpoint contract

| 檔案 | 角色 | SHA-256 |
| --- | --- | --- |
| `pwl-final-best.pt` | 正式推論／交付 winner | `c989aeed…de0123` |
| `weights/v1-br-best.pt` | 原始 v1-br training parent | `9e2ca0d9…9c3f` |
| `artifacts/runs/s0-phase-b-bittrue/checkpoints/evaluated-variant.pt` | LR sweep／recovery immutable parent | `2f5a18e3…d8ac` |

三個 checkpoint 的角色不同，不可當成重複檔案刪除。`doctor` 會驗證完整 SHA-256。

### 4. Config contract

| Config | 使用時機 | 關鍵差異 |
| --- | --- | --- |
| `float-pwl-final.yaml` | training | Binary sign 使用 clipped STE；Float-PWL 可微分 |
| `bittrue-pwl-final.yaml` | gate／evaluation／delivery | deterministic sign；Q8.8/UQ1.15 Bit-True numerator |
| `configs/training/*.yaml` | 各階段 training | scope、LR、epochs、patience、optimizer |
| `configs/evaluation/coco2017.yaml` | formal val | COCO2017 val、imgsz、batch、device |

## 安裝與第一次啟動

```bash
cd final
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest ruff
python -m pip install -e . --no-deps
```

正式結果環境為 Python 3.12.3、PyTorch 2.11.0+cu128、Ultralytics 8.4.90、
CUDA 12.8、NVIDIA RTX 5090。其他 GPU／CUDA 組合需安裝相容的 PyTorch wheel。

## 驗證 workspace

設定資料後：

```bash
python run.py doctor
python run.py check
```

不需要資料集即可跑 functional tests：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
ruff check --no-cache .
```

CPU model smoke：

```bash
python scripts/smoke_cpu.py
```

## 推論

```bash
python run.py predict path/to/images --device 0 --imgsz 640 --conf 0.25 --save
```

## 訓練與續跑

先預覽，不啟動 GPU：

```bash
python run.py recipe
python run.py train
```

明確建立或續跑 LR sweep＋staged recovery：

```bash
python run.py train --execute
python run.py status
```

訓練使用 Float-PWL surrogate；每個候選會重新 materialize 成 Bit-True PWL，在完整
COCO2017 val 5,000 images 上驗證。child 相對直接 parent 下降超過 `0.001`
mAP50-95 就 rollback。`--execute` 會啟動長時間 GPU 工作。

## 創新摘要

本 workspace 的主要創新集中在：

1. deterministic Binary Q/K 與 XNOR＋popcount reference；
2. identity＋Walsh–Hadamard 雙 binary basis；
3. per-head fixed power-of-two score coefficients；
4. decomposed 2D relative-position bias；
5. Q8.8 score、UQ1.15 endpoints、20-segment Bit-True PWL exponential numerator；
6. Float-PWL training／Bit-True gate 的雙路徑設計；
7. fail-closed 兩-site conversion、Bit-True phase gate 與 rollback；
8. immutable queue、checkpoint hash 與 provenance contract。

完整公式、實作對照、創新價值與限制統一寫在
[BINARY_QK_PWL_ALGORITHM.md](BINARY_QK_PWL_ALGORITHM.md)。

## 正式結果與限制

- Formal winner mAP50-95：`0.5067369`。
- Best-observed seed-0 mAP50-95：`0.5069387`。
- Best-observed 相對 audited parent：`+0.0001944`，未達 `+0.001` 正式門檻。
- 只完成 seed 0，沒有 three-seed mean/std。
- PWL numerator 是 Bit-True；denominator 仍是 exact floating-point reference。
- 尚未完成 P/V/projection 全整數化、HLS、上板、功耗與實體 latency 測量。
