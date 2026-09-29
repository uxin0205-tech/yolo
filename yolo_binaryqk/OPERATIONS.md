# YOLO11 BinaryAttention 操作與維護文件

本文件是 `yolo_binaryqk/` 的主要操作手冊。它說明實驗產物的位置、Python
模組責任、常用維護指令及權重保存規則。實驗設計本身請參考
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)，結果解讀請參考
[`artifacts/reports/binary_attention_final_report.md`](artifacts/reports/binary_attention_final_report.md)。

## 1. 執行環境與固定路徑

所有指令都應從本目錄執行：

```bash
cd /home/hplab/uxin/yolo11/yolo_binaryqk
```

主要依賴與唯讀來源如下：

| 項目 | 路徑 | 維護規則 |
|---|---|---|
| Python 環境 | `../.venv/bin/python` | 使用既有環境，不在實驗目錄建立第二套環境 |
| FP 初始權重 | `../original/weight/yolo11m.pt` | 來源資產，不可覆寫或刪除 |
| COCO2017 | `../coco2017/` | 完整 train2017/val2017 資料 |
| Dataset YAML | `data/coco-full.yaml` | 正式訓練與驗證設定 |
| Train manifest | `data/coco_full.txt`、`data/coco_full.json` | 固定資料順序與 hash，不可任意重建後混用舊結果 |

### 固定 seed

本批 26 個 canonical experiments 全部使用 `seed=0`；22 個 T/N 訓練 runs 另有
`deterministic=True`。永久紀錄位於 [`EXPERIMENT_SEED.md`](EXPERIMENT_SEED.md)，
機器可讀版本位於 `artifacts/reports/experiment_seed.json`。未來若執行不同 seed，
應建立新的 run UUID 並分開彙整，不可覆寫這批 seed 0 artifacts。

## 2. 專案目錄

```text
yolo_binaryqk/
├── README.md                         # 專案入口與目前狀態
├── OPERATIONS.md                     # 本操作文件
├── EXPERIMENT_SEED.md                # 固定 seed 與資料 manifest 可重現紀錄
├── IMPLEMENTATION_PLAN.md            # 實驗矩陣與研究契約
├── binary_attention/                 # 模型、訓練、artifact 與 audit 程式
│   ├── attention/                    # attention 實作與 fake quantizers
│   ├── variants/                     # variant 定義、繼承及 config hash
│   └── verification/                 # strict checkpoint 格式與重載驗證
├── configs/                          # YOLO11m BinaryAttention YAML template
├── data/                             # dataset YAML、manifest 與資料說明
├── tests/                            # 單元測試、模型建構與計畫契約測試
├── services/                         # systemd user service 範本
├── artifacts/
│   ├── runs/<variant>/<stage>/<uuid>/# 每次執行的原始、不可混合 artifact
│   ├── reports/                      # 彙整表、圖、selection 與完整報告
│   └── final_weights/                # 26 組 canonical 正式權重封存
├── logs/formal-plan/                 # runner、report 與 final audit 紀錄
├── run_formal_plan.sh                # 可續跑的完整矩陣排程器
└── audit_after_formal.sh             # 等待完成後重建報告並 audit
```

`.cache/`、`.pytest_cache/` 與 `__pycache__/` 是可重建快取，不是研究產物。
`artifacts/runs/`、`artifacts/reports/`、`artifacts/final_weights/` 及
`logs/formal-plan/` 則包含研究證據，不可使用廣域清理指令。

## 3. Python 模組責任

| 模組 | 責任 | 修改時需要一起檢查 |
|---|---|---|
| `binary_attention/attention/base.py` | FP、sign、scaled-sign、三種 dual-basis、N4 magnitude attention 與 C2PSA wrapper | 模型建構、variant contract、strict reload 測試 |
| `binary_attention/attention/quantizers.py` | clipped STE、P8/V8、INT8/INT4 fake quantization | quantization contract 與數值測試 |
| `binary_attention/variants/definitions.py` | 所有 variant 的唯一來源、繼承規則與 config hash | `run_formal_plan.sh`、audit 預期矩陣、報告名稱 |
| `binary_attention/model.py` | 將自訂 attention 註冊到 Ultralytics parser，建立 student 並載入 FP source | Ultralytics 版本相容性與 construction tests |
| `binary_attention/trainer.py` | attention-only freeze、effective batch accumulation、teacher/KD 接線與 trainer | frozen delta、EMA 與 batch contract |
| `binary_attention/kd.py` | positional、feature、attention KD loss | T6 materialization 與 KD target diagnostics |
| `binary_attention/training_profiles.py` | 10-epoch 正式超參數與 stage overrides | `training_args.json` 及 plan-contract tests |
| `binary_attention/workflow.py` | 建立 run、執行 validation、完成 artifact、diagnostics 與 strict checkpoint | artifact schema、audit 與 report reader |
| `binary_attention/verification/checkpoint.py` | checkpoint manifest、SHA-256、儲存與 schema-3 strict reload | 權重格式或 model key 改動 |
| `binary_attention/formal_audit.py` | 對 26 個 canonical runs、選擇結果與封存權重做 fail-closed audit | 新增 variant、artifact 欄位或 tolerance 時 |
| `binary_attention/report.py` | 從 artifacts 重建 CSV/JSON、圖表與完整 Markdown 報告 | metrics 欄位、分組或 selection 邏輯 |
| `binary_attention/area_metrics.py` | 用 canonical weights 與官方 COCOeval 補算 mAP、APs、APm、APl，支援中斷續跑 | annotation、checkpoint hash 與報告 metrics |
| `binary_attention/weight_archive.py` | 封存每個 canonical strict weight，複製重建 metadata 並驗證載入 | audit 通過後才可執行 |
| `binary_attention/checkpoint_cleanup.py` | 驗證封存後，精確清除 Ultralytics `best.pt`/`last.pt` | 這是破壞性操作，必須先 dry-run |
| `binary_attention/ema_repair.py` | 修復舊式 Ultralytics EMA checkpoint metadata | 僅供舊 artifact 修復，不是一般載入流程 |
| `binary_attention/data.py` | COCO layout、YOLO label conversion、manifest 與 dataset YAML | 資料 hash 和資料集版本 |
| `binary_attention/config.py` | 將 variant 寫入模型 YAML | variant/model schema |
| `binary_attention/cli.py` | 對外 CLI：verify、run、data、report、audit | 新操作應先考慮是否納入 CLI |

維護原則：variant 行為應先寫入 `definitions.py`，不要只在 shell runner 或報告中
硬編碼；artifact 格式變更時，必須同步更新 workflow、audit、report 與測試。

## 4. 實驗與權重命名

| 階段 | Variants | 用途 |
|---|---|---|
| E | E0、E1-S、E1、E2-DUAL | 零訓練 validation |
| T0–T5 | T0、T1、T2、T3、T4、T5 | FP/sign/scaled-sign 與 dual-basis 基礎消融 |
| T6 | T6-O、T6-F、T6-A、T6-O/F、T6-O/A、T6-F/A、T6 | positional/feature/attention KD 選擇 |
| T7 | T7-D、T7-R、T7-P、T7-V、T7-PV | relative-position bias 與 P/V 8-bit fake quantization |
| N4 | N4-FP、N4-I8、N4-I4、N4-PV | 非 KD magnitude side-channel 消融 |

檔案系統名稱會把 `/` 轉為 `-`，因此 `T6-O/F` 的目錄是 `T6-O-F/`，
`T6-F/A` 是 `T6-F-A/`。正式 variant ID 仍保存在 `resolved_config.json`。

## 5. Artifact 契約

每個 canonical run 位於：

```text
artifacts/runs/<artifact-name>/<validation|full>/<uuid>/
```

至少應保留下列研究證據：

- `status.json`：完成狀態、stage、來源權重及有效性。
- `resolved_config.json`：完整 variant 定義與 config hash。
- `training_args.json`、`environment.json`：訓練與執行環境。
- `model.yaml`、`architecture_manifest.json`：模型重建資料。
- `validation_metrics.json`、`training_curves.csv`：正式結果。
- `attention_diagnostics.json`、`parameter_delta_diagnostics.json`：attention 與 freeze 證據。
- `checkpoint_manifest.json`：strict checkpoint 的 identity 與 schema。
- `checkpoints/strict-validation.pt` 或 `checkpoints/strict-last.pt`：正式權重。
- `experiment_report.md`、`formal.log`：單一 run 報告與紀錄。

`ultralytics/train/weights/best.pt` 與 `last.pt` 是 optimizer/resume 用檔案，不是
canonical 權重。本批 58 個檔案已在封存後刪除；刪除紀錄位於
`artifacts/final_weights/checkpoint_cleanup.json`。

## 6. 日常驗證

修改程式後先執行：

```bash
../.venv/bin/python -m pytest -q
../.venv/bin/python -m binary_attention.cli verify --variant T2
../.venv/bin/python -m binary_attention.cli audit
git diff --check
```

正式 audit 成功條件包括：26/26 canonical variants、26/26 封存權重、無
errors，且 selection、config hash、metrics、freeze delta 與權重 SHA-256 一致。

需要更新保存的 audit JSON 時，只把 stdout 寫入檔案，避免將環境 warning 混入 JSON：

```bash
../.venv/bin/python -m binary_attention.cli audit \
  > logs/formal-plan/final-audit.json
```

補算或續跑 COCO 尺寸分組 AP：

```bash
../.venv/bin/python -m binary_attention.area_metrics --device 0 --batch 16
../.venv/bin/python -m binary_attention.cli report
```

每組完成後會保存 `artifacts/area_metrics/<variant>/coco_area_metrics.json`，並同步更新
canonical run 與 `final_weights` 的 `validation_metrics.json`。總表位於
`artifacts/reports/coco_area_metrics.csv/json`；預設不保留體積較大的 predictions JSON。

## 7. 重建報告

報告完全由 artifacts 產生，不應手動修改 generated report：

```bash
../.venv/bin/python -m binary_attention.cli report
```

主要輸出：

| 輸出 | 用途 |
|---|---|
| `artifacts/reports/binary_attention_final_report.md` | 完整研究報告 |
| `binary_attention_summary.csv/json` | 26 組結果的機器可讀表格 |
| `experiment_seed.json` | 26 組實驗 seed 與 COCO manifest 的機器可讀紀錄 |
| `paper_qat_selection.json` | T6、T7、N4 與 final-family 選擇依據 |
| `run_reports_index.md` | canonical run 路徑索引 |
| `figures/` | loss 與各階段比較圖 |

若更新 audit 或權重 manifest，應再次執行 report，確保報告附錄同步。

## 8. Canonical 權重封存與載入

`artifacts/final_weights/` 是長期保存入口。每個 variant 目錄包含：

```text
weight.pt
model.yaml
resolved_config.json
validation_metrics.json
checkpoint_manifest.json
status.json
```

封存與逐一載入驗證：

```bash
../.venv/bin/python -m binary_attention.weight_archive --verify-load
```

該命令要求 final audit 已通過，並更新 `manifest.json`、`manifest.csv` 與
`README.md`。目前權重多數以 hardlink 連到 run 內的 strict checkpoint；刪除其中
一個檔名不會立即刪除另一個，但**不可原地修改任何一端**，因為 hardlink 共享同一
inode。需要搬移到其他檔案系統或備份媒體時，應複製整個 `final_weights/` 並保留
`manifest.json`。

最小載入範例：

```python
import json
from pathlib import Path

import torch

from binary_attention.model import build_student
from binary_attention.variants.definitions import variant_from_resolved_config

bundle = Path("artifacts/final_weights/T7-D")
resolved = json.loads((bundle / "resolved_config.json").read_text())
variant = variant_from_resolved_config(resolved)
model = build_student(bundle / "model.yaml", variant)
payload = torch.load(bundle / "weight.pt", map_location="cpu", weights_only=True)
model.load_state_dict(payload["state_dict"], strict=True)
model.eval()
```

## 9. Checkpoint 清理

只有在 26 組 archive 完整、載入驗證通過且 final audit 為 `ok=true` 時才可清理。

先做 dry-run：

```bash
../.venv/bin/python -m binary_attention.checkpoint_cleanup
```

檢查輸出的 `targets` 僅包含：

```text
artifacts/runs/**/ultralytics/train/weights/best.pt
artifacts/runs/**/ultralytics/train/weights/last.pt
```

確認後才執行：

```bash
../.venv/bin/python -m binary_attention.checkpoint_cleanup --execute
```

清理會失去 optimizer、scheduler 與 resume 狀態，只有重新訓練才能重建；不會刪除
`strict-last.pt`、`strict-validation.pt` 或 `final_weights/*/weight.pt`。不要自行用
`find ... -delete`、`rm -r artifacts/runs` 或其他廣域指令取代此工具。

## 10. 完整矩陣重新執行

只有需要重做實驗時才啟動：

```bash
./run_formal_plan.sh
```

Runner 使用 `logs/formal-plan/.formal-plan.lock` 避免重複排程，並只重用已完成、
config hash 相符且 strict checkpoint 存在的 run。執行順序、動態 selection 與參數
請以 `IMPLEMENTATION_PLAN.md` 和 script 內容為準。

`services/` 內是特定本機路徑的 systemd user service 範本。搬移 workspace 後必須
先修改 `WorkingDirectory` 與 `ExecStart`，不要直接安裝舊的絕對路徑。一般維護、
報告重建與 audit 不需要啟動 service。

## 11. 修改、新增 variant 的檢查清單

1. 在 `variants/definitions.py` 定義行為、繼承及 quantization contract。
2. 若有新 attention 計算，更新 `attention/base.py` 或 `quantizers.py`。
3. 確認 `model.py` 能重建模型，且 checkpoint manifest 能描述新架構。
4. 更新 runner 的順序與 selection，但不要只在 shell 中定義 variant。
5. 更新 `formal_audit.py` 的預期矩陣與 artifact 契約。
6. 更新 `report.py` 的名稱、比較組與結論生成方式。
7. 增加或修改 tests，至少涵蓋數值、模型建構與 plan contract。
8. 完成 run 後依序執行 audit、report、weight archive，再考慮 checkpoint cleanup。

## 12. 不可刪除與可重建項目

不可刪除或覆寫：

- `../original/weight/yolo11m.pt`
- `artifacts/final_weights/*/weight.pt` 與 `manifest.json`
- canonical run 的 `strict-last.pt`／`strict-validation.pt`
- canonical run 的 JSON/CSV diagnostics 與 metrics
- `logs/formal-plan/final-audit.json`、完整報告與 selection JSON

可重建，但刪除前仍應確認沒有除錯需求：

- `.cache/`、`.pytest_cache/`、`__pycache__/`
- `artifacts/reports/figures/`（由 report 重建）
- Ultralytics `best.pt`／`last.pt`（只在不需要 resume 且已完成封存時）

大型權重與資料集不應提交到一般 Git history；若要同步，使用專案既定的大檔案或
外部儲存流程，並讓 manifest、報告與程式碼維持同一版本。
