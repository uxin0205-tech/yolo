# SIPA–BCSP 訓練架構

這個目錄保存 generic planner、activation manifest、模型替換、Pareto placement 支援與 Full35 production
adapter。詳細 epoch、batch、學習率與正式 queue 請看 [Full35 實驗配置](full35/README.md)。

## 目前完成狀態

| 部分 | 狀態 | 實際成果 |
| --- | --- | --- |
| SIPA 數學與 activation 實作 | 完成 | `poly_quality`、`poly_shift`、fixed-point 與 ONNX／介面驗證 |
| Full35 production adapter | 完成 | accepted checkpoint、190-site manifest、COCO／BBAT5 validation 與 recovery |
| BCSP 支援架構 | 完成 | 11 regions、policy schema、bounded queue、dependency 與 gate propagation |
| BCSP mixed-policy 搜尋 | 未完成 | `poly_shift` prerequisite 未過，14 個 region／mixed jobs 依規則 blocked |
| Activation finalist | 部分完成 | SiLU control 完成；qSiLU 依使用者要求人工停止 |
| PTQ／QAT／硬體實測 | 未執行 | 交由後續量化與部署工作列 |

因此不能把 BCSP 說成已找到最佳 activation placement；本專案完成的是可重現的搜尋支援架構與正確的
停止機制。

## SIPA 是什麼

SIPA 全名是 Symmetry-preserving Integral Polynomial Activation。它先固定下列共同結構：

```text
activation 輸出 = 輸入的一半 + 只與輸入絕對值有關的修正量
A(x) = x / 2 + H(|x|)
```

正、負輸入共用同一個 `H(|x|)`，所以修正量相減後會抵消：

```text
A(x) - A(-x) = x
```

這不是分別儲存正半邊與負半邊。部署資料路徑可理解為：

```text
輸入 x
  ├─ 取絕對值 → 截斷範圍 → 計算多項式 H(|x|) ─┐
  └─ 算術右移一位，得到 x / 2 ───────────────┤
                                                   └─ 相加得到輸出
```

`poly_shift` 使用門檻 8，中央修正量為：

```text
H(u) = (9/32)u² - (15/256)u³ + (11/2048)u⁴ - (3/16384)u⁵
```

所有常數都是 2 的冪次分母，可用 shift／add 表示；但仍要形成 `u²` 到 `u⁵`，所以不能誤稱為零乘法器。
完整推導請看[可讀版數學文件](../docs/research/full35-activation-mathematical-derivations.md)。

## BCSP 是什麼

BCSP 全名是 Budget-Constrained Static Placement。它不是 activation，而是安排「哪些區域換 activation」
的有限預算搜尋架構。規劃順序為：

```text
資料與 checkpoint 契約檢查
→ 重現 SiLU baseline
→ 全訓練集 activation profiling
→ 每個 region 的 zero-shot sensitivity
→ uniform activation 的相同預算 recovery
→ prerequisite 通過後才做逐 region recovery
→ 再做有限寬度的 mixed-policy 搜尋
→ 最多選兩個 finalists
```

節省實驗成本的機制：

- planner 每次只回傳下一個必要 stage，不預先展開所有組合。
- ReLU 只作最低成本診斷，不進昂貴 recovery。
- region search 原本只讓 `poly_shift` 進場；`poly_quality` 用於 uniform accuracy 對照。
- 只有 prerequisite 通過，後續 region／mixed jobs 才能執行。
- latency 只有在指定 target 且資料完整時才能成為 objective；目前只保存明確標記的 operator proxy。

本輪 `poly_shift` 的 uniform 10-epoch gate 失敗，因此 14 個後續 jobs 正確封鎖。這是預註冊停止規則，
不是 queue 故障。

## Manifest 安全替換

系統只替換 manifest 中原本為 `nn.SiLU` 且 `eligible=true` 的 module path：

1. `inspect()` 依 model source、hash 與 region rules 產生 draft。
2. 未匹配、重複匹配或空 manifest 會阻擋 planning。
3. 人工核對 path、region 與 eligibility 後才可 `approve()`。
4. `apply()` 先檢查所有 path；任一路徑失效就完全不改模型。
5. 每個實驗從同一 accepted checkpoint 乾淨載入，不沿用前一候選的 mutable model 或 optimizer。

## 程式介面

```python
from activation_lab.training import (
    TrainingArchitecture,
    load_manifest,
    load_observations,
)

architecture = TrainingArchitecture.from_yaml("training/configs/pipeline.yaml")
manifest = load_manifest("/delivery/activation-manifest.yaml")
observations = load_observations("/run/observations.yaml")
next_plan = architecture.plan(manifest, observations)
```

對外介面只有：

- `inspect(model, model_id, region_rules)`
- `apply(model, manifest, policy)`
- `plan(manifest, observations)`
- `audit_delivery(mapping)`

硬體 cost、Pareto dominance、beam expansion、路徑與 rounding 規則留在 module implementation。

## 目錄契約

```text
training/
├── README.md
├── configs/                 # generic pipeline 與 region 範例
├── contracts/               # manifest／delivery／observation 範例
├── schemas/                 # machine-readable JSON schema
└── full35/                  # 正式 Full35 recipe、manifest、queue 與結果狀態
```

`delivery.example.yaml` 故意保留 placeholders，audit 必須失敗；不能把它當成正式 baseline。

## 資料契約

- Canonical BBAT5 v1：`/home/uxin/yolo/original/pose/derived/bbat5-v1/`。
- COCO80 Detect：`/home/uxin/yolo/coco2017.yaml`。
- 不建立新 split、不抽樣、不改 labels，也不把 BBAT5 二類 YAML 接到 COCO80 head。
- COCO 與 BBAT5 分別產生指標，再各自對 accepted SiLU baseline 計算差值。

## 安全指令

```bash
# 不碰正式資料與 checkpoint 的完整 wiring dry-run
/home/uxin/yolo/.venv/bin/python scripts/activation_training.py toy-dry-run

# 稽核 delivery；placeholder、錯誤 YAML、hash 或不存在路徑會 exit 2
/home/uxin/yolo/.venv/bin/python scripts/activation_training.py \
  audit-delivery --delivery /delivery/delivery.yaml

# 只編譯下一個必要實驗 stage；未核准 manifest 會被阻擋
/home/uxin/yolo/.venv/bin/python scripts/activation_training.py \
  compile-plan \
  --manifest /delivery/activation-manifest.yaml \
  --observations /run/observations.yaml
```

Activation-only 階段目前已停止。下一個正確入口是使用[已發布的 SiLU／qSiLU 權重](../release/weights/README.md)
進行相同規則的量化比較，而不是重新啟動 BCSP 或延長 activation epoch。
