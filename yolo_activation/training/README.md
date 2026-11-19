# SIPA-BCSP Activation 訓練架構

狀態：generic planner、Pareto placement 與 toy dry-run 已可執行；Full35 accepted baseline 的 production
adapter、190-site manifest、完整 validation 與 checkpoint recovery 已接通。固定數值與執行方式見
[Full35 全量正式實驗配置](full35/README.md)。

`SIPA-BCSP` 是內部工作名稱，不是新穎性或論文貢獻宣告：

- **SIPA — Symmetry-preserving Integral Polynomial Activation**：利用 SiLU 的差分對稱性，直接
  建構非負半軸 even residual。
- **BCSP — Budget-Constrained Static Placement**：以精度損失與硬體成本向量做受限 Pareto 搜尋，
  只產生下一個必要實驗 stage，最後凍結靜態 profile ID。

學術主張邊界、stage-matched SiLU controls、claim-critical ablations、統計方法、完整 job budget 與
上板驗證順序見[學術主張與完整實驗計畫](../docs/research/sipa-bcsp-experiment-plan.md)。

## 1. SIPA 如何使用對稱性

核心式為：

\[
A(x)=\frac{x}{2}+H(|x|),\qquad A(x)-A(-x)=x.
\]

這不是在正負半軸各保存一套 activation。兩側共用同一個 \(H(|x|)\)，部署資料路徑為：

```text
                    ┌─ clip(|x|, T) ─ powers ─ fixed coefficients ─┐
x ──┬─ abs ─────────┤                                               ├─ add ─ A(x)
    └─ arithmetic /2 ───────────────────────────────────────────────┘
```

`poly_shift` 使用 `a=9/2,T=8`：

\[
H(u)=\frac9{32}u^2-\frac{15}{256}u^3
     +\frac{11}{2048}u^4-\frac3{16384}u^5,\quad 0\le u\le8.
\]

特殊優化如下：

1. `abs` 後只計算一條 shared residual，解析上保有 `A(x)-A(-x)=x`。
2. `T=8` 是 power-of-two threshold；固定係數皆為 dyadic/APoT，可由 shift/add 組成。
3. 仍誠實保留四次資料相乘建立 powers，不把它們誤報成 shift。
4. 先 clip 再算 powers，避免 exact-ReLU tail 上無用的高次項溢位。
5. `q(1)=1/2`、`q'(1)=0` 與積分條件使函數、一階、二階導數接到 finite ReLU tail。
6. Q16.10 emulator 以 complementary rounding 讓離散 symmetry 與兩側 tail 都達到 0 LSB 誤差。

## 2. BCSP 特殊搜尋優化

BCSP 不用單一人工加權分數把 AP、AP_S 與硬體 proxy 混成一個數字，而是保存非支配 Pareto
frontier。每個資料集有自己的 observation 與 frontier：

```text
contract gate
    ↓
SiLU baseline reproduction
    ↓
full-train activation profiling
    ↓
poly_shift pre-recovery region zero-shot sensitivity
    ↓
uniform zero-shot: Hardswish / ReLU / Shift / Quality
    ↓
equal-budget recovery: SiLU / Hardswish / Shift / Quality
    ↓
poly_shift region one-at-a-time recovery
    ↓
Pareto beam expansion（最多 3 個 changed regions、3 種 kernels）
    ↓
accuracy extreme + hardware extreme（最多 2 finalists）
    ↓
seed 1 full recovery → 資源允許才補 seed 2
```

節省實驗的機制：

- planner 每次只回傳**下一個必要 stage**，不預先展開 `K^L` 全部組合。
- ReLU 固定停在 cheap control，不進 recovery。
- Region search 預設只讓 `poly_shift` 進場；`poly_quality` 已在 uniform recovery 提供 accuracy
  ablation，不自動倍增逐區域實驗。
- Beam 只從當前非支配 policy 擴一個 region，且受 `beam_width`、changed-region 與 kernel budget
  共同限制。
- 若沒有凍結 accuracy budget，planner 只回報 Pareto，不擅自發明「可接受 AP loss」。
- 有真實 target latency 且所有比較列都完整時，latency 才加入 objective；否則只使用明確標記的
  operator proxy。

## 3. Manifest 安全替換

只替換 manifest 中原本就是 `nn.SiLU` 且 `eligible=true` 的 module path。`inspect()` 只盤點
`nn.SiLU`，因此 softmax、output sigmoid、raw Conv 與其他 activation 不會被候選名稱誤傷。

安全順序：

1. 對交付模型執行 `named_modules()`，依 region rules 產生 `reviewed=false` draft。
2. 未匹配、重複匹配、空 manifest 都會阻擋 planning。
3. 人工核對 model source/hash、module path、region 與 eligibility 後才呼叫 `approve()`。
4. `apply()` 先對所有 path 做一次 preflight；只要有 path 消失或不再是 SiLU，完全不變更模型。
5. 預設 `deepcopy` 後替換，原始 baseline model 保持乾淨；每個 experiment 必須重新載入同一
   checkpoint，不得沿用上一個候選的 mutated model。

## 4. 深模組介面

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

外部 seam 只有：

- `inspect(model, model_id, region_rules)`
- `apply(model, manifest, policy)`
- `plan(manifest, observations)`
- `audit_delivery(mapping)`

硬體 cost、Pareto dominance、beam expansion、路徑設定與 rounding 留在 module implementation。
generic planner 仍不假設任意未知 baseline；目前第一個真實 seam 是
`Full35ActivationExperiment`，固定載入 `yolo_combine/final/full35` 的 source bundle、checkpoint 與
canonical data，並共用同一個 production loader 進行測試、validation 與 recovery。

## 5. 目錄契約

```text
training/
├── README.md
├── configs/
│   ├── pipeline.yaml
│   └── region-rules.example.yaml
├── contracts/
│   ├── activation-manifest.example.yaml
│   ├── delivery.example.yaml
│   └── observations.example.yaml
└── schemas/
    ├── activation-manifest.schema.json
    ├── delivery.schema.json
    ├── observations.schema.json
    └── pipeline.schema.json
```

`delivery.example.yaml` 故意包含 placeholders，audit 必須失敗；它不能被誤當真實 baseline。

## 6. 資料契約

- Canonical BBAT5 v1 Detect 唯一入口：
  `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml`，`nc=2`。
- COCO80 Detect 唯一入口：`/home/uxin/yolo/coco2017.yaml`，`nc=80`。
- planner 對兩者分別產生 experiment ID、observations 與 frontier，不平均 raw AP。
- 不建立 split、不抽樣、不改 labels、不將 COCO head 指向 BBAT5，也不把 Runtime Dataset View
  當成新版本。

## 7. 指令

```bash
# 不碰資料集／checkpoint 的完整 wiring dry-run
/home/uxin/yolo/.venv/bin/python scripts/activation_training.py toy-dry-run

# 交付資料夾到位後先稽核；有 placeholder、錯誤 YAML、hash 或不存在路徑就 exit 2
/home/uxin/yolo/.venv/bin/python scripts/activation_training.py \
  audit-delivery --delivery /delivery/delivery.yaml

# 只編譯下一個實驗 stage；未 reviewed manifest 會被阻擋
/home/uxin/yolo/.venv/bin/python scripts/activation_training.py \
  compile-plan \
  --manifest /delivery/activation-manifest.yaml \
  --observations /run/observations.yaml
```

## 8. Full35 已接手與仍缺項目

Full35 的 source/config、checkpoint hashes、套件版本、baseline metrics、完整 recipe 與人工核准 manifest
均已固化。誤啟動的 1.0× LR SiLU recovery 已停止且不列正式結果；目前先執行完整 profiling 與
pre-recovery region sensitivity。後續 short recovery 使用相同 10-epoch、0.1× J3 LR；from-source
或 QAT 只給最後 1–2 個 policy。

仍缺 target FPGA/ASIC 的 precision、compiler、warm-up／repetition／同步、power 與 resource protocol；
這些到位前不做硬體加速宣稱。其他 baseline 仍必須各自完成 delivery audit，不能套用 Full35 契約。
