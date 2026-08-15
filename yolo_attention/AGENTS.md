# YOLO26m Hardware-Friendly Attention 開發規範

本檔是此 repository 的程式維護契約。研究內容以 `plan.md` 為準；程式邊界以 `docs/architecture.md` 為準。修改程式、設定或實驗流程時，必須遵守以下規則。

## 1. 固定研究邊界

- 正式模型是 `yolo26m.pt`，資料是 COCO2017，先做演算法與 CPU reference，不自動啟動 GPU、HLS 或上板。
- 只重新設計兩個 Attention 的 Q/K score 與 normalization path：
  - `model.10.m.0.attn`
  - `model.22.m.0.1.attn`
- 正式實驗必須同時替換兩處。少一處就 fail closed，不得靜默退化成單點實驗。
- 保留 V、`PV + PE(V)`、output projection、PSABlock residual/FFN，以及外層 C2PSA/C3k2/Detect 結構。
- 不使用 KD 或 `Q⊙K`。量化是 A-FINAL 後的 optional phase，沒有明確指示時不得啟動。

## 2. 單一執行入口

所有使用者操作統一走：

~~~bash
python -m yolo_attention.cli <command>
~~~

安裝後也可用等價命令 `yolo26-attention <command>`。`scripts/main.py` 只是同一 CLI 的薄包裝。其他 `scripts/*.py` 不得含模型、演算法、實驗順序或路徑判斷；它們只能轉送參數到 package API。

主要命令：

~~~bash
python -m yolo_attention.cli workflow             # 顯示主線及 optional 量化
python -m yolo_attention.cli list                 # 顯示可直接取得的 registry runs
python -m yolo_attention.cli validate-config PATH
python -m yolo_attention.cli profile
python -m yolo_attention.cli smoke --model yolo26m.yaml --variant PATH --imgsz 64
python -m yolo_attention.cli train --variant PATH --training PATH --run-id ID
python -m yolo_attention.cli queue init|status|validate|next
python -m yolo_attention.cli queue run-next [--execute]
python -m yolo_attention.cli queue retry JOB_ID
python -m yolo_attention.cli queue rewind SELECTION_JOB_ID
python -m yolo_attention.cli queue extend-d1  # legacy 5-epoch queue migration only
~~~

`train` 預設只能 dry-run；只有使用者明確要求並加入 `--execute` 才能訓練。

queue 也遵守同一 gate：`run-next`／`run` 不加 `--execute` 只能預覽。不得另寫第二支 queue script 或直接在 shell 串接實驗；所有 job 都由同一 CLI、`QueueExecutor` 與 `ResearchQueueBackend` 依序執行。

## 3. 程式模組責任

~~~text
config.py          唯一演算法設定 seam；enum、驗證、YAML round-trip
projection.py      官方 fused QKV gather/interleave、Conv+BN split/fold
binary_basis.py    FP、Identity、Hadamard MDB、T5 score
relative_bias.py   none、dense 2D、decomposed 2D bias
normalization.py   scores → normalized P；所有 Softmax/Multimax 候選與 factory
bdcn.py            distance codebook、PoT projection、denominator 與 fused bucket-PV
attention.py       組合上述元件並保留官方 V/PE/projection path
integration.py     兩個 Attention 的 fail-closed 模型轉換、既有 checkpoint reconfiguration、freeze scope
workflow.py        主線步驟、依賴、選 winner 與 optional phase；不執行訓練
experiments.py     可直接索引的 immutable experiment registry
run_config.py      訓練資源設定，與演算法設定分離
training.py        Ultralytics trainer adapter；依官方/custom parent 決定正確的 load/convert 順序
runner.py          dry-run request 與明確 gated launch
artifacts.py       immutable run、manifest 與標準結果資料夾
profiling.py       演算法運算量 reference
queue_model.py     queue job/state/result schema 與 invariants
queue_store.py     atomic JSON、JSONL events、單 worker file lock
queue_policy.py    無 filesystem side effect 的 winner/gate 規則
queue_workflow.py  readiness、動態 graph、winner-derived YAML
queue_backend.py   train/evaluate/P0/select 的唯一 live dispatch
queue_executor.py  dry-run gate、狀態轉移、failure/retry 與 selection rewind/archive
cli.py             唯一公開 orchestration entry point
~~~

模組之間以 `VariantConfig`、`WorkflowStep`、`TrainingRecipe` 等小型資料介面連接。不要複製整個 Attention class 來做新方法，也不要在 CLI 寫演算法細節。

## 4. 新方法或程式轉寫流程

新增 basis、bias、normalization 或量化方法時，依序做：

1. 先在 `tests/` 寫會失敗的 reference、shape、normalization、gradient 或 integration test。
2. 在 `config.py` 新增 enum/欄位與不合法組合驗證。
3. 在方法所屬模組實作一個責任清楚的 `nn.Module`；輸入輸出 shape 不得任意改變。
4. 只在 factory 或 `HardwareFriendlyAttention` 的 composition seam 接線。
5. 若是正式實驗，更新 `workflow.py`；若可獨立執行，再加入 registry/variant YAML。
6. 更新 `plan.md`、`docs/architecture.md`、根 README 與受影響資料夾 README。
7. 跑焦點測試、全測試、Ruff、CLI smoke；記錄尚未執行的 GPU/COCO 工作，不得把 CPU smoke 說成精度驗證。

近似論文方法時，class/docstring 必須標示是 faithful reproduction、bit-true reference 或 project approximation。沒有公式或 codebook 來源時不得自行猜測。

## 5. 實驗主線

~~~text
B26-FP baseline → P0 equivalence
→ I/H/T5 各 10 ep screening → architecture winner
→ winner 做 Direct / Progressive 20–40 ep recovery → A0
→ N0 normalization zero-train screening
→ 最多兩個候選做 N1 5 ep attention-only PMP recovery
→ BDCN D0 → D1 三候選 staged 5+5 ep codebook-only → conditional winner seed 1 → D2/R0/R1/R2
→ A-FINAL 與完整 COCO/operation/error report
→ optional quantization（未核准時停止）
~~~

N0/N1 必須沿用相同 A0 checkpoint、兩個 Attention sites、資料/evaluator/seed 與未修改模組。N0 不量化 P/V/projection。N1 的 `ρ` 是 probability-level blend，不能和 score-level `λ` 混名。

BDCN 報告必須把 denominator 寫清楚：R0 exact division/reciprocal 是 reference；R1 reciprocal LUT 是 primary；R2 PoT denominator shift 才是 division-free diagnostic。不得把整個 BDCN 方法籠統稱為 division-free。fused bucket-PV 不 materialize P，但仍要計算並回報 bucket-add、codebook weighting 與每列 reciprocal 成本。

BDCN fused bucket-PV 的軟體 AMP reference 必須停用 autocast，以 FP32 累加 bucket partial sums、codebook weighting 與 reciprocal，最後才轉回 V dtype；硬體版另計 accumulator guard bits。Trainer 發現任一 batch loss 為 NaN/Inf 必須立即 fail closed。

## 6. 結果與 provenance

每個正式 run 使用唯一小寫 `run_id`，不可覆寫：

~~~text
artifacts/runs/<run_id>/
├── manifest.json       # variant、training、環境、Git revision
├── variant.yaml
├── training.yaml
├── checkpoints/        # best/last 與重新 BN-fold export
├── metrics/            # COCO metrics、per-site/normalization errors
├── profiles/           # MAC、binary word ops、LUT、memory/latency proxy
├── exports/            # bit-true/export artifacts
├── logs/
└── ultralytics/
~~~

比較表至少記錄 parent run、兩個 converted paths、best epoch、mAP50-95、mAP50、參數量、估算運算量、row-sum/error statistics 與失敗原因。數值不存在就寫 `not_run`，不得以預估值冒充量測值。

Queue 固定放在 `artifacts/queue/`。`queue.json` 是唯一排程狀態來源，`events.jsonl` 是 append-only 事件紀錄，`generated/<job-id>/variant.yaml` 必須由實際 winner 派生。不可手改 job 為 succeeded、不可複製前一個 checkpoint 假裝是新結果，也不可同時啟動兩個 worker。

每個 evaluate/train job 必須產生 `metrics/queue-result.json`；variant job 另產生 `profiles/analytical.json`。`estimated_memory_traffic` 與 `arithmetic_cost_proxy` 只是固定假設下的比較 proxy，不能寫成 GPU/FPGA 實測。P0 是 implementation validation，不算研究貢獻。

Fixed-scale validation 可能讓 Ultralytics 原地進行 Conv+BN fusion；`calibrated.pt` 必須從 validation 前保留的未融合 training graph 匯出，再寫入校準係數。Custom parent 的 state coverage 低於 95%，或 child mAP 低於 model parent 的 50%，都必須 fail closed，禁止繼續選 winner。

## 7. 文件與資料夾規則

- 新增任何資料夾時，同時加入 README，說明用途、輸入、輸出及是否提交 Git。
- 根 README 是全專案導航；`plan.md` 是正式研究順序；`docs/architecture.md` 是程式 seam；資料夾 README 是局部操作說明。
- 程式行為、命令、檔案樹或實驗名稱變更時，同一個 patch 內同步文件。
- LaTeX 公式使用 `$...$` 或 `$$...$$`；流程圖可用純文字，避免用無法閱讀的 Unicode 數學碎片代替公式。

## 8. 品質與安全門檻

- 使用 typed dataclass/enum，不散落 magic strings。
- 路徑、Attention 數量、tensor shape 與 config 組合都要 fail closed。
- 保留現有使用者修改；禁止 destructive Git 指令與覆寫 artifact run。
- GPU 訓練、下載大型資料/權重、外部發布及上板都需要明確授權。
- 一般程式修改與測試不得使用 `queue ... --execute`；只有使用者明確允許 GPU/COCO 執行時才可使用。
- 完成前至少執行 `pytest -q` 與 `ruff check src tests scripts`；若無法執行，要明確列出原因。
