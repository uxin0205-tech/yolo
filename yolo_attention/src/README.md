# src

production package 位於 `src/yolo_attention/`。外部主要 interface：

1. `VariantConfig`：選擇 FP/I/H/T5、bias、scale 與 normalization。
2. `HardwareFriendlyAttention.from_ultralytics()`：轉換官方 Attention。
3. `convert_yolo26_model()`：fail-closed 轉換 YOLO26m 兩個 Attention，不碰 Detect。
4. `ResearchWorkflow`：集中定義主線、winner gate 與 optional quantization。

| 模組 | 責任 |
|---|---|
| `config.py` | typed variant schema 與 YAML |
| `projection.py` | per-head QKV gather/interleave、BN fold |
| `binary_basis.py` | sign/STE、Hadamard、T5、XNOR-popcount |
| `relative_bias.py` | none/dense/decomposed bias |
| `normalization.py` | N0 normalization factory、N1 PMP、Optional integer LUT |
| `reciprocal.py` | 整數 leading-one、LUT256／PWL16／Newton64、UQ1.24 probability |
| `stage3_checkpoint.py` | S1 parent 繼承、完整 manifest 與嚴格重建 |
| `stage3_diagnostics.py` | 定點 reference、真實 score／exp／reciprocal／P／PV 分項誤差 |
| `stage3_workflow.py`／`stage3_cli.py` | 第 3 階段既有 queue 接線、來源凍結、600 秒監測 |
| `stage3_report.py` | 正式 AP 選型、交付及無 GPU／checkpoint 重畫 |
| `bdcn.py` | distance bucket、fixed/unbounded/anchored monotonic codebook、PoT projection、R0/R1/R2 與 fused bucket-PV |
| `quantization.py` | U8/S8 fake quant |
| `attention.py` | 保留 official PV+PE(V)+projection |
| `integration.py` | 兩個 Attention 的 YOLO conversion、D1→D2/R trained-bank state transfer、trainable scope |
| `experiments.py` | 正式漏斗 registry |
| `workflow.py` | 主線階段與 optional phase |
| `training.py` | Ultralytics custom trainer seam；保留官方與 custom parent state |
| `runner.py` | dry-run request 與明確 training launch |
| `artifacts.py` | immutable provenance |
| `profiling.py` | dense/binary 與 BDCN lookup、bucket-add、reciprocal operation accounting |
| `queue_model.py` | persistent job/state/result schema 與 invariant |
| `queue_store.py` | atomic queue JSON、event log、single-worker lock |
| `queue_policy.py` | deterministic accuracy/cost gates；無 I/O |
| `queue_workflow.py` | readiness 與 winner-dependent graph/YAML expansion |
| `queue_backend.py` | `--execute` 後的 train/evaluate/P0/select dispatch |
| `queue_executor.py` | dry preview、狀態轉移、失敗 retry、selection rewind 與 archive |
| `cli.py` | 唯一公開 orchestration CLI |

新增 basis 時擴充 `BasisKind`、`BinaryScore` 與 tests；新增 normalization 時擴充 `NormalizationKind`、`normalization.py` 與 construction mapping。不要把資料路徑或 trainer side effect 寫進數學模組。

Queue seam 的資料流固定為：

~~~text
queue_workflow (pure graph)
  → queue_store (persistent state)
  → queue_executor (one-job state machine)
  → queue_backend (live dispatch only)
  → runner / evaluation / P0
~~~

新增實驗時，先新增 policy test，再在 `queue_workflow.py` 建 job；不要把 winner 判斷寫進 CLI 或 backend。新增結果欄位時要同步 `QueueResult`、JSON contract、executor validation 與 README。
