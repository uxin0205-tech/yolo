# YOLO11m Binary Q/K Attention Research

這個目錄保存一套已完成的 YOLO11m BinaryAttention 消融研究：將偵測器中的
attention Q/K 從 full precision 改成 sign、scaled-sign、dual-basis 等二值形式，
再依序測試短期 QAT fine-tuning、知識蒸餾（KD）、2D relative-position bias、
P/V 8-bit fake quantization 與 magnitude side channel。

正式矩陣共有 **26 組 canonical experiments**。每組實驗的設定、指標、模型結構、
strict checkpoint 與 SHA-256 都已封存；final audit 目前為 **26/26 通過、0 errors**。

> 本研究是 COCO detection 上的 **10-epoch attention-only QAT adaptation**，不是
> BinaryAttention 論文的 300-epoch full-model ImageNet recipe。這份結果適合回答本專案
> 內部的相對消融問題，不應表述成論文完整重現。

## 1. 快速導覽

| 想找的內容 | 文件或目錄 |
|---|---|
| 完整實驗結果、表格與結論 | [`artifacts/reports/binary_attention_final_report.md`](artifacts/reports/binary_attention_final_report.md) |
| 日常操作、維護、清理與載入權重 | [`OPERATIONS.md`](OPERATIONS.md) |
| 實驗矩陣、選擇流程與訓練契約 | [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) |
| 固定 seed 與資料 manifest | [`EXPERIMENT_SEED.md`](EXPERIMENT_SEED.md) |
| 26 組正式權重 | [`artifacts/final_weights/`](artifacts/final_weights/) |
| 26 組機器可讀彙整 | [`artifacts/reports/binary_attention_summary.csv`](artifacts/reports/binary_attention_summary.csv) |
| COCO mAP、APs、APm、APl | [`artifacts/reports/coco_area_metrics.csv`](artifacts/reports/coco_area_metrics.csv) |
| 動態選擇結果 | [`artifacts/reports/paper_qat_selection.json`](artifacts/reports/paper_qat_selection.json) |
| 最終 audit | [`logs/formal-plan/final-audit.json`](logs/formal-plan/final-audit.json) |
| 每個 canonical run 的路徑 | [`artifacts/reports/run_reports_index.md`](artifacts/reports/run_reports_index.md) |

## 2. 研究問題與方法

### 2.1 Binary Q/K 是什麼

標準 attention 以 full-precision query、key 計算相似度。Binary Q/K 將 Q/K 投影成
`sign(x)` 或帶尺度的 `scale(x) × sign(x)`，降低乘法表示成本，但直接替換可能造成
準確率下降。本研究比較下列恢復方法：

- **sign-only**：只保留正負號。
- **scaled-sign**：在符號之外保留 per-sample、per-head 的平均幅度尺度。
- **dual-basis**：用兩組或多組 binary basis 近似原始 Q/K。
- **QAT fine-tuning**：forward 使用二值／假量化運算，backward 使用 STE，讓模型適應量化誤差。
- **KD**：讓 student 對齊 full-precision teacher 的 positional、feature 或 attention 訊息。
- **relative-position bias**：向 attention logits 加入 2D 空間位置偏置。
- **P/V fake quantization**：將 softmax probability P 或 value V 以 8-bit 假量化模擬。
- **magnitude side channel**：在 binary attention 外補充 FP、INT8 或 INT4 幅度訊息。

### 2.2 固定實驗契約

| 項目 | 正式設定 |
|---|---|
| 模型 | YOLO11m |
| 初始化 | 每個 T/N run 都從同一 FP source checkpoint 獨立初始化 |
| 資料 | COCO2017 train2017 118,287 張、val2017 5,000 張 |
| 輸入尺寸 | 640 px |
| 訓練長度 | 10 epochs；E 系列只驗證、不訓練 |
| 可訓練範圍 | attention-only，其餘 backbone/head 參數凍結 |
| Optimizer | AdamW |
| Learning rate | `5e-5`，cosine decay 至 `5e-6` |
| Weight decay | `0.02` |
| Batch | micro batch 16，gradient accumulation 8，effective batch 128 |
| 執行 | AMP、8 workers、disk cache |
| Seed | 所有 26 組 canonical experiments 都是 `0` |
| 決定性 | 22 個 T/N training runs 使用 `deterministic=True` |
| 指標權重 | epoch-last EMA；strict checkpoint 與 validation 使用同一份權重 |

一般 T variants 約有 200,960 個 attention parameters 可訓練、約 19.91M 個參數
凍結；dense bias 與 N4 variants 約有 217k 個可訓練參數。完整 training arguments
保存在各 run 的 `training_args.json`，不可只靠本表重建實驗。

### 2.3 指標口徑

本專案同時保存兩種 mAP，兩者不可混為一談：

- **原 mAP**：訓練完成時由 Ultralytics evaluator 保存的 `mAP50-95`。
- **COCO mAP / APs / APm / APl**：使用同一 canonical strict weight 在 5,000 張
  val2017 重新推論，再由官方 COCOeval 計算。small、medium、large 的 bbox area
  分界分別為 `<32²`、`32²–96²`、`>=96²` pixels。

兩個 evaluator 的處理細節不同，因此 `COCO mAP − 原 mAP` 不是重新訓練帶來的增益。

## 3. 完整實驗矩陣

### E：不訓練的直接替換

| Variant | 內容 |
|---|---|
| E0 | FP attention 零訓練基準 |
| E1-S | sign-only binary Q/K |
| E1 | scaled-sign binary Q/K |
| E2-DUAL | matched residual dual-basis；只加入使用者指定的 dual basis，不訓練 |

### T0–T5：基礎 attention 與 QAT 消融

| Variant | 內容 | QK / Softmax / PV 次數 |
|---|---|---:|
| T0 | FP attention-only fine-tuning 控制組 | 0 / 1 / 1 |
| T1 | sign-only QAT | 1 / 1 / 1 |
| T2 | scaled-sign QAT | 1 / 1 / 1 |
| T3 | parallel dual binary attention | 2 / 2 / 2 |
| T4 | full-basis residual dual attention | 4 / 1 / 1 |
| T5 | matched-basis residual dual attention | 2 / 1 / 1 |

T3、T4、T5 對應早期規劃中的 N1、N2、N3。這三組本身不使用 KD；完成後以原
`mAP50-95` 從 T1–T5 選出 T4，作為 T6 KD 消融的基礎結構。

### T6：KD component 消融

| Variant | 內容 |
|---|---|
| T6-O | 只加入 positional encoding KD |
| T6-F | 只加入 feature KD |
| T6-A | 只加入 attention KD |
| T6-O/F | positional + feature KD |
| T6-O/A | positional + attention KD |
| T6-F/A | feature + attention KD |
| T6 | 將選出的 positional + feature 組合正式重現 |

T6-O/F 以原 mAP 成為預先定義的最佳 KD 組合，T6 是其 deterministic 正式重現。
若改看官方 COCOeval，T6-F/A 的 COCO mAP 與 APs 略高；差距很小且只有單一 seed，
因此 README 同時保留兩個觀察，不事後改動原選擇規則。

### T7：位置偏置與 P/V 量化

| Variant | 內容 |
|---|---|
| T7-D | T6 配置 + dense 2D relative-position bias |
| T7-R | T6 配置 + decomposed 2D relative-position bias |
| T7-P | 選定 dense bias，P 分支做 8-bit fake quantization |
| T7-V | 選定 dense bias，V 分支做 8-bit fake quantization |
| T7-PV | 選定 dense bias，P、V 同時做 8-bit fake quantization |

T7-D/T7-R 比較 bias 參數化；正式流程依原 mAP 選 dense 2D，再在同一分支做 P/V
量化。P 使用 static unsigned `1/255` scale，V 使用 per-sample/per-head token-channel
max-abs scale。

### N4：不使用 KD 的 magnitude side channel

| Variant | 內容 |
|---|---|
| N4-FP | FP magnitude side channel |
| N4-I8 | 8-bit magnitude |
| N4-I4 | 4-bit magnitude |
| N4-PV | 最佳 N4-I4 再加 P/V 8-bit fake quantization |

N4 使用 T7 + T1–T5 中最佳的非 KD 結構選擇，但不繼承 T6/T7 的 KD loss 或訓練
checkpoint；各 run 仍從相同 FP source 獨立初始化。正式 parent/bias 為 T7-D 的
dense 2D 結構，N4 內部以 I4 最佳。

## 4. 結果統整

下表的「原 mAP」來自 Ultralytics；其餘欄位來自 canonical strict weight 的官方
COCOeval。完整 precision、recall、AP50、AP75、差值與 SHA-256 請看
[`完整報告`](artifacts/reports/binary_attention_final_report.md)。

| Variant | 原 mAP | COCO mAP | APs | APm | APl |
|---|---:|---:|---:|---:|---:|
| E0 | 0.50481 | 0.51085 | 0.33363 | 0.56644 | 0.66956 |
| E1-S | 0.45070 | 0.45930 | 0.28321 | 0.50373 | 0.62014 |
| E1 | 0.48121 | 0.48056 | 0.29818 | 0.53223 | 0.64129 |
| E2-DUAL | 0.48710 | 0.49565 | 0.31405 | 0.54803 | 0.65191 |
| T0 | 0.50672 | 0.51267 | 0.33102 | 0.56532 | 0.67330 |
| T1 | 0.50188 | 0.50789 | 0.33074 | 0.55952 | 0.66853 |
| T2 | 0.50346 | 0.50951 | 0.33201 | 0.56311 | 0.66997 |
| T3 | 0.50211 | 0.50798 | 0.32850 | 0.55988 | 0.66805 |
| T4 | 0.50586 | 0.51163 | 0.33103 | 0.56511 | 0.67134 |
| T5 | 0.50447 | 0.51007 | 0.32888 | 0.56377 | 0.66793 |
| T6-O | 0.50570 | 0.51131 | 0.33076 | 0.56481 | 0.67016 |
| T6-F | 0.50547 | 0.51139 | 0.33020 | 0.56481 | 0.66946 |
| T6-A | 0.50539 | 0.51151 | 0.33252 | 0.56336 | 0.66843 |
| T6-O/F | 0.50624 | 0.51176 | 0.33126 | 0.56445 | 0.67106 |
| T6-O/A | 0.50549 | 0.51146 | 0.33251 | 0.56308 | 0.66911 |
| T6-F/A | 0.50563 | 0.51184 | 0.33378 | 0.56275 | 0.67061 |
| T6 | 0.50624 | 0.51176 | 0.33126 | 0.56445 | 0.67106 |
| T7-D | 0.50616 | 0.51165 | 0.33050 | 0.56429 | 0.67057 |
| T7-R | 0.50577 | 0.51166 | 0.33218 | 0.56512 | 0.67014 |
| T7-P | 0.50562 | 0.51143 | 0.33122 | 0.56340 | 0.67007 |
| T7-V | 0.50506 | 0.51082 | 0.32882 | 0.56415 | 0.66931 |
| T7-PV | 0.50513 | 0.51114 | 0.33196 | 0.56401 | 0.66936 |
| N4-FP | 0.50384 | 0.50949 | 0.32854 | 0.56243 | 0.66922 |
| N4-I8 | 0.50374 | 0.50918 | 0.32825 | 0.56307 | 0.66789 |
| N4-I4 | 0.50406 | 0.50986 | 0.32985 | 0.56332 | 0.66830 |
| N4-PV | 0.50402 | 0.50976 | 0.32896 | 0.56311 | 0.66836 |

### 主要結論

- 全矩陣 COCO mAP 最高是 FP 控制組 **T0 = 0.51267**。
- T1–T5 最佳 binary structure 是 **T4**；原 mAP `0.50586`，只比 T0 低 `0.00086`。
- 正式 KD 選擇 **T6-O/F / T6**；原 mAP `0.50624`，與 T0 相差 `0.00048`。
- 官方 COCOeval 下，binary variants 中 **T6-F/A = 0.51184** 最高，且
  **APs = 0.33378** 是全矩陣最高，但它不是依原 selection metric 選出的正式 T6。
- 預先定義 final-family 的正式候選是 **T7-D**；原 mAP `0.50616`、COCO mAP
  `0.51165`。T7-R 的 COCO mAP 高 `0.00001`，遠小於單一 seed 可支持的結論強度。
- N4 族群以 **N4-I4** 最佳，但所有 N4 variants 都未超過 T7-D。
- 零訓練 E 系列顯示 binary Q/K 直接替換會明顯降準確率；scaled-sign、dual-basis
  與 10-epoch QAT 能逐步縮小差距。

## 5. 專案結構

```text
yolo_binaryqk/
├── README.md                          # 本文件：研究入口與統整
├── OPERATIONS.md                      # 維護、驗證、封存與清理手冊
├── IMPLEMENTATION_PLAN.md             # 實驗設計與正式流程契約
├── EXPERIMENT_SEED.md                 # seed、dataset manifest 與 hash 紀錄
├── binary_attention_executable_spec.md# 初始可執行規格
├── relative_position_bias_report.md   # relative-position bias 專項分析
├── configs/                           # YOLO11m BinaryAttention 模型 YAML
├── data/                              # COCO YAML、固定 manifest 與資料說明
├── binary_attention/                  # 研究程式套件
│   ├── attention/                     # attention kernels 與 fake quantizers
│   ├── variants/                      # 26 組 variant 定義與 materialization
│   └── verification/                  # strict checkpoint schema 與重載驗證
├── tests/                             # 數值、模型建構與流程契約測試
├── artifacts/
│   ├── runs/<variant>/<stage>/<uuid>/ # 每次 run 的完整原始證據
│   ├── reports/                       # 彙整、selection、圖與完整報告
│   ├── area_metrics/                  # 每組官方 COCOeval 尺寸指標
│   └── final_weights/                 # 26 組 canonical 權重 bundle
├── logs/formal-plan/                  # runner、report 與 final audit 紀錄
├── services/                          # 本機 systemd user service 範本
├── run_formal_plan.sh                 # 可續跑、單一 owner 的矩陣 runner
├── audit_after_formal.sh              # 完成後重建報告並 audit
└── repair_current_t0_ema.sh           # 舊 T0 EMA metadata 修復工具
```

### 5.1 其他檔案與目錄

- `.cache/`、`.pytest_cache/`、`__pycache__/`：可重建快取，不是研究證據。
- `logs/T0.*`：T0 早期背景執行與除錯紀錄；正式結果以 canonical run 為準。
- `data/*seed42*`：早期或候選資料 manifest；本批 canonical 26 runs 的正式 seed 是
  `0`，正式 full dataset manifest 是 `data/coco_full.txt/json`。
- `binary_attention_executable_spec.md:Zone.Identifier`：從其他平台複製規格時保留的
  metadata sidecar，不參與訓練或驗證。
- `yolo26n.pt`：輔助 Ultralytics 模型資產；正式研究初始化來源仍是
  `../original/weight/yolo11m.pt`，不可用它取代 canonical source。
- `services/` 內含目前工作站的絕對路徑；換機後需先修改 `WorkingDirectory` 與
  `ExecStart`，一般報告重建不需要安裝 service。

## 6. Python 模組說明

| 模組 | 作用 |
|---|---|
| `attention/base.py` | FP、sign、scaled-sign、三種 dual-basis、N4 magnitude attention，以及插入 YOLO C2PSA 的 wrapper |
| `attention/quantizers.py` | clipped STE、P8、V8、INT8、INT4 fake quantizer 與 scale 規則 |
| `variants/definitions.py` | 所有 variant 的唯一來源；定義繼承、config hash、T6/T7/N4 materialization 與 quantization contract |
| `config.py` | 把 resolved variant 寫入 YOLO model YAML |
| `model.py` | 向 Ultralytics parser 註冊自訂 attention、建構 student、載入 FP source |
| `trainer.py` | attention-only freeze、gradient accumulation、EMA、teacher/KD 接線與訓練器 |
| `training_profiles.py` | 10-epoch 正式超參數與不同 stage 的 override |
| `kd.py` | positional、feature、attention distillation losses 與 target diagnostics |
| `data.py` | COCO layout、label conversion、manifest、dataset YAML 與 hash |
| `workflow.py` | 建立 run、訓練／驗證、完成 artifact、診斷與 strict checkpoint |
| `verification/checkpoint.py` | schema-3 checkpoint、manifest、SHA-256 與 strict reload |
| `formal_audit.py` | 對 26 個 canonical runs、selection、freeze delta、metrics 與 archived weights 做 fail-closed audit |
| `report.py` | 從 artifacts 重建 CSV、JSON、圖與 Markdown 報告 |
| `area_metrics.py` | 以 canonical weights 重新推論並計算官方 COCO mAP、APs、APm、APl，可中斷續跑 |
| `weight_archive.py` | 封存 canonical strict weights、metadata、manifest 並逐一驗證載入 |
| `checkpoint_cleanup.py` | 在封存與 audit 通過後精確清除 Ultralytics optimizer/resume checkpoints |
| `ema_repair.py` | 修復舊式 Ultralytics EMA checkpoint metadata；不屬於一般流程 |
| `cli.py` | 對外命令入口：data、verify、run、report、audit |

修改 variant 時，至少要一起檢查 `definitions.py`、attention 實作、runner、audit、
report 與 tests，避免 shell 排程和模型定義出現兩套互相矛盾的真相來源。

## 7. Artifact 與權重契約

每個 canonical run 位於：

```text
artifacts/runs/<artifact-name>/<validation|full>/<uuid>/
```

其中應包含：

| 檔案 | 作用 |
|---|---|
| `status.json` | completed、stage、valid-for-research 與來源資訊 |
| `resolved_config.json` | materialized variant、quantization contract、config hash |
| `training_args.json` | epochs、optimizer、batch、freeze 與 weight-source 契約 |
| `environment.json` | 執行環境資訊 |
| `model.yaml` | 可重建模型架構 |
| `architecture_manifest.json` | 模組、參數與 attention 結構證據 |
| `validation_metrics.json` | Ultralytics 與補算後 COCO metrics |
| `training_curves.csv` | 每個 epoch 的訓練曲線 |
| `attention_diagnostics.json` | binary/softmax/PV 計算與 attention 診斷 |
| `parameter_delta_diagnostics.json` | attention 確實更新、其餘參數維持凍結的證據 |
| `checkpoint_manifest.json` | checkpoint identity、schema 與 SHA-256 |
| `checkpoints/strict-*.pt` | 與報告 metrics 對應的正式權重 |
| `experiment_report.md` | 單一 run 的摘要 |

長期使用請從 [`artifacts/final_weights/`](artifacts/final_weights/) 取權重。每個 variant
bundle 都包含 `weight.pt`、`model.yaml`、`resolved_config.json`、
`validation_metrics.json`、`checkpoint_manifest.json` 與 `status.json`。

這些 `weight.pt` 多數以 hardlink 連到 run 內 strict checkpoint；兩個檔名可能共享
同一 inode，因此**不要原地修改任一端**。跨檔案系統備份時應複製整個
`final_weights/` 並一起保存 [`manifest.json`](artifacts/final_weights/manifest.json)。

Ultralytics `best.pt` / `last.pt` 是帶 optimizer/resume 狀態的暫存 checkpoint，不是
canonical 權重。本批 58 個冗餘檔案已在驗證封存後清除，共釋放 2.554 GiB；26 組
strict weights 均保留。

## 8. 使用與驗證

所有指令從本目錄執行：

```bash
cd /home/hplab/uxin/yolo11/yolo_binaryqk
```

### 快速測試與 final audit

```bash
../.venv/bin/python -m pytest -q
../.venv/bin/python -m binary_attention.cli verify --variant T2
../.venv/bin/python -m binary_attention.cli audit
```

成功的 final audit 必須同時符合：`ok=true`、26 variants、26 archived weights、
26 area metrics、`errors=[]`。目前 freeze delta audit 使用 `1e-5` absolute tolerance，
原因是 epoch EMA 對理論上不變的 FP tensors 仍可能產生 ulp 級捨入；本批觀察到的
最大 frozen parameter delta 為 `6.67572021484375e-06`，沒有超過門檻。

### 重建報告

```bash
../.venv/bin/python -m binary_attention.cli report
```

`artifacts/reports/` 內的 generated report 應由 artifacts 重建，不要手動修改數值。

### 補算或續跑 COCO 尺寸 AP

```bash
../.venv/bin/python -m binary_attention.area_metrics --device 0 --batch 16
../.venv/bin/python -m binary_attention.cli report
```

每組完成後會寫入 `artifacts/area_metrics/<variant>/coco_area_metrics.json`，可以中斷後
續跑；預設不保存體積較大的 predictions JSON。

### 驗證全部 canonical 權重可載入

```bash
../.venv/bin/python -m binary_attention.weight_archive --verify-load
```

此命令會讀取 final audit、重建每個模型、strict-load 權重並更新 archive manifest。

### 重新執行完整矩陣

```bash
./run_formal_plan.sh
```

Runner 以 `logs/formal-plan/.formal-plan.lock` 保證單一 owner，並只重用 completed、
config hash 相符且 strict checkpoint 存在的 run。這會啟動大型 GPU 工作；日常查看
報告或載入權重不需要重跑。

## 9. 可重現性與限制

- 目前所有 canonical 結果只有 **seed 0**；低於約 `0.001 AP` 的差異只應視為趨勢。
- 本實驗只訓練 attention 10 epochs，不能外推成 full-model 長期訓練結論。
- Q/K、P/V、INT8/INT4 都是 PyTorch QAT/fake quantization；尚未測量實際硬體的
  latency、energy、memory bandwidth 或部署後模型大小。
- T7-D 是依預先定義流程選出的 final-family 模型，不是全矩陣純準確率最高模型；
  論文或報告應同時列出 T0、T6、T7-D，避免混淆控制組、最佳 KD 與最終候選。
- 若要主張小幅提升具穩定性，應補多 seed 或更長訓練，而不是只比較小數點後差異。

## 10. 目前完成狀態

- Final audit：`ok=true`
- Canonical experiments：`26/26`
- Canonical weight bundles：`26/26`
- COCO area metrics：`26/26`
- Audit errors：`0`
- 固定 seed：`0`
- T1–T5 最佳結構：`T4`
- 正式 KD 組合：`positional + feature`
- 正式 T7 bias：`dense_2d`
- N4 最佳 magnitude：`INT4`
- Final-family candidate：`T7-D`

研究數據的最終解讀以 [`完整報告`](artifacts/reports/binary_attention_final_report.md)
為準；維護、清理和擴充前請先閱讀 [`OPERATIONS.md`](OPERATIONS.md)。
