# Codex 任務規格：YOLO26m Activation 選擇與硬體友善共同設計

> **2026-08-25 Phase 0 決策覆寫**
>
> - 未來由使用者交付的不可變 baseline 資料夾決定實際模型；本文 YOLO26m mapping 目前只作
>   template，不得在未檢查 manifest 前硬套。
> - detection 評估同時涵蓋 Canonical BBAT5 v1 ball/bat Detect 與 COCO2017 COCO80 Detect；兩者
>   使用各自 YAML、checkpoint 與 metrics，分開訓練／報告，不合併 raw mAP。
> - 候選不限定 PWL。文獻短名單與 proposed integral-polynomial working hypothesis 見
>   [Phase 0 研究索引](research/README.md)。
> - 搜尋先 1 seed，有餘裕時 finalists 補第 2 seed；因此本階段只作探索性結論。原文三 seeds
>   保留為未來論文等級驗證門檻。
> - 已完成純函數 package、Q16.10 整數 emulator、ONNX graph 稽核與測試；模型整合與訓練仍須
>   等待交付資料夾。
> - 第一輪既有函數 baseline 只保留 SiLU 與 Hardswish；ReLU 僅作便宜 primitive floor。Proposed
>   只保留 `poly_shift`（主候選）與 `poly_quality`（accuracy ablation），其他候選暫不進昂貴
>   recovery。實作與結果見[硬體友善原型報告](research/phase0-hardware-activation-prototype.md)。
> - 已完成 SIPA 對稱 activation 與 BCSP 受限靜態 placement 訓練架構：manifest 必須 reviewed、
>   replacement 先完整 preflight、planner 每次只產生下一個 Pareto stage。入口見
>   [training/README](../training/README.md)；production adapter 仍等待交付 framework commit。
>
> 若本覆寫與下方早期規格衝突，以本覆寫及 Phase 0 文件為準；BBAT5 全域不可變規範仍具有最高
> 優先權。


## 0. 任務定位

本任務針對 **Ultralytics YOLO26m detection** 建立一套可重現的 activation 分析、候選生成、區域選擇、微調、匯出與硬體驗證流程。

目前「確認過的 activation 資料資料夾」尚未提供，因此本階段只能：

1. 建立程式框架、資料契約、候選 activation、測試與 dry-run。
2. 能載入 YOLO26m，自動盤點 activation 並建立 region manifest。
3. 能用合成資料完成單元測試與端到端 smoke test。
4. **不得憑空產生真實統計、不得先宣告最佳 activation、不得開始正式選型。**

資料到位後，才執行完整的 sensitivity scan、擬合、policy search、fine-tuning 與硬體驗證。

---

## 1. 研究目標

不要只回答「ReLU、HardSwish、SiLU 哪一個最好」。本研究要回答：

1. YOLO26m 不同區域對 activation 的敏感度是否不同？
2. 能否使用少量共享 activation profile，而非每層各自一套，取得較好的 accuracy–hardware Pareto？
3. 能否利用 SiLU 的恆等式，只近似正半邊，降低係數儲存與硬體路徑？
4. 能否把斜率限制為 power-of-two 或 additive power-of-two，使乘法轉成 shift/add？
5. 能否利用 YOLO26 的 one-to-many 訓練輔助 head 與 one-to-one 部署 head，採用不對稱 activation policy？
6. 能否把 activation 與 requantization 融合成 integer-only datapath？

最終輸出不是單一函數名稱，而是：

- 一份 `selected_policy.yaml`
- 一組 activation 係數
- 可重現的訓練與驗證結果
- 匯出的模型
- 軟體與目標硬體的 Pareto 報告

---

## 2. 主要創新候選：DH-RA-DySiLU（暫名）

全名：**Dual-Head-aware Region-Adaptive Dyadic SiLU**。

這只是研究工作名稱，不得在結果尚未完成前宣稱為新方法。

### 2.1 核心函數

SiLU：

\[
\operatorname{SiLU}(x)=x\sigma(x)
\]

對任意 \(u\ge 0\)：

\[
\operatorname{SiLU}(-u)=\operatorname{SiLU}(u)-u
\]

因此只擬合正半邊 \(g(u)\approx \operatorname{SiLU}(u)\)：

\[
\hat f(x)=
\begin{cases}
g(x), & x\ge 0\\
g(-x)+x, & x<0
\end{cases}
\]

等價硬體路徑：

`ABS -> segment select -> shift/add PWL -> negative input 時 subtract abs(x)`

### 2.2 正半邊 PWL

\[
g(u)=a_k u+b_k,\quad t_k\le u<t_{k+1}
\]

約束：

- `t0 = 0`
- `g(0) = 0`
- segment 間連續
- 尾端 `u >= T` 使用 identity：`g(u)=u`
- breakpoint 必須可量化到 input fixed-point grid
- slope 僅允許：
  - PoT：`±2^-q`
  - APoT2：`±2^-q1 ± 2^-q2`
  - 必要時允許 `0` 與 `1`
- intercept 由連續性推導並量化

### 2.3 兩套共享 profile

不要一開始做每層一套係數。先限定兩套：

- `hi_profile`：6–8 segments，允許 APoT2，給高敏感區域。
- `lo_profile`：2–4 segments，優先單一 PoT slope，給低敏感區域。

最終 segment 數不得固定寫死，需由 config 控制並做 ablation。

### 2.4 YOLO26 dual-head aware policy

YOLO26 detection 同時存在：

- `one2many`：訓練輔助 head，部署 fuse 後移除。
- `one2one`：預設 end-to-end inference head。

必須比較：

- `H0`：兩個 head 都是 SiLU。
- `H1`：兩個 head 都替換成相同硬體 activation。
- `H2`：`one2many = SiLU`、`one2one = hi/lo profile`。
- `H3`：`one2many = hi profile`、`one2one = lo profile`。

`H2` 是優先研究候選：保留訓練輔助 head 的高精度非線性，但部署模型只留下硬體友善 one-to-one head。

不得假設 H2 一定較好，必須由 end-to-end mAP、收斂與硬體結果判斷。

### 2.5 activation + requantization 融合

第二階段硬體候選必須支援直接在量化整數域計算：

\[
q_y=\operatorname{sat}\left(\sum_j s_j(q_x \gg r_j)+c_k\right)
\]

其中：

- breakpoint 直接表示成 integer threshold。
- slope 以 shift code 表示。
- intercept 以 output-scale 整數表示。
- 不先 dequantize 成 float，再做 activation，再 requantize。

軟體 reference、integer emulator、ONNX/export path 必須分離，避免把理論 shift/add 與實際 backend latency 混為一談。

---

## 3. YOLO26m region 定義

不得只靠字串猜測，必須先載入實際模型，自動建立 manifest。以下只作為目前官方 detection YAML 的預設 mapping；若 commit 或模型結構不同，程式應 fail fast 或要求更新 mapping。

### 3.1 預設 top-level mapping

- `early_backbone`：top-level layer 0–4
- `deep_backbone`：top-level layer 5–9
- `attention_ffn`：
  - top-level layer 10 的 C2PSA 內 eligible SiLU
  - 任何其他位置的 `PSABlock` descendant，例如末端 attention-enabled C3k2
- `neck`：top-level layer 11–22，但排除已歸入 `attention_ffn` 的 descendant
- `head_one2many`：Detect 的 `cv2`、`cv3`
- `head_one2one`：Detect 的 `one2one_cv2`、`one2one_cv3`

### 3.2 不在本階段替換的非線性

以下不屬於本次 SiLU activation selection：

- attention softmax
- detection output sigmoid
- raw output `nn.Conv2d`
- `act=False` 的 Conv
- pooling、concat、upsample

不得將 softmax 或 final sigmoid 當成一般 SiLU 一併替換。

### 3.3 eligible activation 盤點規則

1. 逐一走訪每個 `Conv`、`DWConv`、`ConvTranspose` 或具有 `act` 欄位的模組。
2. 只有原始 activation 是 `nn.SiLU` 時，預設標為 eligible。
3. 每一個 eligible node 必須有唯一名稱、region、parent top-level index、input/output shape。
4. Ultralytics 的 `Conv.default_act` 可能是共享的 stateless module；不得僅對共享 `nn.SiLU` 註冊 hook。
5. profiling 應：
   - 在每個 Conv 的 BN output 上掛 hook，取得 activation input；或
   - 把每個 `conv.act` 包成唯一的 `ActivationProbe`。
6. 不得改動 `act=False`。
7. 每次實驗從乾淨 baseline deepcopy 或重新載入 checkpoint，避免 activation 替換累積。

---

## 4. 未來「確認資料資料夾」契約

Codex 現在需實作 validator 與 template generator。資料不存在時只能輸出缺失清單，不得自動填入假資料。

建議結構：

```text
confirmed_activation_data/
├── run_manifest.json
├── baseline_metrics.json
├── activation_manifest.json
├── activation_stats.parquet
├── histograms/
│   └── <safe_module_name>.npz
├── samples/
│   └── <safe_module_name>.npz        # optional reservoir sample
├── sensitivity/
│   └── prior_scan.csv                # optional
└── hardware/
    └── target_profile.yaml           # optional, but required for hardware claim
```

### 4.1 `run_manifest.json` 必填欄位

```json
{
  "schema_version": "1.0",
  "model_id": "yolo26m",
  "task": "detect",
  "checkpoint_sha256": "...",
  "ultralytics_commit": "...",
  "torch_version": "...",
  "end2end": true,
  "dataset_yaml": "...",
  "dataset_fingerprint": "...",
  "split": "val",
  "imgsz": 640,
  "batch": 1,
  "seed": 0,
  "calibration_image_count": 0,
  "calibration_image_list_sha256": "..."
}
```

禁止把 `checkpoint_sha256`、commit、dataset fingerprint 留空後繼續正式比較。

### 4.2 `activation_stats.parquet` 每列欄位

至少包含：

- `module_name`
- `region`
- `top_level_index`
- `count`
- `mean`
- `std`
- `min`
- `max`
- `p0001`, `p001`, `p01`, `p50`, `p99`, `p999`, `p9999`
- `negative_fraction`
- `zero_fraction`
- `histogram_path`
- `sample_path`（可空）

histogram 必須是 streaming 累積，禁止把所有 feature tensor 全部存入記憶體或磁碟。

### 4.3 分布資料一致性檢查

validator 必須檢查：

- manifest 中的 module 是否與實際 checkpoint 相符。
- 每個 eligible module 是否都有統計。
- histogram count 是否與 row count 一致。
- bins 是否遞增、counts 是否非負。
- NaN/Inf。
- model、dataset、imgsz、end2end、seed 是否一致。
- 若資料來自 fused model，需明確標記；預設只接受 unfused profiling。

---

## 5. 候選 activation library

### 5.1 必須保留的基準

- `SiLUExact`
- `ReLU`
- `ReLU6`
- `Hardswish`
- `LeakyReLU`（僅作比較；硬體成本需獨立計）

### 5.2 PWL 基準

- `PWLRealSiLU-K`：雙側或一般 float slope PWL，K = 3/4/6/8。
- `SymPWLRealSiLU-K`：使用 SiLU 正負恆等式，但 slope 為 float。
- `SymPWL-PoT-K`
- `SymPWL-APoT2-K`

### 5.3 Proposed family

- `RADySiLUHi`
- `RADySiLULo`
- `FusedIntRADySiLU`

### 5.4 訓練 backward ablation

預設使用 PWL 的真實分段梯度。

另做一個可選 ablation：

- forward：硬體 PWL
- backward：SiLU smooth surrogate gradient

此選項只能標為 `surrogate_backward`，不可默認開啟，也不可把它與 exact PWL 結果混在同一列。

---

## 6. 分布感知擬合

### 6.1 折疊正負樣本

由於對稱恆等式使正負半邊近似誤差相同，將所有 activation input 轉為：

```text
u = abs(x)
weight(u) = positive_count(u) + negative_count(-u)
```

擬合目標只需比較：

```text
g(u) vs SiLU(u), u >= 0
```

這一點必須有單元測試，證明 folded objective 與完整雙側 objective 一致至數值容忍度。

### 6.2 breakpoint 候選

候選集合應由下列聯集產生：

- `abs(x)` 的 quantiles
- 均勻 grid
- SiLU 高曲率區的固定 anchor
- fixed-point 可表示的 threshold grid
- `0` 與 tail threshold `T`

所有 threshold 最後需 snap 到硬體 input grid。

### 6.3 fitting 流程

每個 K 與 slope family：

1. 先做 unconstrained continuous PWL fit，作為 lower-bound reference。
2. 將每段 slope 投影到最近的 PoT/APoT 候選集合。
3. 以 local search 或 beam search 同時修正 breakpoint 與 slope code。
4. 由連續性重新推導 intercept。
5. 檢查 `g(0)=0`、continuity、tail identity。
6. 輸出 weighted MSE、MAE、max error、p99.9 error。
7. 保存所有係數與固定點編碼，不只保存圖。

不得只用全域均勻 MSE；至少同時輸出：

- uniform-grid error
- empirical-distribution-weighted error
- tail error

### 6.4 profile sharing

流程：

1. 先為每個 region 擬合獨立 oracle PWL，僅用來了解需求。
2. 根據 region 的分布特徵、敏感度與 oracle 係數差異，初始化兩群。
3. 聯合擬合 `hi_profile` 與 `lo_profile`。
4. 以完整 validation task metric 重新搜尋 region assignment。

最終部署只允許兩套共享 profile，除非 ablation 證明第三套帶來明顯 Pareto 優勢。

---

## 7. Sensitivity scan 與 policy search

### 7.1 第一層：簡單替換矩陣

建立以下實驗：

- `E0`：all SiLU baseline
- `E1`：all ReLU
- `E2`：all Hardswish
- 每個 region 單獨換成 ReLU，其他保留 SiLU
- 每個 region 單獨換成 Hardswish，其他保留 SiLU
- head 必須拆成 one2many / one2one

輸出：

```text
region, candidate, delta_map5095, delta_map50, delta_recall,
ap_s_delta, ap_m_delta, ap_l_delta,
latency_delta, memory_delta, export_ok
```

### 7.2 三階段 fidelity

1. `zero_shot`：不微調，固定小型 validation subset。
2. `short_recovery`：短期微調，固定 epoch/iteration budget。
3. `full_validation`：完整 validation，只保留前幾名。

不得在不同候選間變更 dataset subset、seed、imgsz、batch 或 augmentation。

### 7.3 選擇原則

優先使用 Pareto front，不要只靠單一任意加權分數。

硬性約束由 config 指定，例如：

- 最大 `mAP50-95` 降幅
- 最大 `AP_S` 降幅
- latency / area / power budget
- export compatibility

若未提供 accuracy budget，程式需輸出多個 Pareto 候選，不得擅自選一個「最佳」。

### 7.4 兩 profile exhaustive search

五個主要 region 使用兩 profile 時只有 `2^5` 組；加上 one-to-one head policy 後仍可窮舉。

先做 region-level exhaustive search，再視結果決定是否需要細化到 layer-level。不得一開始進入每層 activation NAS，避免與既有工作重複且搜索空間過大。

---

## 8. Fine-tuning 與最終訓練

### 8.1 權重初始化

- 從相同 YOLO26m baseline checkpoint 載入。
- 每個候選使用獨立 run directory。
- 記錄所有 config、git diff、seed、checkpoint hash。

### 8.2 訓練層次

1. activation-only recovery：先固定大部分權重，確認數值穩定。
2. full fine-tuning：低 learning rate 恢復。
3. final candidates：至少三個 seed。
4. 只有最終 1–2 個方法需要做 from-scratch 或完整官方 recipe 驗證。

### 8.3 dual-head 注意事項

- 確認 one2one 與 one2many activation 都被正確盤點。
- `H2` 中 one2many 保留 SiLU、one2one 使用 proposed activation。
- 部署驗證前執行 fuse，確認 one2many 已移除。
- 保存 fuse 前後 graph report。

### 8.4 穩定性紀錄

至少記錄：

- train/val loss
- gradient norm
- NaN/Inf 次數
- activation saturation ratio
- 每個 epoch 的 mAP
- convergence epoch

---

## 9. Export 與硬體驗證

### 9.1 三種實作必須分開

1. `reference_float`：研究用 PyTorch 精確參考。
2. `export_float_pwl`：ONNX 友善的 unrolled compare/where/add/mul graph。
3. `integer_shift_add_emulator`：只允許 compare、abs、shift、add/sub、clamp。

不得因 PyTorch custom activation 變慢，就否定硬體設計；也不得因理論運算少，就宣稱真實硬體加速。

### 9.2 ONNX graph gate

Proposed activation 的部署 graph 中，activation 路徑不得包含：

- `Exp`
- `Sigmoid`
- `Div`

float PWL 可暫時存在 constant `Mul`；只有 integer emulator / custom kernel 才能驗證 shift/add。

### 9.3 integer golden test

對每個 profile：

- 產生完整可表示 input domain 或大規模隨機向量。
- 比較 PyTorch fixed-point reference 與 integer emulator。
- 最大誤差不得超過 config 中指定的 LSB tolerance。
- breakpoint 邊界值、飽和值、正負零都要有測試。

### 9.4 backend 測量

至少支援可插拔 adapter：

- PyTorch CUDA/CPU
- ONNX Runtime
- OpenVINO
- TensorRT
- FPGA/HLS/RTL（目標硬體確定後）

只有在實際 target backend 上有 latency、resource 或 energy 結果時，報告才可使用「hardware-efficient」。否則使用「hardware-oriented」或「shift-add compatible」。

---

## 10. 建議專案結構

```text
activation_lab/
├── README.md
├── pyproject.toml
├── configs/
│   ├── activation_search.template.yaml
│   └── hardware_profiles/
│       └── generic_shift_add.yaml
├── src/activation_lab/
│   ├── __init__.py
│   ├── provenance.py
│   ├── model_inspector.py
│   ├── region_mapper.py
│   ├── data_contract.py
│   ├── profiler.py
│   ├── replacement.py
│   ├── candidates/
│   │   ├── standard.py
│   │   ├── symmetric_pwl.py
│   │   ├── dyadic.py
│   │   └── fused_integer.py
│   ├── fitting/
│   │   ├── fold_distribution.py
│   │   ├── breakpoint_search.py
│   │   ├── slope_projection.py
│   │   └── objective.py
│   ├── search/
│   │   ├── sensitivity.py
│   │   ├── region_policy.py
│   │   └── pareto.py
│   ├── training/
│   │   ├── finetune.py
│   │   └── surrogate_backward.py
│   ├── export/
│   │   ├── onnx_check.py
│   │   ├── integer_emulator.py
│   │   └── coefficient_export.py
│   ├── evaluation/
│   │   ├── accuracy.py
│   │   ├── latency.py
│   │   └── report.py
│   └── cli.py
├── scripts/
│   ├── 00_inspect_model.py
│   ├── 01_make_data_template.py
│   ├── 02_validate_confirmed_data.py
│   ├── 03_profile_activations.py
│   ├── 04_scan_region_sensitivity.py
│   ├── 05_fit_dysilu_profiles.py
│   ├── 06_search_activation_policy.py
│   ├── 07_finetune_finalists.py
│   ├── 08_export_and_validate.py
│   └── 09_build_report.py
└── tests/
    ├── test_silu_identity.py
    ├── test_folded_objective.py
    ├── test_pwl_continuity.py
    ├── test_dyadic_encoding.py
    ├── test_integer_emulator.py
    ├── test_region_mapping.py
    ├── test_detect_dual_head.py
    ├── test_replacement_isolation.py
    └── test_onnx_gate.py
```

若現有 repo 已有清楚結構，可整合進既有 package，但上述責任邊界不得消失。

---

## 11. CLI 契約

```bash
python scripts/00_inspect_model.py \
  --model yolo26m.pt \
  --config configs/activation_search.yaml \
  --out runs/activation/inspect

python scripts/01_make_data_template.py \
  --manifest runs/activation/inspect/activation_manifest.json \
  --out confirmed_activation_data_template

python scripts/02_validate_confirmed_data.py \
  --input confirmed_activation_data

python scripts/04_scan_region_sensitivity.py \
  --config configs/activation_search.yaml \
  --input confirmed_activation_data

python scripts/05_fit_dysilu_profiles.py \
  --config configs/activation_search.yaml \
  --input confirmed_activation_data

python scripts/06_search_activation_policy.py \
  --config configs/activation_search.yaml \
  --runs runs/activation

python scripts/08_export_and_validate.py \
  --policy runs/activation/<run_id>/selected_policy.yaml
```

所有 CLI 都需：

- `--help`
- 非零 exit code 處理錯誤
- logging 到檔案
- 不覆寫既有 run，除非明確 `--overwrite`
- config snapshot

---

## 12. 每個 run 的輸出

```text
runs/activation/<run_id>/
├── provenance.json
├── config_resolved.yaml
├── activation_manifest.json
├── baseline_metrics.json
├── region_sensitivity.csv
├── fit_metrics.csv
├── coefficients/
│   ├── hi_profile.json
│   ├── lo_profile.json
│   └── integer_codes.json
├── policy_candidates.csv
├── pareto_front.csv
├── selected_policy.yaml
├── checkpoints/
├── exports/
│   ├── model.pt
│   ├── model.onnx
│   └── graph_report.json
├── plots/
└── report.md
```

`report.md` 至少包含：

- baseline reproduction
- region sensitivity heatmap
- activation input distributions
- approximation curves與誤差
- policy Pareto front
- dual-head ablation
- quantization/fusion ablation
- backend latency/resource table
- 失敗案例與限制

---

## 13. 單元測試與驗收條件

### 13.1 數學測試

- `SiLU(-u) == SiLU(u) - u`
- folded objective 與完整雙側 objective 一致
- breakpoint continuity
- `g(0)=0`
- tail identity
- PoT/APoT code decode 正確

### 13.2 模型測試

- 能載入 YOLO26m 並建立 activation manifest
- 每個 eligible activation 有唯一 ID
- `act=False` 不被改動
- one2one / one2many 可分開替換
- 每個實驗互不污染
- baseline state_dict 可完整還原

### 13.3 匯出測試

- reference 與 ONNX output 在 tolerance 內一致
- proposed activation graph 無 Exp/Sigmoid/Div
- integer emulator 邊界測試通過
- fuse 後 one2many head 不在部署 graph

### 13.4 研究結果驗收

不得只報 activation 函數 MSE。最終方法至少必須同時具備：

1. task accuracy 結果
2. 目標 backend 效能或硬體資源結果
3. 與 standard activation、一般 PWL、PoT PWL、mixed activation baseline 比較
4. 關鍵組件 ablation
5. 三個 seed 或合理的變異估計
6. 可重現的 commit/config/checkpoint/dataset fingerprint

---

## 14. 必做 ablation matrix

| 類別 | 設定 |
|---|---|
| Uniform activation | SiLU / ReLU / Hardswish |
| Mixed standard | region-level standard activation search |
| PWL symmetry | full two-sided vs positive-half symmetry |
| Breakpoints | uniform vs empirical quantile vs optimized |
| Slopes | float vs PoT vs APoT2 |
| Profiles | per-region oracle vs two shared profiles |
| Search objective | curve MSE only vs task-aware selection |
| Head policy | H0 / H1 / H2 / H3 |
| Backward | exact PWL vs SiLU surrogate |
| Quantization | separate activation+requantization vs fused integer |
| Deployment | unfused model vs fused one-to-one deployment graph |

---

## 15. Prior-art 防線與論文敘事限制

以下單獨拿出來都不足以主張新穎性：

- YOLO 各層混合 activation
- PWL 近似 SiLU/Swish
- power-of-two 或 APoT slope
- 每層 activation search
- piecewise quadratic approximation
- forward approximation、backward smooth surrogate

較有機會形成 contribution 的組合是：

1. YOLO26m-specific region sensitivity，而非通用 layer NAS。
2. SiLU 正半邊恆等式約束，配合 empirical distribution folding。
3. 只用兩套共享 dyadic profile，降低係數與控制邏輯。
4. 利用 YOLO26 one2many/one2one 的部署不對稱性做 activation policy。
5. activation 與 requantization 的 integer-only fusion。
6. 在真實 target hardware 上證明 accuracy–latency–area/energy Pareto。

Codex 在程式、註解與報告中一律使用：

- `proposed candidate`
- `working hypothesis`
- `hardware-oriented`

在 prior-art 排查與完整結果完成前，不得使用：

- `novel`
- `first`
- `state of the art`
- `proven hardware-efficient`

---

## 16. 執行順序

### Milestone A：現在完成，不需要確認資料

1. 建立 package 與 config schema。
2. 實作 YOLO26m model inspector、dynamic region mapper。
3. 實作 activation replacement，正確處理共享 SiLU 與 dual head。
4. 實作候選 activation reference。
5. 實作 data template 與 validator。
6. 實作數學、replacement、integer emulator 單元測試。
7. 使用合成 histogram 與 toy tensor 做 dry-run。
8. 產出 `IMPLEMENTATION_STATUS.md`，列出尚未執行的正式步驟。

### Milestone B：確認資料到位後

1. 驗證 provenance 與資料完整性。
2. 重現 baseline。
3. 跑 region sensitivity matrix。
4. 擬合 per-region oracle PWL。
5. 擬合兩套共享 hi/lo profile。
6. 搜尋 region 與 dual-head policy。
7. short recovery，淘汰不佳候選。
8. full fine-tuning 與三 seed 驗證。
9. export、fuse、ONNX gate、integer golden test。
10. target hardware benchmark。
11. 產出 Pareto report 與論文 ablation table。

---

## 17. Codex 最終交付標準

Codex 完成本階段時，必須回報：

1. 新增與修改的檔案清單。
2. model inspector 找到的 activation 數量與 region 分布。
3. one2one / one2many 是否正確分開。
4. 所有測試結果。
5. 尚缺哪些真實資料欄位。
6. 一組可以直接執行的下一步命令。
7. 不得提供任何未由真實資料支持的「最佳 activation」結論。
