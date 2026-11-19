# 2026-08-27：Full35 Activation 全量實驗接手與啟動

> **後續修正：** 本紀錄中的 1.0× LR recovery 與 queue 已停止，不列正式結果。根因、正確
> COMBINE joint contract 與 sensitivity-first 狀態見
> [Full35 Recovery 修正紀錄](2026-08-27-full35-recovery-correction-sensitivity.md)。

## 任務與範圍

- 直接接手 `/home/uxin/yolo/yolo_combine/final/full35` 的 accepted J3 `best_joint`。
- 停用先前規劃中的 30% 抽樣，所有正式訓練與驗證均使用完整 COCO2017 與 Canonical BBAT5 v1。
- 補上真實模型的 activation 盤點、替換、驗證與 recovery seam，並把每個實驗部分的 epoch、
  batch、optimizer、LR、seed、patience 與 warm-up 固化為 machine-readable recipe。
- 本次未重新切分、抽樣、替換資料來源，也未改動影像或標註。

## 變更內容與原因

1. 新增 `training/full35/` 正式設定：
   - `activation-recipe.yaml`：完整實驗階段與預算。
   - `configs/joint.yaml`：Full35 loader/trainer 的資料、batch、augmentation、BN、XNOR 與 log 設定。
   - `configs/region-rules.yaml`：11 個靜態 activation regions。
   - `contracts/accepted-full35-baseline.yaml`：accepted checkpoint 與 BitTrue 指標契約。
   - `contracts/activation-manifest.yaml`：人工核准的 190 個 SiLU reference。
2. 新增 `Full35ActivationExperiment` 生產 adapter 與 `scripts/full35_activation.py` CLI，統一提供
   preflight、manifest、smoke、完整 validation 與 recovery。
3. 修正共用 activation 的盤點：Ultralytics 多個 Conv path 共用 stateless
   `Conv.default_act`，原本 `named_modules()` 會去重而只找到 31 個 module objects；改為
   `remove_duplicate=False` 後，能保留 190 個真實 path references。
4. validation graph 會由 Full35 source 重新 materialize，因此新增 policy-aware source adapter，確保
   Detect/Pose 的 Float 與 BitTrue graph 都套用同一份 manifest，而不是驗證時退回 SiLU。
5. recovery 每個候選都從同一個 accepted checkpoint 乾淨載入，重新建立 optimizer、scheduler、
   EMA 與 RNG，避免候選間污染。
6. 依深模組設計原則，把 Full35 bundle 的版本與資料細節封裝在 adapter 內，外部只需操作
   `preflight/build_manifest/load_policy_model/validate/run_recovery`；production loader 與測試使用同一
   個公開 seam。

## 固定資料與基線

| 項目 | 正式值 |
| --- | --- |
| COCO train／val | 118,287／5,000；`/home/uxin/yolo/coco2017.yaml` |
| BBAT5 train／val | 5,964／683；`bbat5-v1` formal assignment |
| BBAT5 Pose YAML | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml` |
| BBAT5 Detect YAML | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml` |
| checkpoint | `combine/final/full35/weights/combined/inference/best_joint.pt` |
| checkpoint SHA-256 | `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c` |
| data fraction／resampling | `1.0`／`false` |

原始 Full35 血緣為 J0 8、J1 20、J2 80、J3 20 epoch 上限；accepted checkpoint 是 global epoch
58，訓練在 global epoch 63 早停。Activation 候選不重跑這 128 epochs，而是由 accepted checkpoint
做 10-epoch J3 short recovery；只有 finalists 通過後才考慮 matched from-source confirmatory run。

## Activation 實驗配置

| 部分 | Epoch | Seed | Patience／warm-up | 備註 |
| --- | ---: | ---: | --- | --- |
| Baseline reproduction | 0 | 0 | 不適用 | 完整 val，Float／BitTrue |
| Profiling | 0 | 1 | 不適用 | 完整 train split，只統計不更新權重 |
| Uniform zero-shot | 0 | 1 | 不適用 | Hardswish、ReLU、Shift、Quality |
| Short recovery | 10 | 1 | 0／1 | SiLU sham + Hardswish + Shift + Quality，固定跑滿 |
| Region sensitivity | 10 | 1 | 0／1 | `poly_shift` one-region-at-a-time |
| BCSP policy search | 10 | 1 | 0／1 | 最多 8 個 mixed-policy trials |
| Finalist seed 1 | 20 | 1 | 5／3 | 最多兩個 finalists + SiLU |
| Finalist seed 2 | 20 | 2 | 5／3 | 資源允許才補 |
| Claim ablation | 20 | 1 | 5／3 | 最終 proposed policy controls |
| PTQ／QAT | 0／10 | 1 | QAT 0／1 | PTQ 超出 1 AP gate 才做 QAT；LR × 0.5 |

共同 recovery 使用 AdamW、weight decay `2.7e-4`、betas `(0.948, 0.999)`、AMP、gradient clip
`10`、cosine final factor `0.5`。J3 LR：backbone `3.8e-6`、neck `1.9e-5`、MASF `3.8e-5`、
attention `5e-7`、Detect/Pose heads `5e-5`。

Detect logical batch 128、physical microbatch 32；每 macro 2 個 logical Detect batches，即 8 次
physical forward、256 images。Pose batch 16、每 macro 1 batch；loss weights 為 Detect `1.0`、Pose
`0.25`。驗證 batch 為 Detect 32、Pose 16。Detect `fliplr=0.5`，Pose `fliplr=0`，兩者
`mosaic=0`；shared BN running statistics 凍結、affine 可訓練。

## 驗證方式與結果

### Production 與程式驗證

- 完整 preflight：`ready=true`，checkpoint/full-resume/release manifest hashes、正式資料路徑、
  Python/PyTorch/Ultralytics 與 GPU 均通過。警告只有 TensorBoard 未安裝，以及正式 run 仍需觀察
  VRAM gate；JSONL/CSV/PNG logging 可用。
- Production graph smoke：SiLU 190/190 unchanged；`poly_shift` 190/190 changed；Detect/Pose outputs
  都是 finite。
- 測試：`30 passed, 2 warnings`；兩個 warning 都是既有 Torch ONNX legacy exporter deprecation。
- Ruff：`ruff check .` 與 `ruff format --check .` 均通過。

### 基線重現

使用 Full35 原生 validator 跑完整 COCO val + BBAT5 formal val、Float + BitTrue，BitTrue 指標與 release
契約逐位相同：COCO box `0.4980223365`、BBAT box `0.6300355218`、BBAT pose
`0.9037171741`。結果位於 Full35 runtime 的
`evaluations/activation-silu-baseline-20260827/`。

### Uniform zero-shot（完整 validation split）

| Activation | BitTrue COCO box | Δ | BitTrue BBAT box | Δ | BitTrue BBAT pose | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SiLU baseline | 0.498022 | 0 | 0.630036 | 0 | 0.903717 | 0 |
| `poly_quality` | 0.498145 | +0.000123 | 0.625238 | -0.004798 | 0.902543 | -0.001174 |
| `poly_shift` | 0.495705 | -0.002317 | 0.611708 | -0.018327 | 0.896160 | -0.007557 |
| Hardswish | 0.420324 | -0.077698 | 0.470703 | -0.159332 | 0.876465 | -0.027252 |
| ReLU control | 0 | -0.498022 | 0 | -0.630036 | 0 | -0.903717 |

`poly_quality` 是唯一三個主要指標都在 `maximum_map50_95_drop=0.01` 內的零訓練候選。
`poly_shift` 的 COCO/pose 接近，但 BBAT box 尚未過 gate。Hardswish 需 recovery 才能判斷，ReLU 依
既定規則淘汰且不進昂貴 recovery。

### 已啟動的正式訓練

`short-recovery-uniform-silu-seed1` 已開始執行 10-epoch SiLU sham。實際 resolved config 顯示
`fraction=1.0`、`resampling=false`、J3 10 epochs、seed 1、patience 0、warm-up 1，且 shared、
Detect head、Pose head 都有 gradients。啟動後 GPU 約 99%、VRAM 約 21 GB；run 位於
`artifacts/runs/full35/short-recovery-uniform-silu-seed1/`。

已另啟動串行 queue；只有 SiLU completion marker 存在才依序執行 `poly_quality`、`poly_shift`、
Hardswish，各 10 epochs。任一 run 非零結束即停止，避免錯誤被後續實驗掩蓋；ReLU 不入列。

## 困難與解法

1. **共享 SiLU 路徑被 PyTorch 去重**：改以 `named_modules(remove_duplicate=False)` 盤點，並新增
   真實 Full35 regression test，確認 190 個 paths 都能被替換。
2. **validator 會重建 task graph**：用 policy-aware source 在 materialization boundary 套用 manifest，
   同時覆蓋 Float 與 BitTrue，並以全量指標確認候選確實生效。
3. **`apply_patch` 的 bwrap helper 無法建立 loopback**：新增檔仍優先使用 `apply_patch`；既有文件的
   小範圍同步改用精確字串替換，並以 Ruff、pytest 與內容檢查防止誤改。
4. **TensorBoard 未安裝**：不阻擋訓練；保留 JSONL、CSV 與 PNG，未擅自下載新依賴。

## 未解事項與風險

- SiLU sham 及三個候選的 10-epoch recovery 尚需依序完成；GPU 不平行跑，避免 VRAM/OOM 與不公平
  contention。完成前不能宣稱最終 activation 勝出。
- `poly_shift` 目前 BBAT box 超出 0.01 gate；是否能由 recovery 恢復仍未知。
- Hardswish zero-shot 落差大，但仍保留一次 stage-matched recovery；若恢復後仍超 gate 即停止。
- 真實 FPGA/ASIC latency、power、resource 尚無 target profile，因此目前只有 accuracy 結果，不能
  宣稱硬體加速。
