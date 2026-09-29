# YOLO26 Research — Codex 專用實驗規範與執行指令

> 用途：將本文件直接提供給 Codex，作為目前與後續 YOLO26 研究的專案級執行規範。
>
> 核心原則：**控制變因、可重現、可回溯、不可靜默改參數。**
>
> 本文件不要求立即開始正式訓練。除非使用者明確下令，Codex 應先完成專案盤點、設定標準化、相容性檢查、smoke test 與實驗配置。

---

# 0. 研究背景與目前方向

目前研究主體為 YOLO26 系列，現階段主要以 **YOLO26m** 為架構修改對象。

目前與預計研究方向包含：

1. YOLO26m baseline。
2. YOLO26m P3 特徵層修改。
3. P3 加入 MFAM。
4. MFAM 可能包含：
   - 3×3 + 5×5
   - 3×3 + 5×5 Partial 75%
   - 後續其他精簡、多分支或 partial-channel 版本
5. 後續會進行 **整體大架構調整**，不會只侷限在 P3。
6. 後續可能進行量化、部署或 inference efficiency 相關研究。
7. 除 object detection 外，可能需要驗證 **另一個資料級 / 任務級能力：Pose**。

Pose 相關既有內容位於：

```text
/home/uxin/yolo/original/pose/
```

使用者表示：
- 權重在此目錄中。
- 資料集相關內容也在此目錄中。

Codex 必須實際盤點此目錄，不可假定檔名、模型版本、dataset YAML 或 task 設定。

---

# 1. 最高優先級原則

## 1.1 Architecture ablation 必須控制變因

對 architecture experiment：

```text
Baseline
vs
MFAM
vs
MFAM Partial75
vs
其他架構版本
```

主要 independent variable 必須是：

```text
MODEL ARCHITECTURE
```

除非使用者明確要求，以下參數不得因模型版本不同而自行調整：

- dataset
- train/val/test split
- imgsz
- batch
- epochs
- patience
- optimizer
- lr0
- lrf
- momentum
- weight_decay
- warmup
- loss weights
- augmentation
- AMP
- seed
- deterministic
- validation protocol
- evaluation metric
- pretrained/fine-tuning policy

若某個模型因 VRAM、NaN、optimizer 不相容、shape mismatch 等原因無法使用既定設定：

**不得直接修改設定來讓它能跑。**

Codex 應先：
1. 找出原因。
2. 報告問題。
3. 提供可選解法。
4. 等使用者決定是否改變實驗條件。

---

# 2. 專案級文件與目錄規劃

Codex 應先檢查 repository 現況，再在不破壞既有結構的前提下，盡量整理成類似：

```text
project/
│
├── AGENTS.md
│
├── configs/
│   ├── detect/
│   │   ├── train_standard.yaml
│   │   ├── train_baseline.yaml
│   │   ├── train_mfam.yaml
│   │   ├── train_mfam_partial75.yaml
│   │   └── ...
│   │
│   └── pose/
│       ├── train_standard.yaml
│       ├── train_baseline.yaml
│       └── ...
│
├── models/
│   ├── detect/
│   │   ├── yolo26m.yaml
│   │   ├── yolo26m_mfam.yaml
│   │   └── ...
│   │
│   └── pose/
│       └── ...
│
├── tools/
│   ├── check_experiment_config.py
│   ├── inspect_model.py
│   └── ...
│
├── experiments/
│   ├── detect/
│   └── pose/
│
└── ...
```

如果 repository 已有清楚結構，不必為了符合這個範例而大規模搬檔。

原則是：

> 使用現有結構優先，新增必要的標準化層即可。

---

# 3. Canonical Training Configuration

## 3.1 Detection

建立或確認：

```text
configs/detect/train_standard.yaml
```

此檔案是所有 YOLO26m detection architecture experiments 的唯一 canonical training configuration。

任何：

- baseline
- P3 修改
- MFAM
- Partial MFAM
- neck redesign
- backbone redesign
- head redesign
- 大架構修改

都應從此標準衍生。

---

# 4. Detection 初始訓練標準

以下數值是目前研究預定基準。

但 Codex 在真正寫入 YAML 前，必須先確認 **目前 repository 所使用的 YOLO26 / Ultralytics 版本實際支援這些 key 與 optimizer**。

```yaml
task: detect
mode: train

epochs: 100
patience: 20

batch: 16
imgsz: 640

optimizer: MuSGD

lr0: 0.00038
lrf: 0.882

momentum: 0.948
weight_decay: 0.00027

warmup_epochs: 1.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

cos_lr: false

amp: true
deterministic: true
seed: 0

box: 9.83
cls: 0.65
dfl: 0.96

hsv_h: 0.013
hsv_s: 0.353
hsv_v: 0.194

degrees: 0.0
translate: 0.275
scale: 0.95
shear: 0.0
perspective: 0.0

flipud: 0.0
fliplr: 0.304

mosaic: 0.992
mixup: 0.427
copy_paste: 0.304

close_mosaic: 10

val: true
plots: true
save: true
save_period: -1
```

以下 runtime / I/O 欄位可依機器條件調整，但不得影響學習邏輯：

```yaml
device: 0
workers: 8
cache: false
project: ...
name: ...
```

---

# 5. 不得自行修改的參數

除非使用者明確要求，Codex 不得自行修改：

```text
epochs
patience
batch
imgsz

optimizer
lr0
lrf
momentum
weight_decay

warmup_epochs
warmup_momentum
warmup_bias_lr
cos_lr

box
cls
dfl

hsv_h
hsv_s
hsv_v
degrees
translate
scale
shear
perspective
flipud
fliplr
mosaic
mixup
copy_paste
close_mosaic

amp
seed
deterministic
```

尤其禁止下列行為：

```text
MuSGD -> SGD
MuSGD -> AdamW
MuSGD -> auto

batch 16 -> 8
batch 16 -> auto

imgsz 640 -> 512

epochs 100 -> 50

因為 MFAM 較大而降低 augmentation

因為模型較難收斂而自行增加 epoch

因為 loss 不漂亮而自行調 loss weight
```

這些都會破壞 architecture ablation 的公平性。

---

# 6. 允許因實驗改變的欄位

Architecture experiment 通常只允許改：

```text
model
name
```

必要時可以改：

```text
project
device
workers
cache
```

`data` 只有在使用者明確指定不同 dataset 時才可以改。

例如：

## Baseline

```yaml
model: models/detect/yolo26m.yaml
name: baseline
```

## MFAM

```yaml
model: models/detect/yolo26m_mfam.yaml
name: mfam_full
```

## MFAM Partial75

```yaml
model: models/detect/yolo26m_mfam_partial75.yaml
name: mfam_partial75
```

除了允許欄位，其他 training hyperparameters 必須完全一致。

---

# 7. MuSGD 規範

研究預定 optimizer：

```text
MuSGD
```

Codex 必須實際檢查目前 repository：

1. MuSGD implementation 是否存在。
2. trainer 是否能解析 `optimizer: MuSGD`。
3. optimizer registry / build_optimizer 是否已註冊。
4. checkpoint resume 是否支援。
5. AMP 是否相容。
6. distributed training 若存在，是否相容。

如果 MuSGD 不存在：

**禁止靜默改成其他 optimizer。**

回報：

```text
MuSGD implementation: FOUND / NOT FOUND
Registration: OK / MISSING
Trainer compatibility: OK / ERROR
```

並指出需要修改的檔案。

---

# 8. YAML Compatibility 規範

Codex 必須以 repository 目前安裝版本為準。

不得只根據網路文件或其他版本猜測。

對每個 training key：

1. 確認目前 parser/trainer 接受。
2. 確認值型態。
3. 確認 task-specific 行為。
4. 確認是否已 deprecated。
5. 確認是否被其他設定覆寫。

若不支援：

不得：

```text
靜默刪除
自行重新命名
自行改成相似參數
```

必須回報：

```text
Unsupported argument:
- key:
- current value:
- installed version:
- reason:
- suggested compatible alternatives:
```

---

# 9. Detection Dataset 規範

Codex 必須確認 detection dataset：

- dataset YAML 路徑
- train 路徑
- val 路徑
- test 路徑
- number of classes
- class names
- annotation format
- missing labels
- corrupted images
- train/val/test 是否有重疊
- 是否存在資料洩漏

禁止不同 architecture experiment 使用不同 split。

如果資料集有既定 train/val/test split，應鎖定。

---

# 10. Pose 任務 — 新增能力驗證

## 10.1 Pose 原始位置

Pose 相關內容位於：

```text
/home/uxin/yolo/original/pose/
```

此目錄可能包含：

- pretrained weights
- fine-tuned weights
- model YAML
- dataset YAML
- train/val/test dataset
- labels
- previous runs
- training scripts
- result logs

**不可假定實際內容。先盤點。**

---

# 11. Pose 第一階段只做盤點

在執行任何 pose training 前，Codex 先輸出：

```text
Pose directory audit
```

至少包含：

## 11.1 權重

找出：

```text
*.pt
*.pth
*.ckpt
*.onnx
*.engine
```

對每個可辨識權重報告：

- path
- file size
- model/task metadata
- model family
- class count（若可讀）
- keypoint count（若可讀）
- input size（若可讀）
- whether it is trainable checkpoint / inference export

---

## 11.2 Dataset YAML

找出所有：

```text
*.yaml
*.yml
```

並區分：

```text
model YAML
dataset YAML
training YAML
other config
```

對 pose dataset YAML 報告：

- path
- train
- val
- test
- number of classes
- class names
- kpt_shape
- flip_idx
- skeleton（若存在）
- root path
- 是否所有路徑存在

---

## 11.3 Dataset 結構

確認：

```text
images/train
images/val
images/test

labels/train
labels/val
labels/test
```

或實際使用的其他結構。

統計：

- train image count
- val image count
- test image count
- label count
- missing label count
- empty annotation count
- corrupted file count

---

# 12. Pose 與 Detection 必須分開

Detection 與 Pose 不得共用同一份 canonical training YAML。

建立概念：

```text
configs/detect/train_standard.yaml
configs/pose/train_standard.yaml
```

原因：

Pose task 可能具有不同：

- task
- loss
- augmentation
- keypoint settings
- model head
- pretrained checkpoint
- metric
- dataset format

因此禁止：

```text
直接把 detect training YAML 複製成 pose YAML 然後只改 task
```

Codex 必須先讀取 pose 既有訓練方式與目前 framework 支援情況，再建立 pose baseline。

---

# 13. Pose 的研究目的

目前 Pose 的目的不是取代 detection 主線。

它是一個 **額外能力 / 泛化能力測試**。

後續若進行大架構修改，希望知道：

> 新架構是否只對 detection 有效，還是對 pose task 也保有或改善能力。

因此未來可能形成：

```text
Original YOLO26m
├── Detection baseline
└── Pose baseline

Modified YOLO26m
├── Detection modified
└── Pose modified
```

如此才能研究：

```text
architecture modification
        ↓
Detection effect
        +
Pose effect
```

---

# 14. 大架構修改時的規範

後續不會只修改 P3。

可能涉及：

- backbone
- neck
- P3/P4/P5
- feature fusion
- multi-scale branch
- partial convolution
- channel allocation
- head
- detect head
- pose head
- shared feature extractor

因此 Codex 每次大架構修改前，必須先輸出「architecture delta」。

格式：

```text
Architecture Delta

Baseline:
- ...

Modified:
- ...

Changed components:
- Backbone:
- Neck:
- P3:
- P4:
- P5:
- Head:

Unchanged components:
- ...

Expected effect:
- Params:
- FLOPs:
- VRAM:
- receptive field:
- small-object feature:
- inference latency:
```

不能直接改完不說。

---

# 15. 模型結構變更前後必須做靜態檢查

修改架構後、正式 training 前必須確認：

1. model can be constructed
2. forward pass works
3. output tensor shapes correct
4. stride correct
5. Detect/Pose head receives expected feature maps
6. loss can execute
7. parameter count
8. GFLOPs / FLOPs（若工具可用）
9. CUDA memory basic estimate / actual smoke test
10. pretrained weights loading result

對 pretrained weights loading 必須報告：

```text
matched keys
missing keys
unexpected keys
shape mismatch keys
```

不得只寫「loaded successfully」而不檢查實際匹配狀況。

---

# 16. Smoke Test 規範

任何新架構正式長訓練前，先做 smoke test。

建議：

```text
1–3 epochs
小量資料或正常資料
正式 imgsz
正式 forward/loss
```

目的只檢查：

- dataloader
- model
- loss
- optimizer
- AMP
- validation
- checkpoint save
- resume
- NaN / Inf
- CUDA OOM

Smoke test 不可作為正式 accuracy 結論。

---

# 17. Early Stopping

Detection 初始標準：

```yaml
epochs: 100
patience: 20
```

意思：

- 最多 100 epochs
- 若 validation fitness 長時間無改善，可提前停止

不同 architecture 不得自行使用不同 patience。

若未來正式決定改為：

```text
epochs = X
patience = Y
```

所有同一組 ablation 必須一起修改。

---

# 18. Reproducibility

最低要求：

```yaml
seed: 0
deterministic: true
```

同一組初步 architecture screening：

```text
Baseline seed = 0
Modified seed = 0
```

若進入正式論文統計：

可考慮：

```text
seed = 0
seed = 1
seed = 2
```

最後報：

```text
mean ± std
```

但沒有使用者指示時，不要自行啟動多 seed 長訓練。

---

# 19. Pretraining 規範

若 baseline 使用 pretrained weights：

所有相同層級 architecture experiment 應盡可能使用相同 pretrained initialization policy。

如果新架構造成：

```text
部分 layer 無法載入 pretrained weights
```

Codex 必須報告：

```text
哪些層成功繼承
哪些層 random init
比例是多少
```

因為這可能成為實驗變因。

不得把：

```text
Baseline = pretrained
Modified = from scratch
```

當成公平 architecture comparison 而不註記。

---

# 20. Batch / VRAM 規範

預定：

```yaml
batch: 16
```

若新架構 OOM：

禁止直接：

```text
batch 16 -> 8
```

應先報：

```text
Model:
Batch:
imgsz:
GPU:
VRAM:
OOM location:
Baseline VRAM:
Modified VRAM:
```

可建議：

1. gradient accumulation
2. checkpointing
3. reduce batch
4. reduce model
5. multi-GPU

但任何會改變實驗條件的方案，都先交由使用者決定。

---

# 21. Experiment Config Validator

建議建立：

```text
tools/check_experiment_config.py
```

功能：

比較同一 ablation group 的 training YAML。

例如：

```text
train_baseline.yaml
train_mfam.yaml
train_mfam_partial75.yaml
```

允許不同：

```text
model
name
project
device
workers
cache
```

其餘若不同：

```text
ERROR
```

輸出例：

```text
Unexpected training configuration difference

field: batch
baseline: 16
mfam: 8

field: lr0
baseline: 0.00038
mfam: 0.001
```

Exit code 非 0。

---

# 22. 建議建立 Experiment Manifest

對每次實驗建立 manifest。

例如：

```yaml
experiment_id: detect_mfam_partial75_v1

task: detect
model: models/detect/yolo26m_mfam_partial75.yaml
train_config: configs/detect/train_standard.yaml
dataset: ...
weights: ...

architecture_change:
  location: P3
  module: MFAM
  branches:
    - 3x3
    - 5x5
  partial_ratio: 0.75

seed: 0

git_commit: ...
date: ...

notes: ...
```

目的：

之後能準確知道某個 `best.pt` 到底是怎麼訓練出來。

---

# 23. 結果至少記錄

Detection：

```text
mAP50-95
mAP50
Precision
Recall
Params
FLOPs
Latency / FPS（若測）
Peak VRAM
Training time
Best epoch
```

如果研究 small-object：

另外保存可取得的：

```text
AP_small
AP_medium
AP_large
```

或對應資料集可用的 size-based metrics。

Pose：

至少記錄 framework 原生主要 pose metrics，並清楚區分：

```text
box metrics
pose/keypoint metrics
```

---

# 24. 禁止用單一 best.pt 結果掩蓋設定差異

所有結果必須能追溯：

```text
best.pt
↓
experiment directory
↓
training YAML
↓
model YAML
↓
dataset YAML
↓
code version / git commit
```

如果缺失，Codex 應指出 reproducibility risk。

---

# 25. Codex 修改程式碼原則

修改前：

1. 先讀現有 implementation。
2. 找出 extension point。
3. 優先最小侵入式修改。
4. 不複製大量 upstream code。
5. 不破壞原始 YOLO26 baseline。

新 module 應盡量：

- 可獨立 import
- 可由 YAML 設定
- 有清楚 class 名稱
- 保留 baseline module
- 可做 ablation

---

# 26. Baseline 不可被覆寫

永遠保留原始 baseline。

例如：

```text
yolo26m.yaml
```

不要直接修改成 MFAM 版本。

應建立：

```text
yolo26m_mfam.yaml
yolo26m_mfam_partial75.yaml
```

如果 Python module 必須修改：

確保 baseline code path 仍然可執行。

---

# 27. 大架構研究命名規則

建議：

```text
baseline
mfam_full
mfam_partial75

arch_v1
arch_v2
arch_v3
```

進一步可以：

```text
arch_v1_p3_mfam
arch_v2_neck_mfam
arch_v3_backbone_neck
```

避免：

```text
test1
test2
new
new2
final
final2
```

---

# 28. Codex 不得自行開始正式長訓練

除非使用者明確說：

```text
開始訓練
run training
正式跑
```

否則預設工作到：

```text
code complete
config complete
validation complete
smoke-test ready / smoke-test complete
```

不主動開啟數十或數百 epoch 長訓練。

---

# 29. 第一個 Codex 任務

請現在執行以下工作，但 **不要開始正式長訓練**。

## A. Repository Audit

檢查：

- repository root
- YOLO/Ultralytics version
- current branch
- modified files
- existing model YAMLs
- existing training YAMLs
- training entry point
- optimizer implementation
- custom modules
- dataset YAMLs

不要破壞使用者目前未 commit 的工作。

---

## B. Detection Standardization

確認目前 YOLO26m baseline。

建立或整理：

```text
configs/detect/train_standard.yaml
```

並建立：

```text
configs/detect/train_baseline.yaml
configs/detect/train_mfam.yaml
configs/detect/train_mfam_partial75.yaml
```

如果 MFAM model 尚未存在：

可以先建立 placeholder / 清楚標註 TODO，
但不要虛構不存在的 architecture。

---

## C. MuSGD Audit

確認：

```text
MuSGD
```

是否真實可用。

輸出：

```text
Implementation:
Registration:
Trainer:
Checkpoint:
AMP:
Status:
```

---

## D. Training Argument Audit

確認 `train_standard.yaml` 所有 key：

```text
SUPPORTED
UNSUPPORTED
DEPRECATED
TASK-SPECIFIC
```

如果存在問題，先報告，不要靜默替代。

---

## E. Pose Audit

完整檢查：

```text
/home/uxin/yolo/original/pose/
```

至少列出：

```text
weights
dataset YAML
model YAML
training YAML
image folders
label folders
existing runs
```

並判斷：

1. 原本使用什麼 pose model。
2. 原本 task 是否為 pose。
3. dataset 是否可直接使用。
4. dataset keypoint format。
5. 權重可否直接 load。
6. 能否建立 YOLO26m Pose baseline。
7. 若要讓未來修改後的大架構同時支援 pose，需要哪些 architecture/interface 修改。

不要直接開始 pose training。

---

## F. Config Validator

建立：

```text
tools/check_experiment_config.py
```

至少可檢查：

```text
baseline vs MFAM vs MFAM Partial75
```

只有允許欄位能不同。

---

## G. Model Inspector

若 repository 尚無同類工具，可建立簡單 inspector：

```text
tools/inspect_model.py
```

可輸出：

- model name
- task
- parameter count
- trainable parameter count
- layer/module summary
- output feature shapes
- stride
- FLOPs（若容易取得）
- pretrained loading summary

---

# 30. 完成後的回報格式

完成第一階段後，請用以下結構回報：

## 1. Repository status

```text
Root:
YOLO/Ultralytics version:
Git branch:
Uncommitted changes:
```

## 2. Detection

```text
Baseline model:
Dataset:
Canonical training config:
```

## 3. MuSGD

```text
Implementation:
Registration:
Trainer compatibility:
AMP:
Resume:
Final status:
```

## 4. YAML compatibility

列出所有：

```text
supported
unsupported
deprecated
uncertain
```

## 5. Ablation configs

表格：

```text
experiment | model | train config | intentional differences
```

## 6. Pose audit

```text
Pose root:
/home/uxin/yolo/original/pose/

Weights found:
Dataset configs:
Model configs:
Train/val/test:
Classes:
Keypoints:
Current pose model:
Current usable checkpoint:
```

## 7. Architecture implications

說明：

```text
如果未來大架構修改後同時支援 detect + pose，
哪些部分可以 shared，
哪些 head/task-specific 部分需要分離。
```

## 8. Risks

列出：

- config risk
- dataset risk
- pretrained-weight risk
- VRAM risk
- compatibility risk
- reproducibility risk

## 9. Files changed

完整列出。

## 10. Next recommended action

只提出下一步建議。

**不要自動開始正式 training。**

---

# 31. 核心判斷準則

任何時候遇到不確定：

```text
不要猜。
先讀 repository。
```

任何時候遇到版本差異：

```text
以目前實際安裝版本與 source code 為準。
```

任何時候 architecture 跑不起來：

```text
先報原因。
不要偷偷改 training recipe。
```

任何時候做 ablation：

```text
只改研究變因。
其他條件固定。
```

任何時候進入 Pose：

```text
Pose 與 Detect 分開建立 baseline。
不要混用 task-specific training config。
```

任何時候準備正式長訓練：

```text
先 smoke test。
```

---

# 32. 最終研究結構目標

希望專案最後可以清楚支援：

```text
YOLO26 Research
│
├── Detection
│   ├── YOLO26m Baseline
│   ├── P3 MFAM
│   ├── MFAM Partial75
│   └── Large Architecture Variants
│
├── Pose
│   ├── Existing Pose Baseline
│   ├── YOLO26m-compatible Pose Baseline
│   └── Modified Architecture Pose Validation
│
├── Quantization / Deployment
│   └── future
│
└── Experiment Infrastructure
    ├── Canonical configs
    ├── Config validator
    ├── Model inspector
    ├── Reproducibility
    └── Experiment manifests
```

研究的最終目的不是只讓某個模型「能跑」。

而是讓所有實驗做到：

```text
可比較
可驗證
可重現
可追蹤
可擴充
```
