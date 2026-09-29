# YOLO11m 棒球小目標偵測：P2 / MFAM / Temporal Motion 實驗計畫書（Agent 執行版 v4）

> 本文件給 coding / training agent 直接執行。所有正式實驗必須可重現、可重新載入 checkpoint 驗證，且一次只改一個主要因素。

---

# 0. 研究目標

本研究以 **Ultralytics YOLO11m Detect** 為基礎，針對棒球偵測中的四個主要問題：

1. 球非常小，經下採樣後特徵容易消失。
2. 球快速移動，單張 RGB 影像資訊不足。
3. Motion blur 使球由圓形變成短線、長條或模糊斑點。
4. 球容易和燈光、白線、防護網交點、反光等靜態背景混淆。

研究主線：

```text
B1：YOLO11m + P2 Detect
        ↓
M0：P2 + Full MFAM（3/5/7/9）
        ↓
M1：P2 + Ball-MFAM-Lite（3/5）
        ↓
M2：M1 只處理 1/2 channels
M3：M1 只處理 1/4 channels
        ↓
BEST_PARTIAL = M2 / M3 最佳者
        ↓
S0：BEST_PARTIAL 放在 P2 + P3
        ↓
TM0 / TM1：檢驗 Temporal Motion Branch
```

---

# 1. 主要參考來源

## 1.1 MASF-YOLO（主要架構參考）

論文：
**MASF-YOLO: An Improved YOLOv11 Network for Small Object Detection on Drone View**

本研究採用／參考：
- P2 高解析度小目標 Detect layer。
- MFAM 的 3×3、5×5、1×7→7×1、1×9→9×1 多尺度 DWConv。
- 640×640。
- 100 epochs。
- SGD。
- initial LR = 0.01。
- momentum = 0.937。
- cosine annealing。

本研究不直接複製完整 MASF-YOLO，而是把 MFAM 限制於 P2/P3 做硬體友善化消融。

arXiv：
https://arxiv.org/abs/2504.18136

IEEE DOI：
https://doi.org/10.1109/SMC58881.2025.11342487

---

## 1.2 Ultralytics 官方

Training：
https://docs.ultralytics.com/modes/train/

Detect：
https://docs.ultralytics.com/tasks/detect/

Model YAML / Custom Module：
https://docs.ultralytics.com/guides/model-yaml-config/

Fine-tuning guide：
https://docs.ultralytics.com/guides/finetuning-guide

Ultralytics 官方重要原則：
- 優先從 pretrained `.pt` 開始，而不是全模型 random initialization。
- Custom YAML 可用 `.load("yolo11m.pt")` 載入所有 shape 相符的 pretrained weights。
- `freeze` 可接受 integer 或 layer list。
- 是否 freeze 取決於 target domain、dataset size 和新結構的位置。
- 可採 two-stage fine-tuning：先 freeze backbone 讓新 head/neck 適應，再 unfreeze all 並以較低 LR 微調。
- `warmup_epochs`、`cos_lr`、`close_mosaic`、`amp`、`seed`、`deterministic` 都是官方支援參數。

---

## 1.3 Partial-channel 參考

FasterNet / Partial Convolution：
https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html

Code：
https://github.com/JierunChen/FasterNet

只用於支持「不是所有 channels 都需要昂貴 spatial operation」的研究動機。

**PartialBallMFAM 是本研究自行提出的 adaptation，不可寫成 FasterNet 或 MASF-YOLO 原本模組。**

---

## 1.4 Temporal tiny-object 參考

WACV 2023：
https://openaccess.thecvf.com/content/WACV2023W/RWS/html/Corsel_Exploiting_Temporal_Context_for_Tiny_Object_Detection_WACVW_2023_paper.html

CVPRW 2023：
https://openaccess.thecvf.com/content/CVPR2023W/Anti-UAV/html/Yang_Video_Tiny-Object_Detection_Guided_by_the_Spatial-Temporal_Motion_Information_CVPRW_2023_paper.html

用於支持：
- tiny moving object 可以從前後幀 motion cue 獲益。
- temporal branch 不一定要使用 optical flow / Transformer。

本研究 TemporalMotionBranch 為硬體友善自行設計版本。

---


# 2. 全域實驗規則

1. Base model：YOLO11m Detect。
2. Formal input：640×640。
3. Dataset split 必須依 video / session / camera 分組，不可把相鄰幀亂數切到 train/val/test。
4. 所有正式 static variant 必須使用相同 train/val/test manifest。
5. 固定 seed：42。
6. `deterministic=True`。
7. `amp=True`。
8. optimizer 原則以 MASF 論文為主：SGD。
9. momentum：0.937。
10. Formal architecture ablation 不可同時改 optimizer、augmentation、loss、input size。
11. 每次訓練完必須：
    - save best.pt
    - 關閉原 Python process
    - 新 process reload
    - assert custom module / Detect scale 還存在
    - 再 val
12. 不允許只在 runtime monkey-patch module 後直接 `model.train()`。
13. Custom module 必須：
    - 寫入 Ultralytics source/module registry
    - 可在 YAML 直接宣告
    - 可被 checkpoint 正常 serialize / reload

---

# 3. Dataset Audit：訓練前一定先做

輸入影像全部依實際 `imgsz=640` letterbox/resize 規則換算後，統計：

- Ball bbox width histogram。
- Ball bbox height histogram。
- Ball short-side。
- Ball area。
- Ball aspect ratio。
- P2 cell size：bbox / 4。
- P3 cell size：bbox / 8。
- P4 cell size：bbox / 16。
- blur/streak proxy：aspect ratio > 2。
- 場景分類：
  - 室內 / 室外
  - 防護網
  - 強燈
  - 白線
  - 反光
  - 不同 camera angle

至少輸出：

```text
dataset_profile/
├── bbox_width_hist.png
├── bbox_height_hist.png
├── bbox_area_hist.png
├── p2_cell_hist.png
├── p3_cell_hist.png
├── p4_cell_hist.png
└── dataset_profile.json
```

目的：
- 資料統計只能「支持減少大 receptive field 的假設」。
- 是否真的能刪 7/9 必須由 M0/M1 ablation 證明。

---

# 4. B1：YOLO11m + P2 Detect Baseline

## 4.1 架構

原始：

```text
Detect(P3, P4, P5)
```

改成：

```text
Detect(P2, P3, P4, P5)
```

640 input：

```text
P2 = 160×160, stride 4
P3 =  80×80, stride 8
P4 =  40×40, stride 16
P5 =  20×20, stride 32
```

P2 neck：

```text
P3 top-down
    ↓ Upsample ×2
    ├───────────┐
Backbone B2 ────┤
                ↓
              Concat
                ↓
              C3k2
                ↓
           P2 Detect
```

---

# 5. B1 的正式訓練策略：總計 100 epochs

B1 是整個後續研究的基準，因此不要只跑 20～30 epochs。

採用：
- MASF-YOLO 的 100-epoch設定作為總訓練長度參考。
- Ultralytics 官方 two-stage fine-tuning 概念：先 freeze backbone，再 full fine-tune。

## 5.1 初始化

必須：

```python
model = YOLO("yolo11m_p2.yaml")
model.load("yolo11m.pt")
```

記錄：
- loaded keys
- missing keys
- unexpected keys
- P2 新增層哪些是 random init

不要假設 `pretrained=True` 會自動正確處理 custom YAML。

---

## 5.2 Phase A：P2 / Neck / Detect 適應期

**10 epochs**

目的：
- 讓新加入的 P2 路徑與 4-scale Detect 先穩定。
- 暫時保留 COCO pretrained backbone feature。

設定：

```yaml
epochs: 10
imgsz: 640
optimizer: SGD
lr0: 0.01
momentum: 0.937
warmup_epochs: 3
amp: true
seed: 42
deterministic: true
```

Freeze：
- freeze **實際 YOLO11m backbone**。
- Agent 不可直接硬寫 `freeze=10`。
- 先 `print(model.model.model)` / inspect YAML。
- 找出當前 Ultralytics 版本 backbone 的完整 module indices。
- 只 freeze backbone，neck、P2 path、Detect 必須 trainable。

YOLO11 常見 backbone 結束於 C2PSA，但 agent 必須依實際安裝版本確認 index。

輸出：
`B1_phaseA_best.pt`

---

## 5.3 Phase B：全模型解凍 fine-tuning

從 `B1_phaseA_best.pt` 載入。

**90 epochs**

設定：

```yaml
epochs: 90
imgsz: 640
optimizer: SGD
lr0: 0.001
lrf: 0.01
momentum: 0.937
cos_lr: true
warmup_epochs: 3
close_mosaic: 10
freeze: null
amp: true
seed: 42
deterministic: true
```

目的：
- 新 P2 已經有初步能力。
- 再讓 backbone 一起適應棒球 domain。
- 使用比 Phase A 小 10× 的 LR 避免大幅破壞 pretrained feature。

總訓練量：

```text
10 freeze
+
90 full fine-tune
=
100 epochs
```

此 10+90 配置是本研究設定；如果 dataset 很大、domain 與 COCO 差異明顯，也需保留一個 `freeze=None from start` 的 diagnostic run 作確認，但正式主線先以 10+90 為主。

---

# 6. B1 必須通過的 Gate

- 640 forward 正常。
- Detect 有 4 scales。
- stride 包含 4。
- `model.info()` FLOPs 非 0。
- save/reload 後仍有 P2。
- training curve 沒有 NaN。
- val mAP 在最後 20 epochs 不再劇烈震盪。
- 保存：
  - mAP50-95
  - mAP50
  - mAP75
  - AP_S / mAP(S)
  - AP_M / mAP(M)
  - AP_L / mAP(L)
  - Precision
  - Recall
  - Ball AP
  - Ball Recall
  - Ball AP_S / Ball AP_M / Ball AP_L（若樣本數足夠）
  - tiny-ball Recall
  - Params
  - GFLOPs

---

# 7. M0：Full MFAM @ P2

位置：

```text
Backbone B2
    ↓
Full MFAM
    ↓
P2 Neck Fusion
    ↓
P2 Detect
```

Full MFAM：

```text
X
├─ DWConv 3×3
├─ DWConv 5×5
├─ DWConv 1×7 → DWConv 7×1
├─ DWConv 1×9 → DWConv 9×1
└─ Identity
      ↓
     Add
      ↓
projection / residual
```

目的：
- 作為 MASF-inspired full context reference。
- 測試 7/9-equivalent branch 在棒球資料上是否值得。

---

# 8. M1：Ball-MFAM-Lite @ P2

M1 只刪：

```text
1×7 → 7×1
1×9 → 9×1
```

保留：

```text
X
├─ DWConv 3×3
├─ DWConv 5×5
└─ Identity
      ↓
     Add
      ↓
same projection/residual skeleton
```

**M0 與 M1 除 kernel branch 外不可改其他東西。**

比較：

```text
M0 vs M1
```

回答：

> 7/9 大 receptive-field branch 是否真的提供足夠收益？

---

# 9. M2 / M3：都以 M1 架構為基礎

## 9.1 M2

```text
X：C channels
│
├─ C/2 → M1 Ball-MFAM-Lite
│
└─ C/2 → Identity
          ↓
        Concat
```

```python
xp, xb = split(x, [C//2, C//2], dim=1)
yp = BallMFAMLite(xp)
y = torch.cat([yp, xb], dim=1)
```

---

## 9.2 M3

```text
X：C channels
│
├─ C/4  → M1 Ball-MFAM-Lite
│
└─ 3C/4 → Identity
           ↓
         Concat
```

```python
xp, xb = split(x, [C//4, 3*C//4], dim=1)
yp = BallMFAMLite(xp)
y = torch.cat([yp, xb], dim=1)
```

---

## 9.3 重要

「M2/M3 用 M1 去做」指：

> M2/M3 的 processed branch 結構就是 M1 Ball-MFAM-Lite。

**不代表正式實驗一定從 M1 checkpoint 繼承。**

為公平比較：

```text
M0
M1
M2
M3
```

全部從同一個 `B1 final best.pt` 初始化所有可對應的 pretrained layers。

新增 MFAM module 的 weight 分別 random init。

---

# 10. M0/M1/M2/M3 正式訓練

因為：
- B1 已經穩定。
- MFAM 插在 backbone B2。
- 如果 freeze 整個 backbone，新增 MFAM 也會被凍住。

所以正式 M0-M3 **預設不 freeze backbone**。

設定：

```yaml
epochs: 100
imgsz: 640
optimizer: SGD
lr0: 0.001
lrf: 0.01
momentum: 0.937
cos_lr: true
warmup_epochs: 3
close_mosaic: 10
freeze: null
amp: true
seed: 42
deterministic: true
```

理由：
- 與 B1 Phase B 同一級 low-LR fine-tuning。
- 保證新增 MFAM 可以學。
- Formal ablation 全部固定 100 epochs，比較最乾淨。

在正式 100 epoch 前，每個 variant 先做：

```text
3～5 epoch engineering smoke test
```

只檢查：
- forward/backward
- loss finite
- checkpoint 可 reload

smoke test 不可拿來做論文結果。

---

# 11. BEST_PARTIAL：M2 / M3 選擇

只比較：

```text
M2 vs M3
```

選擇順序：

1. mAP50-95。
2. AP_S / mAP(S)。
3. Ball Recall。
4. Ball AP_S（若樣本數足夠）。
5. Tiny-ball Recall。
6. Motion-blur proxy Recall。
7. GFLOPs。
8. Params。
9. peak activation。
10. hardware operator / memory cost。

若：

```text
|M2 mAP - M3 mAP| <= 0.2 AP
```

而 Ball Recall 差 < 1 percentage point：

> 優先選 M3（1/4 processed），因為硬體成本更低。

記錄：

```text
BEST_PARTIAL = M2 或 M3
```

---

# 12. S0：BEST_PARTIAL 再加到 P3

S0 定義：

```text
P2 → BEST_PARTIAL
P3 → BEST_PARTIAL
P4 → unchanged
P5 → unchanged
```

例如 M3 勝：

```text
B2 → PartialBallMFAM(r=0.25)
B3 → PartialBallMFAM(r=0.25)
```

不是把 P2 的 module 移掉。

是：

```text
P2 保留
+
P3 新增一個
```

訓練：

```yaml
epochs: 100
optimizer: SGD
lr0: 0.001
momentum: 0.937
cos_lr: true
warmup_epochs: 3
close_mosaic: 10
freeze: null
```

初始化：

```text
從 BEST_PARTIAL checkpoint 載入相符 weight
P3 新 module random init
```

S0 保留條件建議：

```text
+0.2 mAP50-95
或
+1.0 percentage point Ball Recall
```

且 GFLOPs 增量不應過大。

若沒有實質收益，最終 spatial model 保留 P2-only。

---

# 13. Temporal Motion Branch：正式改成硬體版

## 13.1 功能

Temporal branch 不直接偵測球。

它提供：

> 前一幀到目前幀「哪裡發生變化」的 motion cue。

對固定棒球場：

```text
燈光 / 白線 / 網子 → 多數為靜態
球                → 快速位移
```

因此可協助 RGB branch 區分：

```text
moving small ball
vs
static ball-like background
```

---

# 14. Temporal Motion Branch 輸入

目前幀：

```text
I_t
```

前一幀：

```text
I_(t-1)
```

先灰階：

```text
G_t
G_(t-1)
```

Difference：

```text
D_t = abs(G_t - G_(t-1))
```

第一幀：

```text
previous = current
```

不可跨 video / camera pairing。

---

# 15. Temporal Motion Branch：硬體友善架構

依本次指定，正式架構改成：

```text
D_t：640×640×1
        ↓
Conv 3×3, stride=2
+ BN + ReLU
        ↓
320×320×Cmid
        ↓
DWConv 3×3, stride=2
+ BN + ReLU
        ↓
160×160×Cmid
        ↓
Conv 1×1
+ BN
(no activation before fusion)
        ↓
160×160×Cp
```

即：

```text
Conv 3×3 stride2
        ↓
DWConv 3×3 stride2
        ↓
1×1 Conv
```

固定原則：

```text
Cmid = 16
Cp = BEST_PARTIAL 的 processed channels
```

若 `Cp < 16`：

```text
Cmid = Cp
```

要求 channels 可被硬體向量寬度整除時，優先使用 8/16 的倍數。

---

# 16. Motion 與 PartialBallMFAM 的接法

假設 M3 勝：

```text
B2 Feature
├─ Xp = 1/4 channels
└─ Xb = 3/4 channels
```

Motion branch 只輸出 `Cp` channels。

融合：

```text
Xp_motion = Xp + alpha * M_t
Yp = BallMFAMLite(Xp_motion)
Y = Concat(Yp, Xb)
```

其中：

```text
alpha = learnable scalar
```

初值：

```text
alpha = 0.1
```

部署時 alpha 是固定常數，不需要動態 attention。

禁止第一版加入：
- Softmax。
- Sigmoid gate。
- Optical flow。
- Temporal Transformer。
- feature warping。
- deformable convolution。

原因：
- 保持 operator set 為 Conv / DWConv / 1×1 / Add / Sub / ABS。
- 方便 FPGA streaming / operator reuse。

---

# 17. TM0 / TM1：Temporal 為什麼需要兩組

## TM0：Temporal Control

架構：

```text
BEST_PARTIAL
```

Data loader：
- 會讀 current / previous pair。
- current/previous 做同步 transform。
- 但 previous 不送進 model。
- 無 motion branch。

目的：
- 控制 temporal data pipeline 與 augmentation 改變。

---

## TM1：真正 Temporal Motion

和 TM0 完全相同：

- dataset。
- split。
- optimizer。
- epochs。
- augmentation。
- initialization。

唯一多：

```text
previous/current
        ↓
difference
        ↓
Conv3×3 s2
        ↓
DWConv3×3 s2
        ↓
1×1 Conv
        ↓
P2 partial channels fusion
```

所以：

```text
TM1 - TM0
```

才能歸因於 motion cue。

---

# 18. Temporal augmentation

Temporal 不可直接使用普通單張 Mosaic。

因為 previous/current 必須幾何對齊。

第一版：

```text
mosaic = 0
mixup = 0
```

可使用：
- resize / letterbox：兩張完全相同參數。
- horizontal flip：兩張同步。
- affine：兩張同步。
- photometric transform：第一版也同步。
- labels 只對 current 更新。

所有 TM0/TM1 使用完全一致 augment。

---

# 19. TM0 / TM1 訓練

初始化：

```text
都從 BEST_PARTIAL final best.pt 開始
```

TM0 沒有新 motion weights。

TM1：
- 只新增 motion branch random init。
- 其他 matching weights 從 BEST_PARTIAL 載入。

正式先跑：

```yaml
epochs: 50
imgsz: 640
optimizer: SGD
lr0: 0.001
lrf: 0.01
momentum: 0.937
cos_lr: true
warmup_epochs: 3
mosaic: 0
mixup: 0
freeze: null
amp: true
seed: 42
deterministic: true
```

不 freeze：
- motion branch 必須學。
- P2 spatial feature 也要一起適應 motion cue。
- TM0/TM1 必須保持相同 freeze policy。

## 是否延長到 100 epochs

如果 TM0 或 TM1 在 epoch 46～50 仍持續刷新 best mAP / Ball Recall：

> TM0 與 TM1 兩組一起延長到 100 epochs。

不可只延長表現較好的那一組。

---

# 20. TM1-shuffle（推薦、免重新訓練）

TM1 train 完後：

正常：

```text
frame 100 ← frame 99
```

shuffle：

```text
frame 100 ← random wrong frame
```

若：

```text
TM1 > TM0
且
TM1-shuffle ≈ TM0
```

表示提升確實依賴正確 temporal correspondence。

---

# 21. 硬體 profiling 必做

每個模型輸出：

- Params。
- GFLOPs。
- MACs。
- Peak activation。
- P2 activation size。
- DWConv 次數。
- 1×1 Conv 次數。
- operator list。
- estimated feature traffic。

Temporal 額外記錄：

```text
previous grayscale frame buffer：
640×640×8 bit ≈ 0.39 MiB
```

60 FPS 時 previous-frame read + current-frame write 的灰階 frame-buffer traffic 需另外估算。

Motion operator：

```text
Gray
Sub
ABS
Conv3×3
DWConv3×3
Conv1×1
Add
```

不得把 optical flow 算進第一版。

---

# 22. Metrics

每個正式 run 必須同時輸出整體 COCO-style 指標與棒球專用指標。

## 22.1 整體 Detection 指標

```text
mAP50-95
mAP50
mAP75
Precision
Recall
per-class AP
```

## 22.2 必須提供 mAP(S/M/L)

正式報告必須額外提供 COCO-style：

```text
AP_S / mAP(S)
AP_M / mAP(M)
AP_L / mAP(L)
```

尺寸定義採 COCO evaluation 慣例，以 bounding-box area 分組：

```text
Small  : area < 32^2 pixels
Medium : 32^2 <= area < 96^2 pixels
Large  : area >= 96^2 pixels
```

重要：
- 此 S/M/L 分組應依「送入 evaluator 的影像座標尺度」一致計算，Agent 不可自行混用原始解析度與 640 resize 後尺度。
- 若使用 COCO JSON / pycocotools evaluator，直接保存其 `AP_small / AP_medium / AP_large`。
- 若目前 Ultralytics validation 不直接輸出 AP_S/M/L，額外匯出 prediction JSON，再使用 COCO-style evaluator 計算。
- 報告中需明確標註這是 **COCO-style object-size AP**，避免與下面自行定義的 baseball tiny/small bins 混淆。

## 22.3 Ball-specific 指標

```text
Ball AP
Ball Recall
Ball AP_S
Ball AP_M
Ball AP_L
```

若 evaluator 支援 category + area-range 同時篩選，需另外計算 baseball 類別自己的 AP_S/AP_M/AP_L。

若資料集中 baseball 幾乎全部落在 Small，導致 Ball AP_M/AP_L 樣本數過少，必須同時輸出每個 size bin 的 GT instance count，禁止只報 AP 而不報樣本數。

## 22.4 Baseball 自訂細尺寸分析

由於棒球往往遠小於 COCO 的 32×32 small threshold，因此另外保留更細的 baseball-only bins：

```text
Tiny-ball Recall：short side < 8 px
Small-ball Recall：8–16 px
Larger-ball Recall：>16 px
Motion-blur proxy Recall：aspect ratio > 2
FP / 1000 frames
FN / 1000 ball frames
Longest consecutive miss streak
```

這些自訂 bins 是應用分析，不可稱為 COCO AP_S/AP_M/AP_L。

Hard negatives：
- 燈
- 白線
- 網子交點
- 反光
- 球棒亮點
- 鞋子
- helmet highlight

---


## 22.5 每個正式 run 的結果 JSON 至少包含

```json
{
  "map50_95": null,
  "map50": null,
  "map75": null,
  "ap_small": null,
  "ap_medium": null,
  "ap_large": null,
  "precision": null,
  "recall": null,
  "ball_ap": null,
  "ball_recall": null,
  "ball_ap_small": null,
  "ball_ap_medium": null,
  "ball_ap_large": null,
  "gt_count_small": null,
  "gt_count_medium": null,
  "gt_count_large": null,
  "ball_gt_count_small": null,
  "ball_gt_count_medium": null,
  "ball_gt_count_large": null
}
```

若某一 size bin 沒有足夠 GT，欄位填 `null` 並在報告註明，不得填 0 冒充有效 AP。

---

# 23. Formal Experiment Matrix

| ID | 架構 | Epoch | 初始化 | Freeze |
|---|---|---:|---|---|
| B1-A | YOLO11m + P2 | 10 | yolo11m.pt | Backbone |
| B1-B | YOLO11m + P2 | 90 | B1-A best | None |
| M0 | B1 + Full MFAM @P2 | 100 | B1 best | None |
| M1 | B1 + 3/5 BallMFAMLite @P2 | 100 | B1 best | None |
| M2 | M1 on 1/2 channels @P2 | 100 | B1 best | None |
| M3 | M1 on 1/4 channels @P2 | 100 | B1 best | None |
| S0 | BEST(M2,M3) @P2 + @P3 | 100 | BEST_PARTIAL best | None |
| TM0 | BEST_PARTIAL + paired data, no motion | 50→100 if needed | BEST_PARTIAL best | None |
| TM1 | TM0 + Motion Branch | 50→100 if needed | BEST_PARTIAL best | None |

---

# 24. Agent 實作順序

## Task 1：環境鎖定
- Git commit。
- Ultralytics version。
- PyTorch/CUDA。
- GPU。
- seed。
- 建立 experiment manifest。

## Task 2：Dataset Audit
- sequence split。
- bbox statistics。
- P2/P3 cell histogram。
- blur proxy。
- 統計 COCO-style S/M/L GT instance count。
- 統計 baseball 類別在 S/M/L 各 bin 的 GT instance count。

## Task 3：B1 P2
- custom YAML。
- 4-scale Detect。
- load yolo11m.pt。
- 10 epoch backbone freeze。
- 90 epoch full fine-tune。
- new-process reload/val。

## Task 4：MFAM modules
- FullMFAM。
- BallMFAMLite。
- PartialBallMFAM。
- unit tests。

## Task 5：M0/M1/M2/M3
- smoke test。
- 100 epoch formal。
- identical settings。

## Task 6：BEST_PARTIAL
- M2/M3 比較。
- accuracy + hardware selection。

## Task 7：S0
- best partial 保留 P2。
- 同一模組新增 P3。
- 100 epochs。

## Task 8：Temporal dataset
- previous/current pairing。
- synchronized transforms。
- difference map。

## Task 9：TemporalMotionBranch
- Conv3×3 s2。
- DWConv3×3 s2。
- 1×1 Conv。
- fuse only processed P2 channels。

## Task 10：TM0/TM1
- same paired pipeline。
- 50 epochs。
- 未收斂則兩組一起 100。
- optional TM1-shuffle。

## Task 11：Hardware Profile
- FLOPs / MACs / activation / operator / bandwidth。

## Task 12：Final Report
- architecture tables。
- training curves。
- mAP50-95 / mAP50 / mAP75。
- AP_S / AP_M / AP_L。
- Ball AP / Ball Recall。
- Ball AP_S / Ball AP_M / Ball AP_L（並附各 bin GT 數量）。
- tiny-ball / 8–16 px / >16 px 分析。
- blur proxy。
- FP/FN。
- hardware cost。
- negative results 也必須保留。

---

# 25. 最後研究論述

最終論文故事應是：

```text
P2：
避免極小棒球在 stride=8 前消失
        ↓
Full MFAM：
先建立論文對照
        ↓
Ball-MFAM-Lite：
測試固定棒球場是否不需要 7/9 大 context
        ↓
Partial MFAM：
測試是否只需部分 channels 進行 multi-scale processing
        ↓
P3 extension：
驗證更深層是否仍需要 MFAM
        ↓
Temporal Motion：
利用「球會動、燈/白線/網子多半不動」的時間線索
```

不可在實驗前宣稱：
- 7/9 一定沒用。
- 1/4 channels 一定最佳。
- temporal 一定提升。

所有結論以消融與實測為準。
