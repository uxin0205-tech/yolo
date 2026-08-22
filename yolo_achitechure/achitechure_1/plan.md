# plan.md — YOLO26m P3-MFAM 研究實作規範

## 目前執行狀態（2026-08-22）

本節記錄已落地且經實跑驗證的契約；若下方早期規劃文字與本節衝突，以本節與 `EXPERIMENT_SPEC.md`
為準。

- 實際架構名稱為 P3 MASF-inspired Full35／Partial75；Partial75 是 25% context、75% bit-exact bypass。
- DW3 與 DW5 相加後必須經 dense `C→C` 1×1 projection，再乘 alpha 做 residual。
- 正式環境固定 Python 3.12、PyTorch 2.11.0+cu128、Ultralytics 8.4.90；GPU 為 RTX 5060 Ti 16 GiB。
- Seed 0、imgsz 640、workers 6、AMP、MuSGD；A1／A2／B physical batch 16，C physical batch 8、
  `nbs=16`，validation batch 8。
- 基準 Phase C 是 70 epochs、patience 9；目前完整資料 Phase C queue 依使用者要求留下 manifest 化的
  patience 7 override。
- 30% B/C 與完整資料 B 已完成，所有 child 均 rollback 到各自 accepted A2。完整資料 Phase C 的
  Full35 嘗試在第一個 epoch 尚未產生 checkpoint 前失敗，不納入結論。
- 目前結論與機器可讀數字在 `results/rtx5060ti-half-0822.md`、`.json`、`.csv`。
- `final/` 已封裝 Full35／Partial75 程式碼、9 個 Bit-True 比較候選、8 個 Float checkpoint，以及
  COCO2017／BBT5 detect_dataset 的 9 × 2 完整 AP 矩陣。

## 0. 目的

本專案將以 **YOLO26m** 為基礎進行架構修改與實驗。

主要目標是在 **P3/8 特徵路徑**上加入輕量化多尺度上下文模組，以提升小物體偵測能力，尤其是 **Ball** 類別，同時盡量維持 YOLO26m 原有的訓練邏輯、控制運算成本，並確保第一輪實驗可以在約 **2–3 天內**得到可用結果。

本文件為 Codex 的主要實作規範。修改任何模型架構、訓練程式或設定前，請先遵守本文件。

---

## 1. 不可違反的限制

1. 基礎模型必須是 **YOLO26m**，並從 YOLO26m pretrained checkpoint 初始化。
2. **不要加入 P2 Detect head**。
3. **不要重新設計 P4/P5**。
4. **不要替換 YOLO26 原本的 Detect head**。
5. 除非後續明確要求，**不要修改 YOLO26 原本的 loss / assignment 邏輯**。
6. 保留 `end2end=True`。
7. 正式主實驗明確使用 **MuSGD**，不要使用 `optimizer=auto`。
8. 第一輪架構比較中，主要變因只能是 P3 enhancement module；不要同時加入大量 optimizer、loss 或 augmentation 變動。
9. 第一輪實驗必須控制在約 **2–3 天**內可取得結果。
10. 必須啟用 Early Stopping。
12. 不要自行猜測 `Partial75` 的意思，請依照第 4.2 節定義。

---

## 2. 修改前先檢查 Repository

開始修改程式碼前，先完成以下確認：

1. 記錄：
   - Ultralytics 版本或 git commit。
   - PyTorch 版本。
   - CUDA 版本。
   - GPU 型號與 VRAM。
2. 找到：
   - YOLO26 model YAML。
   - `C3k2`。
   - model parser / module registry。
   - YOLO detection trainer。
   - MuSGD 實作位置。
   - training configuration / default YAML。
3. 確認目前 YOLO26m 的實際計算圖。
4. 確認 P3/8、P4/16、P5/32 如何送入 `Detect`。
5. 確認目前 repository 版本是否已經包含：
   - YOLO26 end-to-end training。
   - MuSGD。
   - 此版本現有的 YOLO26 assignment / loss 行為，例如 STAL、progressive one-to-many weighting 等。

**不要重新實作 repository 中已經存在的功能。**

如果目前 repository 與官方 YOLO26 預期架構不同，請依照實際 feature graph 調整，不要只靠固定 layer index 硬改。

---

## 3. YOLO26m 預期的 P3 插入位置

依照目前官方 YOLO26 架構：

- Layer 16：P3/8 feature block。
- Layer 19：P4/16 feature block。
- Layer 22：P5/32 feature block。
- `Detect` 使用 `[16, 19, 22]`。

P3 enhancement 必須作用在：

> **P3 head block 的輸出之後，且在該 P3 feature 被後續 neck 與 Detect 使用之前。**

### Pretrained 權重相容性規則

優先考慮：

> **在原本 P3 block 的相同 graph index 上包裝或替換模組，而不是插入一個全新 layer 導致後方 index 全部位移。**

建議形式：

```text
原始：
layer 16 = C3k2 -> P3 feature

修改後：
layer 16 = C3k2P3MFAM -> enhanced P3 feature
```

修改後模組應盡量保留原始 `C3k2` 的行為與 parameter naming，之後再接 MFAM enhancement。

建議使用：

- subclass；
- wrapper；
- composition；

以確保原本 `C3k2` 的 pretrained weights 能正常載入，而只有新增 MFAM parameters 使用新初始化。

**不要因為新增 MFAM 而無意間丟失 P4/P5/Detect head 的 pretrained 權重。**

---

## 4. 第一輪要實作的兩個架構

第一輪只實作：

1. Baseline YOLO26m。
2. Full35。
3. Partial75。

不要額外加入第三、第四種 MFAM。

---

## 4.1 架構 A — Full35

暫定名稱：

```text
YOLO26m-P3-MFAM-Full35
```

對 P3 tensor：

```text
x: [B, C, H, W]
```

YOLO26m 在此位置預期 `C≈256`，但實作時應盡量從模型建構參數取得 channel 數，不要能避免時仍硬寫 `256`。

核心結構： 可參考MASF-YOLO An Improved YOLOv11 Network for Small Object第3-A節

```text
                 ┌─ DWConv 3x3 ─┐
x ───────────────┤              ├─ sum ─ conv 1x1 ─ alpha ─┐
                 └─ DWConv 5x5 ─┘                          │
                                                           +
x ──────────────────────────────────────────────-----------┘
```

數學表示：

```text
delta = Conv1x1(DW3(x) + DW5(x))
y = x + alpha * delta
```

### 必要條件

- 3x3 branch 使用 **Depthwise Convolution**。
- 5x5 branch 使用 **Depthwise Convolution**。
- 不改變 spatial size。
- 兩個 branch 以 **element-wise sum** 融合。
- 第一版不要使用 concat。
- 第一版不要加入完整 channel 的 `1x1 2C -> C` fusion convolution。
- branch sum 後必須有 dense `C→C` 1×1 projection；它不是 concat 後的 `2C→C` fusion。
- norm / activation 優先沿用 repository 內 YOLO 原本慣例。
- `alpha` 是可學習 scalar。
- `alpha` 初始值為 `0.01`。
- 輸入與輸出 shape 必須相同。

建議：

```python
alpha = nn.Parameter(torch.tensor(0.01))
```

---

## 4.2 架構 B — Partial75

暫定名稱：

```text
YOLO26m-P3-MFAM-Partial75
```

### 本專案目前正式定義

> **25% 的 P3 channels 進入 MFAM context path，另外 75% channels 做 identity bypass。**

若 `C=256`：

```text
context channels = 64
bypass channels  = 192
```


### 核心結構

```text
x
├─ 前 25% channels -> context path
│      ├─ DWConv 3x3
│      └─ DWConv 5x5
│          -> sum
│          -> 1x1 projection
│          -> alpha residual
│
└─ 後 75% channels -> identity bypass

concat(enhanced_context, identity_bypass) -> y
```

數學表示：

```text
x_ctx, x_id = split(x)

delta_ctx = Conv1x1(DW3(x_ctx) + DW5(x_ctx))
y_ctx = x_ctx + alpha * delta_ctx

y = concat(y_ctx, x_id, dim=1)
```

### 必要條件

- `partial_ratio = 0.75`。
- channel 數要依 `C` 安全計算。
- output channel 數必須與 input 相同。
- bypass channels 必須維持原值，不做額外卷積。
- 只有 context subset 使用 DWConv 3x3 / 5x5。
- 不使用 dynamic gate。
- 不使用 full-width 1x1 fusion。
- `alpha` 可學習且初始值為 `0.01`。

---

## 5. 為什麼第一版必須使用 Depthwise Branch

本研究目標不只有 accuracy，也包含：

- 訓練時間；
- inference cost；
- FPGA 硬體成本。

因此第一輪優先：

```text
DWConv 3x3
+
DWConv 5x5
+
element-wise sum
+
residual
```

而不是：

```text
standard Conv 3x3
+
standard Conv 5x5
```



---

## 6. Module 實作要求

不要把研究邏輯直接塞進大量 upstream 原始程式碼。

建議獨立成模組，例如：

```text
ultralytics/
  nn/
    modules/
      p3_mfam.py
```

建議 class：

```python
class P3MFAMFull35(nn.Module):
    ...

class P3MFAMPartial75(nn.Module):
    ...
```

如果考量 pretrained compatibility，採用 `C3k2` subclass / wrapper 更合適，則可使用：

```python
class C3k2P3MFAMFull35(C3k2):
    ...

class C3k2P3MFAMPartial75(C3k2):
    ...
```

不要在可以 inheritance / composition 的情況下複製整份 `C3k2` 原始實作。

只在必要位置加入：

- module import；
- parser registration；
- YAML support。

---

## 7. Pretrained 權重相容性測試

正式訓練前，必須證明修改後模型確實有使用 YOLO26m pretrained weights。

對每個修改後模型：

1. Instantiate modified model。
2. 載入 YOLO26m pretrained checkpoint。
3. 輸出：
   - transferred tensor 數量；
   - missing tensor 數量；
   - unexpected tensor 數量。
4. 確認 missing tensors 主要是新增的 MFAM parameters。
5. 如果 P4/P5/head 出現大量權重無法 transfer：
   - 停止訓練；
   - 先修正 graph / module naming / wrapper 設計。

**Pretrained transfer 尚未確認成功之前，不要直接跑 60–80 epochs。**

---

## 8. 必須通過的 Unit / Smoke Tests

### 8.1 Shape Test

建立代表性 P3 input：

```python
x = torch.randn(2, C, 80, 80)
```

確認：

```text
Full35(x).shape == x.shape
Partial75(x).shape == x.shape
```

---

### 8.2 Identity-safe Initialization Test

使用：

```text
alpha = 0.01
```

確認初始輸出沒有嚴重偏離原始 feature。

至少記錄：

```text
mean(abs(y - x))
std(y)
std(x)
```

此測試目的為診斷，不需要自行發明沒有實驗根據的 hard threshold。

---

### 8.3 Gradient Test

執行 dummy forward / backward，確認：

- `alpha.grad` 存在。
- `alpha.grad` finite。
- DW3 parameters 有 finite gradient。
- DW5 parameters 有 finite gradient。
- Partial75 bypass 沒有因錯誤實作導致整個主 graph detach。

---

### 8.4 Model Construction Test

確認：

- P3 shape 正常。
- P4 shape 正常。
- P5 shape 正常。
- Detect 仍然接收 3 個 scale。
- `end2end=True`。
- 沒有新增 P2 head。

---

### 8.5 Cost Report

對以下三個模型：

```text
Baseline
Full35
Partial75
```

至少記錄：

- parameters；
- FLOPs / GFLOPs（若現有工具支援）；
- P3 module MACs（可取得時）；
- smoke training step 的 peak CUDA memory；
- 短時間 warm benchmark 的平均 train-step time。

---

## 9. Training Policy

### 9.1 Optimizer

正式第一輪架構比較使用：

```yaml
optimizer: MuSGD
```

不要使用：

```yaml
optimizer: auto
```

目的：

> 在比較 Full35 與 Partial75 時，盡量保留 YOLO26m 原生訓練行為，並讓 architecture 成為主要變因。

第一輪不要另外實作複雜 differential learning-rate groups，除非目前 repository baseline 已經原生、穩定地使用這套設定。

---

## 9.2 YOLO26m 導向的初始訓練設定

建議起點：

```yaml
optimizer: MuSGD

lr0: 0.00038
lrf: 0.882
momentum: 0.948
weight_decay: 0.00027

warmup_epochs: 1.0

imgsz: 640
amp: true
end2end: true
```

第一輪不要改 YOLO26 現有 loss / assignment implementation。

---

## 9.3 Epoch 與 Early Stopping

目前 staged 主設定：

| Phase | Max epochs | Patience |
|---|---:|---:|
| A1 | 5 | 固定跑完 |
| A2 | 10 | 4 |
| B | 10 | 4 |
| C | 70 | 9（完整資料對照 queue 覆寫為 7） |

A2/B 若 Bit-True validation mAP50-95 連續 4 epochs 沒有改善就提早停止；Phase C 基準 patience 為 9。

不要使用：

```yaml
patience: 0
```

---

## 9.4 Batch Size

本機實測後的固定值：

```text
A1 / A2 / B: physical batch 16, nbs 16
C: physical batch 8, nbs 16, accumulate 2
Validation: batch 8
```

原則：

> 使用 GPU 可穩定支援的最大 batch，同時保留適當 VRAM headroom 給新增 P3 module。

最後結果必須記錄實際使用 batch size。

---


## 10. Augmentation Policy

架構比較不能被無關 augmentation 變動污染。

以下三者必須使用同一套 augmentation：

```text
Baseline
Full35
Partial75
```

如果目前專案已經有驗證過的 baseline augmentation，優先完全沿用。

如果必須建立一套偏小物體的 fine-tuning recipe，三個模型都要完全一致。

目前固定設定：

```yaml
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
scale: 0.5
close_mosaic: 10
```

注意：

> 這是目前專案的小物體導向設定，不要宣稱是 YOLO26m 官方 COCO pretraining recipe。

---

## 11. 第一輪實驗矩陣

| ID | Model | P3 Module | Context Channels | Seed |
|---|---|---|---:|---:|
| A0 | YOLO26m | None | 0% | 0 |
| A1 | YOLO26m-P3-MASF-Full35 | DW3 + DW5 + projection | 100% | 0 |
| A2 | YOLO26m-P3-MASF-Partial75 | DW3 + DW5 + projection | 25% | 0 |

以下條件必須保持一致：

```text
dataset
train/val split
imgsz
batch size（能固定時）
optimizer
LR schedule
loss
assignment
augmentation
max epochs
patience
seed
device configuration
```

---

## 12. 2–3 天 Runtime Budget

第一輪可用結果必須控制在約 **2–3 天**。

正式跑滿之前：

1. 先讓第一個修改後模型跑 5 epochs，或固定數量的代表性 training steps。
2. 計算平均 epoch time。
3. 估算：

```text
estimated_run_hours =
average_epoch_minutes * planned_epochs / 60
```

建議規則：

```text
估算 80 epochs <= 24 小時：
    保留 max_epochs = 80

估算 24–30 小時：
    max_epochs 降到約 70

估算 > 30 小時：
    max_epochs 降到約 60
    並檢查 module / data pipeline bottleneck
```

無論最大 epoch 如何調整：

```text
Early Stopping 必須保留
```

第一輪不要單純為了省時間就降低 input image size，除非後續明確要求。

---

## 13. 必須收集的 Metrics

不要只看 total mAP。

### Accuracy

至少收集：

- mAP50-95；
- mAP50；
- Ball AP50-95；
- Ball AP50；
- Ball Recall；
- 若資料集中有 Bat，記錄相關 Bat metrics。

### Cost

至少收集：

- parameters；
- GFLOPs / FLOPs；
- P3-MFAM MACs（可取得時）；
- training epoch time；
- inference latency；
- peak GPU memory。

### Training Behavior

記錄：

- best epoch；
- stop epoch；
- best validation fitness；
- train losses；
- val losses；
- 最終 `alpha`；
- 是否發生 NaN / Inf。

對 Partial75 額外記錄：

```text
context channel count
bypass channel count
```

---

## 14. Architecture Selection Rule

Full35 與 Partial75 先比較 Bit-True COCO2017 mAP50-95；差距在 `0.001` 內時，依序以同一張正式 GPU、
相同設定的 FP16 p50 latency、GFLOPs、Params 打破平手。Ball AP 是必須報告的研究指標，但不能繞過
overall gate 單獨選 winner。Peak GPU memory 只作容量診斷；目前正式比較 GPU 為 RTX 5060 Ti，但 VRAM
不參與 winner 排名。

例如：

> 如果 Partial75 只犧牲極少 accuracy，但能明顯降低 compute / latency，則 Partial75 可以優先成為最終硬體導向設計。

本研究真正要回答：

> **YOLO26m 的 P3 到底需要多少 contextual enhancement，才可以改善小物體偵測，同時避免不必要的運算與硬體成本？**

---

## 15. 第一輪禁止加入的內容

除非後續明確要求，第一輪不要實作：

- P2 head；
- P2 neck；
- P2-specific warmup；
- 7x7 branch；
- 9x9 branch；


第一輪重點是保持 controlled experiment。

---

## 16. Codex 必須交付的內容

完成後請提供：

1. 修改過的檔案清單。
2. Full35 module 實作。
3. Partial75 module 實作。
4. 三份 model/config：
   - baseline；
   - Full35；
   - Partial75。
5. 使用 MuSGD、A2/B `patience=4`、C 最多 70 epochs／基準 `patience=9` 的 training config / script。
6. Smoke-test script。
7. Pretrained weight transfer 驗證結果或驗證 script。
8. Parameters / FLOPs / latency benchmark script 或執行命令。
9. `RUNBOOK.md`，包含完整指令：
   - environment check；
   - smoke tests；
   - A0/A1/A2 training；
   - validation；
   - metrics export。
10. A0/A1/A2 結果表格模板。
11. 所有 implementation assumptions 與 unresolved conflicts。

---

## 17. Definition of Done

只有以下條件全部成立，才視為完成：

- YOLO26m baseline 可以正常 build / run。
- Full35 可以正常 build / run。
- Partial75 可以正常 build / run。
- P3 output shape 不變。
- Detect 仍接收 P3/P4/P5。
- `end2end=True` 保留。
- 明確使用 MuSGD。
- YOLO26m pretrained weight transfer 已驗證。
- 新增 MFAM parameters 有 gradient。
- 沒有大範圍 P4/P5/head pretrained transfer failure。
- Smoke inference 成功。
- 短程 smoke training 成功。
- Early Stopping 已設定。
- 完整實驗命令已寫入文件。
- 沒有新增 P2 或其他非本輪研究內容。

---

## 18. 遇到以下情況，不要猜，停止並回報

如果出現以下任一情況，先停止並回報：

1. Repository 中的 `Partial75` 被改成不是 25% context / 75% bypass。
2. 目前 repository 不支援 MuSGD。
3. 目前 YOLO26 graph 與預期 P3/P4/P5 差異很大。
4. 修改 YAML 後造成大量 pretrained weight transfer failure。
5. 找不到 dataset YAML。
6. 找不到或無法明確判定 Ball class。
7. 修改需求會迫使替換 YOLO26 loss / assignment / head 行為。
8. GPU VRAM 無法支援規劃中的 batch size。
9. Runtime projection 明顯超過 2–3 天。

不要自行改變研究設計來掩蓋上述問題。

---

## 19. 建議執行順序

```text
1. 檢查 repo / version / GPU
2. 確認 YOLO26m P3 graph
3. 實作 Full35
4. 實作 Partial75
5. 加入 module/parser registration
6. 驗證 pretrained transfer
7. 執行 shape / gradient / model smoke tests
8. Benchmark 額外 cost
9. 跑 5-epoch timing probe
10. 若沒有可直接比較的 baseline，訓練 A0
11. 訓練 A1 Full35
12. 訓練 A2 Partial75
13. 比較 Ball AP + overall mAP + compute
14. 選出 winner
15. winner 確定後才考慮額外 seeds 或後續最佳化
```

---

## 20. 目前專案決策摘要

### A1 — Full35

```text
YOLO26m + P3 Full35

100% P3 channels
DWConv 3x3
+
DWConv 5x5
+
sum fusion
+
1x1 projection
+
residual alpha = 0.01
```

### A2 — Partial75

```text
YOLO26m + P3 Partial75

25% P3 channels 進行 context enhancement
75% identity bypass

context path:
DWConv 3x3
+
DWConv 5x5
+
sum fusion
+
1x1 projection
+
residual alpha = 0.01
```

### Training

```text
pretrained YOLO26m
optimizer = MuSGD
Phase C max epochs = 70
patience = A2/B: 4, C baseline: 9
imgsz = 640
AMP = on
end2end = true
first screening seed = 0
```

### 研究優先順序

```text
Ball / small-object accuracy
+
overall detection accuracy
+
training / inference cost
+
未來 FPGA 適用性
```

---

## 21. Codex 執行原則

實作時遵守以下原則：

1. 先確認現況，再修改。
2. 盡量做最小侵入式修改。
3. 不要為了「看起來更完整」自行增加本文件沒有要求的新架構。
4. 不要在沒有驗證 pretrained transfer 的情況下開始長時間訓練。
5. 不要讓 architecture comparison 同時混入大量 training-policy 變因。
6. 遇到定義衝突或 repository 差異時，先回報，不要自行猜測。
7. 所有實驗都應可以透過明確命令重現。
