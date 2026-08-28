# architecture_2 C0–C3 正式實驗報告

狀態：尚未執行。此範本不得預填虛構精度。

## 1. 結論摘要

- Handoff revision：full35-final-j3-seed0-d67fb45c
- yolo_combine winner：full35-j3-best-joint-seed0
- Fusion kind：shared_dual_head
- 實際 resolved candidates：
- Detect 是否完成：
- Pose 是否經使用者 opt-in 並完成：
- Formal comparison ready：
- C_best：null（等待使用者決定）
- 量化核准候選：

先寫觀察到的精度—成本關係，不使用 PASS／REJECT 自動替使用者選型。

## 2. Handoff 與可重現性

| 欄位 | 值 |
|---|---|
| spec version／SHA256 | |
| yolo_combine revision／winner | |
| fusion kind | |
| checkpoint SHA256 | |
| builder SHA256 | |
| architecture SHA256 | |
| training recipe SHA256 | |
| COCO／BBAT5 dataset SHA256 | |
| selection／fresh-process report SHA256 | |
| torch／Ultralytics | |
| git revision | |
| hardware | |

附上 materialized graph 驗收、state-dict strict reload、detect／pose／both 輸出等價與所有
protected／frozen path 證據。

## 3. 候選矩陣與 Architecture Delta

| Candidate | Region | 唯一變更 | Changed paths | Protected paths | Transfer summary |
|---|---|---|---|---|---|
| C0-Handoff | | 無，不訓練 | | | |
| C0-Control | | 無，相同恢復預算 | | | |
| C1 | | e 0.5 → 0.375 | | | |
| C2 | | inner_n 2 → 1 | | | |
| C3 | | 3x3_3x3 → 1x1_3x3 | | | |

本輪region固定為`shared-c3k2`，changed paths固定為`graph.model.6/8/13/19`。記錄
matched、missing、unexpected、shape-mismatch tensors；融合topology、heads、MASF、BinaryQK、
PWL attention必須不變，不加入routed／partial-shared候選。

## 4. 資料與公平性

- COCO2017 Detect：COCO80，應用主要關注 person。
- BBAT5 Pose：ball／bat，kpt_shape=[2,3]。
- BBAT5 2-class Detect：只作 paired diagnostic。
- BBAT5 version：bbat5-v1。
- formal train／val：5,964／683 images。
- search train／val：5,364／600 images，只取 formal train。
- architecture-screen-20-v1：COCO train/search-val 23,657／5,000；BBAT5 train/search-val 1,073／600；screening manifest SHA256：。
- group leakage：0。
- patched negative coordinates：4。
- test split：null。

列出每個候選完全相同的 optimizer steps、validation events、seed、head seed、batch、fraction、
scale、cache、imgsz、task ratio、augmentation 與 freeze policy。Runtime 的 device、workers、
project、name、cache 差異需揭露但不得改變 learning budget。
另列Detect logical128、selected physical32或fallback16、accumulation、Pose train16，以及
Detect/Pose validation16；不可只寫一個含糊的batch欄位。

## 5. 精度與 F1

### 5.1 COCO Detect

| Candidate | mAP50 | mAP50-95 | person AP50 | person AP50-95 |
|---|---:|---:|---:|---:|
| C0 | | | | |
| C1 | | | | |
| C2 | | | | |
| C3 | | | | |

### 5.2 BBAT5 Pose

若使用者未啟用 Pose，填寫「未執行；完整融合排名尚未成立」，不要填 0。

| Candidate | Box mAP50 | Box mAP50-95 | Keypoint mAP50 | Keypoint mAP50-95 | Official combined fitness | Research checkpoint selection |
|---|---:|---:|---:|---:|---:|---|
| C0 | | | | | | |
| C1 | | | | | | |
| C2 | | | | | | |
| C3 | | | | | | |

### 5.3 Per-class 與 F1

| Candidate | Class | Box AP50 | Box AP50-95 | Kpt AP50 | Kpt AP50-95 | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | ball | | | | | | | |
| C0 | bat | | | | | | | |

| Candidate | Fixed confidence threshold | Macro F1 | Micro F1 | Micro F1 method |
|---|---:|---:|---:|---|
| C0 | | | | estimated_from_precision_recall_curves_and_supports |
| C1 | | | | estimated_from_precision_recall_curves_and_supports |
| C2 | | | | estimated_from_precision_recall_curves_and_supports |
| C3 | | | | estimated_from_precision_recall_curves_and_supports |

說明AP50、AP50-95、mAP與F1不可互相替代；附metric exporter與固定threshold證據，並明記
Micro F1是由curves/supports估算，不是逐預測exact重新配對。

## 6. 成本

| Candidate | Params | GFLOPs | Latency ms | Peak VRAM MB | Device／precision |
|---|---:|---:|---:|---:|---|
| C0 | | | | | |
| C1 | | | | | |
| C2 | | | | | |
| C3 | | | | | |

CPU dry-run 不填 GPU latency／VRAM。量測需固定 warmup、iterations、batch、imgsz、precision 與
synchronization 方法。

## 7. C0 差值、敏感度與 Pareto

列出每個候選相對 C0 的所有精度與成本差值。0.005／0.008 只標記描述性敏感度，不自動觸發
C3-P5、R1、e=0.25 或因素組合。附 Pareto front 與被支配理由。

## 8. 使用者決策

- selection_status：pending_user_decision
- c_best：null
- 使用者可接受的精度／成本取捨：
- 是否需要補跑 Pose：
- 是否新增未來候選：
- 核准進 Q1／Q2L／Q2 的候選：

此節只能在使用者看完完整結果後更新。

## 9. 量化

對每個已核准 candidate 分開記錄 Q0 Float、Q1 PTQ simulation、Q2L QAT-lite simulation、Q2完整QAT simulation：

| Candidate | Stage | mAP／F1 gap | Quantized nodes | Excluded custom nodes | simulation_only |
|---|---|---:|---|---|---|
| | | | | | true |

不得以 simulation latency 宣稱真實 INT8 deployment。Q2L／Q2 都必須另有 GPU 執行授權；Q2L
固定200 steps且只作相容性檢查。

## 10. 限制與下一步

明確列出未執行項目、hardware、資料限制、Pose opt-in 狀態、integration skips 與不確定性。
只提出本次證據支持的下一個實驗。
