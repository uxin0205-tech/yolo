# COCO2017 person-only 結構特化研究

- 日期：2026-09-04
- 研究問題：若 COCO2017 Detect 實際只需要 `person`（COCO `category_id=1`、YOLO class `0`），哪些結構調整真正值得先做
- 本輪範圍：唯讀稽核本地模型、資料設定、標註與既有 P2 結果；查核第一手論文與官方實作；提出可否證的最小實驗
- 未執行事項：沒有訓練、推論評估、資料重切、production code 修改、checkpoint 修改或下載新標註

## 結論先行

1. **最小順序固定為 `H0 Detect80 → H1 Detect1 → H2 Detect1-Lite64`。** `H0` 是現行正式對照；`H1`
   只把輸出改成 personness，能從原 head 精確搬移 class-0 row，是任務契約控制；`H2` 保留相同 P3/P4/P5
   與 bbox tower，只把每個 classification tower 的 hidden width 從 256 降至 64，是第一個真正值得訓練的
   結構候選。它回答的是「只剩一個 person class 時，256-channel 分類 tower 是否過寬」，不是再堆新模組。
2. **`H1 Detect1` 的卷積節省很小。** Full35 shared graph 有 `26,529,701` parameters；end-to-end two-branch
   training graph 的 `nc=80 → 1` 只省 `121,818` parameters（Full35 的 `0.4592%`）與 `0.6795264 GFLOPs`。
   fused inference 只省其一半。它仍值得做，因為 class-score tensor 從 `8400×80` 變成 `8400×1`，但不能把
   它宣稱成主要加速或精度創新。
3. **`H2 Detect1-Lite64` 才有實質 head 縮減。** 不改 bbox width `c2=64`，令 class width也為
   `c3=64`；相對 `H0` two-branch graph 解析省 `1,000,410` parameters（Full35 的 `3.7709%`）。純 Conv
   `2 ops/MAC` 推導省 `4.8531456 GFLOPs`，本地 Ultralytics `get_flops()` 整圖建模差為
   `4.9305600 GFLOPs`；fused active-head 解析／建模差均為 `2.4265728 GFLOPs`。這些不是目標硬體 latency，
   必須另測；而且 H2 無法像 H1 一樣保持 epoch-0 函數完全等價，所以有真實精度風險。
4. **P3/P4/P5 先全部保留；不先加 P2，也不刪 P5。** 現有正式 bit-true predictions 以 canonical COCO API
   重算 person AP，standalone 到 J3 的 AP_S／AP_M／AP_L 分別下降 `0.003344 / 0.007119 / 0.006800`，不是
   small-only 退化，不能據此把 P2 當第一解。資料本身又同時含大量 small 與 large persons，刪 P5同樣缺乏證據。
   完整 P2 Detect在本地 fused `nc=1` 模型另增 `14.2706 GFLOPs`（`+21.04%`），只適合有尺度診斷後再測。
5. **資料過濾是成敗前提。** 不可對完整 COCO80 labels直接設 `single_cls=True`；Ultralytics會把車、狗等
   所有類別都重寫成 class 0，等於把它們錯教成人。必須建立可追溯的 person task view：只保留原 class 0
   labels、名稱改為 person，但保留全部 `118,287` 張 train images，包括 `54,172` 張 person-negative images。
6. **P2 center-scale auxiliary、centerness、pose/keypoint、localization KD、crowd loss全降為條件式後續。**
   先只跑 H1/H2 matched training。若 H2守住 AP且有硬體收益，才算此方向成立；只有誤差分析真的指出小人、
   score–IoU失配或 crowd/occlusion瓶頸時，才啟動對應的一個額外實驗。

> **主裁決：** 最值得先做的不是完整 P2或人體專用模組，而是 `H2 Detect1-Lite64`。它直接利用
> person-only 的單一分類語意，改動集中在 Detect classification tower，解析成本清楚、YAML parser已支援，
> P3/P4/P5與 bbox路徑不變。`P2-HCS-Aux` 保留為有 small-person 證據時的第二階段假說，且不主張學術首創。

## 一、證據標籤與任務邊界

本文使用四種標籤，避免把資料統計、論文結果與設計猜測混在一起：

- **【本地證據】**：本機正式設定、原始碼、annotation、labels 或已完成實驗直接支持。
- **【外部證據】**：論文、作者 repository、COCO／Ultralytics 官方資料直接支持。
- **【工程推論】**：由上述證據推導，但尚未在目前 YOLO26 lineage 做 matched experiment。
- **【待驗證假說】**：已定義對照、指標與停止條件，可被實驗否證。

### 1.1 現在不是從零開始的單純 person detector

**【本地證據】** 現行正式 Full35 架構是 shared YOLO26m layers 0–22，之後分成 COCO80 Detect 與 BBAT5
Pose26；應用層才從 COCO80 輸出取 person。正式 contract 是 `detect_nc=80`、Detect inputs
`features=[256,512,512]`、`strides=[8,16,32]`、`reg_max=1`、`end2end=True`，shared model
`26,529,701` parameters；來源接受稽核見
[`source-acceptance.md`](<../../../yolo_combine/docs/audits/2026-08-22-source-acceptance.md>)。完整血緣另見
[既有整體研究報告](<2026-08-31-repconv-binaryqk.md>)。因此 `COCO80 → person-only` 是**任務契約與 head
shape 的改變**，不是 final checkpoint 後的一個無訓練開關：

```text
shared YOLO26m layers 0–22
          ├─> COCO80 Detect ──> application filter class 0/person
          └─> BBAT5 Pose26
```

改成 `nc=1` 後，COCO80 overall mAP 不再是合法主指標，既有 joint score也不能原封不動比較。新的正式 gate
必須以 COCO `category_id=1` person AP為 Detect指標，並繼續保留 BBAT5 box／pose gate。H2雖只改 Detect
classification tower，訓練仍會反向更新 shared trunk；任何後續 P2 auxiliary也一樣，故都要檢查 BBAT5。

**【工程推論】** 這個方向應放在「任務定義／BASE-FP 訓練」階段，而不是 BinaryQK calibration、PTQ 或 export
之後。安全順序是：

```text
鎖定 person-only task/evaluator
    → H0 Detect80 canonical person baseline
    → H1 Detect1 graft + class-0 exact transfer
    → H1 / H2 Detect1-Lite64 matched training
    → H2 fused/export與目標硬體 gate
    → 只有診斷支持時才做 training-only P2-HCS-Aux
    → 若用了 auxiliary，移除後做 export equivalence gate
    → shared Detect+Pose integration gate
    → MASF / RepConv / BinaryQK / PTQ 各自重新校準與單因子驗證
```

不能把 person head、MASF seam、RepConv、BinaryQK 與量化同時更換，否則即使 AP 改善也無法歸因。

## 二、本地 COCO person 資料稽核

### 2.1 稽核來源與方法

**【本地證據】** 本機正式 COCO80 YAML 是
[`/home/uxin/yolo/coco2017.yaml`](<../../../coco2017.yaml>)，指向既有 `train2017.txt`／
`val2017.txt`，class `0` 是 person。沒有建立新 split，也沒有改動影像或標註。本輪可用來源：

| 來源 | 用途 | 稽核結果 |
|---|---|---|
| `/home/uxin/yolo/yolo_p2/p2_study/data/annotations/instances_train2017.json` | 官方 train instance annotation 的本地保存 | 118,287 images；SHA-256 `610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d` |
| `/home/uxin/yolo/coco2017/annotations/instances_val2017.json` | 現行 val instance annotation | 5,000 images；SHA-256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |
| `/home/uxin/yolo/coco2017/labels/{train2017,val2017}` | 現行 Ultralytics YOLO labels | class-0 person labels 可直接計數 |

全機唯讀搜尋沒有找到 `person_keypoints_train2017.json` 或 `person_keypoints_val2017.json`，所以本輪沒有假裝
本地已具備 pose supervision。COCO 官方 API 明確涵蓋 instance detection 與 person keypoints，但要做 keypoint
arm，仍須另行取得同一 2017 split 的官方 keypoint annotation，不能自行生成新 split。
[COCO 官方 API](https://github.com/cocodataset/cocoapi)是 annotation 解析與評估的第一手介面。

統計定義：

- 僅取 `category_id=1`。
- `non-crowd` 指 `iscrowd=0`；crowd annotation 是區域，不當成可逐人計數的 instance。
- COCO size 直接按 annotation `area`：small `<32²`、medium `32²–96²`、large `≥96²`。
- 640 尺寸以等比例 letterbox 的縮放率 `r=min(640/W,640/H)` 計算；padding 不改 bbox 寬高。
- 擁擠度以同張影像 non-crowd person bbox 的兩兩 IoU 計算；對每個人取與其他人的最大 IoU。

### 2.2 數量與尺寸分布

| split | person annotations（含 crowd region） | non-crowd person | person-positive images | 每個 positive image：mean / median / P90 / max |
|---|---:|---:|---:|---:|
| train2017 | 262,465 | 257,253 | 64,115 / 118,287（54.20%） | 4.01 / 2 / 12 / 19 |
| val2017 | 11,004 | 10,777 | 2,693 / 5,000（53.86%） | 4.00 / 2 / 12 / 13 |

現行 YOLO labels 實際有 train class-0 `257,252` 筆、val class-0 `10,777` 筆。train 比 JSON non-crowd 少一筆
不是抽樣：annotation id `2206849` 的 bbox 寬或高非正值，而本地官方 converter 會排除非正 bbox。
Converter 也在寫 labels 前直接跳過 `iscrowd`，見
[`converter.py:298`](<../../../yolo_p2/ultralytics/data/converter.py>)。

train 的 `64,115` 張 person-positive以外，另有 `54,172` 張 person-negative images；原 COCO labels還有
`592,690` 個其他類別 instances。person-only訓練必須保留全部負影像，只移除其他類別 label。**不可直接對
完整80類 labels設 `single_cls=True`**：本地 Ultralytics pipeline先依 `include_class`過濾，之後
`single_cls`會執行 `cls[:, 0] = 0`，因此未先過濾時，車、狗、球等全部會被改標成 person。正確契約是可重建
且可稽核的 class-0 runtime view：沿用原 train/val image manifests，不改 split；只保留原 class 0，資料
`names={0: person}`，空 label影像照常存在。canonical COCO資料本身保持唯讀。

以下統計固定使用 non-crowd person：

| 指標 | train2017 | val2017 | 對結構的含義 |
|---|---:|---:|---|
| COCO annotation `area` small / medium / large | 40.98% / 33.47% / 25.55% | 39.97% / 34.55% / 25.48% | 官方 AP_S/M/L 分桶；不能只保留小尺度，也不能先刪 P5 |
| bbox `w×h` small / medium / large | 30.79% / 33.51% / 35.70% | 29.75% / 34.54% / 35.71% | annotation area受 segmentation/crowd定義影響；幾何尺度判斷需另看 bbox |
| 640 後短邊 `<8 px` | 6.96% | 6.63% | P3/8 上短邊不足 1 cell 的明顯困難樣本 |
| 640 後短邊 `<16 px` | 20.66% | 19.57% | 約五分之一在 P3 上不足 2 cells，P2 supervision有合理動機 |
| 640 後短邊 `<32 px` | 40.65% | 38.87% | 小人物是主問題之一，但不是全部資料 |
| 短邊 P10 / P50 / P90（px@640） | 9.73 / 43.12 / 206.02 | 9.98 / 45.47 / 204.50 | 跨尺度範圍很大 |
| bbox `h/w` P10 / P50 / P90 | 0.943 / 1.855 / 3.329 | 0.954 / 1.822 / 3.326 | 多數偏直立，但坐姿、近景、截斷造成寬框；不可固定 ratio |
| `h/w ≥1.5` | 64.89% | 64.04% | center/height cues有用，但 width仍應獨立預測 |

**【工程推論】** P2 的訊號是存在的，但這些數字只證明「高解析度監督值得測」，不證明「完整 P2 Detect
必然提升 person AP」。由於仍有四分之一 large persons，合理方案是保留 P3/P4/P5 lead head，在訓練期借用
P2，而不是把 pyramid 硬裁成單一人體尺度。

### 2.3 擁擠、重疊與 `iscrowd`

| 指標 | train2017 | val2017 |
|---|---:|---:|
| person-category crowd regions | 5,212（涵蓋 4.41% images） | 227（涵蓋 4.54% images） |
| 每張至少 5 個 non-crowd persons | 18,775 images（15.87%） | 796 images（15.92%） |
| 每張至少 10 個 non-crowd persons | 9,604 images（8.12%） | 403 images（8.06%） |
| person bbox max pair-IoU `≥0.1` | 29.86% persons | 31.43% persons |
| person bbox max pair-IoU `≥0.3` | 5.87% persons | 6.25% persons |
| person bbox max pair-IoU `≥0.5` | 0.94% persons | 1.01% persons |

**【本地證據】** 現行 converter 對 `iscrowd` 是 `continue`，YOLO txt 又沒有 ignore-region 語意；如果保留該
影像，crowd 區域可能在一般 dense loss 中成為未標正例的背景。這是比堆 CrowdDet head 更先要稽核的資料／loss
契約問題。

**【工程推論】** COCO person 有實質密度，但真正 bbox IoU `≥0.5` 的 non-crowd person 只有約 1%。因此首輪
導入「一個 location 多個 prediction + Set NMS」過重，且現行 YOLO26 是 end-to-end NMS-free head，介面不匹配。
CrowdDet 在 CrowdHuman 有強結果，但論文也只稱在較不擁擠的 COCO 有 moderate improvement；它更適合成為
條件式研究，不是預設主幹。
[CrowdDet 原論文](https://openaccess.thecvf.com/content_CVPR_2020/html/Chu_Detection_in_Crowded_Scenes_One_Proposal_Multiple_Predictions_CVPR_2020_paper.html)

## 三、目前 YOLO26m 與 `nc=1` 的真實成本

### 3.1 目前 graph

**【本地證據】** Vendored YOLO26 YAML 明列 `nc=80`、`end2end=True`、`reg_max=1`，Detect 輸入是
layers 16/19/22 的 P3/P4/P5，見
[`yolo26.yaml:8`](<../../../yolo_p2/ultralytics/cfg/models/26/yolo26.yaml>)與
[`yolo26.yaml:52`](<../../../yolo_p2/ultralytics/cfg/models/26/yolo26.yaml>)。Head 原始碼會在
end-to-end 模式複製 one-to-many 與 one-to-one box/class towers；分類 tower 寬度是
`c3=max(ch[0], min(nc,100))`，見
[`head.py:89`](<../../../yolo_p2/ultralytics/nn/modules/head.py>)。對 YOLO26m，`ch[0]=256`，
所以 `nc=80` 與 `nc=1` 的 `c3` 都是 256；只有每個尺度最後的 `1×1 Conv(256→nc)` 會縮小。

```text
layer16：P3/8 ──────────────────────────────┐
              └─> layer17 → layer19：P4/16 ─┤
                               └─> layer20 → layer22：P5/32 ─┤
                                                             ↓
                     Detect80(one-to-many + one-to-one)
                                                             ↓
                                  application保留 class 0/person
```

Ultralytics 官方文件也說明 YOLO26 訓練使用 dual heads，而 fuse/export 會移除 training-only one-to-many
head；因此 checkpoint counts 與 fused deployment counts 必須分開報。
[YOLO26 end-to-end 官方說明](https://docs.ultralytics.com/guides/end2end-detection/)

### 3.2 `nc=80 → 1` 精算與 CPU 建模驗證

對一個 active head，三尺度位置數是 `80²+40²+20²=8,400`。分類末層從 80 類縮成 1 類：

```text
Δparams_active = 3 × (80-1) × (256 weights + 1 bias)
               = 60,909

ΔMACs_active   = (80-1) × 256 × 8,400
               = 169,881,600 MACs
               = 0.3397632 GFLOPs（每 MAC以2 ops計）
```

未 fuse training graph 有 two heads，因此正好是兩倍。本輪以 vendored Ultralytics `DetectionModel` 與
`get_flops(imgsz=640)` 在 CPU 記憶體中建模，沒有保存 checkpoint：

| graph | nc=80 | nc=1 | 差值 | 相對降幅 |
|---|---:|---:|---:|---:|
| 未 fuse parameters | 21,896,248 | 21,774,430 | -121,818 | -0.556% |
| 未 fuse GFLOPs | 75.3893376 | 74.7098112 | -0.6795264 | -0.902% |
| fused parameters | 20,411,132 | 20,350,223 | -60,909 | -0.298% |
| fused GFLOPs | 68.1799680 | 67.8402048 | -0.3397632 | -0.498% |

這是 plain YOLO26m config 的分析值，不是 Full35 J3 的端到端 latency benchmark。可支持的結論是：

- `nc=1` 對卷積主體的節省很小；不能宣稱模型因此大幅加速。
- active class logits 從 672,000 elements 降為 8,400，分類輸出／top-k bandwidth 有 80 倍維度縮減；
  實際 latency 仍需按目標 backend 測量。
- Box tower、P3/P4/P5 neck 與 BinaryQK 都不會因 `nc=1` 自動變小。

### 3.3 權重轉移不能隨機重設整個 Detect

**【工程推論】** `H1-DETECT1` 應從同一 80-class parent 做 shape-aware transfer：

1. Backbone、neck、box towers、classification tower 中間層完整複製。
2. 對每個 P3/P4/P5，以及 one-to-many／one-to-one 的 class output conv，只取原 `[0,:,:,:]` 與
   bias `[0]` 搬到新的單輸出 conv。
3. 不要在搬完後再呼叫通用 `bias_init()`；`nc` 改變會改初始化先驗，覆蓋 pretrained person bias。
4. epoch-0 用同一批影像核對：新 head 的 raw person logit 必須與原 80-class head 的 class-0 raw logit一致；
   box outputs也必須一致。未達 FP32 tolerance就停止。

這樣 `H1-DETECT1` 的 epoch-0 person 函數可以等價，後續差異才來自 matched training，而不是「整個 head
隨機重建」。

## 四、候選方案排序

| 排序 | 候選 | 判定 | 依據與限制 |
|---:|---|---|---|
| H0 | `Detect80(c3=256)` P3/P4/P5 | 現行對照 | 用既有 bit-true predictions做 canonical person AP，不重訓 |
| H1 | `Detect1(c3=256)` P3/P4/P5 | 必做任務控制 | class-0 exact transfer；節省小，但建立正確 person-only契約 |
| H2 | **`Detect1-Lite64(c3=64)` P3/P4/P5** | **第一個結構實驗** | bbox與尺度不變，只縮單類 classification tower；有實質 head成本差 |
| 後續1 | `P2-HCS-Aux` training-only | 條件式 | 只有 short-box／AP_S診斷顯示獨立瓶頸才測；部署零增量 |
| 後續2 | P3/P4/P5 localization-quality／centerness | 條件式 | 只在 score–IoU失配證據成立時測 |
| 後續3 | Pose/keypoint auxiliary或localization KD | 條件式 | 部署可零增量，但本地 keypoint JSON缺失；不可與H2首輪混測 |
| 後續4 | `iscrowd` ignore + Repulsion Loss | 條件式 | 先修 crowd ignore；只有 dense/overlap slice有獨立問題才加 loss |
| 不先做 | 完整 P2 Detect(P2–P5) | 暫不做 | person-only YOLO26m fused `+21.04% GFLOPs`；既有跨模型 P2收益小，且無 person AP |
| 不先做 | 刪 P5／只留 P3-P4 | 不建議 | 25.55% train non-crowd persons是 large；無本地 person-only ablation |
| 不先做 | 固定人體 aspect ratio／全面 asymmetric conv | 不建議 | `h/w`分布很寬；姿態、截斷與近景不符合單一行人模板 |
| 不先做 | CrowdDet multi-prediction／Set NMS | 不建議首輪 | 高 IoU重疊比例低，且與 YOLO26 end-to-end NMS-free介面衝突 |

人類專用 center/scale 表示不是憑空猜測：CSP 把 pedestrian detection 明確表成 center 與 scale prediction，
並在行人 benchmark 驗證；CenterNet 也用 center heatmap、size 與 offset 表示一般物件。
[CSP 原論文](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.html)、
[CenterNet 作者專案／論文入口](https://tubb-lab.github.io/CenterNet/)。
但 [Generalizable Pedestrian Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Hasan_Generalizable_Pedestrian_Detection_The_Elephant_in_the_Room_CVPR_2021_paper.html)
也顯示過度 benchmark-specific 的 pedestrian adaptation 可能泛化較差；因此本方案只把人體先驗放在可裁除的
training auxiliary branch，不改 deployment trunk 的基本多尺度能力。

## 五、首選結構：`Detect1-Lite64`

### 5.1 原本、任務控制與首選結構

```text
H0：P3/P4/P5 → Detect80(c3=256) → application filter class 0/person
                         │
                         ▼
H1：P3/P4/P5 → Detect1(c3=256)  → person
                         │
                         ▼
H2：P3/P4/P5 → Detect1(c3=64)   → person
```

H1 保留 P3/P4/P5、bbox towers 與 classification hidden layers，只把每尺度、one-to-many／one-to-one 的
final class output由80改1。它可從 H0精確搬移 class-0 row，因此先建立正確單類 task control。

H2再把每尺度 classification tower hidden width由256縮到64；bbox tower仍使用 `c2=64`，P3/P4/P5與shared
backbone／neck完全不動。它檢驗的是「單一 personness output是否仍需要為80類分離而保留的256-channel
classification space」，不是固定人體比例，也不是刪除多尺度能力。

### 5.2 H1能精確搬移，H2不能冒稱 epoch-0等價

```text
H0 final class weight [80,256,1,1] ── take row 0 ──> H1 [1,256,1,1]
H0 final class bias   [80]           ── take row 0 ──> H1 [1]
```

此映射對 P3/P4/P5與 two branches共6個 final class conv執行；其餘 trunk、neck、box、class hidden tensors
逐一複製。相同輸入下，H1 raw person logit與 H0 class-0 raw logit、box outputs都必須在 tolerance內一致。

H2的 hidden shape變成64，沒有一般的 exact function-preserving row slice。首輪保留相同 H0 trunk／neck／box
weights，窄 class tower固定seed初始化；H1/H2使用相同訓練 recipe。若 H2失敗，只能說這個實用候選未過 gate；
teacher/KD若要區分容量與優化困難，必須另作 matched control，不能只加在H2。

### 5.3 參數與運算量推導

三尺度 class towers的本地解析總數：

| 架構 | one active branch params | two-branch params |
|---|---:|---:|
| H0 `Detect80(c3=256)` | 611,568 | 1,223,136 |
| H1 `Detect1(c3=256)` | 550,659 | 1,101,318 |
| H2 `Detect1(c3=64)` | 111,363 | 222,726 |

所以 H1→H2省878,592 parameters；H0→H2省1,000,410，約占現行 Full35 26,529,701 parameters的
3.7709%。以640輸入及`2 ops/MAC`，two-branch H0→H2解析差為4.8531456 GFLOPs，fused active branch為
2.4265728 GFLOPs。本地整圖 `get_flops()`建模得到相同 fused差值；仍需 target profile才能宣稱真實加速。

H1/H2 fused class-score elements都是8,400，H0則為672,000；H1已取得輸出與top-k bandwidth縮減，H2再取得
hidden tower縮減。完整圖、公式與最小實驗見
[方向架構報告](<../../proposals/coco-person-specialized-head/architecture-report.md>)。

## 六、其他指定候選的判定

### 6.1 P2／P5 尺度配置

**完整 P2。** Ultralytics 官方確實提供 P2/4–P5/32 YAML，並把 P2定位為 small-object variant。
[官方 `yolo26-p2.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p2.yaml)
在本地 `nc=1` CPU 建模中：

| person-only model | 未 fuse params | 未 fuse GFLOPs | fused params | fused GFLOPs |
|---|---:|---:|---:|---:|
| standard P3/P4/P5 | 21,774,430 | 74.7098 | 20,350,223 | 67.8402 |
| full P2/P3/P4/P5 | 21,062,760 | 89.9590 | 20,324,884 | 82.1108 |

P2版參數反而略少是 YAML neck channel配置不同，不代表便宜；高解析度 feature MACs仍讓 fused GFLOPs
增加 21.04%。只有 H1/H2 完成後，短邊 `<16px`／`<32px` recall顯示獨立瓶頸，才先另案測可裁除的
`P2-HCS-Aux`；它有效但 P3 lead head仍無法轉成 recall時，才把 full P2 Detect列為更後面的候選。

`P2-HCS-Aux` 的條件式構想，是用 P2/P3各自投影到32 channels後相加，暫時預測 person center、`log(w,h)`
與 center offset；export前完全移除。它約17,029 parameters、0.5472256 forward GFLOPs，但 P2 activation與
backward的真實 VRAM／wall time未知。這只保留為有small-person證據後的獨立方向，不是本報告首輪。

**刪 P5。** 目前沒有本地 person AP by level、assignment histogram或 P5 ablation；25.55% large person 已足以
否決「因為只偵測人，所以自然不需要 P5」這個假設。若未來硬體預算強迫裁尺度，應先記錄每個 GT 的 positive
assignment level與 P5 contribution，再做 validation-only mask，不應直接重訓二尺度模型。

### 6.2 person-specific localization／centerness

FCOS 將 centerness target定義為 location到四邊距離的對稱比例，推論時乘上 class score，以壓低遠離物件中心
的低品質框；該論文直接支持「location quality scalar可改善 score與 box quality對齊」。
[FCOS 原論文](https://openaccess.thecvf.com/content_ICCV_2019/html/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.html)

對 person-only，class score本質上接近 personness，額外 quality scalar可能特別適合改善 AP75／ranking；但
YOLO26目前是 one-to-one NMS-free inference，乘上一個新 scalar會改 top-k、loss與 export contract。初步若從
256-channel class feature各接一個 1-channel `1×1`，active fused head只約 771 parameters、`<0.005 GFLOPs`，
算力不是主要風險，**排序與校準才是風險**。

因此先做以下唯讀／evaluation診斷：計算 true-positive prediction 的 confidence–IoU Spearman、ECE或分箱平均
IoU，以及 AP50與 AP75差。只有 H1/H2 winner明顯是「classification score高但 localization quality差」，才新增
單一 `PERSON-QUALITY` arm；不可跟 H2或 P2-HCS同輪一起開。

### 6.3 training-only pose/keypoint auxiliary supervision或蒸餾

COCO與 Mask R-CNN第一手資料證明，同一 person instance框架可平行學 bbox與人體 keypoints；Mask R-CNN
本身可延伸為 person pose head。
[Mask R-CNN 原論文](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)
MultiPoseNet也展示 person detection、segmentation、pose的 joint multi-task可行性。
[MultiPoseNet 原論文](https://openaccess.thecvf.com/content_ECCV_2018/html/Muhammed_Kocabas_MultiPoseNet_Fast_Multi-Person_ECCV_2018_paper.html)

但這些來源**不直接證明**在本專案 YOLO26加入 17-keypoint loss必然提高 person bbox AP。本地又沒有 keypoint
JSON；小、crowd、嚴重遮擋 person的 keypoints缺失不是隨機缺失，硬把無標註當 `(0,0)` 會產生偏差。因此：

- 若日後另開 `P2-HCS-Aux`，第一輪只用 bbox。
- 它獨立通過後，才取得同 split官方 `person_keypoints_train2017.json`，以 annotation/image id join；不得重切。
- 最小 keypoint擴充不是永久 17點 pose head，而是 training-only `TORSO-CENTER`：只在 shoulders/hips中至少
  兩個有效點時，用可見 torso centroid約束 HCS center；缺失 instance把該 loss mask掉。
- 若不想改 target，可用 frozen pose teacher把 P2/P3 feature或 torso heatmap蒸餾給 auxiliary branch；teacher
  只在訓練存在，部署不增加成本，但會增加 training wall time與 cache/provenance負擔。

Localization Distillation顯示，dense detector可在不增加 inference成本下從 teacher轉移 localization knowledge；
但它用的是 localization distribution，而本地 YOLO26 `reg_max=1`、DFL-free，不能原封不動照搬。這裡能採的是
「valuable localization region + teacher box/feature target」概念，必須重新定義適合 YOLO26 direct box head的
loss。
[Localization Distillation 原論文](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)

### 6.4 crowd／occlusion 處理

Repulsion Loss針對 pedestrian crowd，讓 proposal靠近自己的 GT並遠離其他人的 GT／proposal，論文直接報告
occlusion case改善。
[Repulsion Loss 原論文](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.html)
它比 multi-prediction head更接近本專案的低推論成本要求，因為是 training loss；但 YOLO26是 dense、end-to-end
assigner，不能直接搬 Faster R-CNN proposal公式。

正確順序：

1. 先讓 COCO `iscrowd=1` region成為 ignore mask，而不是在 person-only labels被當背景；不把 crowd region當
   獨立 person正例。
2. 固定建立三個 evaluation slices：`persons/image ≥5`、`≥10`、`max pair-IoU ≥0.3`，只做評估索引，不改 split。
3. 若 H1/H2 winner在 dense slice的 AP/AR相對一般 slice有額外顯著缺口，才做一個 YOLO-compatible repulsion
   loss arm；只對已指派 positives與鄰近非 target GT計算，且 loss weight從零 warm up。
4. 若 dense slice沒有改善或一般 person AP下降超過 gate，立即停止，不上 Set NMS／multi-prediction head。

## 七、最小對照矩陣

第一輪只需要一個 read-only baseline validation與兩個新 training arms：

| ID | Head／尺度 | 初始化 | 用途 |
|---|---|---|---|
| `H0-BASE80-PERSON` | 現行 `nc=80,c3=256`，P3/P4/P5 | authoritative parent原樣 | COCO API `catIds=[1]` 現況；不訓練 |
| `H1-DETECT1` | `nc=1,c3=256`，P3/P4/P5 | 同一 parent；class-0 exact transfer | 單類任務／容量 control；新job 1 |
| `H2-LITE64` | `nc=1,c3=64`，P3/P4/P5 | 同一 trunk/neck/box；窄class tower固定seed初始化 | 唯一結構候選；新job 2 |

H1與H2必須同 train/val manifests、augment、image size、batch、optimizer、LR、epoch、patience、seed、parent revision
與 evaluator。所有沒有 person的 COCO影像仍保留作真實 negatives；只過濾其他類別 labels，不抽掉 negative
images。H2−H1才是「縮窄person class tower」的實用效果；H2無法和H1保持epoch-0 hidden function等價，這項
初始化／可優化性風險必須如實列入，不能偷偷只替H2加KD。

首輪 seed 0 做 pilot；只有通過下面 gate，才對H1/H2補 seeds 1、2。不要一開始就加入 full P2、P2-HCS、
centerness、pose KD或 repulsion。

### 7.1 指標契約

Primary metric：官方 COCO bbox AP@[.50:.95]、`catIds=[1]`。同時報：

- person AP50、AP75、AP_S、AP_M、AP_L、AR100。
- bbox短邊 `<16px` 與 `<32px` 的診斷 recall；這是本地自訂 slice，不能標成官方 COCO AP_S。
- density `≥5`、`≥10` 與 pair-IoU `≥0.3` slices。
- Params、GFLOPs、batch-1 latency、peak inference VRAM，並分未 fuse與 fused/export。
- Training wall time／epoch、peak training VRAM。
- 若整合 Full35：既有 BBAT5 box／pose gates全部保留。

不得把 `H0-BASE80-PERSON` 的單類 AP與舊 COCO80 mean AP當同一指標，也不得用 Ultralytics internal mAP與 canonical
COCO API mAP混欄。COCO API與資料格式以[官方 cocoapi](https://github.com/cocodataset/cocoapi)為準。

### 7.2 預先宣告 gate與停止條件

以下是本專案工程 gate，不是論文定律：

1. **Epoch-0 transfer gate：** `H1-DETECT1` raw person logits與 boxes相對 `H0-BASE80-PERSON` 對應輸出
   `max_abs ≤1e-5`；否則不訓練。
2. **H1 task gate：** 相對H0 canonical person AP不得低於`-0.001`，negative-image FP不得明顯惡化；否則先修
   task view／mapping，不進H2升格。
3. **H2 pilot gate：** seed 0 person AP相對H1不得低於`-0.001`；AP_S/M/L任一不得低於`-0.002`；失敗即停。
4. **H2正式 gate：** 三-seed paired mean套相同accuracy gate，且fused target latency改善超過noise並達事前
   登錄門檻；只有GFLOPs下降不能升格。
5. **共享任務 gate：** Full35整合後任一既有 BBAT5 box／pose gate失敗，保留standalone結果但不升格joint；
   不用COCO person gain抵銷另一任務退化。
6. **graph gate：** fused/export只保留一組64-channel person class tower；不得殘留80-class logits或錯誤
   256-channel tower，prediction class 0必須映射COCO category 1。
7. **crowd停止條件：** dense slice沒有獨立退化證據，就不做 Repulsion/CrowdDet；ignore-region正確性仍須修。

若 H2失敗，保留通過task gate的H1，不用P2、attention、deformable conv或KD來救同一首輪。若H2通過，才按
誤差診斷另立一個P2-HCS、quality、pose/KD或crowd方向，且一次只加一項。

## 八、精度、硬體與研究風險

| 風險 | 為何存在 | 控制方式 |
|---|---|---|
| `nc=1` 被誤認為大幅減算 | classification tower寬度仍是256，只有末層縮小 | 分開報 raw/fused params、GFLOPs、score tensor與實測 latency |
| H2窄塔容量不足 | person姿態、遮擋與尺度仍高度多樣 | AP_S/M/L、dense slices與三paired seeds；失敗保留H1 |
| H1/H2初始化不完全對稱 | H1可exact transfer，H2 hidden shape改變 | 固定初始化與matched recipe；失敗不得偷偷只替H2加KD |
| 負影像被丟棄 | 54,172張train images沒有person | 保留完整image manifests與empty labels；檢查FP/image |
| shared Pose退化 | person-only recovery仍會更新共同 backbone／neck | standalone先決選；joint整合後保留全部BBAT5 gates |
| crowd region成背景 | 現行converter直接略過 `iscrowd` | 在loss加入ignore mask；官方person AP仍使用原JSON |
| export仍保留舊寬塔 | parser／checkpoint可能建立錯誤c3或雙branch | graph/state audit；fused只准一組64-channel person class tower |
| 過度人體特化傷泛化 | COCO person姿態／尺度很廣，行人benchmark先驗未必泛化 | H2只縮class tower，保留P3/P4/P5與box自由度；另報泛化限制 |
| 與量化交互 | 新head與recovery會改 activation／score分布 | person架構先定版，再重做BinaryQK/PTQ calibration與bit-true gate |

## 九、最終建議與可執行決策

**現在應該做：**

1. 先對 authoritative parent跑一次 canonical `H0-BASE80-PERSON` validation，得到真正的 person AP baseline。
2. 建 `H1-DETECT1`，只改 Detect class output shape並精確搬 class-0 rows；不要同時動 P2/P5、MASF或BinaryQK。
3. 唯一首輪結構 arm做 `H2-LITE64`；只把 classification hidden width由256縮64，bbox與P3/P4/P5不動。
4. 通過 seed-0 accuracy與target-profile gate才補三 seed；H2失敗就保留H1，不加其他模組救援。
5. 密集人群先處理 crowd ignore語意與分層評估；只有 localization error確實集中在 crowd才做Repulsion Loss。

**現在不應該做：**

- 不要因為「只偵測人」就刪 P5。
- 不要直接把整個 Detect 改成 P2/P3/P4/P5並長訓，既有證據顯示成本大、person收益未知。
- 不要固定人體 aspect ratio，也不要全面換成非對稱卷積。
- 不要把 P2-HCS、centerness、pose、KD或repulsion與H2一次疊上去。
- 不要從 COCO80 overall AP推斷 person-only AP，也不要改 split來讓結果變好看。

一句話總結：**person-only最值得利用的不是「砍掉所有通用結構」，而是把 head語意先校正為 `nc=1`，
保留P3/P4/P5與bbox多尺度能力，再把只服務一個person output的 classification tower從256縮到64。**

## 十、第一手來源

1. Ultralytics, **YOLO26** 官方說明：<https://docs.ultralytics.com/models/yolo26/>
2. Ultralytics, **YOLO26 End-to-End NMS-Free Detection**：<https://docs.ultralytics.com/guides/end2end-detection/>
3. Ultralytics, **`yolo26-p2.yaml`** 官方原始碼：<https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p2.yaml>
4. COCO, **COCO API**：<https://github.com/cocodataset/cocoapi>
5. Tsung-Yi Lin et al., **Feature Pyramid Networks for Object Detection**, CVPR 2017：<https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html>
6. Wei Liu et al., **High-Level Semantic Feature Detection: A New Perspective for Pedestrian Detection**, CVPR 2019：<https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.html>
7. Xingyi Zhou et al., **Objects as Points / CenterNet**：<https://tubb-lab.github.io/CenterNet/>
8. Zhi Tian et al., **FCOS: Fully Convolutional One-Stage Object Detection**, ICCV 2019：<https://openaccess.thecvf.com/content_ICCV_2019/html/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.html>
9. Chien-Yao Wang et al., **YOLOv7: Trainable Bag-of-Freebies**, CVPR 2023：<https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html>
10. Kaiming He et al., **Mask R-CNN**, ICCV 2017：<https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html>
11. Muhammed Kocabas et al., **MultiPoseNet**, ECCV 2018：<https://openaccess.thecvf.com/content_ECCV_2018/html/Muhammed_Kocabas_MultiPoseNet_Fast_Multi-Person_ECCV_2018_paper.html>
12. Zhaohui Zheng et al., **Localization Distillation for Dense Object Detection**, CVPR 2022：<https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html>
13. Xinlong Wang et al., **Repulsion Loss: Detecting Pedestrians in a Crowd**, CVPR 2018：<https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.html>
14. Xuangeng Chu et al., **Detection in Crowded Scenes: One Proposal, Multiple Predictions**, CVPR 2020：<https://openaccess.thecvf.com/content_CVPR_2020/html/Chu_Detection_in_Crowded_Scenes_One_Proposal_Multiple_Predictions_CVPR_2020_paper.html>
15. Irtiza Hasan et al., **Generalizable Pedestrian Detection: The Elephant in the Room**, CVPR 2021：<https://openaccess.thecvf.com/content/CVPR2021/html/Hasan_Generalizable_Pedestrian_Detection_The_Elephant_in_the_Room_CVPR_2021_paper.html>

## 十一、本輪驗證與未解事項

### 驗證方式與結果

- 直接解析 train／val instance JSON，重算 person count、COCO size、640短邊、aspect ratio、每圖密度與 pair-IoU：
  train／val分布一致，支持統計沒有明顯 split drift。
- 逐一掃描現行 YOLO labels：train class-0 `257,252`、val class-0 `10,777`；與 converter排除 crowd／
  非正 bbox規則一致。
- 用 vendored Ultralytics在 CPU記憶體建立 `yolo26m nc=80/1` 與 `yolo26m-p2 nc=1`，並用官方 `fuse()`、
  `get_flops(imgsz=640)`重算 parameters/GFLOPs；沒有保存權重或啟動訓練。
- 所有外部架構主張只採論文、COCO／Ultralytics官方文件或作者專案；沒有把第三方摘要當證據。

### 困難與解法

- 困難：根 COCO目錄缺 `instances_train2017.json`。解法：唯讀使用既有正式 P2研究保存的同 split官方
  train annotation，記錄實際路徑與SHA-256；沒有複製或重建資料。
- 困難：本地沒有 COCO person keypoint JSON。解法：把 pose/keypoint與P2-HCS列為條件式後續，首輪不使用。
- 困難：既有 P2封存結果沒有 per-class AP或 prediction JSON。解法：只把它用作 COCO80成本／overall效果證據，
  明記不能外推 person增益。

### 未解事項／風險

- 現有 standalone與Full35 J3 predictions已可重算 canonical person AP；正式H0仍須凍結exact parent、digest與輸出。
- H2 Lite64的實際person AP、target latency、training穩定性與三seed變異尚未實驗。
- P2-HCS-Aux的VRAM、梯度干擾、center collision與person AP只屬條件式未知，不是首輪阻礙。
- Keypoint annotation coverage、可見 torso target品質及 pose-teacher成本尚未稽核。
- `iscrowd` ignore mask如何接進目前 YOLO26 loss／augment pipeline尚未實作。
- 尚未做廣泛 novelty／專利檢索，因此不得把 Lite64或條件式P2-HCS宣稱為文獻首創。
