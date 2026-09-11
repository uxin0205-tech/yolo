# 論文第 3.1 節到 YOLO26M：目前、原論文與建議架構

> 2026-09-08 本輪整合版為 F2-PRE-HOG9，在共同J0之後的J1/J2監督raw P3，MASF留主訓練後；本文post-MASF版本保留，不混用結果。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

本文件只畫資料流與訓練差異。完整證據見[研究報告](<../../docs/research/2026-09-04-paper31-to-yolo26m-training-adaptation.md>)，執行 gate見[最小計畫](<plan.md>)。

## 1. 論文原本怎麼做

論文的 filter augmentation不是只在訓練時使用；filter channels會成為 detector的正式輸入：

```text
RGB image ──> channel mean ──> gray identity ───────────────┐
                         └──> HOG / Canny / WST ────────────┤
                                                            ↓
                                            concat(gray, filter channels)
                                                            ↓
                                            modified-input backbone
                                                            ↓
                                               Neck ──> Detect

HOG例：1個gray channel + 9個HOG channels = 10-channel input
訓練與推論：都必須計算同一條 filter 路徑
```

論文的 multi-stage transfer則是：

```text
ImageNet backbone
       ↓
COCO / optical detector
       ↓
DOTA optical remote-sensing detector
       ↓
SAR detector
```

這個 bridge針對 optical→SAR 的影像域與模型結構落差；目前 COCO person-only仍是同一批 COCO RGB影像，不能把 DOTA直接當合理中介。

## 2. 目前 YOLO26M／Full35 怎麼做

目前 RGB 主圖與雙 head：

```text
RGB [B,3,H,W]
       ↓
layers 0…15：Backbone + Neck early
       ↓
layer16：C3k2 → P3 MASF → p3_raw/post-MASF ────┬──> Detect P3
                     │                         └──> Pose   P3
                     │
                     └──> layer17 → layer19：p4_raw ──────┬──> Detect P4
                                   │                      └──> Pose   P4
                                   └──> layer20 → layer22：p5_raw ─┬──> Detect P5
                                                                  └──> Pose   P5

DualHeadPredictionModule([layer16, layer19, layer22])
```

用使用者偏好的共用節點表示：

```text
layer16：p3_post_masf ─────┬─> layer17 → P4 → layer20 → P5
                           │
                           ├─> Detect([p3_post_masf, p4_raw, p5_raw])
                           │
                           └─> Pose  ([p3_post_masf, p4_raw, p5_raw])
```

目前 loss與 optimizer step：

```text
COCO80 Detect batch ─> native Detect loss ─┐
                                           ├─> macro scaling ─> backward
BBAT5 Pose batch    ─> native Pose loss ───┘                      ↓
                                                       unscale → clip → AdamW
```

已有的 J0→J3不是空白 baseline：

```text
J0：Pose head only
 ↓
J1：Neck + Detect/Pose heads
 ↓
J2：layer9+ backbone + Neck + MASF + heads
 ↓
J3：full low-LR refinement + differentiable attention
```

## 3. 不建議的直接照搬版本

```text
RGB ─> gray ───────────┐
       └─> HOG9 ───────┴─> concat 10ch ─> 改造 layer0 stem ─> 現有YOLO26M
```

問題沿資料流傳遞：

```text
改成10-channel input
       ├─> pretrained RGB stem無法原樣載入
       ├─> 色彩線索被灰階路徑削弱
       ├─> 每張推論圖都要重算HOG
       └─> ONNX / FPGA / Bit-True輸入與成本契約全部改變
```

即使 AP提高，也無法分辨是額外 channels、stem重訓、更多計算，還是 HOG方向本身造成。

## 4. 建議改後：HOG只當訓練標靶

### 4.1 訓練圖

```text
同一個完成augmentation的 RGB tensor
       │
       ├──────────────────────────────────────────────────────┐
       │                                                      │
       ↓                                                      ↓ no_grad
現有 YOLO26M shared graph                              RGB → luminance
       │                                                      ↓
       ├─ layer16：post-MASF P3 ─> 1×1 Conv(256→9)        fixed HOG9
       │                         ↓                            ↓
       │                   predicted orientation       target orientation
       │                         │                            │
       │                         └──── box-aware CE ──────────┘
       │                                  ↓
       ├─ layer19：P4                   L_hog
       ├─ layer22：P5                     │
       │                                  │
       └─ Detect/Pose heads ─> L_native ──┴─> L_raw → 原本macro engine
```

同樣用共用節點表示：

```text
layer16：p3_post_masf ─────┬─> layer17 → P4 → layer20 → P5
                           │
                           ├─> Detect([p3_post_masf, p4_raw, p5_raw])
                           │
                           ├─> Pose  ([p3_post_masf, p4_raw, p5_raw])
                           │
                           └─> train-only 1×1 Conv(256→9) ─> P_hog ─┐
                                                                    ├─> L_hog
augmented RGB ─> luminance ─> fixed HOG9 ─> T_hog ─> GT box mask ──┘
```

重要的是 `P_hog`**不會餵回 P3/P4/P5**，所以 auxiliary activation不會污染主任務 forward；它只經 loss的 gradient暫時約束 P3 producer。

### 4.2 精確 loss

亮度與停止梯度標靶：

\[
Y=0.299R+0.587G+0.114B,
\qquad
T=\operatorname{sg}(\operatorname{HOG}_{9,cell=8}(Y)).
\]

P3 side head：

\[
P=\operatorname{softmax}_{bin}
  (\operatorname{Conv}_{1\times1}^{256\to9}(F_3)).
\]

每張圖以 GT box soft mask `M` 和 local gradient energy `A`加權：

\[
\ell_{hog}^{(b)}=
-\frac{\sum_{u,v,k}M_{buv}A_{buv}T_{bkuv}
                 \log(P_{bkuv}+\epsilon)}
       {\sum_{u,v}M_{buv}A_{buv}+\epsilon}.
\]

最後保持現有 raw loss 的 batch-sum語義：

\[
L_{raw}=L_{native,raw}+\mu(t)\sum_b\ell_{hog}^{(b)}.
\]

不能先對 batch取 mean再直接相加，否則最後一個短 batch或 microbatch切法會改變 auxiliary權重。

## 5. 什麼時候加、什麼時候拿掉

```text
J0                 J1 warmup       J1 active          J2前8 epochs       J2其餘 / J3
│                     │                 │                    │                 │
mu=0 ───────────────> mu=0 ─────────> 0→mu0→hold ───────> mu0→0 ───────────> mu=0
Pose head only          等主訓練穩定       約束P3 representation     逐步退出            主任務收尾
```

推導理由：

```text
J0 shared P3不更新        → 加HOG沒有正確作用點
J1首次開Neck/P3 producer   → 最適合提供局部形狀先驗
J2開late backbone與MASF    → 只保留短過渡，避免突然拿掉
J3低LR task refinement     → 應只由原生Detect/Pose loss決定終點
```

## 6. 最終部署圖

訓練完成並 strip後：

```text
RGB [B,3,H,W]
       ↓
layers0…22 shared YOLO26M
       ├─ layer16：P3 ─┐
       ├─ layer19：P4 ─┼─> Detect / Pose
       └─ layer22：P5 ─┘

不存在：HOG target generator、GT mask、1×1 auxiliary head、extra input channel
正式新增 parameters / FLOPs / latency：0
```

訓練前後的差別應只留在 shared weights；正式 graph、輸入、輸出schema與 baseline完全相同。

## 7. 三臂如何回答原因

```text
同一 dormant-aux J0 parent
          │
          ├─ F0-MATCH：不做aux forward，mu=0
          │              └─ 回答「新程式是否自己造成差異」
          │
          ├─ F1-LUMA9：相同head/mask/loss，預測local luminance
          │              └─ 回答「一般deep supervision是否已足夠」
          │
          └─ F2-HOG9 ：相同head/mask/loss，預測HOG orientation
                         └─ 只有F2>F1，才支持「方向先驗」
```

如果 `F2≈F1>F0`，結論是 companion supervision可能有效，不是 HOG特別有效；如果 `F2≤F0`，本方向停止，而不是把 filter、位置與超參數無限擴張。

## 8. 最後建議位置

```text
T0：person task/head定案
       ↓
T1：conflict-safe（只有新梯度screen過trigger才跑）
       ↓
A0：P3 MASF位置／最終FP graph定案
       ↓
T2：本方向 F0/F1/F2；首輪不疊gradient projection
       ↓
strip auxiliary，凍結最終Float winner
       ↓
Q0：BinaryQK site isolation / QAT
```

原因是 HOG supervision直接作用在 P3；如果之後再搬 MASF、換 head或改 RepConv，先前的 P3 training證據就不再代表最後部署 lineage。

返回[方向說明](<README.md>)。
