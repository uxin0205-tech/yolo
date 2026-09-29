# OPT-COCO-PERSON-SPECIALIZED-HEAD：COCO person-only 專用 Detect head

> 2026-09-08 使用者暫緩；本文保留為歷史提案，不列本輪主線，不建立 View 或改 head。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

| 欄位 | 內容 |
|---|---|
| 狀態 | `proposed` |
| 全域序位 | T0；先定義任務與 FP head，再做 joint training／MASF／BinaryQK |
| 現行做法 | 訓練 COCO80 `Detect(c3=256)`，應用端才過濾 class 0/person |
| 任務控制 | `H1 Detect1(c3=256)`：只改輸出語意，精確搬移原 person row |
| 首選結構 | `H2 Detect1-Lite64(c3=64)`：保留 P3/P4/P5與 bbox tower，只縮分類 tower |
| 第一輪新訓練 | H1、H2 各 1 個 matched job；H0重用既有 validation |
| 不先做 | P2、刪 P5、固定人體比例、centerness、pose/KD、crowd head |
| 詳細研究 | [COCO person-only 結構特化研究](<../../docs/research/2026-09-04-coco-person-only-structural-specialization.md>) |
| 架構圖 | [H0／H1／H2目前與預計架構](<architecture-report.md>) |
| 執行方式 | [最小實驗計畫](<plan.md>) |

## 方向結論

只需要 person 時，第一件事不是加人體專用大模組，而是把 supervision與輸出契約真正改成一類。接著才測
`Detect1-Lite64`：YOLO26m在 `nc=80` 與 `nc=1` 時，預設 classification hidden width都仍是256；所以單純
把 `nc` 改為1，只縮最後一層，主分類塔幾乎沒變小。

最小順序固定為：

~~~text
H0：Detect80(c3=256)，application filter person
                 │
                 ▼
H1：Detect1(c3=256)，建立正確單類任務控制
                 │
                 ▼
H2：Detect1-Lite64(c3=64)，第一個真正的結構候選
~~~

P3/P4/P5全部保留。COCO person橫跨小、中、大尺度；既有 standalone→J3的 canonical person AP退化也不是
small-only，所以不能先加 P2或刪 P5。

## 為什麼 `nc=1` 本身不夠

現行 Detect在三個尺度各有 box tower與 class tower，end-to-end training又複製 one-to-many／one-to-one兩組。
預設：

\[
c_3=\max(ch_{P3},\min(n_c,100)).
\]

目前 `ch_P3=256`，因此：

~~~text
nc=80 → c3=max(256,80)=256
nc=1  → c3=max(256,1) =256
~~~

H1只把每尺度最後 `Conv(256→80)` 改成 `Conv(256→1)`。它是必要的任務控制與輸出／top-k bandwidth優化，
但不是大幅 head壓縮。

## 解析成本

以 640輸入、P3/P4/P5位置數 `80²+40²+20²=8,400` 計：

| 比較 | two-branch params差 | two-branch理論差（2 ops/MAC） | Full35比例／角色 |
|---|---:|---:|---|
| H0 Detect80 → H1 Detect1 | -121,818 | -0.6795264 GFLOPs | -0.4592% params；主要是任務契約 |
| H1 Detect1 → H2 Lite64 | -878,592 | -4.1736192 GFLOPs | 真正縮 classification hidden tower |
| H0 Detect80 → H2 Lite64 | -1,000,410 | -4.8531456 GFLOPs | -3.7709% Full35 params |

fused inference只保留一組 active head，所以 H0→H2解析差約 `2.4265728 GFLOPs`。本地整圖建模得到相同
fused差值；它仍不是目標硬體 latency，必須 profile後才可宣稱加速。

class-score elements則由 fused H0的 `8,400×80=672,000` 降成 H1/H2的 `8,400`；FP32 tensor約少
2.53 MiB，且 top-k不用在80類上競爭。實際 memory與後處理效益仍須量測。

## 正確的 person-only 資料契約

本地 COCO2017 train共有118,287張影像，其中：

- 64,115張含 person，現行 YOLO labels有257,252個 person instances。
- 54,172張沒有 person，必須全部保留作真實負樣本。
- 原 labels另有592,690個其他類 instances；person task view只移除這些 label，不能移除影像。

不可在完整 COCO80 labels上直接設 `single_cls=True`。現行
[`update_labels()`](<../../../yolo_p2/ultralytics/data/base.py>)會把仍存在的每個 class id重寫成0；若未先
過濾，車、狗、球等都會被錯教成人。

正式做法是可重建的 COCO person Runtime Dataset View：

~~~text
原 train2017.txt / val2017.txt：完整保留影像身分與 split
原 YOLO labels：             只保留 class 0 rows
person-negative images：      保留，對應空 label
新 data YAML names：          {0: person}
model nc：                    1
prediction category mapping：head 0 → COCO category_id 1
~~~

它不是新的 COCO split；必須保存來源 manifest、逐檔映射與 digest，原 COCO資料維持唯讀。

## 為何保留 P3／P4／P5

COCO val的 non-crowd persons依官方 annotation `area`約為39.97% small、34.55% medium、25.48% large；按
bbox `w×h`則約29.75%／34.54%／35.71%。三個尺度都有大量樣本。

既有正式 bit-true predictions以 COCO API只取 `category_id=1`：

| 模型 | person AP | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|
| standalone Detect | 0.643591 | 0.432398 | 0.722596 | 0.838519 |
| Full35 J3 | 0.636740 | 0.429055 | 0.715476 | 0.831719 |
| J3−standalone | -0.006852 | -0.003344 | -0.007119 | -0.006800 |

J3的中、大人物下降不小於小人物，因此 P2不對應主要回歸。完整 P2 Detect在本地 fused nc=1分析反而增加
約14.2706 GFLOPs（+21.04%）；只在後續證據顯示短邊小人物是獨立瓶頸時才另案。

## 權重轉移

H1可作 epoch-0精確 graft：

- Backbone、Neck、box towers與 classification hidden layers全部複製。
- P3/P4/P5的 one-to-many與one-to-one class output conv只複製原 class-0 weight row與bias。
- 搬移後不可再呼叫通用 `bias_init()`覆蓋 person bias。
- 相同輸入下，raw person logits與boxes應在 FP32 tolerance內對齊 H0。

H2的 hidden width從256改64，無法保持完整 class tower函數等價。它保留相同 trunk／neck／box weights，窄
classification tower採固定seed初始化；這項優化風險本來就是 H2 intervention的一部分。若 H2只因明顯優化困難
失敗，teacher/KD必須另立後續 matched control，不能在首輪只替 H2加 KD。

## 首輪範圍

首輪納入：

- H0 canonical person-only evaluation。
- H1 class-0 exact graft、zero-train equivalence與標準 Detect1 control。
- H2只設定 `nc=1, cls_channels=64`；P3/P4/P5、bbox tower、Neck與backbone不變。
- H1/H2使用同一 person Runtime Dataset View與 matched training recipe。

首輪不納入：

- 完整 P2 Detect、P2-HCS auxiliary、刪 P5、固定人體 aspect ratio。
- centerness／quality scalar、pose/keypoint supervision、KD、Repulsion Loss或CrowdDet。
- MASF seam、RepConv、BinaryQK、PTQ或INT8 calibration。

## 風險與停止邊界

- H2變窄可能損失擁擠／姿態多樣性；必須看 AP_S/M/L、dense slices與負樣本誤報，不只overall AP。
- `iscrowd`目前未出現在YOLO txt；crowd區域可能被視為背景。首輪保持H1/H2一致，另記為loss/data後續方向。
- H1/H2一旦改 Detect supervision，舊 joint gradient、MASF、BinaryQK與PTQ證據都不能當 same-lineage最終證據。
- 沒有 target backend profile前，GFLOPs與tensor大小只能稱解析估算。
- 本方向尚未訓練，不能先宣稱 Lite64提高精度或速度。

返回[優化方向索引](<../README.md>)。
