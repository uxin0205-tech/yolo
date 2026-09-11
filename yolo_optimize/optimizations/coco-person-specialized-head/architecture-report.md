# COCO person-only：H0、H1、H2目前與預計架構

> 2026-09-08 使用者暫緩；本文保留為歷史提案，不列本輪主線，不建立 View 或改 head。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

> 狀態：結構分析／尚未修改 production model／尚未以 matched training驗證
>
> 方向 ID：`OPT-COCO-PERSON-SPECIALIZED-HEAD`

在 `yolo_optimize` 目錄可用下列命令於 terminal 重看：

~~~bash
sed -n '1,420p' optimizations/coco-person-specialized-head/architecture-report.md
~~~

## 0. 一句話判斷

只需要 person，不代表可以砍掉 P5或只用 P3。先把80類輸出改成1類，再把「仍然維持256 channels」的 person
classification tower縮成64；P3/P4/P5與 bbox tower先完全不動。

## 1. H0：一開始／目前的架構

~~~text
layer16：p3_raw ────────┬─> layer17 → layer19：p4_raw → layer20 → layer22：p5_raw
                         │
                         └──────────────────────────────────────────────────────────────┐
layer19：p4_raw ────────────────────────────────────────────────────────────────────────┤
layer22：p5_raw ────────────────────────────────────────────────────────────────────────┤
                                                                                        ↓
                                        Detect80([p3_raw, p4_raw, p5_raw], c3=256)
                                                                                        │
                                                                                        ▼
                                                      8,400 locations × 80 class scores
                                                                                        │
                                                                                        ▼
                                                           application filter class 0/person
~~~

Full35同一組 shared P3/P4/P5另供 Pose26 head使用；本方向只畫 Detect branch。Detect是 end-to-end dual-head：

~~~text
training： one-to-many box/class towers + one-to-one box/class towers
fused：   只保留 one-to-one active towers
~~~

## 2. H0單一尺度內部

以某尺度輸入 `x∈R^(C×H×W)` 為例：

~~~text
                         ┌─> Box tower：Conv(C→64) → Conv(64→64) → Conv(64→4)
x(P3/P4/P5) ─────────────┤
                         └─> Class tower：DWConv(C) → PWConv(C→256)
                                          → DWConv(256) → PWConv(256→256)
                                          → Conv(256→80)
~~~

三尺度各一組，one-to-many與one-to-one再各一份。`nc=80`的最後一層雖明顯，但大部分 class tower計算在前面的
256-channel hidden layers。

## 3. H1：先校正成真正的 person-only任務

~~~text
layer16：p3_raw ────────┬─> layer17 → layer19：p4_raw → layer20 → layer22：p5_raw
                         │
                         └──────────────────────────────────────────────────────────────┐
layer19：p4_raw ────────────────────────────────────────────────────────────────────────┤
layer22：p5_raw ────────────────────────────────────────────────────────────────────────┤
                                                                                        ↓
                                         Detect1([p3_raw, p4_raw, p5_raw], c3=256)
                                                                                        │
                                                                                        ▼
                                                        8,400 locations × 1 person score
~~~

單尺度：

~~~text
                         ┌─> Box tower：完全不變，64→4
x(P3/P4/P5) ─────────────┤
                         └─> Class tower：DW/PW → 256 → DW/PW → 256 → Conv(256→1)
~~~

H1只將 final conv的80 outputs改為1，hidden width仍是256。這一步的重要性是 task/loss/output/top-k語意正確，
不是主幹大幅加速。

### 3.1 H1 epoch-0搬移

~~~text
H0 final class weight [80,256,1,1] ── take row 0 ──> H1 [1,256,1,1]
H0 final class bias   [80]           ── take row 0 ──> H1 [1]

對 P3/P4/P5 × one-to-many/one-to-one 共6個 final class conv都做同一映射
~~~

box與 person raw logits應對齊；如果不對齊，先修 graft，不可用 recovery training掩蓋搬移錯誤。

## 4. H2：希望的 Detect1-Lite64

~~~text
layer16：p3_raw ────────┬─> layer17 → layer19：p4_raw → layer20 → layer22：p5_raw
                         │
                         └──────────────────────────────────────────────────────────────┐
layer19：p4_raw ────────────────────────────────────────────────────────────────────────┤
layer22：p5_raw ────────────────────────────────────────────────────────────────────────┤
                                                                                        ↓
                                     Detect1-Lite64([p3_raw, p4_raw, p5_raw], c3=64)
                                                                                        │
                                                                                        ▼
                                                        8,400 locations × 1 person score
~~~

單尺度：

~~~text
                         ┌─> Box tower：完全不變，Conv(C→64) → Conv(64→64) → 64→4
x(P3/P4/P5) ─────────────┤
                         └─> Class tower：DWConv(C) → PWConv(C→64)
                                          → DWConv(64) → PWConv(64→64)
                                          → Conv(64→1)
~~~

H2利用的先驗很單純：只有一個 personness／quality-aligned classification output，不一定需要為80類分離而保留
256-channel hidden space。它不是固定人體模板；框的 `x,y,w,h`、多尺度特徵與 assigner全部保留。

## 5. 為何 `nc=1` 不會自動得到 H2

現行 [`Detect.__init__()`](<../../../yolo_p2/ultralytics/nn/modules/head.py>)預設：

\[
c_2=\max(16,ch_0/4,4\,reg\_max)=64,
\]

\[
c_3=\max(ch_0,\min(n_c,100))=256.
\]

所以：

~~~text
H0：nc=80, cls_channels=None → c3=256
H1：nc=1,  cls_channels=None → c3=256
H2：nc=1,  cls_channels=64   → c3=64
~~~

H2必須顯式傳入 `cls_channels=64`；不能只改 YAML的 `nc`後假定分類塔已變窄。

## 6. 參數量推導

對每個尺度的一個 class branch，計入 Conv weights、BN affine與final bias。本地解析總數：

| 架構 | 每組 active class towers | two branches |
|---|---:|---:|
| H0 Detect80, c3=256 | 611,568 | 1,223,136 |
| H1 Detect1, c3=256 | 550,659 | 1,101,318 |
| H2 Detect1, c3=64 | 111,363 | 222,726 |

因此：

~~~text
H0 → H1：1,223,136 - 1,101,318 =   121,818 params
H1 → H2：1,101,318 -   222,726 =   878,592 params
H0 → H2：1,223,136 -   222,726 = 1,000,410 params
~~~

H0→H2約等於現行 Full35 shared model `26,529,701` parameters的3.7709%。fuse後只保留一組 active tower，
對應差值減半。

## 7. 運算量推導

三尺度位置總數：

\[
N=80^2+40^2+20^2=8,400.
\]

H0→H1只差 final class conv：

\[
\Delta MAC_{active}=79\times256\times8,400=169,881,600.
\]

以 `2 ops/MAC`，two-branch training graph差 `0.6795264 GFLOPs`。H0→H2的 class tower解析差為
`4.8531456 GFLOPs`，其中 H1→H2為 `4.1736192 GFLOPs`；fused H0→H2為 `2.4265728 GFLOPs`。

~~~text
解析值回答：Conv乘加少多少
get_flops回答：目前框架建模少多少
target profile回答：真實 latency／throughput／energy少多少
~~~

三者不能互相冒充。本方向升格必須有第三項。

## 8. 為何不先加 P2，也不刪 P5

~~~text
COCO person bbox尺度：small + medium + large皆占大量比例
                         │
                         ├─ P3：保留小／中尺度入口
                         ├─ P4：保留中尺度入口
                         └─ P5：保留大人物與全域語意入口
~~~

standalone→J3的 person AP差：

~~~text
overall -0.006852
AP_S    -0.003344
AP_M    -0.007119
AP_L    -0.006800
~~~

主要 joint回歸不是只集中在small。完整 P2 Detect會大幅增加160×160路徑成本；它只能在 H1/H2完成後、short-box
recall仍獨立落後時另案。刪 P5則會在沒有assignment證據時犧牲大量large persons。

## 9. 資料流不能犯的錯

錯誤：

~~~text
COCO80 labels ── single_cls=True ──> car/dog/ball/... 全部被改成 class 0/person
~~~

正確：

~~~text
同一 train/val image manifests
       │
       ├─ 原 class 0 rows ───────────> person positives
       ├─ 其他79類 rows ─────────────> 從 person task labels移除
       └─ 沒有 class 0的images ──────> 保留為 empty-label negatives
~~~

這個 Runtime Dataset View只隔離 label/cache，不擁有新的split語意。

## 10. Full35整合後

~~~text
shared YOLO26 layers0–22
          ├─> winner：H1 Detect1 或 H2 Detect1-Lite64 ──> person boxes
          └─> 原 BBAT5 Pose26 ──────────────────────────> ball/bat box + keypoints
~~~

接著才重新量 `g_detect ↔ g_pose`。若達[衝突安全訓練方向](<../training-conflict-safe/README.md>)的trigger，跑
G0/G1；否則直接凍結 BASE-FP。MASF、RepConv與BinaryQK都在此之後，不能沿用舊 head的最終 calibration。

## 11. 最小決策圖

~~~text
H0 Detect80 canonical person baseline
                │
                ▼
H1 Detect1 exact graft + zero-train equivalence
                │
       ┌────────┴────────┐
       │失敗             │通過
       ▼                 ▼
   修資料／graft      H1 vs H2 Lite64 matched seed 0
                              │
                   ┌──────────┴──────────┐
                   │H2 accuracy/成本失敗 │通過
                   ▼                     ▼
             保留 H1 winner        補 paired seeds 1/2
                                         │
                                  ┌──────┴──────┐
                                  │失敗         │通過
                                  ▼             ▼
                              保留 H1       升格 H2
~~~

P2-HCS、centerness、keypoint/KD、crowd loss都不在這張首輪決策圖；它們需要各自的診斷與 matched control。

返回[方向說明](<README.md>)、[最小計畫](<plan.md>)或[優化方向索引](<../README.md>)。
