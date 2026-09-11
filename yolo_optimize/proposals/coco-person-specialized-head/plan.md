# COCO person-only 專用 head：最小實驗計畫

> 2026-09-08 使用者暫緩；本文保留為歷史提案，不列本輪主線，不建立 View 或改 head。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

本計畫先把任務從 COCO80應用端過濾，改成真正的單類 person Detect；接著只測一個結構因子：classification
hidden width `256→64`。詳細證據見[方向說明](<README.md>)，圖與成本推導見[架構圖報告](<architecture-report.md>)。

## 一、整體位置

~~~text
現行 COCO80 / Full35 parent
              │
              ▼
person Runtime Dataset View + canonical evaluator
              │
              ▼
H0 Detect80 read-only baseline → H1 Detect1 exact graft
                                      │
                                      ▼
                             H1 vs H2 Lite64 matched training
                                      │
                                      ▼
                         凍結 standalone person head winner
                                      │
                                      ▼
                     joint training + conflict-safe方向（若觸發）
                                      │
                                      ▼
                       MASF／RepConv／BinaryQK／PTQ／export
~~~

person head是任務定義的一部分，不能留到已完成 BinaryQK或PTQ後才改。任何 head winner進 joint graph後，都需
重新訓練／校正下游方向，舊 COCO80結果只保留為歷史 baseline。

## 二、Phase 0：先建立資料與 evaluator契約

建立可重建的 COCO person Runtime Dataset View，不建立新 split：

1. 原樣沿用 `train2017.txt`的118,287張與 `val2017.txt`的5,000張影像身分、順序與split。
2. 每個 label檔只保留原 class 0 row；輸出 class id仍為0。
3. person-negative影像保留空 label，包括原本只有其他類別的影像。
4. 保存 source→view逐檔manifest、source/view label digests、positive/negative/image/instance counts。
5. YAML只宣告 `{0: person}`；training不得再設 `single_cls=True`。
6. prediction輸出把 head class 0明確映射回 COCO `category_id=1`。

硬 gate：

| 檢查 | 預期值 |
|---|---:|
| train images | 118,287 |
| train person-positive／negative | 64,115／54,172 |
| train YOLO person instances | 257,252 |
| val images | 5,000 |
| val person-positive／negative | 2,693／2,307 |
| val YOLO person instances | 10,777 |
| view中非0 class rows | 0 |

任一數值不符就停止；不能藉重切或丟棄 negative images規避。

canonical evaluator固定使用官方 COCO API的 bbox AP@[.50:.95]與 `catIds=[1]`。Ultralytics internal AP只能另欄
報告，不能與 COCO API AP混成同一數列。

## 三、Phase 1：H0與 H1 zero-train gate

### H0：既有對照

H0不新增訓練；凍結要採用的 exact parent、prediction JSON與digest，重算 person AP。現有可參考的 bit-true
數值是 standalone `0.643591`、Full35 J3 `0.636740`，但正式實驗仍須清楚指定哪一個 parent進下一步。

### H1：標準 Detect1 graft

從同一 H0 parent建立 `nc=1, cls_channels=None`：

- trunk、neck、box towers與 class hidden layers逐 tensor複製。
- one-to-many／one-to-one、P3/P4/P5每個 final class conv只取 H0 class-0 weight row與bias。
- 搬完不可再執行通用 `bias_init()`。

zero-train gate：

- 相同輸入與前處理下，H1 raw boxes等於 H0 boxes。
- H1 raw class logit等於 H0 class-0 raw logit，`max_abs<=1e-5`。
- state差異只允許六個 final class conv的 output shape由80變1；其餘 expected tensors與digest一致。
- save/reload、EMA、one-to-many／one-to-one、fuse/export與 class-id mapping全部通過。

zero-train canonical AP可能因 H0的80類 top-k競爭被移除而略變；這要另記為輸出契約效果，不可稱為訓練增益。

## 四、Phase 2：唯一首輪 training矩陣

| Arm | Detect設定 | 初始化 | 唯一角色 | seed-0新 jobs |
|---|---|---|---|---:|
| `H1-DETECT1` | `nc=1, c3=256, P3/P4/P5` | H0 class-0 exact graft | 單類任務與容量控制 | 1 |
| `H2-LITE64` | `nc=1, c3=64, P3/P4/P5` | 相同 H0 trunk/neck/box；窄 cls tower固定seed初始化 | 測單類分類塔能否縮窄 | 1 |

兩臂完全相同：person view、train/val manifests、parent revision、seed、sampler、augmentation、imgsz、batch、
optimizer、LR、scheduler、epochs、patience、AMP、BN、optimizer steps與 evaluator。不能只替 H2加 KD、長 warmup或
更多 epochs。

首輪建議先做 standalone Detect，避免把 BBAT5 Pose與 shared-gradient policy混入 head容量判斷。winner確認後才進
joint graph，並重新走 J0→J3／衝突安全方向的必要 gate。

## 五、H2實作 gate

現行 [`Detect`](<../../../yolo_p2/ultralytics/nn/modules/head.py>)已有 `cls_channels`參數；parser也有
對應傳遞 seam。H2只允許：

~~~text
nc = 1
cls_channels = 64
ch = [256, 512, 512]
strides = [8, 16, 32]
reg_max = 1
end2end = true
~~~

必測：

- P3/P4/P5 box tower state與H1 shape完全相同。
- one-to-many／one-to-one兩套 class tower均為64，不能只縮一支。
- raw output channel、decode、loss target、top-k與 COCO category mapping正確。
- fused graph移除 training-only branch後只有一組64-channel class tower。
- H0→H1、H1→H2、H0→H2 parameters／FLOPs解析值與建模值一致。
- export後不存在256-channel person cls tower或80-class final logits。

## 六、指標與 seed-0停止條件

主指標與診斷：

- canonical person AP、AP50、AP75、AP_S、AP_M、AP_L、AR100。
- 短邊 `<16px`／`<32px` recall；另標自訂 slice，不冒稱官方 AP_S。
- persons/image `>=5`、`>=10`與 pair-IoU `>=0.3` dense slices。
- person-negative images的 FP/image，在固定 score thresholds與 matched max_det下報告。
- raw/fused Params、GFLOPs、score tensor bytes、batch-1 latency、throughput、peak inference VRAM。
- training wall time／epoch與 peak VRAM。

H1先決條件：相對 H0 canonical person AP不得下降超過 `0.001`，且負樣本 FP/image不得惡化超過事前容許值；
否則先修 task view／loss／mapping，不進 H2升格。

H2 seed-0相對 H1：

- person AP不得下降超過 `0.001`。
- AP_S／AP_M／AP_L任一不得下降超過 `0.002`。
- AP75、dense slices與negative FP不得出現未解釋的重大退化。
- fused target latency改善至少超過量測noise，建議預先登錄 `>=3%`；若只有GFLOPs下降而真實 latency不變，
  不升格為部署優化。
- graph/state/export與成本計數全部通過。

任一 accuracy或工程 gate失敗即停止 H2，不加入 P2、centerness、pose/KD或不同 LR來救它。H1若過 gate，仍可
單獨成為正確的 person-only winner。

H2 seed 0全過後，補 H1/H2 paired seeds 1、2，以三組 mean/std套相同 gate。三 seeds前只能標
`provisional`。

## 七、條件式後續，不在首輪執行

只有 winner的誤差分解支持單一瓶頸，才另開一個方向：

| 診斷 | 可考慮的下一項 | 為何現在不做 |
|---|---|---|
| 短邊人物 recall獨立落後 | training-only P2-HCS auxiliary | 完整 P2 fused成本約+21%；現有joint回歸不是small-only |
| confidence與IoU排序失配 | 1-channel quality／centerness | 會改NMS-free top-k、loss與export語意 |
| 遮擋／姿態造成定位錯誤 | training-only keypoint或teacher KD | 本地尚無COCO keypoint JSON；增加teacher變因 |
| crowd區域誤當背景 | `iscrowd` ignore contract | 屬data/loss語意，不能混成head寬度效果 |
| dense slice仍特殊退化 | YOLO-compatible repulsion loss | 先證明ignore修正後仍有問題 |

不因只偵測人就刪 P5，也不固定人體 aspect ratio；COCO人物尺度與姿態分布不支持這兩個捷徑。

## 八、交付與結果契約

每個 arm保存：

- source/view manifests與digests、positive/negative/instance統計。
- parent、code、config、seed、sampler與trainable names。
- H1 transfer map；H2 initialization policy與 state schema。
- raw/fused模型計數、export graph、target profile與 prediction JSON。
- canonical person metrics、全部 slices、negative FP與失敗理由。
- 若進 joint：BBAT5正式 gates、Float／Bit-True parity與新 shared-gradient統計。

摘要寫到本方向未來的 `results.md`；大型 checkpoint與原始 artifacts留在正式實驗目錄。

返回[方向說明](<README.md>)或[優化方向索引](<../README.md>)。
