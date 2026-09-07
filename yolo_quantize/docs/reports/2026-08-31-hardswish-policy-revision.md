# Full35 Quantization：Hardswish Policy 修訂報告

日期：2026-08-31

## 結論先行

本次已完成規畫與CPU實作修訂，沒有使用GPU、沒有跑新mAP validation、沒有QAT或正式訓練：

1. 現行uniform A8 parent改為qSiLU、Hardswish、poly_shift；`poly_quality`退出future matrix、active CLI、racing與QAT promotion。
2. 舊`poly_quality`的V0–V2 JSON、view manifest、報告、圖表與hash全部保留為歷史事實，不刪除、不改名，也不把2026-08-31決策倒寫到舊實驗。
3. 使用者指定的[Q3報告](https://github.com/uxin0205-tech/yolo/blob/main/yolo_activation/reports/full35-q3-cpu-final-report.md)確實需要取用，但它支持的是「qSiLU checkpoint＋單區Hardswish」，不是uniform Hardswish勝出。本專案因此另設Q3 regional branch。
4. Uniform Hardswish使用本機epoch-8 `best_joint.pt`作experimental parent，已補checkpoint-level intake、dual-view parity及2,960筆CPU weight analysis；它不等於Q3 parent，也不是正式activation winner。
5. Q3優先的`neck_attention`、`masf`、`backbone_attention`三個單區policy都已通過CPU dual-view parity；尚未做A8 observer calibration、W8 PTQ或mAP。
6. Active三parent重新計算Fixed SD4 routing後，保守交集仍是34層，與歷史v1集合相同；因來源parent hashes已變，另發v2 manifest，不能沿用舊標籤。

## 現行與歷史邊界

| 身分 | 狀態 | 用途 |
|---|---|---|
| qSiLU＋A8，SHA `767918…6190e` | active uniform parent | accuracy fallback；也是Q3 regional Hardswish的checkpoint root |
| Uniform Hardswish＋A8，SHA `79e0e4…7731` | active experimental parent | 標準硬體operator軸；必須綁epoch-8 best-joint selector |
| poly_shift＋A8，SHA `878324…9713` | active uniform parent | dyadic／APoT hardware extreme |
| `poly_quality`＋A8，SHA `eedd48…c968` | historical only | 保留已完成numeric proxy與V1/V2證據，不再排程 |
| Accepted SiLU | historical control | 不新增SiLU實驗 |

使用者寫的`poly_quantity`在程式與上游registry中的正式識別字是`poly_quality`；本報告以正式名稱記錄排除範圍。

## 為什麼Hardswish可以納入，但必須分成兩條證據

### Uniform Hardswish checkpoint

本機Hardswish short recovery完成10 epochs。上游run-level frozen selector看final epoch，所以原結論仍是fail；但量化要使用的`inference/best_joint.pt`是epoch 8，不能把final-epoch數值錯綁到該檔案。

| Selector | Epoch（zero-based） | 八項最差delta | 判定 |
|---|---:|---:|---|
| `best_joint.pt`實際metadata | 8 | `-0.014121`，COCO box | 通過舊`-0.015`與本專案`-0.04`screen |
| 上游run final epoch | 9 | `-0.016884`，BBAT bat pose | 依上游final-epoch契約失敗 |

兩個判定同時為真，不能互相覆蓋。現行量化parent使用者明確選擇前者，但仍標為experimental、非上游正式release。八項checkpoint metrics、來源檔hash、CIoU precision與selector邊界已凍結於[`hardswish-uniform-parent-intake-v1.yaml`](../../artifacts/manifests/hardswish-uniform-parent-intake-v1.yaml)。

另有一個公平性風險：Hardswish與poly_shift recovery使用FP32 CIoU數值穩定修正，qSiLU recovery早於該修正。後續PTQ本身不會再訓練這些權重，但若進QAT，matched sham與optimizer／loss precision必須重新凍結，不能把差異歸因給量化。

### Q3 regional Hardswish

Q3以qSiLU checkpoint `767918…6190e`為唯一權重根，190個activation sites全部非SiLU；每次只有一個region改成Hardswish，其餘保持qSiLU。完整CPU／Bit-True、COCO 5,000張val與BBAT5 683張formal val的三個優先單區為：

| Region | Hardswish sites | qSiLU sites | 八項最差global delta | Q3舊`-0.015`gate |
|---|---:|---:|---:|---|
| `neck_attention` | 1 | 189 | `-0.008581` | 通過 |
| `masf` | 3 | 187 | `-0.010301` | 通過 |
| `backbone_attention` | 3 | 187 | `-0.013518` | 通過 |

這些數字可以決定placement優先序，但不能直接當成A8、W8、GPU或硬體效能結果。`neck_attention + masf`尚未量測，單區delta不可相加。更完整的來源、遠端commit/blob與不可跨用邊界見[Q3證據查核](2026-08-31-q3-activation-parent-evidence.md)。

## 已完成的程式與CPU證據

### Regional policy seam

`Full35ActivationPolicy`現在可表達：

```text
default activation + sorted region assignments + LSQ+ output bit width
```

例如：

```text
qsilu_pq--regional--neck_attention=hardswish--lsq-plus-a8
```

adapter會在修改前檢查重複／未知region，透過上游190-site manifest建立static policy，並保存每種activation的site count。Activation-output quantizer仍只包住124個deployment-eligible sites；66個training-only paths不計部署收益。

`prepare_v1_v3.py`新增：

- active `--parent` choices：`qsilu_pq`、`hardswish`、`poly_shift`；
- `--historical-parent poly_quality`：只允許重建凍結歷史；
- `--activation-region REGION=ACTIVATION --view-only`：建立regional dual-view parity，不重跑相同checkpoint的weight-only sweep。

### Uniform Hardswish V1/V2

Hardswish已完成179→0 BN fold、148條deployment path與one-to-one CPU forward parity：

| 指標 | 結果 |
|---|---:|
| Forward NRMSE | `5.759842e-6` |
| Max error／reference peak | `3.202221e-5` |
| Source unchanged | true |
| Inference contract parity | true |
| GPU used | false |

V2 main profile為2 views × 148 layers，共2,960筆：uniform 1,480、Fixed SD4 1,184、Paper-TWN 296；沒有NaN／Inf。

| Format | Hardswish deployment NRMSE | Active三parent範圍 |
|---|---:|---:|
| W8 | `0.006240` | `0.006240–0.006272` |
| W7 | `0.012459` | `0.012459–0.012514` |
| W6 | `0.024462` | `0.024462–0.024547` |
| W5 | `0.049417` | `0.049417–0.049521` |
| W4 | `0.095724` | `0.095724–0.095840` |
| Fixed SD4 per-channel MSE | `0.160400` | `0.160327–0.160400` |
| Paper-TWN | `0.677474` | `0.677474–0.677598` |

這些都是weight reconstruction proxy，不是mAP。Hardswish的W8 NRMSE略低，不代表它的activation-output或完整模型精度更好。

### Regional Hardswish dual-view parity

| Policy | Forward NRMSE | Max error／reference peak | 結果 |
|---|---:|---:|---|
| qSiLU＋Hardswish `neck_attention` | `3.395364e-6` | `7.118448e-6` | pass |
| qSiLU＋Hardswish `masf` | `3.423228e-6` | `6.440059e-6` | pass |
| qSiLU＋Hardswish `backbone_attention` | `3.809068e-6` | `5.286480e-6` | pass |

三者的BN fold、deployment path與inference contract都通過；只建立view manifest，因它們共用完全相同的qSiLU權重根，重複宣稱另外3×2,960筆weight-only結果沒有資訊增益。

## Fixed SD4重新路由

把active parent換成qSiLU／Hardswish／poly_shift後重新逐層比較`Fixed SD4 per-channel MSE`與matched uniform W4：

- 三個active parent在deployment各有37／148層SD4勝W4，集合完全相同。
- master各有35／148層，集合完全相同。
- 三parent、兩view的保守交集仍為34層：backbone early 4、backbone deep 4、neck 19、neck attention-safe 1、Detect one-to-one tower 6。
- 34層hybrid相對all-W4的deployment NRMSE改善約`1.116%–1.173%`，幅度小，仍只是V6候選。

清單雖與歷史v1相同，證據來源已改，因此現行入口是[`fixed-sd4-routing-candidates-v2.yaml`](../../artifacts/manifests/fixed-sd4-routing-candidates-v2.yaml)。這不代表Fixed SD4已通過mAP，也不代表LS-SD4或A-SD4有效。

## 修訂後實驗矩陣

### V4 uniform bridge：15個量化cell

```text
[qSiLU, uniform Hardswish, poly_shift]
× [W8, W7, W6, W5, W4]
= 15
```

另要求每個parent的FP32-weight matched control，但不把control混算成15個量化cell。先做三parent W8 graph／probe／完整八指標，再依序W7、W6、W5、W4；successive racing只節省full validation，不省略artifact或sentinel稽核。

### V4H regional bridge：先6個、最多再2個條件式cell

```text
3個Q3單區policy × [matched FP32-weight control, W8] = 6
```

只有`neck_attention`與`masf`的單區W8都通過，才新增其組合的control／W8兩格。Regional policy通過W8後才可升級W7–W4或進V5；目前不預先建立大型笛卡兒積。

### V5 uniform weight-region sensitivity：150個量化cell

```text
3 uniform parents × 10 weight regions × 5 weight bits = 150
```

注意Q3的activation region與V5 weight region不是同一組module paths：例如`neck_attention`只有1個activation site，而`neck_attention_safe`有5個Conv／Linear weights。兩個軸必須分開記錄，不能因名稱接近就視為等價。

完整未執行設定見[`v4-plus-prepared-plan-v3.yaml`](../../configs/experiments/v4-plus-prepared-plan-v3.yaml)，其中`execution_authorized: false`。

## Gate與停止線

每個未來cell仍要求八項mAP50-95：COCO box／person、BBAT overall box／pose、ball box／pose、bat box／pose。

- total drop：每項不得低於accepted `-0.04`；
- W8 incremental：每項相對同一activation policy的FP32-weight matched control不得低於`-0.01`；
- QAT sham drift：每項absolute drift不得超過`0.01`；
- `-0.04`到`-0.06`只進recovery，不算通過。

Q3原`-0.015`標籤保持來源語意，不取代本專案gate。Regional policy的matched control必須是相同region assignments＋FP32 weights，不可直接拿uniform qSiLU或uniform Hardswish作incremental baseline。

本次停止於CPU準備完成：

- 未使用GPU；
- 未跑新PTQ calibration／probe forward或完整mAP；
- 未啟動QAT、正式訓練、export或硬體benchmark；
- 未建立`neck_attention + masf`組合結果；
- 未宣稱Hardswish、qSiLU、poly_shift、W8、SD4或ternary勝出。

## 下一個可執行順序

取得新的GPU執行授權後：

1. V4三個uniform parent各跑matched FP32-weight control與W8，先驗證graph、observer及八項指標。
2. V4H依Q3順序跑`neck_attention`、`masf`、`backbone_attention`的matched control／W8。
3. Uniform W8與regional W8結果完整報告後，再決定是否展開W7／W6；W5／W4保留但可經racing停止full validation。
4. 只有通過的policy進V5；Fixed SD4先測34層router＋matched W4，不做全網SD4。
5. PTQ超出預算才進matched QAT；不會自動開始。

## 主要產物

- [現行machine-readable計畫](../../configs/experiments/full35-quantization-plan-v3.yaml)
- [未執行V4+矩陣](../../configs/experiments/v4-plus-prepared-plan-v3.yaml)
- [Uniform Hardswish checkpoint intake](../../artifacts/manifests/hardswish-uniform-parent-intake-v1.yaml)
- [Uniform Hardswish dual-view manifest](../../artifacts/manifests/weight-views-hardswish-a8-v3.json)
- [Uniform Hardswish 2,960筆分析](../../artifacts/reports/weight-format-analysis-hardswish-a8-main-v3.json)
- [Active Fixed SD4 routing v2](../../artifacts/manifests/fixed-sd4-routing-candidates-v2.yaml)
- [Q3 parent證據查核](2026-08-31-q3-activation-parent-evidence.md)

最終測試、hash總表與工作樹風險見[`v1-v3-cpu-delivery-v2.yaml`](../../artifacts/manifests/v1-v3-cpu-delivery-v2.yaml)及[2026-08-31工作紀錄](../worklogs/2026-08-31-hardswish-policy-revision.md)。
