# Full35 Quantization V0–V3 CPU實作與靜態分析報告

> 歷史狀態：本文凍結2026-08-30當時的qSiLU／`poly_quality`／poly_shift結果。2026-08-31後active parent已改為qSiLU／Hardswish／poly_shift；本文數值與hash不回寫，現行結論見[Hardswish policy修訂報告](2026-08-31-hardswish-policy-revision.md)。

日期：2026-08-30

## 結論先行

V0–V3已完成；本輪沒有使用GPU，沒有跑mAP validation、QAT、正式訓練或export。三個A8 activation parent各完成2,960筆靜態測量，合計8,880筆：

- uniform W8／W7／W6／W5／W4已全部納入，不再只看W8／W6／W4。
- Fixed SD4與Paper-TWN靜態式已實作並量測；LS-SD4、Channel-TWN、TTQ仍屬後續QAT，沒有偷跑。
- W8是目前最穩健的下一階段起點；W7與W6提供有意義的非2冪次位寬中間點。W5／W4保留在V4矩陣，但靜態重建誤差明顯升高。
- 全網Fixed SD4不勝uniform W4；不應整網替換。不過有34層在三個activation parent、master與deployment兩個view都穩定勝過matched W4，可作未來逐層routing候選。
- Paper-TWN在任何parent／view都沒有一層的靜態MSE勝過W4；它目前只適合作為約16×容量proxy與後續progressive ternary QAT研究，不適合作PTQ主線。
- V4已準備成`3 parents × 5 bits = 15`格，V5為`3 × 10 regions × 5 bits = 150`格；兩者都未執行。

這裡的NRMSE、SQNR與容量都是weight reconstruction proxy，不是mAP，也不能直接換算成準確率下降或GPU速度。

## V0：遠端publication與本機PTQ整合

`5090 Profile 0829`的公開activation報告已逐位元整合回本機；本機與publication commit `bf56249b`中的報告SHA-256同為`89c968ecc47f32f59da0d325f617549de3871f07d20038ce0df5f228bddf0709`。

歷史publication manifest的21個檔案中，15個仍逐位元一致、0個遺失；6個README／索引檔因本輪新增V2內容而正常演進。舊`backbone_early`三parent × W8／W4六格PTQ仍保留，但它使用修正前151-layer catalog與極小probe，只能當歷史診斷，不能和本輪V2混成正式結果。

Git工作樹仍是`main`相對遠端ahead 1／behind 52，整個子專案在根repo為untracked；本輪沒有checkout、reset、commit或push。`gh issue view 11 --comments`因本機未登入而不可用，因此實作權限來自本對話已確認的grilling決策，不假裝取得Issue #11內容。

完整血緣見[V0 manifest](../../artifacts/manifests/v0-lineage-reconciliation-v1.yaml)。

## V1：corrected catalog與dual views

### Graph分類修正

官方inference fuse會移除`pose_head.one2one_cv4_sigma.[0-2]`；舊catalog把它們算成部署predictor是錯的。本輪改成training-only後，真實分類為：

| 分類 | Modules | Weight elements |
|---|---:|---:|
| 部署候選 | 148 | 22,571,840 |
| Training-only | 99 | 3,748,480 |
| Binary Q/K protected | 4 | 131,072 |
| 合計 | 251 | 26,451,392 |

Binary Q/K、attention PWL、MASF拓樸、TopK／gather／decode都沒有被一般weight quantizer改寫。`reg_max=1`，因此沒有有效DFL量化；one-to-one end-to-end維持NMS-free。

### 為什麼需要兩個view

`Full35WeightViewAdapter`建立兩個互不污染的模型：

1. `master`：unfused FP32，保留BN與training supervision，供未來QAT。
2. `deployment`：BN-folded、移除training-only支路，供PTQ、static routing與export。

三個parent的179個BN都成功fold成0；148條deployment weight path、inference contract與one-to-one forward parity全部通過，來源模型也未被修改。

| Parent | Forward NRMSE | Max error／reference peak | 判定 |
|---|---:|---:|---|
| qSiLU＋A8 | 0.000003391 | 0.000007106 | pass |
| poly_quality＋A8 | 0.000004143 | 0.000014448 | pass |
| poly_shift＋A8 | 0.000004893 | 0.000022362 | pass |

只看單一view會導致錯誤判斷。例如qSiLU parent的uniform W4 weight NRMSE在master／deployment分別是`0.163698`／`0.095840`；Paper-TWN則相反，為`0.581763`／`0.677598`。BN folding確實改變格式適配性，因此QAT與export結果不得混表。

## V2：W8到W4、SD4與ternary靜態分析

### 公平矩陣

三個parent都使用自己的checkpoint；沒有把activation單獨winner和weight單獨winner事後拼接。主比較固定：

```text
per-output-channel + MSE grid-selected scale
W8 → W7 → W6 → W5 → W4
Fixed SD4：per-tensor/per-channel × max/MSE
Paper-TWN：delta=0.7×mean(abs(W))，layer-wise alpha
```

這裡的「MSE scale」是從`[1.0, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5] × max-range`十點grid中逐group選擇，不是連續解析最優值。

每個parent包含148層 × 2 views：uniform 1,480筆、Fixed SD4 1,184筆、Paper-TWN 296筆，共2,960筆。

### 全網deployment結果

下表的NRMSE／SQNR列出三parent範圍；容量包含weight codes與FP32 scales，FP32參考為90,287,360 bytes。

| Format | NRMSE range | SQNR dB range | Code＋scale | 對FP32壓縮 |
|---|---:|---:|---:|---:|
| W8 | 0.006267–0.006272 | 44.052–44.058 | 22,687,936 B | 3.98× |
| W7 | 0.012509–0.012514 | 38.052–38.056 | 19,866,456 B | 4.54× |
| W6 | 0.024540–0.024547 | 32.200–32.203 | 17,044,976 B | 5.30× |
| W5 | 0.049508–0.049521 | 26.104–26.107 | 14,223,496 B | 6.35× |
| W4 | 0.095830–0.095840 | 20.369–20.370 | 11,402,016 B | 7.92× |
| Fixed SD4 per-channel MSE | 0.160325–0.160336 | 15.899–15.900 | 11,402,016 B | 7.92× |
| Paper-TWN layer-wise | 0.677546–0.677598 | 3.381 | 5,643,552 B | 16.00× |

三個parent的weight-only結果非常接近，表示靜態routing訊號穩定；這不代表activation可互換。它們的activation-output誤差與上游mAP不同，V4仍必須保留三條完整policy。

### 從backbone到head的region指標

下表為三parent中較差的deployment weight NRMSE。數值只衡量該region的weight重建，語意上重要但參數很少的predictor仍可能對mAP極敏感。

| Region | Elements | W8 | W7 | W6 | W5 | W4 | Fixed SD4 | Paper-TWN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| backbone_early | 1,218,240 | .00545 | .01094 | .02158 | .04496 | .08950 | .15968 | .68302 |
| backbone_deep | 8,126,464 | .01384 | .02737 | .05319 | .09904 | .17328 | .18657 | .53895 |
| backbone_attention_safe | 919,808 | .01049 | .02085 | .04149 | .08032 | .14632 | .18302 | .52029 |
| neck | 8,142,848 | .01503 | .02978 | .05821 | .10811 | .18705 | .18488 | .58473 |
| MASF | 74,240 | .00393 | .00772 | .01559 | .03058 | .06357 | .15079 | .50873 |
| neck_attention_safe | 395,520 | .01103 | .02201 | .04382 | .08575 | .15890 | .17659 | .55923 |
| Detect one-to-one tower | 1,390,592 | .01243 | .02395 | .04469 | .07916 | .13849 | .14526 | .79450 |
| Detect predictor | 62,208 | .01074 | .02115 | .04217 | .08163 | .15044 | .17677 | .58415 |
| Pose one-to-one tower | 2,238,464 | .00428 | .00839 | .01682 | .03291 | .06535 | .14530 | .59531 |
| Pose predictor | 3,456 | .00581 | .01106 | .02197 | .04412 | .08631 | .17506 | .46042 |

靜態上neck、backbone_deep與Detect tower較敏感；Pose tower、MASF與backbone_early較耐uniform低位元。但V5仍要按八指標做isolated-region sensitivity，不能因weight NRMSE較低就直接量化predictor。

### SD4應該放哪裡

全網Fixed SD4的NRMSE比W4差，因此「小權重就用SD4」不是足夠規則。逐層matched比較則得到：

- deployment：三parent各37／148層SD4 MSE低於W4，而且三個集合完全相同。
- master：三parent各35／148層，而且三個集合完全相同。
- 三parent與兩view共同支持的保守交集：34層，分布為backbone_early 4、backbone_deep 4、neck 19、neck attention-safe 1、Detect tower 6。
- 保守34層使用SD4、其餘W4時，deployment全網NRMSE約從`0.09583–0.09584`降到`0.09471–0.09472`，相對改善只有約1.16–1.17%。

所以建議是「保留34層作V6 layer-level router候選」，不是把整個neck或全網改成SD4。候選清單見[Fixed SD4 routing manifest](../../artifacts/manifests/fixed-sd4-routing-candidates-v1.yaml)。

本輪完成的是一般Fixed SD4。使用learnable scale／STE的LS-SD4仍在V8 QAT，必須和matched W4 QAT、sham及相同budget比較；A-SD4則是更後面的activation-format研究，不能拿W-SD4結果直接套用。

### Ternary應該怎麼處理

Paper-TWN靜態式提供約16×容量，但三parent × 兩view中，沒有任何一層的MSE勝過W4，全網deployment NRMSE約0.678。第一版不能把它當PTQ主線。

它仍保留兩個價值：

1. 作為PDF第33–38頁原式的可重現static control。
2. 未來只對V5證明較耐量化的層做progressive ternary QAT；Paper-TWN、Channel-TWN與TTQ必須分開命名與比較。

## V3：固定診斷資料與八指標gate

固定manifest使用seed `20260830`，沒有建立新BBAT5 split：

| Split/task | 樣本 | Coverage |
|---|---:|---|
| Calibration COCO train | 32 | person影像16張 |
| Calibration BBAT formal-train | 32 | ball/bat皆有、44 pose rows、32個不同`.rf.` group |
| Probe COCO val | 64 | person影像42張 |
| Probe BBAT formal-val | 64 | ball/bat皆有、85 pose rows、64個不同`.rf.` group |

BBAT train／val source-group overlap為0；assignment與labels未更動。semantic SHA-256為`b0eb2a…eef2d`，實體檔SHA-256為`76b439…4356`，三次重建後沒有漂移。

正式gate要求以下八項全部存在：COCO box、COCO person box、BBAT overall box／pose、ball box／pose、bat box／pose。每項total drop不得低於`-0.04`；W8對matched activation parent每項incremental drop不得低於`-0.01`；QAT sham每項absolute drift不得超過`0.01`。`-0.04`到`-0.06`只進recover，不算通過。

本輪只實作與測試gate，沒有產生任何量化後mAP。

## 下一步方向與停止線

等GPU重新獲得明確授權後，建議V4仍跑完整15格，但採以下讀取順序：

1. 三parent的W8先驗證graph、probe與八指標；這是最穩健baseline。
2. W7、W6作非2冪次中間點；它們的static NRMSE約每降1 bit增加一倍，但是否值得取決於mAP與目標硬體packing。
3. W5、W4保留完整格，不先淘汰；若probe collapse可透過successive racing停止full validation。
4. Fixed SD4先只看34層保守router與matched W4，不做全網SD4。
5. Paper-TWN不進PTQ主線；等uniform sensitivity與QAT baseline完成後才考慮progressive QAT。

V4與V5設定已在[`v4-plus-prepared-plan-v2.yaml`](../../configs/experiments/v4-plus-prepared-plan-v2.yaml)，其中`execution_authorized: false`。目前明確停在GPU validation之前，也不會自動進QAT。

## 產物與驗證

- [V0–V3 delivery manifest](../../artifacts/manifests/v1-v3-cpu-delivery-v1.yaml)
- [固定32／64 diagnostic manifest](../../artifacts/manifests/full35-diagnostic-cal32-probe64-v1.json)
- [qSiLU V2完整JSON](../../artifacts/reports/weight-format-analysis-qsilu-pq-a8-main-v2.json)
- [poly_quality V2完整JSON](../../artifacts/reports/weight-format-analysis-poly-quality-a8-main-v2.json)
- [poly_shift V2完整JSON](../../artifacts/reports/weight-format-analysis-poly-shift-a8-main-v2.json)
- [Fixed SD4 routing候選](../../artifacts/manifests/fixed-sd4-routing-candidates-v1.yaml)

驗證結果：`53 passed`；Ruff check通過；32個Python檔皆符合formatter；12份YAML與7份JSON解析及交叉hash稽核通過。所有pytest與完整分析都在`CUDA_VISIBLE_DEVICES=-1`下執行。

## 限制

- V2只看weight本身，不是layer-output NRMSE、TopK overlap或mAP。
- V3 diagnostic manifest已建立，但尚未跑實際probe forward或validation。
- 尚無PTQ/QAT量化後八指標、latency、power、resource或bit-true export結果。
- W5／W6／W7沒有目標硬體kernel前，只能報容量與運算proxy，不能宣稱速度提升。
- Fixed SD4的34層清單是候選，不是已接受policy。
