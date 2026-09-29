# 2026-08-31 Q3 Activation Parent 證據查核

> 狀態：唯讀研究完成；沒有使用 GPU、沒有訓練、沒有執行 validation，也沒有修改
> `yolo_activation`。本文只提供 `yolo_quantize` 修訂依據，不宣稱已完成新 PTQ／QAT。

## 結論

1. **Hardswish 應納入後續量化研究，但應先納入為 qSiLU 權重根上的 regional
   activation policy，不應把 Q3 誤寫成 uniform Hardswish 已勝出。** Q3 實測支持的三個
   deployment placement 依序是 `neck_attention`、`masf`、`backbone_attention`；三區組合
   尚未量測。[指定 Q3 報告](https://github.com/uxin0205-tech/yolo/blob/30bd50e2c9e16feb7adb989cb042e1a13091c067/yolo_activation/reports/full35-q3-cpu-final-report.md#hardswish11-region-%E5%AE%8C%E6%95%B4%E8%A1%A8)
   與[機器可讀 summary](../../../yolo_activation/reports/q3-nonsilu-cpu/summary.json)一致。
2. **`poly_quality`（使用者寫作 `poly_quantity`）應從 2026-08-31 之後的 active parent
   matrix、racing、QAT promotion 與 CLI choices 排除；歷史證據不得刪除或改寫。** 它曾有
   zero-shot／A8 proxy 優勢，但 10-epoch run 的 final-epoch 八項 gate 未通過；更重要的是，
   使用者已明確改變研究範圍。刪除既有 V2 JSON、view manifest 或報告會破壞 V0–V2 血緣，
   也會讓既有 hash 無法重建。[上游最終分析](../../../yolo_activation/reports/full35-activation-final-analysis.md#L96)
3. 建議把新 active policy 拆成兩層身分：

   - checkpoint root：`qsilu_pq`、`poly_shift`；
   - activation placement：uniform qSiLU、uniform poly_shift，以及以 qSiLU checkpoint 為根的
     Hardswish `neck_attention`／`masf`／`backbone_attention` 單區 policy。

   這比把「Hardswish」當成第三個沒有 region 語意的 uniform parent 更忠於 Q3 證據，也避免
   把 activation 與 weight checkpoint 的耦合關係抹掉。

## 證據身分與完整性

- 查核的 GitHub `main` commit 是
  [`30bd50e2c9e16feb7adb989cb042e1a13091c067`](https://github.com/uxin0205-tech/yolo/tree/30bd50e2c9e16feb7adb989cb042e1a13091c067)。
- 遠端 Q3 報告 Git blob 是 `cf21ddd028f527f0d6a00bdd28531b3fc9014a7c`、13,794 bytes；
  本機[同名報告](../../../yolo_activation/reports/full35-q3-cpu-final-report.md)的 Git blob 與 byte
  size 完全相同。遠端 metadata 可由
  [GitHub Contents API](https://api.github.com/repos/uxin0205-tech/yolo/contents/yolo_activation/reports/full35-q3-cpu-final-report.md?ref=30bd50e2c9e16feb7adb989cb042e1a13091c067)
  查核。
- Q3 報告明定 22/22 是完整 CPU zero-shot validation，不含 optimizer、backward、epoch 或權重
  更新；每次只替換一個 region，其他 190-site policy 保持 recovered qSiLU。參見
  [Q3 報告實驗定義](https://github.com/uxin0205-tech/yolo/blob/30bd50e2c9e16feb7adb989cb042e1a13091c067/yolo_activation/reports/full35-q3-cpu-final-report.md#q3-%E5%AF%A6%E9%9A%9B%E5%9C%A8%E6%B8%AC%E4%BB%80%E9%BA%BC)
  與[runner 的 qSiLU checkpoint、候選及 region 常數](../../../yolo_activation/scripts/full35_nonsilu_q3.py#L42)。
- [190-site manifest](../../../yolo_activation/reports/q3-nonsilu-cpu/q3-qsilu-recovery-activation-manifest.yaml)
  固定 module path 與 region；runner 會驗證 site 數為 190、region 集合完全相符、checkpoint
  SHA 不漂移，並以 `default_activation="qsilu_pq"` 加單一 `region_assignments` 建立 candidate。
  參見[runner 契約檢查](../../../yolo_activation/scripts/full35_nonsilu_q3.py#L288)與
  [policy 建構](../../../yolo_activation/scripts/full35_nonsilu_q3.py#L907)。
- Q3 的 CPU fail-closed 防護、完整資料與不重新切分紀錄見
  [Q3 工作紀錄](../../../yolo_activation/docs/worklogs/2026-08-31-full35-nonsilu-q3-cpu-sensitivity.md#L10)。

## Hardswish 應如何納入

### 已量測

Q3 的 Hardswish candidate 全都使用下列**同一份 qSiLU 權重**，不是 Hardswish recovery 權重：

```text
/home/uxin/yolo/yolo_activation/artifacts/runs/full35/
short-recovery-v2-lr01-uniform-qsilu-pq-seed1/inference/best_joint.pt
SHA-256: 7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e
```

完整 CPU／Bit-True evaluator 的 matched 結果如下；八項 `global Δ` 都是相對同裝置 accepted SiLU
baseline，而不是相對既有 `yolo_quantize` GPU 或 diagnostic 數字：

| Policy | Hardswish sites | COCO box | COCO person | BBAT box | BBAT pose | Ball box | Bat box | Ball pose | Bat pose | 最差 global Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform qSiLU reference | 0 | 0.495313 | 0.619684 | 0.625257 | 0.901469 | 0.497670 | 0.752845 | 0.857103 | 0.945834 | −0.008810 |
| qSiLU + Hardswish `neck_attention` | 1 | 0.495274 | 0.619765 | 0.625374 | 0.901553 | 0.497899 | 0.752848 | 0.857271 | 0.945834 | **−0.008581** |
| qSiLU + Hardswish `masf` | 3 | 0.495153 | 0.619612 | 0.624241 | 0.899952 | 0.496179 | 0.752304 | 0.853939 | 0.945966 | **−0.010301** |
| qSiLU + Hardswish `backbone_attention` | 3 | 0.492225 | 0.618365 | 0.623216 | 0.903307 | 0.492962 | 0.753469 | 0.863688 | 0.942927 | **−0.013518** |

原始未四捨五入值、local delta、activation counts、hardware proxy 與 gate bool 都保存在
[Q3 summary](../../../yolo_activation/reports/q3-nonsilu-cpu/summary.json)。三者在上游 `−0.015`
activation-only gate 均通過；`neck_attention` 對四項 headline 幾乎無變化，因此是第一優先。

### 推論出的實作決策

- 第一個 Hardswish 量化 policy 應是
  `base=qsilu_pq@767918…6190e + region(neck_attention)=hardswish + A8 + Wn`。
- `masf` 與 `backbone_attention` 必須是另外兩個單區 policy；不可把三個單區 delta 相加。
- `neck_attention + masf` 只能列為後續新 policy，必須先做 matched PTQ；Q3 明確說此組合未實測。
- `detect_one2many`、`pose_flow`、`pose_one2many` 是 training-only graph，zero delta 不可列成
  deployment Hardswish 候選。

### 未量測

- Hardswish placement 加上 LSQ+ A8、W8–W4、Fixed SD4、ternary 後的 layer output error、Top-K
  overlap 與八項 mAP。
- 多區 Hardswish 組合。
- Hardswish 的 INT activation rounding、saturation、完整碼域或 target hardware latency／power。

因此「納入」只能表示進入候選矩陣，不能寫成已成為 activation winner。

## Checkpoint、SHA、指標與 parent 語意

### 建議 active roots

| 身分 | 可用 checkpoint 與 SHA-256 | 已量測指標 | 正確 parent 語意 |
|---|---|---|---|
| qSiLU | [`inference/best_joint.pt`](../../../yolo_activation/artifacts/runs/full35/short-recovery-v2-lr01-uniform-qsilu-pq-seed1/inference/best_joint.pt)，`7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e` | Q3 同 checkpoint CPU 八項最差 Δ `−0.008810`；上游 10-epoch final-epoch selector為 `−0.008635`，兩者裝置／baseline不同 | 可交付 non-SiLU checkpoint root；uniform qSiLU fallback與Q3 Hardswish regional policies共用此權重根 |
| poly_shift | [`inference/best_joint.pt`](../../../yolo_activation/artifacts/runs/full35/short-recovery-v2-lr01-uniform-poly-shift-seed1/inference/best_joint.pt)，`8783248a513329ae80e3f659e6f3ce617e4bcca9fca9530b1f95448de5e19713` | 本機 gate log 的最高且通過 best-joint 是 epoch 2，八項最差 Δ `−0.013650`；run 的 final epoch則為 `−0.030138`、未通過 | 僅本機 hardware-oriented experimental root；沒有被上游 release 發布，必須保存 checkpoint-level selector metadata |
| Hardswish regional（推薦） | **使用 qSiLU checkpoint** `767918…6190e` | 上表三個單區 policy 的最差 global Δ 分別 `−0.008581`／`−0.010301`／`−0.013518` | Hardswish 是 activation placement；不是另一份 weight checkpoint。policy id 必須帶 region，不能只叫 `hardswish--a8` |

qSiLU 與 poly_shift 的上游 run-level數值見
[最終 machine-readable 結果](../../../yolo_activation/reports/full35-activation-results.json)；上游公開 handoff
只發布 accepted SiLU 與 qSiLU，沒有發布 gate 淘汰候選權重，見
[最終分析的權重發行界線](../../../yolo_activation/reports/full35-activation-final-analysis.md#L165)。

poly_shift best-joint epoch 2 的八項 mAP50-95 是：COCO `0.496157`、person `0.620492`、
BBAT box `0.621601`、pose `0.901843`、ball box `0.493787`、bat box `0.749416`、
ball pose `0.859114`、bat pose `0.944573`。來源是
[epoch-0002 metrics](../../../yolo_activation/artifacts/runs/full35/short-recovery-v2-lr01-uniform-poly-shift-seed1/validation/epoch-0002/bittrue/metrics.json)
與[gate log](../../../yolo_activation/artifacts/runs/full35/short-recovery-v2-lr01-uniform-poly-shift-seed1/logs/gate.csv)。

### Uniform Hardswish 的替代路徑

本機另有一份 uniform Hardswish recovery export：

```text
/home/uxin/yolo/yolo_activation/artifacts/runs/full35/
short-recovery-v2-lr01-uniform-hardswish-seed1/inference/best_joint.pt
SHA-256: 79e0e4f615a7d8b82da4fd244165d2c035d7a523681833392d30b66e57177731
```

這份權重**可以作明確標為 experimental 的 uniform Hardswish parent，但不是 Q3 的 parent**。
本機 gate log 顯示 best-joint score 最高的 epoch 8 通過 `−0.015` gate，八項最差 Δ 是
`−0.014121`（COCO box）；對應 mAP50-95 為 COCO `0.483902`、person `0.607785`、
BBAT box `0.620591`、pose `0.900707`、ball box `0.499461`、bat box `0.741720`、
ball pose `0.866918`、bat pose `0.934496`。來源是
[epoch-0008 metrics](../../../yolo_activation/artifacts/runs/full35/short-recovery-v2-lr01-uniform-hardswish-seed1/validation/epoch-0008/bittrue/metrics.json)
與[gate log](../../../yolo_activation/artifacts/runs/full35/short-recovery-v2-lr01-uniform-hardswish-seed1/logs/gate.csv)。

但上游 frozen report 對 short-recovery job 使用 **final epoch** selector：epoch 9 的最差 Δ 是
`−0.016884`（BBAT bat pose），因此 run-level 結論是未通過，且該權重未發布。queue source 明定
未設定時 `metric_selection="final_epoch"`，而 `best_joint` 是另一種可選 selector；參見
[queue parser與selector](../../../yolo_activation/scripts/full35_queue.py#L90)及
[short-recovery jobs](../../../yolo_activation/training/full35/experiment-queue.yaml#L39)。

因此若主線採 uniform Hardswish，必須新增一份 checkpoint-level intake manifest，明寫
`selection=best_joint`、`epoch=8`、checkpoint SHA 與八項 metrics，並標為使用者覆寫上游
final-epoch promotion 結論；不可只把舊報告的 `−0.016884` 綁到 `best_joint.pt`。基於 Q3 的直接
證據與可重現性，本文仍建議先採 regional Hardswish。

## `poly_quality` 的排除邊界

### 從 active future matrix 排除

應移除：

- active parent registry 與 preparation CLI choices；
- V4／V5 未執行矩陣；
- V6 routing promotion、V7+ QAT shortlist及「numeric parent」角色；
- 現行文件中「三 parent 仍為 qSiLU／poly_quality／poly_shift」的現在式敘述。

### 歷史證據保留

不得刪除或重寫：

- `weight-views-poly-quality-a8-v2.json`；
- `weight-format-analysis-poly-quality-a8-main-v2.json`；
- V1–V3 delivery manifest中的既有 hash／2,960筆計數；
- 2026-08-29 activation-output smoke／preselection report；
- 2026-08-30 V1–V3報告與 Fixed SD4 v1 routing manifest。

這些檔案應標成 `historical_excluded_from_active_matrix_after_2026-08-31`。原因有二：第一，它們是
已完成實驗的真實結果；第二，舊 Fixed SD4 v1 manifest 的 rule 明確要求
`[qsilu_pq, poly_quality, poly_shift]`，若原地改名就會讓來源 hash 與「34層跨三 parent」語意失真。
[舊 SD4 manifest](../../artifacts/manifests/fixed-sd4-routing-candidates-v1.yaml#L10)

`poly_quality` 的排除是**研究決策**，不是「它從未有價值」的科學結論：上游 uniform zero-shot
最差 Δ `−0.005501` 曾通過，但 10-epoch final epoch 的 worst Δ `−0.020970` 未通過；既有 A8
numeric proxy也曾領先。這些相反證據更說明歷史資料必須保留，而不能為配合新方向刪掉。

## Q3 哪些內容可直接取用

| Q3內容 | 可否直接取用 | 在 `yolo_quantize` 的用法 |
|---|---|---|
| qSiLU checkpoint path／SHA | 可以 | 作 qSiLU root及Hardswish regional root |
| 190-site module path／11-region manifest | 可以，但需複製為有來源 hash的 intake contract | 建 regional policy；不得自行用名稱猜 path |
| training-only region分類 | 可以 | 不列 deployment節省，仍保留訓練 graph |
| 八項 metric keys與「matched baseline相減」 | 可以 | 和既有八指標 gate相容；保留 COCO person與 ball／bat class-level |
| Hardswish三個單區排序 | 可以作候選優先序 | `neck_attention` → `masf` → `backbone_attention` |
| 完整 COCO 5,000／BBAT5 683與 canonical資料契約 | 可以 | 正式 validation時沿用；diagnostic 32／64不能冒充完整結果 |
| CPU fail-closed guard模式 | 可以 | 必須在 import torch前隱藏 CUDA，並在執行期檢查 availability/count |
| operation-count proxy | 只可作排序 | 不可宣稱 latency、power、LUT、DSP或晶片面積改善 |

## Q3 哪些內容不能跨專案直接套用

1. **Q3 沒有 A8/W8。** `Bit-True weights backend` 是上游 Full35 evaluator 身分，不是本專案的
   LSQ+ A8、uniform W8或W4 PTQ結果。
2. **單區不能推算多區。** `neck_attention + masf` 尚未量測，三個通過 delta不能相加。
3. **regional Hardswish不能證明 uniform Hardswish。** 二者 checkpoint、activation counts與政策
   不同。
4. **CPU raw mAP不能和既有其他裝置 baseline混減。** Q3 accepted CPU COCO box為 `0.498460`，
   既有凍結 accepted selector為 `0.498022`；只能在各自 matched evaluator內解讀 delta。
5. **`−0.015`不能覆蓋本專案 gate。** 它可保留為 activation-only intake gate；本專案仍需同時
   執行 total drop `−0.04`、W8 matched-parent incremental `−0.01`及 sham drift `0.01`契約。
6. **hardware proxy不是硬體實測。** Hardswish雖是標準 fused op，也仍需目標 backend/export/profile。
7. **Q3 沒有 activation-aware PTQ／QAT證據。** 任何 Hardswish W-bit結果仍是未量測。

## 對 V0–V5 契約與既有 V2 的影響

### V0：只追加 amendment

- 保留既有 publication、PTQ v1、V1–V3 lineage與 hashes。
- 新增 Q3 remote commit/blob、summary SHA、Hardswish checkpoint或qSiLU-root policy的來源欄位。
- 不把 2026-08-31決策回寫成「2026-08-30當時已排除 poly_quality」。

### V1：catalog不變，policy/view需版本化

- corrected 148-layer catalog、training-only／binary protected分類不受 activation更動影響。
- regional Hardswish需要把 `Full35ActivationPolicy` 從單一 uniform `activation` 擴為
  `default_activation + region_assignments`；現行 adapter雖支援 uniform `hardswish`名稱，build時仍只
  呼叫 `uniform_full35_policy(...)`，因此尚不能重建Q3 mixed policy。
  [現行 adapter](../../src/yolo_quantize/full35_adapter.py#L30)
- qSiLU與regional Hardswish共用權重，weight path與state SHA相同；但 activation graph不同，仍需新的
  CPU forward/fuse parity manifest。這是尚未量測的新V1 policy證據。

### V2：8,880筆不失效，但不再等於 active matrix

- 已完成三 parent × 2 views共8,880筆是歷史事實，不可改成「已完成Hardswish」。
- active qSiLU + poly_shift已有5,920筆可繼續使用；poly_quality的2,960筆轉為 historical。
- regional Hardswish使用同一 qSiLU checkpoint時，**weight-only reconstruction數學上與qSiLU相同**；
  這可避免重複宣稱另一批獨立weight測量，但不能據此推論activation-output或mAP相同。
- 若改選uniform Hardswish本機checkpoint `79e0…7731`，則它是不同權重，必須另跑V1 dual view與
  V2 2,960筆CPU static analysis；目前未量測。
- Fixed SD4 v1的34層候選保持歷史。新active routing須另發v2 manifest；不可原地把
  `required_parents`改掉。因舊報告顯示三parent各自winner集合相同，移除poly_quality後34層集合
  **推論上可能不變**，但仍要用新規則與來源hash重新產出，不能直接改標籤。

### V3：資料與八指標keys可沿用，baseline不可混用

- 32／64 diagnostic manifest與八項key不需改。
- Q3的完整CPU baseline只可作activation intake evidence；正式W-bit candidate仍對自己的matched
  activation parent及accepted Full35走既有gate。
- 建議新增activation-policy identity gate，至少要求 checkpoint SHA、default activation、完整
  region assignments、190-site manifest SHA與training-only exclusions。

### V4：由固定15格改為分階段、最多25格

不建議用一個模糊的「Hardswish parent」直接取代poly_quality後仍宣稱15格已完整。證據一致的設計是：

1. V4A：uniform qSiLU與uniform poly_shift × W8/W7/W6/W5/W4，共10格。
2. V4B bridge：三個Hardswish單區policy先各跑W8，共3格。
3. 只有W8通過的Hardswish policy才展開W7–W4，每個最多再4格；V4總上限為25格。
4. `neck_attention + masf`是額外未量測組合，不納入第一批25格；待兩個單區W8都通過才建立。

此設計既納入Hardswish，也利用Q3排序減少無證據的笛卡兒積。若專案必須維持恰好15格，唯一可清楚
解釋的縮減版是先把 `neck_attention` Hardswish當第三條policy；但需明記它不是uniform parent，且
`masf`／`backbone_attention`仍在候選佇列。

### V5：不要把activation region與weight region混成一個軸

- 原150格公式的parent軸已失效。
- Hardswish placement與weight quantization region是兩個獨立軸；若直接用5個policy × 10 weight
  regions × 5 bits會膨脹至250格，且大多沒有必要。
- 建議只讓V4通過的policy進V5 successive racing，沿用10個weight regions；每個stratum保留
  sentinel。Q3的training-only activation regions不參與deployment placement。

## 最終建議

新的現行規畫應使用下列語意，而不是只做名稱替換：

```text
active checkpoint roots:
  - qsilu_pq @ 767918...6190e
  - poly_shift @ 878324...9713  # local experimental

active activation policies:
  - uniform qsilu_pq
  - uniform poly_shift
  - qsilu_pq root + hardswish at neck_attention
  - qsilu_pq root + hardswish at masf
  - qsilu_pq root + hardswish at backbone_attention

excluded from future promotion:
  - poly_quality

historical evidence retained:
  - all completed poly_quality V0-V2 artifacts and hashes
```

這個方案完整吸收Q3的新證據，又不把未量測的Hardswish × A8 × W-bit結果包裝成完成。下一個可執行
工作應先是CPU-only policy intake、manifest與dual-view parity；GPU PTQ仍需另外授權。

