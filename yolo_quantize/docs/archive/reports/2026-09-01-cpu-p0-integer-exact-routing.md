# Full35 量化 CPU P0：整數邊界契約與 exact W4／Fixed-SD4 routing v3

日期：2026-09-01  
狀態：CPU 前置完成，停止於第一個需要 GPU 的 V4 W8 bridge 前  
執行限制：`CUDA_VISIBLE_DEVICES=-1`；未跑 calibration、mAP validation、QAT、訓練或 export

## 結論

本輪完成兩個原本阻擋後續實驗的 CPU 項目：

1. 建立 Full35 hybrid-integer boundary 的 CPU reference Interface，綁定 qSiLU＋A8 checkpoint 與 BN-folded deployment state；Add／Concat、RNE、saturation、INT32 MAC bound與保護島都有可稽核契約。
2. 對 qSiLU、Hardswish、poly_shift 三個 active parent 完成 exact uniform W4／Fixed SD4 全層分析，並發布 cross-parent、cross-view routing v3。

下一步是 V4 W8 GPU bridge：凍結 LSQ+ activation scale／offset、量 branch saturation與輸出尺度、做 layer-output NRMSE／Top-300 overlap，再跑含 COCO person 的八指標 validation。本輪沒有啟動該步。

## 一、IntegerBoundaryContract

### Module 與 Interface

`IntegerBoundaryContract`是一個深 Module：呼叫端只需提供 affine integer tensor spec；Module 內部統一處理 nearest-even、clamp、requant、Add、Concat與累加器上界。`inspect_full35` Adapter把真實 BN-folded graph轉成 hash-bound manifest，不讓 runner或報告各自猜邊界。

CPU reference 支援兩種 affine 語意：

- 一般 zero-point：`x = (q - z) × scale`。
- LSQ+ arbitrary offset：`x = q × scale + offset`。

後者很重要。現有 LSQ+ 的 learned offset通常不能假設等於整數 zero-point；若靜默套一般公式，qSiLU邊界會建模錯誤。真正整數 Conv需要在 calibration凍結scale／offset後，把offset correction與padding zero語意納入per-output-channel INT32 bias，再以 GPU fake-quant結果逐邊界核對。

### 真實 graph 盤點

| 項目 | 數量 |
|---|---:|
| BN-folded deployment weight sites | 148 |
| protected Binary Q/K weight sites | 4 |
| activation-output LSQ+ quantizers | 124 |
| top-level Concat modules | 4 |
| reviewed core Concat operations | 21 |
| reviewed core residual Add operations | 20 |
| MASF islands | 1 |
| attention／BinaryScore／PWL islands | 2／2／2 |
| Detect／Pose output islands | 2 |

Add policy是所有輸入先requant到註冊output scale，再以INT32相加、RNE並saturate；Concat policy是各branch先對齊共同output scale再沿channel維度串接。MASF與attention不是假裝成一般INT8 operator，而是保留明確Q/DQ ingress／egress的float或custom-kernel island。Binary Q/K不進一般weight quantizer；PWL denominator仍是float reference，沒有宣稱完整全整數attention。

148個W8/A8部署站點都通過保守raw-code INT32 MAC bound。最大上界為`150,405,120`，出現在`graph.model.5.conv`，低於signed INT32最大值`2,147,483,647`。這只證明MAC accumulator靜態上界；calibrated bias correction仍待下一個GPU stage。

## 二、Exact W4／Fixed-SD4 靜態矩陣

每個active parent固定同一公平矩陣：

```text
148 layers × 2 views × 4 formats = 1,184 cells

formats:
  uniform W4 per-output-channel mse_grid_v1
  uniform W4 per-output-channel optimal_scaled_codebook
  Fixed SD4 per-output-channel mse_grid_v1
  Fixed SD4 per-output-channel optimal_scaled_codebook
```

三個parent共`3,552`筆新measurement；舊8,880筆與routing v2沒有覆寫。routing比較的exact W4與exact Fixed SD4均為4-bit code、同per-output-channel scale數、同code bytes與metadata bytes。

### Exact solver對舊十點grid的影響

以下是所有master＋deployment layers的element-weighted SSE；百分比表示exact相對grid的SSE下降，不是mAP增益。

| Parent | uniform W4 SSE下降 | Fixed SD4 SSE下降 |
|---|---:|---:|
| qSiLU | 1.6092% | 4.1595% |
| Hardswish | 1.5882% | 4.1659% |
| poly_shift | 1.6049% | 4.1576% |

三個parent × 兩個view × 148層的888個uniform cells與888個Fixed SD4 cells中，exact皆不劣於`mse_grid_v1`。因此舊grid可保留作ablation，但不能再當最強baseline。

## 三、Fixed-SD4 routing v3

一層只有在三個parent、master與deployment共六個比較中，exact Fixed SD4 MSE都嚴格小於exact uniform W4，才進stable routing。

| 結果 | v2 grid | v3 exact |
|---|---:|---:|
| cross-parent、cross-view stable layers | 34 | 36 |

v3保留全部34個舊候選，新增：

- `graph.model.16.m.0.cv3.conv`
- `graph.model.23.detect_head.one2one_cv3.2.2`

36層region分布為backbone_early 4、backbone_deep 4、neck 20、neck_attention_safe 1、detect_one2one_tower 6、detect_one2one_predictor 1。

若只在這36層用exact Fixed SD4、其餘維持exact W4，deployment static NRMSE相對all-W4下降約1.10%–1.15%：

| Parent | all exact W4 NRMSE | hybrid NRMSE | 相對下降 |
|---|---:|---:|---:|
| qSiLU | 0.095077 | 0.093979 | 1.1547% |
| Hardswish | 0.094973 | 0.093925 | 1.1028% |
| poly_shift | 0.095069 | 0.093980 | 1.1455% |

NRMSE是weight reconstruction proxy，不是layer output error，更不是COCO／BBAT mAP。v3的`execution_authorized=false`、`map_validation_run=false`；36層只作下一階段候選，不構成policy promotion。

## 四、Artifact 與 SHA-256

| Artifact | SHA-256 |
|---|---|
| `artifacts/reports/weight-format-analysis-qsilu-pq-a8-exact-w4-sd4-v1.json` | `26247cf4bb8b88253e3438d45c2797b5b8eb948a1d3761b75bcf46a2923a86b2` |
| `artifacts/reports/weight-format-analysis-hardswish-a8-exact-w4-sd4-v1.json` | `8a24e5fabd87542ae28a9bfd4c3081ff7156d9673e46a29569a7bea4f6abe18d` |
| `artifacts/reports/weight-format-analysis-poly-shift-a8-exact-w4-sd4-v1.json` | `e013b3cc78ad8e470ec89bf935f84d37f051282696949cd8657425fcdc6c7def` |
| `artifacts/manifests/full35-integer-boundary-contract-qsilu-pq-a8-v1.json` | `db0ef0ade253e09701907dd1235c9ab56dea44883dc6d52bb189b0c6a8254e3b` |
| `artifacts/manifests/fixed-sd4-routing-candidates-v3.yaml` | `afbefae48e3bafd87a2842da205c3f7a5b5d081f5351a9caf480ecacb1ed98f4` |
| `artifacts/manifests/exact-w4-sd4-cpu-delivery-v1.yaml` | `158388aed80e6ae1490ab2dbaf6de6be4138b2bbff44d42e25b1ca9f2d607abe` |

## 五、停止線與未解風險

- 已完成CPU reference semantics，不等於已有target multiplier／shift lowering或native integer kernel。
- LSQ+ scale／offset、Add／Concat output scale與實際saturation需要GPU calibration資料；這是下一個stage，不是本輪遺漏執行。
- PWL denominator、MASF與attention目前仍有protected island；沒有HLS／RTL／上板latency、power或resource結果。
- fold-aware effective-weight contract只阻擋QAT，未在本輪為了PTQ先猜一個方案；任何QAT前仍須選`fold_aware_shadow_qat`或`folded_graph_qat`。
- routing v3只允許候選進GPU output sensitivity；必須通過八指標gate，其中含COCO person，才可進一步選擇。

## 六、驗證

- 最終完整CPU回歸：`100 passed in 11.08s`，環境固定`CUDA_VISIBLE_DEVICES=-1`。
- `ruff check src tests scripts`通過；`ruff format --check src tests scripts`回報44個檔案均已格式化。
- 使用拒絕duplicate key的loader解析22份YAML，並解析20份JSON；delivery內5個artifact SHA全部重算一致。
- 檢查31份Markdown的120個本地連結，缺失為0。
- 三個parent CPU sweep每份1,184 cells；執行時間約63.8–67.7秒、峰值RSS約1.26–1.29 GB。
