# Activation 報告圖表

本目錄保存權威整合報告使用的高解析度 PNG、可縮放 SVG 與對應 CSV。最終比較依使用者要求排除
ReLU；ReLU 的歷史診斷結果仍留在 `../full35-activation-results.json`，沒有從原始證據刪除。

## 圖表與資料

| 主題 | PNG／SVG | 機器可讀資料 | 口徑 |
| --- | --- | --- | --- |
| Full35 四項精度長條圖 | `activation-accuracy-overview.*` | `activation-accuracy-summary.csv` | COCO／Person／BBAT5 box／pose 的 Bit-True zero-shot 與 recovery raw、delta；正式 gate 仍使用八項 |
| 中央函數與誤差放大 | `activation-central-error-enlarged.*` | `activation-function-metrics.csv` | 圖為 `[-4,4]`；表為 `[-8,8]`、400,001 點 float64 |
| 運算量解析估計 | `activation-operation-estimate.*` | `activation-operation-estimate.csv` | 每元素靜態 proxy 與 640×640 deployment elements 投影 |

運算量圖不是 latency、功耗或 FPGA／ASIC 資源實測。不同種類的乘法、shift-add、比較、sigmoid 與
記憶體操作不能直接相加成單一 FLOP 數；qSiLU 的 eager reference 與預期 fused coefficient-select
資料路徑也分開標示。

## 重新產生

```bash
cd /home/uxin/yolo/yolo_activation
/home/uxin/yolo/.venv/bin/python scripts/generate_activation_report_figures.py
```

產生器讀取 `reports/full35-activation-results.json`、`src/activation_lab/activations.py`、
`src/activation_lab/hardware.py`，並在本機 raw profiling artifact 存在時核對 Detect／Pose deployment
activation elements。圖表產生不會啟動 GPU、訓練、PTQ 或 QAT，也不會改動 COCO2017 或
Canonical BBAT5 v1。

返回[權威整合報告](../completed-activation-integrated-report.md)。
