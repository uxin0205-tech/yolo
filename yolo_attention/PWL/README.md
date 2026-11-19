# PWL final normalization validation

本資料夾只驗證固定 YOLO26m Binary Attention 的 Exact Softmax、Float PWL 與
Q8.8 Bit-True PWL。Binary QK、PoT scale、decomposed 2D bias、V、`PV + PE(V)`
及 projection 均不變，也不包含訓練、BDCN、P/V quantization 或其他 normalization。

固定 parent 是 `artifacts/runs/v1-br/ultralytics/weights/best.pt`，資料與 evaluator
是 `configs/evaluation/coco2017.yaml`。正式 queue extension：

```bash
python -m yolo_attention.cli queue append-pwl-validation --json
python -m yolo_attention.cli queue run-next --json            # dry-run
python -m yolo_attention.cli queue run-next --execute --json  # 需要明確 GPU 授權
```

正式 queue 的兩個 jobs 已在 RTX 5090 上完成；再次執行 append 會 fail closed。
最終選定 `[-10,0]`、20 segments、$\Delta=0.5$，Exact／Float／Bit-True
mAP50-95 為 0.506658／0.506730／0.506737。


流程：

```text
pwl-score-analysis
  ├─ Exact Softmax COCO validation
  ├─ two-site/per-head centered-score histogram
  └─ [-8,0] 與 [-10,0] 的 numerical error reference
          ↓
pwl-compare
  ├─ tail mass <= 0.1%：選 [-8,0], 16 segments
  ├─ tail mass >  0.1%：選 [-10,0], 20 segments
  └─ 同一 parent/val 比較 Float PWL 與 Bit-True PWL
```

`configs/` 是人工維護的設定；`results/` 保存整理後的 CSV、SVG 與報告並提交 Git；
完整 raw provenance 放在 `artifacts/runs/pwl-*`。本專案 Bit-True 規格沿用論文參數，
但 rounding、UQ1.15 endpoints 與 saturation 是 project-derived specification，因論文
未公開完整 RTL 細節。

完整結論見 [results/REPORT.md](results/REPORT.md)；機器可讀比較見
[results/comparison.csv](results/comparison.csv)，score/head 分布與圖表也位於同一資料夾。
